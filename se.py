import sys
import io
import os
import re
import gzip
import csv
import time
import json
import pickle
import numpy as np
import pandas as pd
import builtins
import math
from time import time
import itertools
from pathlib import Path
import nltk
from nltk.stem.porter import *
from nltk.corpus import stopwords
from nltk.util import ngrams
from nltk.corpus import wordnet

nltk.download("wordnet")
from datetime import datetime
import operator
from itertools import islice, count, islice, count, groupby
from contextlib import closing
from io import StringIO
from pathlib import Path
from operator import itemgetter
from google.cloud import storage

import google.auth
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

from scipy.sparse import csr_matrix
from numpy import dot
from numpy.linalg import norm

from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter, OrderedDict, defaultdict

from inverted_index_gcp import InvertedIndex, MultiFileReader

TUPLE_SIZE = 6
english_stopwords = frozenset(stopwords.words("english"))
corpus_stopwords = [
    "category",
    "references",
    "also",
    "external",
    "links",
    "may",
    "first",
    "see",
    "history",
    "people",
    "one",
    "two",
    "part",
    "thumb",
    "including",
    "second",
    "following",
    "many",
    "however",
    "would",
    "became",
    "make",
    "accordingly",
    "hence",
    "namely",
    "therefore",
    "thus",
    "consequently",
    "meanwhile",
    "accordingly",
    "likewise",
    "similarly",
    "notwithstanding",
    "nonetheless",
    "despite",
    "whereas",
    "furthermore",
    "moreover",
    "nevertheless",
    "although",
    "notably",
    "notwithstanding",
    "nonetheless",
    "despite",
    "whereas",
    "furthermore",
    "moreover",
    "notably",
    "hence",
]

all_stopwords = english_stopwords.union(corpus_stopwords)
RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)


class search_engine:
    corpus_size = 6348910

    def __init__(self):
        """
        Search engine initialzation. init dicts without loading the indecies this done at load_index function.
        """
        self.idf_dict = dict()
        self.index_dict = dict()
        self.bm25_dict = dict()
        self.weights = dict()
        self.stemmer = PorterStemmer()

        self.candidates_term = dict()

    def calculate_index_idf(self, index_name):
        """
        calc index idf in-order to imporve performance. this done only once at initialization.
        """
        try:
            idf_dict = {}
            if len(self.index_dict[index_name].dl) == 0:
                self.index_dict[index_name].dl = self.index_dict[index_name].DL

            for term, freq in self.index_dict[index_name].df.items():
                idf = math.log(
                    len(self.index_dict[index_name].dl) / (freq + 0.0000001), 10
                )
                idf_dict[term] = idf
            return idf_dict
        except Exception as e:
            print("error when calling calculate_index_idf")
            raise e

    def load_index(self, in_index, in_index_name):
        """
        load an index and keep it in a dict objects. This done only once at initialization
        """
        try:
            self.index_dict[in_index_name] = in_index
            self.idf_dict[in_index_name] = self.calculate_index_idf(in_index_name)
            self.bm25_dict[in_index_name] = BM25_from_index(in_index, k1=1.5, b=0.75)
        except Exception as e:
            print("error when calling load_index. index_name {}".format(in_index_name))
            raise e

    def load_all_indecies_from_bucket(self, load_only_title=False):
        """
        load all indices from buckets and keep them in a dict objects. This done only once at initialization
        """
        try:
            client = storage.Client()
            bucket = client.get_bucket("bx_pickles")

            print("loading title indecies")
            title_index = pickle.loads(
                bucket.get_blob("bx_title_index_index.pkl").download_as_string()
            )
            title_bigram_index = pickle.loads(
                bucket.get_blob("bx_title_bigram_index_index.pkl").download_as_string()
            )
            self.load_index(title_index, "title_index")
            self.load_index(title_bigram_index, "title_bigram_index")

            if load_only_title == False:
                print("loading rest of the indecies")

                body_index = pickle.loads(
                    bucket.get_blob("bx_body_index_index.pkl").download_as_string()
                )
                body_bigram_index = pickle.loads(
                    bucket.get_blob(
                        "bx_body_bigram_index_index.pkl"
                    ).download_as_string()
                )
                idx_body_nf = pickle.loads(
                    bucket.get_blob("bx_body_index_nf_index.pkl").download_as_string()
                )
                anchor_index = pickle.loads(
                    bucket.get_blob("bx_anchor_index_index.pkl").download_as_string()
                )
                body_no_stem_index = pickle.loads(
                    bucket.get_blob(
                        "bx_body_index_no_stemming_index.pkl"
                    ).download_as_string()
                )
                title_no_stem_index = pickle.loads(
                    bucket.get_blob(
                        "bx_title_index_no_stemming_index.pkl"
                    ).download_as_string()
                )

                self.load_index(body_index, "body_index")
                self.load_index(body_bigram_index, "body_bigram_index")
                self.load_index(anchor_index, "anchor_index")
                self.load_index(idx_body_nf, "body_index_nf")
                self.load_index(body_no_stem_index, "body_no_stem_index")
                self.load_index(title_no_stem_index, "title_no_stem_index")

            # loading title dictionary
            self.title_dict = pickle.loads(
                bucket.get_blob("title_dict.pkl").download_as_string()
            )

            print("loading page views")
            # loading pageview
            self.pv = pickle.loads(bucket.get_blob("wid2pv.pkl").download_as_string())

            # page ranks to dict
            print("loading page rank")
            prgz = bucket.get_blob("pagerank.csv.gz").download_as_string()
            decompressed_file = gzip.decompress(prgz)
            csv_reader = csv.reader(io.StringIO(decompressed_file.decode("utf-8")))
            self.pr = {int(row[0]): float(row[1]) for row in csv_reader}
        except Exception as e:
            print("error when calling load_all_indecies_from_bucket")
            raise e

    def init_engine(self, load_only_title=False):
        """
        Initilizae search engine
        """
        self.load_all_indecies_from_bucket(load_only_title)

    def get_posting_list(self, index_name, w):
        """
        get posting list from index - reading related bin file using the MultiFileReader
        """
        with closing(MultiFileReader()) as reader:
            locs = self.index_dict[index_name].posting_locs[w]
            b = reader.read(
                locs,
                self.index_dict[index_name].df[w] * TUPLE_SIZE,
                self.index_dict[index_name].bucket_name,
            )
            posting_list = []
            for i in range(self.index_dict[index_name].df[w]):
                doc_id = int.from_bytes(b[i * TUPLE_SIZE : i * TUPLE_SIZE + 4], "big")
                tf = int.from_bytes(b[i * TUPLE_SIZE + 4 : (i + 1) * TUPLE_SIZE], "big")
                posting_list.append((doc_id, tf))
            return posting_list

    def get_candidate_documents_and_scores(
        self, query_to_search, index_name, partial_doc_list=None
    ):
        """
        Generate a dictionary representing a pool of candidate documents for a given query. This function will go through every token in query_to_search
        and fetch the corresponding information (e.g., term frequency, document frequency, etc.') needed to calculate TF-IDF from the posting list.
        Then it will populate the dictionary 'candidates.'
        For calculation of IDF, use log with base 10.
        tf will be normalized based on the length of the document.

        Parameters:
        -----------
        query_to_search: list of tokens (str). This list will be preprocessed in advance (e.g., lower case, filtering stopwords, etc.').
                         Example: 'Hello, I love information retrival' --->  ['hello','love','information','retrieval']

        index_name:           inverted index name

        partial_doc_list: use only this list of docs if provided

        Returns:
        -----------
        dictionary of candidates. In the following format:
                                                                   key: pair (doc_id,term)
                                                                   value: tfidf score.
        """
        # return index.get_candidate_documents(query_to_search,index)
        candidates = {}
        candidates_doc_tfidf = {}
        corpus_size = len(self.index_dict[index_name].dl)
        for term in np.unique(query_to_search):
            if term in self.index_dict[index_name].df:
                list_of_doc = self.get_posting_list(index_name, term)
                if partial_doc_list is not None:
                    list_of_doc = [
                        value for value in list_of_doc if value in partial_doc_list
                    ]

                normlized_tfidf = [
                    (
                        doc_id,
                        (freq / self.index_dict[index_name].dl[doc_id])
                        * math.log10(
                            corpus_size / self.index_dict[index_name].df[term]
                        ),
                    )
                    for doc_id, freq in list_of_doc
                ]

                #                 for doc_id, tfidf in normlized_tfidf:
                #                     candidates_doc_tfidf[doc_id] =  candidates_doc_tfidf.get(doc_id,0) + tfidf
                #                     candidates[(doc_id,term)] = candidates.get((doc_id,term),0) + tfidf

                for doc_id, tfidf in normlized_tfidf:
                    if doc_id in candidates_doc_tfidf.keys():
                        candidates_doc_tfidf[doc_id] = (
                            candidates_doc_tfidf[doc_id] + tfidf
                        )
                        if (doc_id, term) in candidates.keys():
                            candidates[(doc_id, term)] = (
                                candidates[(doc_id, term)] + tfidf
                            )
                        else:
                            candidates[(doc_id, term)] = tfidf
                    else:
                        candidates_doc_tfidf[doc_id] = tfidf
                        candidates[(doc_id, term)] = tfidf

        docs_filtered = dict(
            sorted(
                candidates_doc_tfidf.items(), key=lambda item: item[1], reverse=True
            )[:2000]
        )
        candidates = {k: v for k, v in candidates.items() if k[0] in docs_filtered}
        candidates = dict(
            sorted(candidates.items(), key=lambda item: item[1], reverse=True)
        )
        return candidates, docs_filtered

    def get_query_vector(self, query, index_name):
        """
        generates and returns a vector of zeros for words not in the query and tfidf scores for query words
        @params:
        query: list of tokens
        index: inverted index object
        idf_dict: a dictionary with term:idf score items
        """
        try:
            words = list(self.index_dict[index_name].df.keys())
            Q = np.zeros(len(self.index_dict[index_name].df))
            counted_query = Counter(query)
            for term, freq in counted_query.items():
                if term not in words:
                    continue
                tf = freq / len(query)
                idf = self.idf_dict[index_name][term]
                idx = words.index(term)
                Q[idx] = tf * idf
            return Q
        except Exception as e:
            print(
                "error when calling get_query_vector query:{}, index_name:{}".format(
                    query, index_name
                )
            )
            raise e

    def get_candidates(self, query, index_name):
        """
        TEMPORARY returns a set of relevant docments for query as well as a dictionary that keeps relevant documents for each term.
        @params
        query: a list of tokens
        index: an inverted index object

        return:
            a set of relevant documents and a dictionary with items in the form of term:list_of_doc_ids
        """
        try:
            candidates = set()
            cadidates_for_term = {term: [] for term in np.unique(query)}
            for term in np.unique(query):
                if term in self.index_dict[index_name].df:
                    if (index_name, term) in self.candidates_term.keys():
                        unique_docs, cadidates_for_term[term] = self.candidates_term[
                            (index_name, term)
                        ]
                    else:
                        pl = self.get_posting_list(index_name, term)
                        unique_docs = set([i[0] for i in pl])
                        cadidates_for_term[term] = dict(pl)
                        self.candidates_term[(index_name, term)] = (
                            unique_docs,
                            cadidates_for_term[term],
                        )
                    candidates = candidates.union(unique_docs)
            return candidates, cadidates_for_term
        except Exception as e:
            print(
                "error when calling get_candidates query:{}, index_name:{}".format(
                    query, index_name
                )
            )
            raise e

    def candidates_tfidf_cosine_scores(self, query, index_name):
        words = list(self.index_dict[index_name].df.keys())
        candidates_vector_items = defaultdict(list)
        candidates_score = {}
        candidates = self.get_candidates(query, index_name)[0]
        Q = self.get_query_vector(query, index_name)
        Qnorm = norm(Q)

        for term in np.unique(query):
            if term not in words:
                continue
            pl_dict = dict(self.get_posting_list(index_name, term))
            idf = self.idf_dict[index_name][term]
            idx = words.index(term)
            for doc_id in candidates:
                if doc_id not in pl_dict:
                    continue
                freq = pl_dict[doc_id]
                tf = freq / self.index_dict[index_name].dl[doc_id]
                pair = (idx, tf * idf)
                candidates_vector_items[doc_id].append(pair)

        for doc_id, scores in candidates_vector_items.items():
            total = 0
            for idx, score in scores:
                total += score * Q[idx]
            cosine_similariy = total / (Qnorm**2)
            candidates_score[doc_id] = cosine_similariy

        return candidates_score

    def binary_score(self, query, term_docs_dict, doc_id):
        """
        return the binary scores for a document based on how many query words it contains
        """
        score = 0
        for term in query:
            if doc_id in term_docs_dict[term]:
                score += 1
        return score

    def seach_titles(self, query_tokens, candidates=None, candidates_dict=None):
        """
        Binary search for titles
        """
        if candidates is None:
            candidates, candidates_dict = self.get_candidates(
                query_tokens, "title_index"
            )

        return self.get_top_n(
            dict(
                [
                    (doc, self.binary_score(query_tokens, candidates_dict, doc))
                    for doc in candidates
                ]
            ),
            N=9999999,
        )

    def search_body(self, query_tokens, candidates=None, candidates_dict=None):
        """
        Binary search for body
        """
        if candidates is None:
            candidates, candidates_dict = self.get_candidates(
                query_tokens, "body_index_nf"
            )

        return self.get_top_n(
            dict(
                [
                    (doc, self.binary_score(query_tokens, candidates_dict, doc))
                    for doc in candidates
                ]
            ),
            N=9999999,
        )

    def full_search(self, query):
        """
        Full search warpper, calling full_search_ with a given/best params
        """
        params = {
            "max_docs_from_binary_title": 4433,
            "max_docs_from_binary_body": 1349,
            "bm25_body_weight": 5.175842946028495,
            "bm25_title_weight": 2.297919942629382,
            "bm25_body_bi_weight": 0.9433857458903415,
            "bm25_title_bi_weight": 5.43406860036612,
            "body_cosine_score": 4.925136475255385,
            "title_cosine_score": 0.29963143927827507,
            "pr_weight": 0.34839354014586377,
            "pv_weight": 4.740913798917137,
        }

        return self.full_search_(query, params)

    def fast_cosine_search(
        self, tokens, index_name, N, stem=True, candidates=None, unique_candidates=None
    ):
        cs_dict = self.fast_cosine_calc(
            query_to_search=tokens,
            index_name=index_name,
            candidates=candidates,
            unique_candidates=unique_candidates,
        )
        return self.get_top_n(cs_dict, N)

    def full_search_(self, query, params):
        try:
            start = datetime.now()
            # step 1 tokenize the query
            tokens = self.tokenize(query, bigram=False, stem=True)
            tokens_bi_gram = self.tokenize(query, bigram=True, stem=True)
            #         print('after tokenizing',tokens,tokens_bi_gram)

            # step 2 call for candidets from all indices
            title_candidates, title_dict = self.get_candidates(tokens, "title_index")
            print("title_candidates:", len(title_candidates))

            body_candidates, body_dict = self.get_candidates(tokens, "body_index")
            print("body_candidates:", len(body_candidates))

            body_candidates_bi_gram, body_bi_gram_dict = self.get_candidates(
                tokens_bi_gram, "body_bigram_index"
            )
            print("body_candidates_bi_gram:", len(body_candidates_bi_gram))

            title_candidates_bi_gram, title_bi_gram_dict = self.get_candidates(
                tokens_bi_gram, "title_bigram_index"
            )
            print("title_candidates_bi_gram:", len(title_candidates_bi_gram))

            if len(body_candidates_bi_gram) > 200:
                body_candidates = body_candidates_bi_gram

            anchor_candidates = set()

            # step 3 combine results (union or intersection with minimum)
            if len(title_candidates) > 0:
                title_candidates = set(
                    list(zip(*self.seach_titles(tokens, title_candidates, title_dict)))[
                        0
                    ][: params["max_docs_from_binary_title"]]
                )

            if (len(body_candidates) > 0) and (len(body_candidates_bi_gram) <= 200):
                body_candidates = set(
                    list(zip(*self.search_body(tokens, body_candidates, body_dict)))[0][
                        : params["max_docs_from_binary_body"]
                    ]
                )
            else:
                body_candidates = body_candidates_bi_gram

            # cosine similarity section
            # body
            cos_candidates_to_add = set()
            if len(body_candidates) < 10:
                (
                    self.cos_candidates,
                    self.cos_unique_candidates,
                ) = self.get_candidate_documents_and_scores(
                    tokens, "body_index", body_candidates
                )
                if len(self.cos_unique_candidates) > 0:
                    cos_candidates_to_add = set(self.cos_unique_candidates.keys())
            else:
                self.cos_unique_candidates = set()

            (
                self.cos_title_candidates,
                self.cos_title_unique_candidates,
            ) = self.get_candidate_documents_and_scores(
                tokens, "title_index", title_candidates
            )

            # title
            if len(self.cos_title_unique_candidates) > 0:
                cos_title_candidates_to_add = set(
                    self.cos_title_unique_candidates.keys()
                )

            else:
                cos_title_candidates_to_add = set()

            # after binary search
            print("after binary search title_candidates:", len(title_candidates))
            print("after binary search body_candidates:", len(body_candidates))
            print("anchor_candidates:", len(anchor_candidates))

            print("cos_candidates_to_add:", len(cos_candidates_to_add))

            self.all_relevent_docs = set().union(
                *[
                    body_candidates,
                    title_candidates,
                    body_candidates_bi_gram,
                    title_candidates_bi_gram,
                    anchor_candidates,
                    cos_candidates_to_add,
                    cos_title_candidates_to_add,
                ]
            )

            if len(self.all_relevent_docs) == 0:
                tokens = self.get_synonyms(query)
                title_candidates, title_dict = self.get_candidates(
                    tokens, "title_index"
                )
                if len(title_candidates) > 0:
                    title_candidates = set(
                        list(
                            zip(
                                *self.seach_titles(tokens, title_candidates, title_dict)
                            )
                        )[0][: params["max_docs_from_binary_title"]]
                    )

                    self.all_relevent_docs = title_candidates
                else:
                    return []

            self.bm25_body_score = self.bm25_dict["body_index"].search_with_candidates(
                tokens, self.all_relevent_docs
            )

            self.bm25_title_score = self.bm25_dict[
                "title_index"
            ].search_with_candidates(tokens, self.all_relevent_docs)

            self.title_cosine_score = dict(
                self.fast_cosine_search(
                    tokens,
                    "title_index",
                    params["max_docs_from_binary_title"],
                    True,
                    self.cos_title_candidates,
                    self.cos_title_unique_candidates,
                )
            )

            if len(self.cos_unique_candidates) > 0:
                self.body_cosine_score = dict(
                    self.fast_cosine_search(
                        tokens,
                        "body_index",
                        params["max_docs_from_binary_body"],
                        True,
                        self.cos_candidates,
                        self.cos_unique_candidates,
                    )
                )
            else:
                self.body_cosine_score = self.title_cosine_score

            # bi-gram calc section
            self.bi_gram_rel_docs = set().union(
                *[body_candidates_bi_gram, title_candidates_bi_gram]
            )

            self.bm25_body_bi_score = self.bm25_dict[
                "body_bigram_index"
            ].search_with_candidates(tokens_bi_gram, self.bi_gram_rel_docs)

            self.bm25_title_bi_score = self.bm25_dict[
                "title_bigram_index"
            ].search_with_candidates(tokens_bi_gram, self.bi_gram_rel_docs)

            self.pr_score = dict(self.pr_scores(tokens, self.all_relevent_docs))
            self.pv_score = dict(self.pv_scores(tokens, self.all_relevent_docs))

            self.scores = self.normlize_scores(
                self.all_relevent_docs,
                self.bm25_body_score,
                self.bm25_title_score,
                self.bm25_body_bi_score,
                self.bm25_title_bi_score,
                self.body_cosine_score,
                self.title_cosine_score,
                self.pr_score,
                self.pv_score,
            )

            self.weights = np.array(
                [
                    params["bm25_body_weight"],
                    params["bm25_title_weight"],
                    params["bm25_body_bi_weight"],
                    params["bm25_title_bi_weight"],
                    params["body_cosine_score"],
                    params["title_cosine_score"],
                    params["pr_weight"],
                    params["pv_weight"],
                ]
            )

            total_time = (datetime.now() - start).total_seconds()
            print("total search time {} seconds".format(total_time))
            
            scores_sorted = sorted(
                [
                    (doc_id, np.dot(self.scores[doc_id], self.weights))
                    for doc_id, score in self.scores.items()
                ],
                key=lambda x: x[1],
                reverse=True,
            )[:100]
            result = [(doc_id, self.title_dict[doc_id]) for doc_id, score in scores_sorted]

            return result

        except Exception as e:
            print("error when calling full_search_ query:{}".format(query))
#             raise e
            return []

    #             raise e

    def normlize_scores(self, docs, *args):
        """
        normailze lists of scores using min-max normalization method.
        todo
        """
        try:
            doc_scores = {doc: np.array([0] * len(args), dtype=float) for doc in docs}
            i = -1
            for score_list in args:
                i += 1
                if len(score_list.values()) == 0:
                    continue
                max_score = builtins.max(score_list.values())
                min_score = builtins.min(score_list.values())
                if max_score == 0:
                    continue
                for doc in score_list:
                    if (max_score - min_score) == 0:
                        doc_scores[doc][i] = score_list[doc]
                    else:
                        doc_scores[doc][i] = (score_list[doc] - min_score) / (
                            max_score - min_score
                        )
            return doc_scores
        except Exception as e:
            print(
                "error when calling get_normelize_union all_relevent_docs len:{}".format(
                    len(docs)
                )
            )
            raise e

    def get_synonyms(self, query_to_search):
        try:
            list_of_tokens = [
                token.group()
                for token in RE_WORD.finditer(query_to_search.lower())
                if token.group() not in all_stopwords
            ]

            synonyms = []
            for token in list_of_tokens:
                count = 0
                for syn in wordnet.synsets(token):
                    for lemma in syn.lemmas():
                        if count < 2:
                            synonyms.append(lemma.name())
                            count += 1
                        else:
                            break
                synonyms = list(set(synonyms))
                list_of_tokens += synonyms

            list_of_tokens = [self.stemmer.stem(token) for token in list_of_tokens]
            return list_of_tokens
        except Exception as e:
            print(
                "error when calling get_synonyms all_relevent_docs len:{}".format(
                    query_to_search
                )
            )
            raise e

    def fast_cosine_calc(
        self, query_to_search, index_name, candidates=None, unique_candidates=None
    ):
        """
        fast cosine calculation, using only related documents and term, as discribed in chapter 7 on the Information Retrival course book.
        """
        if candidates is None:
            candidates, unique_candidates = self.get_candidate_documents_and_scores(
                query_to_search, index_name
            )

        D = dict()
        token_counter = Counter(query_to_search)
        for token in query_to_search:
            if token in self.index_dict[index_name].df:
                # tf idf of the query
                #                 tf_q = token_counter[token] / len(query_to_search)
                idf_q = self.idf_dict[index_name][token]
                w_t_q = idf_q
                # tf idf of the term
                for doc_id in unique_candidates:
                    w_t_d = candidates.get((doc_id, token), 0)
                    if doc_id in D.keys():
                        D[doc_id] += w_t_q * w_t_d
                    else:
                        D[doc_id] = w_t_q * w_t_d

        for doc_id in unique_candidates:
            D[doc_id] = D[doc_id] / self.index_dict[index_name].dl[doc_id]
        return D

    def get_top_n(self, sim_dict, N=3):
        """
        Sort and return the highest N documents according to the cosine similarity score.
        Generate a dictionary of cosine similarity scores

        Parameters:
        -----------
        sim_dict: a dictionary of similarity score as follows:
                                                                    key: document id (e.g., doc_id)
                                                                    value: similarity score. We keep up to 5 digits after the decimal point. (e.g., round(score,5))

        N: Integer (how many documents to retrieve). By default N = 3

        Returns:
        -----------
        a ranked list of pairs (doc_id, score) in the length of N.
        """

        return sorted(
            [(doc_id, round(score, 5)) for doc_id, score in sim_dict.items()],
            key=lambda x: x[1],
            reverse=True,
        )[:N]

    def titles(self, score_list):
        """
        gets a list of tuples (doc_id, score) or dict {doc_id:score} sorted by score and return a list of tuples (doc_id, title)
        """
        if isinstance(score_list, dict):
            return [
                (doc_id, self.title_dict[doc_id])
                for doc_id, score in score_list.items()
            ]
        return [(doc_id, self.title_dict[doc_id]) for doc_id, score in score_list]

    def normalize_scores(self, score_list):
        ids, scores = zip(*score_list)
        min_score, max_score = min(scores), max(scores)
        result = [
            (doc_id, (score - min_score) / (max_score - min_score))
            for doc_id, score in score_list
        ]
        return result

    def bm25_search(self, query, index_name):
        return self.bm25_dict[index_name].search(query)

    def pv_scores(self, query, candidates=None, index_name=None):
        if candidates == None:
            candidates = self.get_candidates(query, self.index_dict[index_name])[0]
        return sorted(
            [
                (doc_id, self.pv[doc_id] if doc_id in self.pr else 0)
                for doc_id in candidates
            ],
            key=itemgetter(1),
            reverse=True,
        )

    def pr_scores(self, query, candidates=None, index_name=None):
        if candidates == None:
            candidates = self.get_candidates(query, self.index_dict[index_name])[0]
        return sorted(
            [
                (doc_id, self.pr[doc_id] if doc_id in self.pr else 0)
                for doc_id in candidates
            ],
            key=itemgetter(1),
            reverse=True,
        )

    def big_search(self, query):
        final_scores = Counter()

        for index_name in self.index_dict.keys():
            print(index_name)
            bm25_scores = self.bm25_search(query, index_name)
            normalized = dict(self.normalize_scores(list(bm25_scores.items())))
            final_scores.update(normalized)
        return self.get_top_n(final_scores, 100)

    def tokenize(self, text, bigram=False, stem=False):
        """
        This function aims in tokenize a text into a list of tokens. Moreover, it filter stopwords.

        Parameters:
        -----------
        text: string , represting the text to tokenize.

        Returns:
        -----------
        list of tokens (e.g., list of tokens).
        """

        list_of_tokens = [
            token.group()
            for token in RE_WORD.finditer(text.lower())
            if token.group() not in all_stopwords
        ]

        if stem:
            list_of_tokens = [self.stemmer.stem(token) for token in list_of_tokens]

        if bigram:
            bigrams = list(ngrams(list_of_tokens, 2))
            bigrams = [" ".join(bigram) for bigram in bigrams]
            return bigrams

        return list_of_tokens

    def search_body_Q2(self, query):
        tokenized_query = self.tokenize(query, bigram=False, stem=False)
        scores = self.fast_cosine_search(
            tokenized_query,
            "body_no_stem_index",
            100,
            stem=False,
            candidates=None,
            unique_candidates=None,
        )
        result = [(doc_id, self.title_dict[doc_id]) for doc_id, score in scores]
        return result

    def search_title_Q3(self, query):
        tokenized_query = self.tokenize(query, bigram=False, stem=False)
        candidates, candidates_for_term = self.get_candidates(
            tokenized_query, "title_no_stem_index"
        )
        scores = [
            (doc_id, self.binary_score(tokenized_query, candidates_for_term, doc_id))
            for doc_id in candidates
        ]
        scores_sorted = sorted(scores, key=itemgetter(1), reverse=True)
        result = [(doc_id, self.title_dict[doc_id]) for doc_id, score in scores_sorted]
        return result

    def search_anchor_Q4(self, query):
        tokenized_query = self.tokenize(query, bigram=False, stem=False)
        candidates, candidates_for_term = self.get_candidates(
            tokenized_query, "anchor_index"
        )
        scores = [
            (doc_id, self.binary_score(tokenized_query, candidates_for_term, doc_id))
            for doc_id in candidates
        ]
        scores_sorted = sorted(scores, key=itemgetter(1), reverse=True)
        result = [(doc_id, self.title_dict[doc_id]) for doc_id, score in scores_sorted]
        return result

    def page_rank_Q5(self, wiki_ids):
        return [self.pr[doc_id] for doc_id in wiki_ids]

    def page_view_Q6(self, wiki_ids):
        return [self.pv[doc_id] for doc_id in wiki_ids]


class BM25_from_index:
    def __init__(self, index, k1=1.5, b=0.75):
        self.b = b
        self.k1 = k1
        self.index = index
        self.N = len(index.dl)
        self.AVGDL = sum(index.dl.values()) / self.N
        self.idf = self.calc_idf(self.index.term_total.keys())

    def calc_idf(self, words):
        idf = {}
        for term in words:
            if term in self.index.df.keys():
                n_ti = self.index.df[term]
                idf[term] = math.log(1 + (self.N - n_ti + 0.5) / (n_ti + 0.5))
            else:
                pass
        return idf

    def get_posting_list(self, index, w):
        with closing(MultiFileReader()) as reader:
            locs = index.posting_locs[w]
            if w not in index.df:
                return
            b = reader.read(locs, index.df[w] * TUPLE_SIZE, index.bucket_name)
            posting_list = []
            for i in range(index.df[w]):
                doc_id = int.from_bytes(b[i * TUPLE_SIZE : i * TUPLE_SIZE + 4], "big")
                tf = int.from_bytes(b[i * TUPLE_SIZE + 4 : (i + 1) * TUPLE_SIZE], "big")
                posting_list.append((doc_id, tf))
            return posting_list

    def get_candidates(self, query, index):
        """
         returns a set of relevant docments for query as well as a dictionary that keeps relevant documents for each term.
        @params
        query: a list of tokens
        index: an inverted index object

        return:
            a set of relevant documents and a dictionary with items in the form of term:list_of_doc_ids
        """
        candidates = set()
        cadidates_for_term = {term: [] for term in np.unique(query)}
        for term in np.unique(query):
            if term in index.df:
                pl = self.get_posting_list(self.index, term)
                unique_docs = set([i[0] for i in pl])
                cadidates_for_term[term] = dict(pl)
                candidates = candidates.union(unique_docs)
            return candidates, cadidates_for_term

    def search_with_candidates(self, query, in_candidates):
        candidates = in_candidates
        doc_scores = {}
        for term in query:
            if term in self.index.df:
                pl_dict = dict(self.get_posting_list(self.index, term))
                for doc_id in candidates:
                    if doc_id in pl_dict:
                        freq = pl_dict[doc_id]
                        numerator = self.idf[term] * freq * (self.k1 + 1)
                        denominator = freq + self.k1 * (
                            1 - self.b + self.b * (self.index.dl[doc_id] / self.AVGDL)
                        )
                        doc_scores[doc_id] = doc_scores.get(doc_id, 0) + (
                            numerator / denominator
                        )
        return doc_scores

    def search(self, query):
        candidates = self.get_candidates(query, self.index)[0]
        doc_scores = {}
        for term in query:
            if term in self.index.df:
                pl_dict = dict(self.get_posting_list(self.index, term))
                for doc_id in candidates:
                    if doc_id in pl_dict:
                        freq = pl_dict[doc_id]
                        numerator = self.idf[term] * freq * (self.k1 + 1)
                        denominator = freq + self.k1 * (
                            1 - self.b + self.b * (self.index.dl[doc_id] / self.AVGDL)
                        )
                        doc_scores[doc_id] = doc_scores.get(doc_id, 0) + (
                            numerator / denominator
                        )
        return doc_scores

