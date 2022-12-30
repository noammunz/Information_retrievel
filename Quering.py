from collections import Counter
from nltk.stem.porter import *
import builtins
import math
import numpy as np
import pandas as pd
from contextlib import closing
from inverted_index_gcp import InvertedIndex, MultiFileReader

TUPLE_SIZE = 6
RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)
stemmer = PorterStemmer()
all_stopwords = ['was', 'how', 'for', 'could', 'please', 'then', 'whereupon', 'so', 'as', 'any', 'used', 'why',
                 'several', 'six', 'by', 'yourself', 'hence', 'did', 'whose', 'doing', 'n’t', 'show', 'her',
                 'yourselves', 'nothing', 'not', '‘ve', 'thereby', 'forty', 'each', 'whether', 'serious', 'perhaps',
                 'me', 'nevertheless', 'list', 'rather', 'elsewhere', 'former', 'seemed', 'up', 'than', 'except', 'via',
                 'can', 'everything', 'already', 'along', 'while', 'anyone', 'our', 'thus', 'must', 'either', 'move',
                 '’ve', 'between', 'ever', 'does', 'whereafter', 'twelve', 'ca', 'in', 'again', 'itself', 'themselves',
                 'until', 'whence', 'she', 'us', 'say', 'no', 'amongst', 'were', 'whole', 'nine', 'under', 'everywhere',
                 'an', 'therein', 'go', 'would', 'beyond', 'become', 's', 'category', 'sixty', 'myself', 'they',
                 'about', 'since', 'put', 'amount', 'whatever', 'whereby', 'though', 'thru', 'a', "'ll", 'never',
                 'their', 'even', 'ours', 'twenty', 'am', 'someone', "'s", 'that', 'last', 'to', 'becoming', 'upon',
                 'one', 'this', 'meanwhile', 'more', 'else', 'himself', 'moreover', 'bottom', 'well', 'became', 'same',
                 'alone', '‘d', 'these', 'before', 'with', 'but', 'because', 'following', 'hers', 'really', 'thereupon',
                 'third', 'always', 'some', "'ve", 'hereby', 'his', 'above', 'thumb', 'part', 'few', 'formerly', "'d",
                 '’d', 'however', 'history', 'seeming', 'having', 'somewhere', 'him', 'neither', 'do', 'whereas',
                 'besides', 'wherein', 'next', 'throughout', 'against', 'is', 'unless', 'yet', 'therefore', 'many',
                 'eight', 'still', 'name', 'nobody', 'top', 'mine', 'somehow', 'you', 'been', 'fifty', 'where',
                 'sometimes', 'get', 'hereafter', 'three', 'quite', 'towards', 'made', 'may', 'side', 'thereafter',
                 'beside', 'noone', 'none', 'be', 'anywhere', 'further', '‘ll', 'links', 'toward', 'due', 'my', 'what',
                 'references', 'beforehand', 'there', 'thence', 'together', 'four', 'among', 'at', "n't", 'such', 't',
                 'external', 'people', 'using', 'something', 'ourselves', 'keep', 'another', 'onto', 'latterly', '‘s',
                 'had', 'or', 'only', 'hundred', 'whenever', 'also', 'see', '‘re', 'whither', 'less', 'much', 'others',
                 'should', 'sometime', 'although', 'around', 'seem', 'of', 'down', 'five', 'them', 'on', '‘m', 'yours',
                 'very', 'various', 'empty', 'through', 'wherever', 'most', 'cannot', 'here', 'namely', 'mostly', 'per',
                 'during', 'whoever', 'out', 'we', 'second', '’re', 'both', 're', 'ten', 'don', 'anyhow', 'after',
                 'back', 'everyone', 'done', 'often', 'call', 'latter', 'seems', 'might', 'now', 'take', 'fifteen',
                 'every', '’m', 'own', 'becomes', 'all', 'make', 'which', 'full', 'indeed', 'nor', 'give', 'other',
                 'almost', 'hereupon', 'who', 'nowhere', '’s', 'including', "'re", 'too', 'i', 'he', 'its', 'those',
                 'the', 'it', 'first', 'herein', '’ll', 'just', 'over', 'have', 'and', "'m", 'when', 'eleven', 'below',
                 'if', 'theirs', 'behind', 'whom', 'are', 'will', 'from', 'into', 'enough', 'your', 'least', 'anyway',
                 'n‘t', 'herself', 'has', 'within', 'once', 'front', 'anything', 'afterwards', 'being', 'off', 'across',
                 'two', 'otherwise', 'regarding', 'without']


class BM25_from_index:
    def __init__(self, index, k1=1.5, b=0.75):
        self.b = b
        self.k1 = k1
        self.index = index
        self.N = len(index.DL)
        self.AVGDL = builtins.sum(index.DL.values()) / self.N

    def calc_idf(self, list_of_tokens):
        idf = {}
        for term in list_of_tokens:
            if term in self.index.df:
                n_ti = self.index.df[term]
                idf[term] = math.log(1 + (self.N - n_ti + 0.5) / (n_ti + 0.5))
            else:
                pass
        return idf

    def search(self, query, relevant_docs, word_dict, N=5000):
        self.idf = self.calc_idf(set([t for t in query]))
        return get_top_n(
            [(relevant_doc, self._score(query, relevant_doc, word_dict)) for relevant_doc in relevant_docs], N)

    def _score(self, query, doc_id, word_dict):
        score = 0.0
        if doc_id == 0.0:
            return 0
        doc_len = self.index.DL[doc_id]
        for term in query:
            if term in self.index.df:
                term_frequencies = word_dict[term]
                if doc_id in term_frequencies:
                    freq = term_frequencies[doc_id]
                    numerator = self.idf[term] * freq * (self.k1 + 1)
                    denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / self.AVGDL)
                    score += (numerator / denominator)
        return score


def get_tokens(text, ngram=False):
    tokens = [token.group() for token in RE_WORD.finditer(text.lower())]
    tokens = [stemmer.stem(term) for term in tokens if term not in all_stopwords]
    if not ngram: return tokens
    if len(tokens) < 2: return []
    return [''.join(x) for x in zip(tokens[0::1], tokens[1::1])]


def get_posting_list(inverted, w, trashold=200):
    trashold = builtins.min(inverted.df[w], trashold)
    with closing(MultiFileReader()) as reader:
        locs = inverted.posting_locs[w]
        b = reader.read(locs, trashold * TUPLE_SIZE, inverted.bucket_name)
        posting_list = []
        for i in range(trashold):
            doc_id = int.from_bytes(b[i * TUPLE_SIZE:i * TUPLE_SIZE + 4], 'big')
            tf = int.from_bytes(b[i * TUPLE_SIZE + 4:(i + 1) * TUPLE_SIZE], 'big')
            posting_list.append((doc_id, tf))
        return posting_list


def get_candidate_documents(query_to_search, index, thrashold=200):
    candidates = set()
    candidates_dict = {}
    for term in np.unique(query_to_search):
        if term in index.df:
            all_res = get_posting_list(index, term, thrashold)
            candidates_dict.update({term: dict(all_res)})
            candidates.update([x[0] for x in all_res])
    return candidates, candidates_dict


def get_top_n(sim_list, N=1000):
    return dict(sorted(sim_list, key=lambda x: x[1], reverse=True)[:N])


def calc_tf_idf(doc_id, query, inverted_index, word_dict):
    score = 0
    if doc_id == 0.0:
        return 0
    token_counter = Counter(query)
    size_q = []
    for token in token_counter:
        if token in inverted_index.df and doc_id in word_dict[token]:
            if doc_id not in inverted_index.DL: return 0
            tf_t = word_dict[token][doc_id] / inverted_index.DL[doc_id]
            idf_t = math.log2(6348910 / inverted_index.df[token])
            tf_q = token_counter[token] / len(query)
            idf_q = idf_t
            score += tf_t * idf_t * tf_q * idf_q
            size_q.append(tf_q * idf_q * tf_q * idf_q)
    return score / (math.sqrt(builtins.sum(size_q)) * inverted_index.DS[doc_id])


def get_page_rank(relevant_docs, pr):
    return list(pr[pr[0].isin(relevant_docs)].itertuples(index=False, name=None))


def get_page_view(relevant_docs, wid2pv):
    return [(doc, wid2pv[doc]) if doc in wid2pv else (doc, 0) for doc in relevant_docs]


def get_normelize_union(all_relevent_docs, *args):
    doc_id_score = {relevent_docs: np.array([0] * len(args), dtype=float) for relevent_docs in all_relevent_docs}
    i = -1
    for feature in args:
        i += 1
        if len(feature.values()) == 0: continue
        max_val = builtins.max(feature.values())
        min_val = builtins.min(feature.values())
        interval = max_val - min_val
        if max_val == 0: continue
        for doc in feature:
            if interval == 0:
                doc_id_score[doc][i] = feature[doc]
            else:
                doc_id_score[doc][i] = (feature[doc]- min_val) / interval
    return doc_id_score


def get_num_of_match(query, term_dict, doc_id):
    score = 0
    for w in query:
        if w in term_dict and doc_id in term_dict[w]:
            score += term_dict[w][doc_id]
    return score


def get_num_of_match_binary(query, term_dict, doc_id):
    score = 0
    for w in query:
        if w in term_dict and doc_id in term_dict[w]:
            score = 1
    return score


def GetResult(q, idx_title, idx_title2, idx_body, idx_body2, bm25_body, pr, pv):
    query = get_tokens(q)
    query2 = get_tokens(q, True)

    body_docs, body_dict = get_candidate_documents(query, idx_body)
    title_docs, title_dict = get_candidate_documents(query, idx_title)
    title_docs = title_docs.intersection(body_docs)

    for w in title_dict:
        for doc_id in list(title_dict[w].keys()):
            if doc_id not in body_docs:
                del title_dict[w][doc_id]

    num_match_title = get_top_n(
        [(relevant_doc, get_num_of_match_binary(query, title_dict, relevant_doc)) for relevant_doc in title_docs])
    # num_match_body = get_top_n(
    #     [(relevant_doc, get_num_of_match(query, body_dict, relevant_doc)) for relevant_doc in body_docs])

    bm25_body_score = bm25_body.search(query, body_docs, body_dict)

    cos_simi_body_score = get_top_n(
        [(relevant_doc, calc_tf_idf(relevant_doc, query, idx_body, body_dict)) for relevant_doc in body_docs])

    body_docs2, body_dict2 = get_candidate_documents(query2,idx_body2)
    title_docs2, title_dict2 = get_candidate_documents(query2, idx_title2)

    title_docs2 = title_docs2.intersection(body_docs)

    for w in title_dict2:
        for doc_id in list(title_dict2[w].keys()):
            if doc_id not in body_docs:
                del title_dict2[w][doc_id]

    num_match_title2 = get_top_n(
        [(relevant_doc, get_num_of_match_binary(query2, title_dict2, relevant_doc)) for relevant_doc in title_docs2])
    num_match_body2 = get_top_n([(relevant_doc,get_num_of_match(query2, body_dict2, relevant_doc)) for relevant_doc in body_docs2])

    all_relevent_docs = set().union(*[body_docs, title_docs, title_docs2, body_docs2])

    pr_score = get_top_n(get_page_rank(all_relevent_docs, pr))
    pv_score = get_top_n(get_page_view(all_relevent_docs, pv))

    norm_scores = get_normelize_union(all_relevent_docs,num_match_title, bm25_body_score,num_match_body2, cos_simi_body_score, num_match_title2, pr_score, pv_score)
    weights = np.array([0.88618747, 3.94137498, 3.26116641, 0.82773864, 2.84060984, 2.12254428, 3.13827727])
    # return norm_scores
    return sorted((norm_scores.keys()), key=lambda x: np.dot(norm_scores[x],weights), reverse=True)