from flask import Flask, request, jsonify
from Quering import *
import pandas as pd
from inverted_index_gcp import InvertedIndex, MultiFileReader
from google.cloud import storage
import pickle
import nltk
import builtins
from nltk.corpus import stopwords
import json
nltk.download('stopwords')

client = storage.Client()
bucket = client.get_bucket('bx_title_index')
idx_title = pickle.loads(bucket.get_blob('postings_gcp/bx_title_index_index.pkl').download_as_string())

bucket = client.get_bucket('bx_body_index')
idx_body = pickle.loads(bucket.get_blob('postings_gcp/bx_body_index_index.pkl').download_as_string())

bucket = client.get_bucket('bx_title_bigram_index')
idx_title2 = pickle.loads(bucket.get_blob('postings_gcp/bx_title_bigram_index_index.pkl').download_as_string())

bucket = client.get_bucket('bx_body_bigram_index')
idx_title2 = pickle.loads(bucket.get_blob('postings_gcp/bx_body_bigram_index_index.pkl').download_as_string())

bucket = client.get_bucket('bx_body_bigram_index')
idx_body2 = pickle.loads(bucket.get_blob('postings_gcp/bx_body_bigram_index_index.pkl').download_as_string())

bucket = client.get_bucket('bx_title_index')
idx_title_simple = pickle.loads(bucket.get_blob('postings_gcp/bx_title_index_index.pkl').download_as_string())

bucket = client.get_bucket('bx_body_index')
idx_body_simple = pickle.loads(bucket.get_blob('postings_gcp/bx_body_index_index.pkl').download_as_string())

# bucket = client.get_bucket('bx_anchor_index')
# idx_anchor = pickle.loads(bucket.get_blob('postings_gcp/bx_anchor_index_index.pkl').download_as_string())

bucket = client.get_bucket('bx_general_bucket')
pv = pickle.loads(bucket.get_blob('wid2pv.pkl').download_as_string())
pr = pd.read_csv(bucket.get_blob('pr/part-00000-a5ad5b2e-27ae-46dc-b038-36bc2f23b2c9-c000.csv.gz').download_as_string())
# bucket = client.get_bucket('bx_body_index')
# idx_body_simple = pickle.loads(bucket.get_blob('postings_gcp/bx_body_index_index.pkl').download_as_string())

# idx_body = pickle.loads(bucket.get_blob('index_body_inverted_index.pkl').download_as_string())
# idx_title2 = pickle.loads(bucket.get_blob('index_title2_inverted_index.pkl').download_as_string())
# idx_body2 = pickle.loads(bucket.get_blob('index_body2_inverted_index.pkl').download_as_string())
# idx_title_simple = pickle.loads(bucket.get_blob('index_simple_title_inverted_index.pkl').download_as_string())
# idx_body_simple = pickle.loads(bucket.get_blob('index_simple_body_inverted_index.pkl').download_as_string())
# idx_anchor = pickle.loads(bucket.get_blob('index_anchor_bucket.pkl').download_as_string())
# pv = pickle.loads(bucket.get_blob('pageviews-202108-user.pkl').download_as_string())
# pr = pickle.loads(bucket.get_blob('PageRankWiki.pkl').download_as_string())
# Mapping = pickle.loads(bucket.get_blob('id_title1.pkl').download_as_string())
bm25_body = BM25_from_index(idx_body)


idx_body_simple.DL=idx_body.DL

class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)



app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False


def _get_tokens(text):
    tokens = [token.group() for token in RE_WORD.finditer(text.lower())]
    tokens = [term for term in tokens if term not in all_stopwords]
    return tokens

# def _idToValuesMapping(ids):
#     return list(Mapping[Mapping[0].isin(ids)].itertuples(index=False, name=None))

english_stopwords = frozenset(stopwords.words('english'))
corpus_stopwords = ["category", "references", "also", "external", "links",
                    "may", "first", "see", "history", "people", "one", "two",
                    "part", "thumb", "including", "second", "following",
                    "many", "however", "would", "became"]
RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)
all_stopwords = english_stopwords.union(corpus_stopwords)

@app.route("/training_LR")
def training_LR():
    with open('queries_train.json') as json_file:
        train_queries = json.load(json_file)
    featurs = []
    labels = []
    for q in list(train_queries):
      result = GetResult(q, idx_title, idx_title2, idx_body, idx_body2, bm25_body, pr, pv)
      featurs += [result[scores] for scores in result]
      labels += [1 if qu in train_queries[q] else 0 for qu in result]
    from sklearn.linear_model import LogisticRegression
    logreg = LogisticRegression()
    logreg.fit(featurs, labels)
    def average_precision(true_list, predicted_list, k=40):
        true_set = frozenset(true_list)
        predicted_list = predicted_list[:k]
        precisions = []
        for i,doc_id in enumerate(predicted_list):
            if doc_id in true_set:
                prec = (len(precisions)+1) / (i+1)
                precisions.append(prec)
        if len(precisions) == 0:
            return 0.0
        return builtins.sum(precisions)/len(precisions)
    avg_model = []
    for i in train_queries:
        result = GetResult(i, idx_title, idx_title2, idx_body, idx_body2, bm25_body, pr, pv)
        res_model = sorted((result.keys()), key=lambda x: logreg.decision_function([result[x]]), reverse = True)
        doc = [i for i in train_queries[i]]
        avg_model.append(average_precision(doc, res_model))
    a = logreg.coef_
    b = logreg.intercept_
    c = builtins.sum(avg_model)/30
    d = len(labels)
    print(a,b,c,d)
    return jsonify([c])



@app.route("/search")
def search():
    ''' Returns up to a 100 of your best search results for the query. This is 
        the place to put forward your best search engine, and you are free to
        implement the retrieval whoever you'd like within the bound of the 
        project requirements (efficiency, quality, etc.). That means it is up to
        you to decide on whether to use stemming, remove stopwords, use 
        PageRank, query expansion, etc.
        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each 
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    res = GetResult(query, idx_title, idx_title2, idx_body, idx_body2, bm25_body, pr, pv)[:100]
    # res = _idToValuesMapping(res)
    # END SOLUTION
    return jsonify(res)


@app.route("/search_body")
def search_body():
    ''' Returns up to a 100 search results for the query using TFIDF AND COSINE
        SIMILARITY OF THE BODY OF ARTICLES ONLY. DO NOT use stemming. DO USE the 
        staff-provided tokenizer from Assignment 3 (GCP part) to do the 
        tokenization and remove stopwords.
        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search_body?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each 
        element is a tuple (wiki_id, title).
    '''
    # BEGIN SOLUTION
    query = request.args.get('query', '')
    res = []
    if len(query) == 0:
        return jsonify(res)
    query = _get_tokens(query)
    rel_docs, candidates_dict = get_candidate_documents(query, idx_body_simple, thrashold=200)
    cos_simi_body_score = get_top_n(
        [(relevant_doc, calc_tf_idf(relevant_doc, query, idx_body_simple, candidates_dict)) for relevant_doc in
         rel_docs], N=100)
    res = [i[0] for i in cos_simi_body_score.items()]
    # res = _idToValuesMapping(res)
    # END SOLUTION
    return jsonify(res)


@app.route("/search_title")
def search_title():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE TITLE of articles, ordered in descending order of the NUMBER OF 
        QUERY WORDS that appear in the title. For example, a document with a 
        title that matches two of the query words will be ranked before a 
        document with a title that matches only one query term. 
        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_title?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).
    '''
    # BEGIN SOLUTION
    query = request.args.get('query', '')
    res = []
    if len(query) == 0:
        return jsonify(res)
    query = _get_tokens(query)
    rel_docs, candidates_dict = get_candidate_documents(query, idx_title_simple, thrashold=999999)
    id_score = get_top_n([(relevant_doc, get_num_of_match_binary(query, candidates_dict, relevant_doc)) for relevant_doc in
                           rel_docs], N=100)
    res = [i[0] for i in id_score.items()]
    # res = _idToValuesMapping(res)
    # END SOLUTION
    return jsonify(res)


@app.route("/search_anchor")
def search_anchor():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE ANCHOR TEXT of articles, ordered in descending order of the 
        NUMBER OF QUERY WORDS that appear in anchor text linking to the page. 
        For example, a document with a anchor text that matches two of the 
        query words will be ranked before a document with anchor text that 
        matches only one query term. 
        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_anchor?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    query = _get_tokens(query)
    rel_docs, candidates_dict = get_candidate_documents(query, idx_anchor, thrashold=999999)
    id_score = get_top_n([(relevant_doc, get_num_of_match_binary(query, candidates_dict, relevant_doc)) for relevant_doc in
                           rel_docs], N=999999999)
    res = [i[0] for i in id_score.items()]
    # res = _idToValuesMapping(res)
    # END SOLUTION
    return jsonify(res)


@app.route("/get_pagerank", methods=['POST'])
def get_pagerank():
    ''' Returns PageRank values for a list of provided wiki article IDs. 
        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pagerank
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pagerank', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of floats:
          list of PageRank scores that correrspond to the provided article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    res = [i[1] for i in get_page_rank(wiki_ids, pr)]
    # END SOLUTION
    return jsonify(res)


@app.route("/get_pageview", methods=['POST'])
def get_pageview():
    ''' Returns the number of page views that each of the provide wiki articles
        had in August 2021.
        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pageview
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pageview', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ints:
          list of page view numbers from August 2021 that correrspond to the 
          provided list article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    res = [i[1] for i in get_page_view(wiki_ids, pv)]
    # END SOLUTION
    return jsonify(res)


if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080, debug=False)