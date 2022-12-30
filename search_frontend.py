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
# bucket = client.get_bucket('all_pkl')
# idx_title = pickle.loads(bucket.get_blob('index_title_inverted_index.pkl').download_as_string())
# idx_body = pickle.loads(bucket.get_blob('index_body_inverted_index.pkl').download_as_string())
# idx_title2 = pickle.loads(bucket.get_blob('index_title2_inverted_index.pkl').download_as_string())
# idx_body2 = pickle.loads(bucket.get_blob('index_body2_inverted_index.pkl').download_as_string())
# idx_title_simple = pickle.loads(bucket.get_blob('index_simple_title_inverted_index.pkl').download_as_string())
# idx_body_simple = pickle.loads(bucket.get_blob('index_simple_body_inverted_index.pkl').download_as_string())
# idx_anchor = pickle.loads(bucket.get_blob('index_anchor_bucket.pkl').download_as_string())
# pv = pickle.loads(bucket.get_blob('pageviews-202108-user.pkl').download_as_string())
# pr = pickle.loads(bucket.get_blob('PageRankWiki.pkl').download_as_string())
# Mapping = pickle.loads(bucket.get_blob('id_title1.pkl').download_as_string())
# bm25_body = BM25_from_index(idx_body)

idx_body_simple.DL=idx_body.DL
class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)

app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False


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
    # res = GetResult(query, idx_title, idx_title2, idx_body, idx_body2, bm25_body, pr, pv)[:100]
    # res = _idToValuesMapping(res)
    res = [(1,"Hello"),(2,"World"),(3,query)]
#     res = GetResult(query, idx_title, idx_title2, idx_body, idx_body2, bm25_body, pr, pv)[:100]
#     res = _idToValuesMapping(res)
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
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
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
    res = _idToValuesMapping(res)
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
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
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
    res = _idToValuesMapping(res)
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
    # BEGIN SOLUTION
    query = _get_tokens(query)
    rel_docs, candidates_dict = get_candidate_documents(query, idx_anchor, thrashold=999999)
    id_score = get_top_n([(relevant_doc, get_num_of_match_binary(query, candidates_dict, relevant_doc)) for relevant_doc in
                           rel_docs], N=999999999)
    res = [i[0] for i in id_score.items()]
    res = _idToValuesMapping(res)
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
    app.run(host='0.0.0.0', port=8080, debug=True)
