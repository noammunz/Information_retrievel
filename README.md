# IR
Wikipedia Search Engine

This is the final project of the Information Retrieval course in BGU.

To optimise result accuracy and retrieval time we applied the following scoring methods:
• TF-IDF
• Cosine similarity
• BM25
• Query expansion
• Binary scoring
• Page rank
• Page views

The following code includes

- se.py - Contains initialization of the search engine, loading related indices, and waiting for query requests. All search methods (BM25, cosine similarity, ect) and all required search functions are available here.

- search_frontend.py - wrapper for Flask that allows HTTP requests to be sent.

- se_test.py - a test script that runs all train queries, returns precision, latency, and other performance measures.

- unit_tests.ipynb - a notebook that calls for train queries and runs optimization modules to find the best parameters for our engine.

- inverted_index_gcp.py - Readers and writers for interacting with the bucket's indices and binary files.

- Google buckets are used to store inverted indices and binary files (links are included in the project report pdf file):

    Anchor inverted index
    Title inverted index
    Title inverted index with no stemming
    Title inverted index with bigram
    Body inverted index
    Body inverted index with no stemming
    Body inverted index with bigram
    Body inverted index without popular words filtering Corpus
    Pickles

Thanks

