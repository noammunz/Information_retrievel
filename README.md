# IR
Wikipedia Search Engine

This is the final project of the Information Retrieval course in BGU.
</br>
To optimise result accuracy and retrieval time we applied the following scoring methods:</br>
• TF-IDF</br>
• Cosine similarity</br>
• BM25</br>
• Query expansion</br>
• Binary scoring</br>
• Page rank</br>
• Page views</br>
</br>
The following code includes:</br>
</br>
- se.py - Contains initialization of the search engine, loading related indices, and waiting for query requests. All search methods (BM25, cosine similarity, ect) and all required search functions are available here.</br>
</br>
- search_frontend.py - wrapper for Flask that allows HTTP requests to be sent.</br>
</br>
- se_test.py - a test script that runs all train queries, returns precision, latency, and other performance measures.</br>
</br>
- unit_tests.ipynb - a notebook that calls for train queries and runs optimization modules to find the best parameters for our engine.</br>
</br>
- inverted_index_gcp.py - Readers and writers for interacting with the bucket's indices and binary files.</br>
</br>
- Google buckets are used to store inverted indices and binary files (links are included in the project report pdf file):</br>
</br>
    Anchor inverted index</br>
    Title inverted index</br>
    Title inverted index with no stemming</br>
    Title inverted index with bigram</br>
    Body inverted index</br>
    Body inverted index with no stemming</br>
    Body inverted index with bigram</br>
    Body inverted index without popular words filtering Corpus</br>
    Pickles</br>
</br>
Thanks</br>

