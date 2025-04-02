# Wikipedia Search Engine

This repository contains our implementation of a search engine for Wikipedia articles, which secured the 2nd place in the Information Retrieval course at Ben-Gurion University with a grade of 98.

## Overview

Our search engine optimizes query response time while maintaining high accuracy by leveraging multiple scoring methods:

- TF-IDF (Term Frequency-Inverse Document Frequency)
- Cosine similarity 
- BM25 ranking function
- Query expansion
- Binary scoring
- PageRank
- Page views analysis

## Architecture

The search engine implementation is based on an inverted index structure and uses Google Cloud Platform services for storage and deployment.

### Key Components

- **Inverted Index Builder** (`index_creation.ipynb`): Creates and stores inverted indices for different document parts (title, body, anchors).
- **Search Engine Core** (`se.py`): Handles query processing and implements the various ranking algorithms.
- **Web Interface** (`search_frontend.py`): Flask-based API that serves search results.
- **Testing Framework** (`se_test.py`): Contains testing scripts and evaluation metrics.
- **Inverted Index Utilities** (`inverted_index_gcp.py`): Provides functionality for reading and writing index files to Google Cloud Storage.

## Features

- Multiple query strategies combining different sources and ranking algorithms
- Support for various types of searches (full text, title-only, anchor text)
- Stemming and stopword removal
- Bigram analysis for better handling of phrases
- PageRank and page view integration for popularity-based ranking
- Query expansion to improve recall

## Implementation Details

- We store indices in GCP buckets for efficient retrieval
- Different indices for various document fields (title, body, anchors)
- Indices with and without stemming to support different query strategies
- Bigram indices for better phrase matching
- Various optimization techniques to ensure fast response times (<1 second per query)

## Performance

Our search engine achieves high precision and recall scores while maintaining excellent response times. The implementation was rigorously tested against a benchmark of queries and evaluated using standard IR metrics.

## Usage

The search engine API provides several endpoints:

- `/search` - Main search endpoint that combines all ranking methods
- `/search_body` - Search using only article body text
- `/search_title` - Search matching query terms in titles
- `/search_anchor` - Search matching query terms in anchor text
- `/get_pagerank` - Retrieve PageRank scores for document IDs
- `/get_pageview` - Retrieve page view statistics for document IDs

## Contributors

This project was developed as the final assignment for the Information Retrieval course at Ben-Gurion University.
