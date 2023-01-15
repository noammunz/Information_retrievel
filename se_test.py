import os
os.chdir('/home/dataproc/production')
import inverted_index_gcp
import se
from datetime import datetime
from se import search_engine
import importlib
importlib.reload(inverted_index_gcp)
importlib.reload(se)

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
    return round(sum(precisions)/len(precisions),3)

def run_se_test():
    # load engine 
    cur_se = search_engine()
    cur_se.init_engine(load_only_title=False)
    # load test queries
    import json
    with open('new_train.json') as json_file:
        official_queries = json.load(json_file)
    # run list of queries with performance monitor
    params = {'max_docs_from_binary_title': 4433,
                'max_docs_from_binary_body': 1349,
                'bm25_body_weight': 5.175842946028495,
                'bm25_title_weight': 2.297919942629382,
                'bm25_body_bi_weight': 0.9433857458903415,
                'bm25_title_bi_weight': 5.43406860036612,
                'body_cosine_score': 4.925136475255385,
                'title_cosine_score': 0.29963143927827507,
                'pr_weight': 0.34839354014586377,
                'pv_weight': 4.740913798917137}

    import json
    # with open('new_train.json') as json_file:
    with open('queries_train_last_year.json') as json_file:
        official_queries = json.load(json_file)

    avg_pre = 0
    counter = 0
    avg_time = 0
    max_time = 0

    for key, value in official_queries.items():
        test_query = key
        print('query:',test_query)
        start = datetime.now()
        b = cur_se.full_search_(test_query,params)
        end = datetime.now()
        total_time = (end - start).total_seconds()
        avg_time +=total_time
        max_time = max(max_time,total_time)

        our_results = list(zip(*b))[0][:100]
        offical_results = official_queries[test_query]
        avg_pre +=average_precision(our_results,offical_results)
        counter +=1
        print('average_precision',average_precision(our_results,offical_results))
        print('*********')
    print('final score:{}'.format(avg_pre/counter))
    print('avg_time score:{}'.format(avg_time/counter))
    print('max_time score:{}'.format(max_time))

run_se_test()
