import json
import os
import pickle

with open(f'{os.pardir}{os.sep}queries_train.json') as f:
    x= json.load(f)
    queries = list(x.keys())
    queries_true = list(x.values())
def print_best_and_worst_queries():
    with open("two_word/final_merge_precision_score_two_word.json") as f:
        query_result_precision = json.load(f)["index_weight:0.98,page_rank_weight:0.0,page_views_weight:0.02"]

    best_queries = sorted(list(enumerate(query_result_precision)), key= lambda x:x[1], reverse=True)[:10]
    worst_queries = sorted(list(enumerate(query_result_precision)), key= lambda x:x[1])[:3]

    print(f"best queries are: {best_queries}")
    print(f"worst queries are: {worst_queries}")

    for i, score in best_queries:
        print(f"good query is: {queries[i]} with score of {score}")
    for i, score in worst_queries:
        print(f"bad query is: {queries[i]} with score of {score}")

def print_query_predicted_and_true_top_10(query_index):
    with open(f"two_word{os.sep}model_best_results_for_each_query_two_word.json") as f:
        query_results = json.load(f)[str(query_index)]
    with open("/Users/eitan/University/Information Retrieval/IR-Project/final_indexes_and_files/dict_of_id_title.pkl","rb") as f:
        doc_id_to_title = pickle.load(f)
    query_results = [doc_id_to_title[int(doc_id)] for doc_id in list(query_results.keys())[:40]]
    print(f"Top 10 predicted query results: {query_results}")
    true_results = [doc_id_to_title[doc_id] for doc_id in queries_true[query_index]]
    print(f"Top 10 true query results: {true_results}")

print_best_and_worst_queries()
print_query_predicted_and_true_top_10(27)