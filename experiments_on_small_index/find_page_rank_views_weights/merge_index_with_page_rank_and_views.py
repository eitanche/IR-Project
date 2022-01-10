import json
import os
import pickle
import time
from collections import defaultdict, Counter
from normalize_page_rank_page_views import normalize_page_rank_views
w_index=[i*0.01 for i in range (101)]
w_page_rank=[i*0.01 for i in range (101)]
w_page_views=[i*0.01 for i in range (101)]

with open(f"normalized_page_views_of_each_doc.pkl", "rb") as f:
    page_views_dict = pickle.load(f)

with open("normalized_page_rank.json",'r') as f:
    page_rank_doc_id_rank_dict = json.load(f)

with open(f"two_word{os.sep}model_best_results_for_each_query_two_word.json", "r") as x:
    query_to_merged_results_dict = json.load(x)

with open(f"{os.pardir}{os.sep}queries_train.json") as f:
    queries_dict = json.load(f)

dict_of_precisions_for_all_queries = defaultdict(list)


def get_top_40_docs_for_single_query(i, index_weight, page_rank_weight, page_views_weight):
    top_300_query_result_dict = query_to_merged_results_dict[str(i)]
    merged_result_dict = Counter()
    for doc_id, score in top_300_query_result_dict.items():
        merged_result_dict[int(doc_id)]=index_weight*top_300_query_result_dict[doc_id]+page_rank_weight*float(page_rank_doc_id_rank_dict.get(doc_id,0))+page_views_weight*page_views_dict.get(int(doc_id),0)
    x = sorted(merged_result_dict.items(), key=lambda x:x[1], reverse=True)[:40]
    return x

def map_at_k(doc_id_oredered_bm_25_scores,k,i):
    sum = 0
    hit = 0
    relevant_docs = list(queries_dict.values())[i] #the relevant documents from the i_th query list
    for i in range(len(doc_id_oredered_bm_25_scores)):#we already cuted the list to size 40 in order_scores funciton
        if doc_id_oredered_bm_25_scores[i][0] in relevant_docs:

            hit += 1
            sum += hit / (i + 1)
    if hit == 0:
        return 0
    return sum/hit

normalize_page_rank_views()

for index_weight in w_index:
    for page_rank_weight in w_page_rank:
        for page_views_weight in w_page_views:
            if 0.99 < index_weight+page_views_weight+page_rank_weight < 1.01:
                list_of_all_queries_precision = []
                for i in range(30): #for each query
                    sorted_docs_and_scores = get_top_40_docs_for_single_query(i,index_weight, page_rank_weight, page_views_weight) #40 sorted list of (doc_id,score)
                    list_of_all_queries_precision.append(map_at_k(sorted_docs_and_scores,40,i)) #takes all the docs we have and makes list which contains 30 precisions
                dict_of_precisions_for_all_queries["index_weight:"+str(index_weight)+","+"page_rank_weight:"+str(page_rank_weight)+","+"page_views_weight:"+str(page_views_weight)] = list_of_all_queries_precision
with open(f"two_word{os.sep}final_merge_precision_score_two_word.json",'w') as f:
    json.dump(dict_of_precisions_for_all_queries,f)

