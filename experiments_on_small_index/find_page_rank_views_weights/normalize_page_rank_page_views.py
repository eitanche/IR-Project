import json
import os
import pickle
from collections import Counter
from math import log
import os

def merge_scores(titles_docs_result_dict, bodies_docs_result_dict, anchors_docs_result_dict):
    """
    dicts of (doc_id,score)
    """
    merged_results = Counter()
    merged_results.update(add_scores_to_merged_dict(0.34,titles_docs_result_dict))
    merged_results.update(add_scores_to_merged_dict(0.13, bodies_docs_result_dict))
    merged_results.update(add_scores_to_merged_dict(0.53, anchors_docs_result_dict))

    return dict(sorted(merged_results.items(), key=lambda x: x[1], reverse=True)[:300])



def add_scores_to_merged_dict(weight, doc_id_scores_dict):
    counter = Counter()
    for doc_id,score in doc_id_scores_dict.items():
        counter[doc_id] = score*weight
    return counter

def merge_weighted_results():
    with open("top_k_results_from_all_indexes_two_word_index.json","r") as f:
        dictionary_of_all_results = json.load(f)
        query_to_merged_results_dict = dict()
        for query_id, query_result_dictionary in dictionary_of_all_results.items():
            title_results = query_result_dictionary["title"]
            body_results = query_result_dictionary["body"]
            anchor_results = query_result_dictionary["anchor"]
            dict_of_merged_results = merge_scores(title_results, body_results, anchor_results)
            query_to_merged_results_dict[query_id]=dict_of_merged_results
        with open("model_best_results_for_each_query_two_word.json","w") as x:
            json.dump(query_to_merged_results_dict  , x)


def print_bm25_distirbution():
    with open("two_word/model_best_results_for_each_query_two_word.json", "r") as f:
        query_to_doc_id_and_score_dict = json.load(f)
        max_results = []
        min_results = []
        for query_id,result_dict in query_to_doc_id_and_score_dict.items():
            max_results.extend(sorted(result_dict.items(), key= lambda x:x[1], reverse=True)[:5])
            min_results.extend(sorted(result_dict.items(), key= lambda x:x[1])[:5])
        print("MAXIMUM")
        for item in max_results:
            print(item)
        print("MINIMUM")
        for item in min_results:
            print(item)
            # minimum 1 1.5 2.5
            # maximum3.5 4 1.5 3.5 4.5 5 6 3.5 6.84 9 5

def print_normalized_page_views():
    with open (f'{os.pardir}{os.sep}{os.pardir}{os.sep}final_indexes_and_files{os.sep}page_views_of_each_doc.pkl',"rb") as f:
        page_views = pickle.load(f)
    sorted_page_views = sorted([(doc_id, log(page_views,10)) for doc_id, page_views in page_views.items()], reverse=True, key=lambda x:x[1])
    for item in sorted_page_views[:5]:
        print(item)
    for item in sorted_page_views[100:105]:
        print(item)
    for item in sorted_page_views[1000:1005]:
        print(item)
    for item in sorted_page_views[10000:10005]:
        print(item)
    for item in sorted_page_views[100000:100000]:
        print(item)
    for item in sorted_page_views[800000:800005]:
        print(item)
    for item in sorted_page_views[6000000:6000005]:
        print(item)

def print_normalized_page_rank():
    with open (f'{os.pardir}{os.sep}{os.pardir}{os.sep}final_indexes_and_files{os.sep}page_rank.json',"r") as f:
        page_views = json.load(f)
    sorted_page_views = sorted([(doc_id, log(float(page_views)+1)) for doc_id, page_views in page_views.items()], reverse=True, key=lambda x:x[1])
    for item in sorted_page_views[:5]:
        print(item)
    for item in sorted_page_views[100:105]:
        print(item)
    for item in sorted_page_views[1000:1005]:
        print(item)
    for item in sorted_page_views[10000:10005]:
        print(item)
    for item in sorted_page_views[100000:100000]:
        print(item)
    for item in sorted_page_views[800000:800005]:
        print(item)
    for item in sorted_page_views[6000000:6000005]:
        print(item)

def normalize_page_rank_views(page_views_log_base = 10, page_rank_log_base = 5):

    with open(f'{os.pardir}{os.sep}{os.pardir}{os.sep}final_indexes_and_files{os.sep}page_views_of_each_doc.pkl',
              "rb") as f:
        page_views_dict = pickle.load(f)
    normalized_page_views_dict = {doc_id: normalize_with_log(page_view, page_views_log_base) for doc_id, page_view in page_views_dict.items()}

    with open(f"normalized_page_views_of_each_doc.pkl", "wb") as f:
        pickle.dump(normalized_page_views_dict, f)

    with open(f'{os.pardir}{os.sep}{os.pardir}{os.sep}final_indexes_and_files{os.sep}page_rank.json', 'r') as f:
        page_rank_doc_id_rank_dict = json.load(f)
    normalized_page_rank_doc_id_rank_dict = {doc_id: normalize_with_log(float(page_rank) + 1, page_rank_log_base) for doc_id, page_rank in
                                             page_rank_doc_id_rank_dict.items()}
    with open("normalized_page_rank.json", 'w') as f:
        json.dump(normalized_page_rank_doc_id_rank_dict, f)

def normalize_with_log(score, log_base):
    if log_base==0:
        print("HERE")
        return score
    return log(score, log_base)