import json
import os
import pickle
from collections import Counter
from math import log


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

#merge_weighted_results()
with open(f"{os.pardir}/final_indexes_and_files/page_views_of_each_doc.pkl", "rb") as f:
    page_views_dict = pickle.load(f)
normalized_page_views_dict = { doc_id:log(page_view,10) for doc_id, page_view in page_views_dict.items()}

with open(f"{os.pardir}/final_indexes_and_files/normalized_page_views_of_each_doc.pkl", "wb") as f:
    pickle.dump(normalized_page_views_dict, f)


with open("small_page_rank.json",'r') as f:
    page_rank_doc_id_rank_dict = json.load(f)
normalized_page_rank_doc_id_rank_dict = { doc_id:log(float(page_rank)+1,5) for doc_id, page_rank in page_rank_doc_id_rank_dict.items()}
with open("normalized_small_page_rank.json",'w') as f:
    json.dump(normalized_page_rank_doc_id_rank_dict, f)
