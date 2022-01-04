import json
from collections import Counter
from inverted_index_gcp import InvertedIndex
from calculate_BM_25 import *


OPTIMAL_K = 1.5
OPTIMAL_B = 0.75

####working on queries
with open("queries_train.json") as f:
    queries_dict = json.load(f)

queries_seperated = get_queries(queries_dict) # return keys of the dict
all_queris_splitted_after_stemming = stem_queries(queries_seperated)


####
queries_results_dict = defaultdict(list)
####

#### load indexes
title_index = InvertedIndex.read_index("postings_small_title_gcp","index")
body_index  = InvertedIndex.read_index("posting_small_text","index")
anchor_index = InvertedIndex.read_index("postings_small_anchor_text_gcp","index")




#### weights

w_titles = [0.03]
w_bodies = [0.3]
w_anchors =[0.5]




def main():
    """

    """
    write_q_id_dict() #creates for each query 900
    dict_of_precisions_for_all_queries = defaultdict(list)
    for w_title in w_titles:
        for w_body in w_bodies:
            for w_anchor in w_anchors:
                list_of_all_queries_precision = []
                for i in range(30): #for each query
                    sorted_docs_and_scores = get_top_40_docs_for_single_query(w_title, w_body, w_anchor, i) #40 sorted list of (doc_id,score)
                    list_of_all_queries_precision.append(map_at_k(sorted_docs_and_scores,40,i)) #takes all the docs we have and makes list which contains 30 precisions
                dict_of_precisions_for_all_queries["title wight:"+str(w_title)+","+"body wight:"+str(w_body)+","+"anchor wight:"+str(w_anchor)] = list_of_all_queries_precision

    with open("only_indexes_merged_weights_precision_scores.json") as f:
        json.dump(dict_of_precisions_for_all_queries,f)




def get_top_40_docs_for_single_query(w_title, w_body, w_anchor, i):
    return sorted(merge_scores(w_title,w_body,w_anchor,queries_results_dict[i]['title'],queries_results_dict[i]['body'],queries_results_dict[i]['anchor'])
                  , reverse=True, key=lambda x: x[1])[:40]




def merge_scores(w_title, w_body, w_anchor, titles_docs_result_dict, bodies_docs_result_dict, anchors_docs_result_dict):
    """
    dicts of (doc_id,score)
    """
    merged_results = Counter()
    merged_results.update(add_scores_to_merged_dict(w_title,titles_docs_result_dict))
    merged_results.update(add_scores_to_merged_dict(w_body, bodies_docs_result_dict))
    merged_results.update(add_scores_to_merged_dict(w_anchor, anchors_docs_result_dict))

    return merged_results.items()



def add_scores_to_merged_dict(weight, doc_id_scores_dict):
    counter = Counter()
    for doc_id,score in doc_id_scores_dict.items():
        counter[doc_id] = score*weight
    return counter




def write_q_id_dict():
    """
    returns dict of dicts for each index, each dict contains k docs which holds the best BM_25 scores for each query id
    """
    for i, query in enumerate(all_queris_splitted_after_stemming):
        title_dict_index = create_index_dict(title_index, query, 300) ##remeber to save 300 docs here!!
        body_dict_index = create_index_dict(body_index, query, 300)
        anchor_dict_index = create_index_dict(anchor_index, query, 300)

        list_of_doc_id_scores_merged = defaultdict(list)
        list_of_doc_id_scores_merged['title'] = title_dict_index
        list_of_doc_id_scores_merged['body'] = body_dict_index
        list_of_doc_id_scores_merged['anchor'] = anchor_dict_index

        queries_results_dict[i] = list_of_doc_id_scores_merged

    with open("top_k_results_from_all_indexes.json", 'w') as f:
        json.dump(queries_results_dict, f)

def create_index_dict(index, query, k=300):
       return calculate_BM_25_of_a_single_query(OPTIMAL_K,OPTIMAL_B,query , index)


def calculate_BM_25_of_a_single_query(k, b, query, idx):
    """
    returns list of tuples, each tuple is (doc_id,score)
    """
    counted_query = Counter(query) #tf of each term in query
    docs_scores = Counter() #holds and counts the scores for each BM_25 score for each doc
    for term, query_tf in counted_query.items():
        if not term in idx.df:
            continue
        pls_of_term = idx.read_posting_list(term)
        for doc_id, tf, max_tf, doc_len in pls_of_term:
            docs_scores[doc_id] += BM_25_of_a_single_term_in_query(k,b,tf,doc_len, query_tf, term)

    return list(docs_scores.items())[:300]

def BM_25_of_a_single_term_in_query(k, b, tf, doc_len, query_tf, term):
    numerator = query_tf*(k+1)*tf
    denumerator = tf+k*(1-b+((b*doc_len)/AVGDL))
    return (numerator/denumerator)*idf[term]


main()




