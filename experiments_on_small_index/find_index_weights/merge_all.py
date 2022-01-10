import json
import os
import time
from collections import Counter,defaultdict
from experiments_on_small_index.inverted_index_gcp import InvertedIndex
from math import log10
from nltk.stem.porter import *
# from calculate_BM_25 import *





stemmer = PorterStemmer()


####title####
OPTIMAL_K_FOR_TITLE = 0.3
OPTIMAL_B_FOR_TITLE = 1
####body####
OPTIMAL_K_FOR_BODY = 2.7
OPTIMAL_B_FOR_BODY = 0.05
####anchor####
OPTIMAL_K_FOR_ANCHOR = 0.1
OPTIMAL_B_FOR_ANCHOR = 0.2

####working on queries
with open("queries_train.json") as f:
    queries_dict = json.load(f)

def get_queries(queries_dictionary):
    return list(queries_dictionary.keys())

def stem_queries(queries_seperated):

    all_queris_splitted_after_stemming = []
    for query in queries_seperated:
        splited_query = query.split(' ')
        splited_stemmed_query = []
        for term_from_query in splited_query:
            splited_stemmed_query.append(stemmer.stem(term_from_query)) #list of query terms after stemming
        all_queris_splitted_after_stemming.append(splited_stemmed_query) #list of lists of queries
    return all_queris_splitted_after_stemming


queries_seperated = get_queries(queries_dict) # return keys of the dict
all_queris_splitted_after_stemming = stem_queries(queries_seperated)


####
queries_results_dict = defaultdict(list)

####

#### load indexes
location_of_index = f"{os.pardir}{os.sep}"
title_index = InvertedIndex.read_index(location_of_index+"title_two_word_index","index")
body_index  = InvertedIndex.read_index(location_of_index+"body_two_word_index","index")
anchor_index = InvertedIndex.read_index(location_of_index+"anchor_two_word_index","index")


### n #####
def calculate_number_of_unique_docs_in_index(index,folder):
    unique_docs_set = []
    for word in index.df:
        pls = index.read_posting_list(word,location_of_index+folder)
        for doc_id,_,_,_ in pls:
            unique_docs_set.append(doc_id)
    return len(set(unique_docs_set))




title_base_dir = "title_two_word_index"
N_TITLE = calculate_number_of_unique_docs_in_index(title_index,"title_two_word_index")
title_index.N = N_TITLE
title_index.BASE_DIR = location_of_index+title_base_dir

N_BODY = calculate_number_of_unique_docs_in_index(body_index, "body_two_word_index")
body_index.N = N_BODY
body_base_dir = "body_two_word_index"
body_index.BASE_DIR = location_of_index+body_base_dir

N_ANCHOR = calculate_number_of_unique_docs_in_index(anchor_index, "anchor_two_word_index")
anchor_index.N = N_ANCHOR
anchor_base_dir = "anchor_two_word_index"
anchor_index.BASE_DIR = location_of_index+anchor_base_dir


#### weights

w_titles = [i*0.01 for i in range (101)]
w_bodies = [i*0.01 for i in range (101)]
w_anchors =[i*0.01 for i in range (101)]






def main():
    """

    """
    write_q_id_dict() #creates for each query 900
    dict_of_precisions_for_all_queries = defaultdict(list)
    for w_title in w_titles:
        for w_body in w_bodies:
            for w_anchor in w_anchors:
                if 0.99 < w_anchor+w_body+w_title < 1.01:
                    list_of_all_queries_precision = []
                    stopper = time.time()
                    for i in range(30): #for each query
                        sorted_docs_and_scores = get_top_40_docs_for_single_query(w_title, w_body, w_anchor, i) #40 sorted list of (doc_id,score)
                        list_of_all_queries_precision.append(map_at_k(sorted_docs_and_scores,40,i)) #takes all the docs we have and makes list which contains 30 precisions
                    dict_of_precisions_for_all_queries["title wight:"+str(w_title)+","+"body wight:"+str(w_body)+","+"anchor wight:"+str(w_anchor)] = list_of_all_queries_precision
                    print(time.time() - stopper)
    with open("only_indexes_merged_weights_precision_scores_two_word_index_precision_new.json", 'w') as f:
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
        title_dict_index = create_index_dict(title_index, query, 300,OPTIMAL_K_FOR_TITLE,OPTIMAL_B_FOR_TITLE) ##remeber to save 300 docs here!!
        body_dict_index = create_index_dict(body_index, query, 300,OPTIMAL_K_FOR_BODY,OPTIMAL_B_FOR_BODY)
        anchor_dict_index = create_index_dict(anchor_index, query, 300,OPTIMAL_K_FOR_ANCHOR,OPTIMAL_B_FOR_ANCHOR)

        list_of_doc_id_scores_merged = defaultdict(list)
        list_of_doc_id_scores_merged['title'] = title_dict_index
        list_of_doc_id_scores_merged['body'] = body_dict_index
        list_of_doc_id_scores_merged['anchor'] = anchor_dict_index

        queries_results_dict[i] = list_of_doc_id_scores_merged

    with open("top_k_results_from_all_indexes_two_word_index.json", 'w') as f:
        json.dump(queries_results_dict, f)

def create_index_dict(index, query, k ,optimal_k,optimal_b):
       return calculate_BM_25_of_a_single_query(optimal_k,optimal_b,query , index)


def calculate_BM_25_of_a_single_query(k, b, query, idx):
    """
    returns list of tuples, each tuple is (doc_id,score)
    """
    counted_query = Counter(query) #tf of each term in query
    docs_scores = Counter() #holds and counts the scores for each BM_25 score for each doc
    for term, query_tf in counted_query.items():
        if not term in idx.df:
            continue
        pls_of_term = idx.read_posting_list(term,idx.BASE_DIR)
        for doc_id, tf, max_tf, doc_len in pls_of_term:
            docs_scores[doc_id] += BM_25_of_a_single_term_in_query(idx,k,b,tf,doc_len, query_tf, term)
    two_word_query = [query[i] + " " + query[i + 1] for i in range(len(query) - 1)]
    counted_query = Counter(two_word_query)
    for term, query_tf in counted_query.items():
        if term not in idx.df:
            continue
        pls_of_term =  idx.read_posting_list(term,idx.BASE_DIR)
        for doc_id, tf, max_tf, doc_len in pls_of_term:
            docs_scores[doc_id] += BM_25_of_a_single_term_in_query(idx, k, b, tf * 0.5, doc_len, query_tf, term)

    return dict(sorted(docs_scores.items(), key=lambda x: x[1], reverse=True)[:300])

def BM_25_of_a_single_term_in_query(index,k, b, tf, doc_len, query_tf, term):
    numerator = query_tf*(k+1)*tf
    denumerator = tf+k*(1-b+((b*doc_len)/index.AVGDL))
    return (numerator/denumerator)*log10((index.N+1)/index.df[term])


def map_at_k(doc_id_ordered,k,i):
    hit = 0
    relevant_docs = list(queries_dict.values())[i]
    for i in range(len(doc_id_ordered)):  # we already cuted the list to size 40 in order_scores funciton
        if doc_id_ordered[i][0] in relevant_docs:
            hit += 1
    return hit/40

# def map_at_k(doc_id_oredered_bm_25_scores,k,i):
#     sum = 0
#     hit = 0
#     relevant_docs = list(queries_dict.values())[i] #the relevant documents from the i_th query list
#     for i in range(len(doc_id_oredered_bm_25_scores)):#we already cuted the list to size 40 in order_scores funciton
#         if doc_id_oredered_bm_25_scores[i][0] in relevant_docs:
#             hit += 1
#             sum += hit / (i + 1)
#     if hit == 0:
#         return 0
#     return sum/hit



main()



