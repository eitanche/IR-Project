import time
from collections import defaultdict, Counter
from inverted_index_gcp import InvertedIndex
from nltk.stem.porter import *
import json
from math import log
import os

FILE_NAME = "all_k_b_scores_body_1200_sampels"
INDEX_FOLDER = "postings_small_body_gcp"


stemmer = PorterStemmer()
index = InvertedIndex.read_index(INDEX_FOLDER,"index")
AVGDL = index.AVGDL
query_terms = ['python', 'data', 'scienc', 'migrain', 'chocol', 'how', 'to', 'make', 'pasta', 'doe', 'pasta', 'have', 'preserv', 'how', 'googl', 'work', 'what', 'is', 'inform', 'retriev', 'nba', 'yoga', 'how', 'to', 'not', 'kill', 'plant', 'mask', 'black', 'friday', 'whi', 'do', 'men', 'have', 'nippl', 'rubber', 'duck', 'michelin', 'what', 'to', 'watch', 'best', 'marvel', 'movi', 'how', 'tall', 'is', 'the', 'eiffel', 'tower', 'where', 'doe', 'vanilla', 'flavor', 'come', 'from', 'best', 'ice', 'cream', 'flavour', 'how', 'to', 'tie', 'a', 'tie', 'how', 'to', 'earn', 'money', 'onlin', 'what', 'is', 'critic', 'race', 'theori', 'what', 'space', 'movi', 'wa', 'made', 'in', '1992', 'how', 'to', 'vote', 'googl', 'trend', 'dim', 'sum', 'ted', 'fairi', 'tale']
# query_terms = ['python', 'data', 'science', 'migraine', 'chocolate', 'how', 'to', 'make', 'pasta', 'Does', 'pasta', 'have', 'preservatives', 'how', 'google', 'works', 'what', 'is', 'information', 'retrieval', 'NBA', 'yoga', 'how', 'to', 'not', 'kill', 'plants', 'masks', 'black', 'friday', 'why', 'do', 'men', 'have', 'nipples', 'rubber', 'duck', 'michelin', 'what', 'to', 'watch', 'best', 'marvel', 'movie', 'how', 'tall', 'is', 'the', 'eiffel', 'tower', 'where', 'does', 'vanilla', 'flavoring', 'come', 'from', 'best', 'ice', 'cream', 'flavour', 'how', 'to', 'tie', 'a', 'tie', 'how', 'to', 'earn', 'money', 'online', 'what', 'is', 'critical', 'race', 'theory', 'what', 'space', 'movie', 'was', 'made', 'in', '1992', 'how', 'to', 'vote', 'google', 'trends', 'dim', 'sum', 'ted', 'fairy', 'tale']
inverted_index_dict = defaultdict(list)
k_b_presicion_values = defaultdict(list)
queries_dict = {}
with open("queries_train.json") as f:
    queries_dict = json.load(f)

docs = set()
idf = defaultdict(list)

"""
load inverted index into dictionary and count all the docs in the corpus int docs set
"""

for term in query_terms:
    if term in index.df.keys():
        inverted_index_dict[term] = index.read_posting_list(term,INDEX_FOLDER)
        docs.update([tup[0] for tup in inverted_index_dict[term]])



N = len(docs) # number of all documents in the inverted index


"""
calculates idf for each term in the index
"""
for term in query_terms:
    if term in index.df.keys():
        idf[term] = log((N+1)/index.df[term])

# print(N)
# print(idf)



# k_s = [i*0.1 for i in range(2)]

"""
lists of k's and b's for training
"""
k_s = [i*0.05 for i in range(10,51)]
b_s = [i*0.05 for i in range(0,11)]

# k_s=[0.5]
# b_s=[0.5]


def main():

    queries_seperated = get_queries(queries_dict) # return keys of the dict
    all_queris_splitted_after_stemming = stem_queries(queries_seperated)

    count = 1
    for k in k_s:
        for b in b_s:
            k_b_presicion_values[str(k)+","+str(b)] = calculate_presicion(k,b,all_queris_splitted_after_stemming) # returns list of presicions
            print(f"Finished {count} of 400")
            count+=1
    with open(f"{FILE_NAME}.json", 'r') as f:
        old_values = json.load(f)
    for old_key,old_value in old_values.items():
        k_b_presicion_values[old_key]=old_value
    with open(f"{FILE_NAME}_added_values.json", 'w') as f:
        json.dump(k_b_presicion_values,f)


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



def calculate_presicion(k,b,queries):
    presicion_list = []
    for i,q in enumerate(queries): # for each query calc score
        presicion_list.append(calculate_score_of_a_single_query(k,b,q,i))
    return presicion_list




def calculate_score_of_a_single_query(k, b, query, i):
    """
    i: i_th query in the training set
    """
    doc_id_bm_25_scores_tuple_list = calculate_BM_25_of_a_single_query(k,b,query)
    doc_id_oredered_bm_25_scores = order_scores(doc_id_bm_25_scores_tuple_list)
    presicion = map_at_k(doc_id_oredered_bm_25_scores, 40, i) # returns the score of the query
    return presicion


def calculate_BM_25_of_a_single_query(k, b, query):
    """
    returns list of tuples, each tuple is (doc_id,score)
    """
    counted_query = Counter(query) #tf of each term in query
    docs_scores = Counter() #holds and counts the scores for each BM_25 score for each doc
    for term, query_tf in counted_query.items():
        pls_of_term = inverted_index_dict[term]
        for doc_id, tf, max_tf, doc_len in pls_of_term:
            docs_scores[doc_id] += BM_25_of_a_single_term_in_query(k,b,tf,doc_len, query_tf, term)

    return docs_scores.items()

def BM_25_of_a_single_term_in_query(k, b, tf, doc_len, query_tf, term):
    numerator = query_tf*(k+1)*tf
    denumerator = tf+k*(1-b+((b*doc_len)/AVGDL))
    return (numerator/denumerator)*idf[term]


def order_scores(bm_25_scores):
    return sorted(bm_25_scores,reverse=True,key=lambda x:x[1])[:40]


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





main()