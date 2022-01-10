import json
import pickle
import re
import time
from collections import defaultdict, Counter
from math import log10

import gensim.downloader
import nltk
from nltk.corpus import stopwords
from inverted_index_gcp import InvertedIndex
from nltk.stem import PorterStemmer
nltk.download('stopwords')

english_stopwords = frozenset(stopwords.words('english'))
corpus_stopwords = ["category", "references", "also", "external", "links",
                    "may", "first", "see", "history", "people", "one", "two",
                    "part", "thumb", "including", "second", "following",
                    "many", "however", "would", "became"]

all_stopwords = english_stopwords.union(corpus_stopwords)
RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)
stemmer = PorterStemmer()
BASE_DIR = "small_indexes_pickles_and_bins/body_two_word_index"
index = InvertedIndex.read_index(BASE_DIR, "index")
wiki_100 = gensim.downloader.load('glove-wiki-gigaword-100')

minimum_score_for_expansion = [0.05*i for i in range(12,21)]


docs = set()
idf = defaultdict(list)
inverted_index_dict = defaultdict(list)
"""
load inverted index into dictionary and count all the docs in the corpus int docs set
"""
for term in index.df.keys():
    inverted_index_dict[term] = index.read_posting_list(term,BASE_DIR)
    docs.update([tup[0] for tup in inverted_index_dict[term]])

#print(docs)
#"train": 0.608095937037738, "test": 0.7142173274622993}


N = len(docs) # number of all documents in the inverted index

"""
calculates idf for each term in the index
"""
for term in index.df.keys():
    idf[term] = log10((N+1)/index.df[term])



queries_dict = {}
with open("queries_train.json") as f:
    queries_dict = json.load(f)



def main():
    dict_of_scores_to_presicions = defaultdict(list)
    for score in minimum_score_for_expansion:
        dict_of_scores_to_presicions[score] = calculate_presicion(score,list(queries_dict.keys()))

    with open("../charts/charts_jsons/query expanssion/all_scores_of_query_expansion.json", 'w') as f:
        json.dump(dict_of_scores_to_presicions,f)


def query_expansion(query,threshhold):
    tokens = [token.group() for token in RE_WORD.finditer(query.lower())]
    # stopper = time.time()
    tokens = [token for token in tokens if token not in all_stopwords]
    # print(stopper - time.time())
    similarities_dict = defaultdict(list)
    for token in tokens:
        if token in wiki_100:
            lst_of_word_score_similarities = wiki_100.most_similar(token)
            counter = 0
            for word,score in lst_of_word_score_similarities:
                if score >= threshhold:
                    counter += 1
                    similarities_dict[stemmer.stem(token)] += [stemmer.stem(word)]
                    if counter > 0:
                        break
                else:
                    break
    two_words_ngrams = []
    for i in range(len(tokens) - 1):
        two_words_ngrams.append((stemmer.stem(tokens[i]), stemmer.stem(tokens[i + 1])))

    new_query = [stemmer.stem(word) for words in similarities_dict.values() for word in words]\
                + [stemmer.stem(token) for token in tokens]

    for first_word, second_word in two_words_ngrams:
        for x in similarities_dict[first_word] + [first_word]:
            for y in similarities_dict[second_word] + [second_word]:
                new_query.append(x + " " + y)

    return new_query






def calculate_presicion(threshhold,queries):
    presicion_list = []
    for i,q in enumerate(queries): # for each query calc score
        new_q = query_expansion(q,threshhold)
        presicion_list.append(calculate_score_of_a_single_query(new_q,i))
    return presicion_list




def calculate_score_of_a_single_query(query, i):
    """
    i: i_th query in the training set
    """
    doc_id_bm_25_scores_tuple_list = calculate_BM_25_of_a_single_query(query)
    doc_id_oredered_bm_25_scores = order_scores(doc_id_bm_25_scores_tuple_list)
    presicion = map_at_k(doc_id_oredered_bm_25_scores, 40, i) # returns the score of the query
    return presicion


def calculate_BM_25_of_a_single_query(query):
    """
    returns list of tuples, each tuple is (doc_id,score)
    """
    counted_query = Counter(query) #tf of each term in query
    docs_scores = Counter() #holds and counts the scores for each BM_25 score for each doc
    for term, query_tf in counted_query.items():
        pls_of_term = inverted_index_dict[term]
        for doc_id, tf, max_tf, doc_len in pls_of_term:
            docs_scores[doc_id] += BM_25_of_a_single_term_in_query(tf,doc_len, query_tf, term)
    # two_word_query = [query[i]+" "+query[i+1] for i in range(len(query)-1)]
    return docs_scores.items()




def BM_25_of_a_single_term_in_query(tf, doc_len, query_tf, term):
    k = 2.7
    b = 0.05
    numerator = query_tf*(k+1)*tf
    denumerator = tf+k*(1-b+((b*doc_len)/319.52423534118395))
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





print('how' in english_stopwords)



