import re
import os
from collections import Counter
from merged_all_inexes_with_bm_25_score import InvertedIndex
RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)
MERGED_BM25_INDEX_FOLDER = "merged_corpus_index"
MERGED_BM25_INDEX_FOLDER_LOCAL =f"{os.pardir}{os.sep}final_indexes_and_files{os.sep}merged_corpus_index_two_words_sorted"
INDEX_WEIGHT = 0.98
BUCKET_NAME = "idx316179928316366087idx"

def get_top_100_best_search_1(query, index, page_views_page_rank_dict, stemmer, stopwords):

    query_tokens = [stemmer.stem(token.group()) for token in RE_WORD.finditer(query.lower()) if token.group() not in stopwords]
    num_of_tokens = len(query_tokens)

    for i in range(num_of_tokens - 1):
        query_tokens.append(query_tokens[i] + " " + query_tokens[i + 1])
    words_in_query_tf = Counter(query_tokens)
    doc_id_to_score = Counter()
    for word,query_tf in words_in_query_tf.items():
        if word in index.df:
            print(f"{word} df is: {index.df[word]}")
            for doc_id, score in index.read_posting_list_from_local_storage(word, MERGED_BM25_INDEX_FOLDER_LOCAL):
                #for doc_id, score in index.read_posting_list(word, MERGED_BM25_INDEX_FOLDER, BUCKET_NAME):
                doc_id_to_score[doc_id]+= (query_tf*score)


    for doc_id,score in doc_id_to_score.items():
        doc_id_to_score[doc_id] = score * INDEX_WEIGHT + page_views_page_rank_dict.get(doc_id, 0) #####the score here is already wighted
    final_list = sorted(doc_id_to_score.items(), key=lambda x:x[1], reverse=True)[:20]
    return final_list



def get_top_100_best_search_2(query, index, page_views_page_rank_dict, stemmer, stopwords):

    query_tokens = [stemmer.stem(token.group()) for token in RE_WORD.finditer(query.lower()) if token.group() not in stopwords]
    num_of_tokens = len(query_tokens)

    # for i in range(num_of_tokens - 1):
    #     query_tokens.append(query_tokens[i] + " " + query_tokens[i + 1])
    words_in_query_tf = Counter(query_tokens)
    doc_id_to_score = Counter()
    for word,query_tf in words_in_query_tf.items():
        if word in index.df:
            print(f"{word} df is: {index.df[word]}")
            for doc_id, score in index.read_posting_list_from_local_storage(word, MERGED_BM25_INDEX_FOLDER_LOCAL):
                #for doc_id, score in index.read_posting_list(word, MERGED_BM25_INDEX_FOLDER, BUCKET_NAME):
                doc_id_to_score[doc_id]+= (query_tf*score)


    for doc_id,score in doc_id_to_score.items():
        doc_id_to_score[doc_id] = score * INDEX_WEIGHT + page_views_page_rank_dict.get(doc_id, 0) #####the score here is already wighted
    final_list = sorted(doc_id_to_score.items(), key=lambda x:x[1], reverse=True)[:100]
    return final_list




def get_top_100_best_search_3(query, index, page_views_page_rank_dict, stemmer, stopwords):

    tokens = [stemmer.stem(token.group()) for token in RE_WORD.finditer(query.lower()) if token.group() not in stopwords]
    counter_tokens = Counter(tokens)
    query_tokens = []
    num_of_tokens = len(tokens)

    for i in range(num_of_tokens - 1):
        query_tokens.append(tokens[i] + " " + tokens[i + 1])
    counter_for_q_t = Counter(query_tokens)
    for k,v in counter_for_q_t.items():
        counter_for_q_t[k] *= 2
    counter_tokens.update(counter_for_q_t)
    doc_id_to_score = Counter()
    for word,query_tf in counter_tokens.items():
        if word in index.df:
            print(f"{word} df is: {index.df[word]}")
            for doc_id, score in index.read_posting_list_from_local_storage(word, MERGED_BM25_INDEX_FOLDER_LOCAL):
                #for doc_id, score in index.read_posting_list(word, MERGED_BM25_INDEX_FOLDER, BUCKET_NAME):
                doc_id_to_score[doc_id]+= (query_tf*score)


    for doc_id,score in doc_id_to_score.items():
        doc_id_to_score[doc_id] = score * INDEX_WEIGHT + page_views_page_rank_dict.get(doc_id, 0) #####the score here is already wighted
    final_list = sorted(doc_id_to_score.items(), key=lambda x:x[1], reverse=True)[:100]
    return final_list




def get_top_100_best_search_4(query, index, page_views_page_rank_dict, stemmer, stopwords):
    tokens = [stemmer.stem(token.group()) for token in RE_WORD.finditer(query.lower()) if
              token.group() not in stopwords]
    counter_tokens = Counter(tokens)
    query_tokens = []
    num_of_tokens = len(tokens)

    for i in range(num_of_tokens - 1):
        query_tokens.append(tokens[i] + " " + tokens[i + 1])
    counter_for_q_t = Counter(query_tokens)
    for k, v in counter_for_q_t.items():
        counter_for_q_t[k] *= 0.5
    counter_tokens.update(counter_for_q_t)
    doc_id_to_score = Counter()
    for word, query_tf in counter_tokens.items():
        if word in index.df:
            print(f"{word} df is: {index.df[word]}")
            for doc_id, score in index.read_posting_list_from_local_storage(word, MERGED_BM25_INDEX_FOLDER_LOCAL):
                # for doc_id, score in index.read_posting_list(word, MERGED_BM25_INDEX_FOLDER, BUCKET_NAME):
                doc_id_to_score[doc_id] += (query_tf * score)

    for doc_id, score in doc_id_to_score.items():
        doc_id_to_score[doc_id] = score * INDEX_WEIGHT + page_views_page_rank_dict.get(doc_id,
                                                                                       0)  #####the score here is already wighted
    final_list = sorted(doc_id_to_score.items(), key=lambda x: x[1], reverse=True)[:100]
    return final_list




