import re
import pickle
from collections import Counter
from merged_all_inexes_with_bm_25_score import InvertedIndex
import nltk
from nltk.corpus import stopwords
RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)
MERGED_BM25_INDEX_FOLDER = "merged_corpus_index"
INDEX_WEIGHT = 0.98
BUCKET_NAME = "idx316179928316366087idx"

def get_top_100_best_search(query, index, page_views_page_rank_dict, stemmer, stopwords):

    query_tokens = [stemmer.stem(token.group()) for token in RE_WORD.finditer(query.lower()) if token.group() not in stopwords]
    print(f"query after tokenization: {query_tokens}")
    num_of_tokens = len(query_tokens)

    for i in range(num_of_tokens - 1):
        query_tokens.append(query_tokens[i] + " " + query_tokens[i + 1])
    print(f"query after two_worsd: {query_tokens}")
    words_in_query_tf = Counter(query_tokens)
    doc_id_to_score = Counter()
    for word,query_tf in words_in_query_tf.items():
        print(f"looking for word {word} in index")
        if word in index.df:
            print(f"found word in index")
            for doc_id, score in index.read_posting_list(word, MERGED_BM25_INDEX_FOLDER, BUCKET_NAME):
                doc_id_to_score[doc_id]+= (query_tf*score)
            print(f'finished iterating word in index')

    print(f"starting merging with page_views")
    for doc_id,score in doc_id_to_score.items():
        doc_id_to_score[doc_id] = score * INDEX_WEIGHT + page_views_page_rank_dict.get(doc_id, 0) #####the score here is already wighted
    print(f"finished merging with page_views")
    final_list = sorted(doc_id_to_score.items(), key=lambda x:x[1], reverse=True)[:100]
    print(f"finished calculatin final_list")
    return final_list





