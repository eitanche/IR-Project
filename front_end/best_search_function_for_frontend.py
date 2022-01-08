import re
import pickle
from collections import Counter
from body_tf_idf_inverted_index_gcp_without_stemming import InvertedIndex

RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)
MERGED_BM25_INDEX_FOLDER = ""

def get_top_100_best_search(query,index,page_vies_page_rank_dict):

    words_in_query_tf = Counter([token.group() for token in RE_WORD.finditer(query.lower())])
    doc_id_to_score = Counter()
    for word,query_tf in words_in_query_tf.items():
        if word in index.df:
            for doc_id, score in index.read_posting_list(word, MERGED_BM25_INDEX_FOLDER):
                doc_id_to_score[doc_id]+= (query_tf*score)

    for doc_id,score in doc_id_to_score.items():
        doc_id_to_score[doc_id] = score*0.78 + page_vies_page_rank_dict.get(doc_id,0) #####the score here is already wighted


    return sorted(doc_id_to_score.items(), key=lambda x:x[1], reverse=True)[:100]





