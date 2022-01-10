import re
import pickle
from collections import Counter
from body_tf_idf_inverted_index_gcp_without_stemming_2 import InvertedIndex
import os

BODY_TF_IDF_INDEX_FOLDER = f"simple_indexes_for_frontend/body_tf_idf_index" #######CHANGE TO BUCKET FOLDER
RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)
BUCKET_NAME = "idx316179928316366087idx"

def get_top_100_tf_idf_scores(query):
    index = InvertedIndex.read_index(BUCKET_NAME, BODY_TF_IDF_INDEX_FOLDER, "index")

    words_in_query_tf = Counter([token.group() for token in RE_WORD.finditer(query.lower())])
    doc_id_to_score = Counter()
    query_len_for_normalization = (sum([query_tf**2 for query_tf in words_in_query_tf.values()]))**0.5

    for word,query_tf in words_in_query_tf.items():
        if word in index.df:
            for doc_id, normalized_tf_idf_score in index.read_posting_list(word, BODY_TF_IDF_INDEX_FOLDER, BUCKET_NAME):
                doc_id_to_score[doc_id]+=query_tf*normalized_tf_idf_score

    for doc_id, score in doc_id_to_score.items():
        doc_id_to_score[doc_id] = score/query_len_for_normalization

    return sorted(doc_id_to_score.items(), key=lambda x:x[1], reverse=True)[:100]
