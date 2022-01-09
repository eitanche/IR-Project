from title_anchor_binary_inverted_index_gcp_without_stemming import InvertedIndex
import re
from collections import Counter

RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)


def get_achor_title_all_docs(query, folder):
    index = InvertedIndex.read_index(folder,"index")

    uniques_words_in_query = set([token.group() for token in RE_WORD.finditer(query.lower())])

    doc_id_to_score = Counter()
    for word in uniques_words_in_query:
        if word in index.df:
            for doc_id in index.read_posting_list(word, folder):
                doc_id_to_score[doc_id]+=1

    return sorted(doc_id_to_score.items(), key=lambda x: x[1], reverse=True )
