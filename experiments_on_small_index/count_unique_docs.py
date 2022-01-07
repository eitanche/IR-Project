from inverted_index_gcp import InvertedIndex
from math import log
import time
import os
import hashlib
import pickle
def _hash(s):
    return hashlib.blake2b(bytes(s, encoding='utf8'), digest_size=5).hexdigest()
idx = InvertedIndex.read_index(f"small_indexes{os.sep}postings_small_title_gcp","new_index")

#print(idx.df["wa"])

NUM_BUCKETS = 124
def token2bucket_id(token):
  return int(_hash(token),16) % NUM_BUCKETS

#print(idx.df["migrain"])
for term in idx.df:
    try:
        x = idx.read_posting_list(term, f"small_indexes{os.sep}postings_small_title_gcp")
    except:
        print(term)
        continue
# for w,pl in idx.posting_lists_iter(f"small_indexes{os.sep}postings_small_text"):
#     x=1



