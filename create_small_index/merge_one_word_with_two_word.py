from inverted_index_gcp import InvertedIndex
import os
ONE_WORD_PATH = "/experiments_on_small_index/anchor_two_word_index"  #CHANGE HERE

two_word_index = InvertedIndex.read_index(ONE_WORD_PATH,"two_word_index")
one_word_index = InvertedIndex.read_index(ONE_WORD_PATH,"one_word_index")

for term in two_word_index.df:
    one_word_index.posting_locs[term]=two_word_index.posting_locs[term]
    one_word_index.df[term] = two_word_index.df[term]
one_word_index.write_index(ONE_WORD_PATH,"index")

# index = InvertedIndex.read_index(ONE_WORD_PATH,"index")
# print(index.read_posting_list("data scienc",ONE_WORD_PATH))