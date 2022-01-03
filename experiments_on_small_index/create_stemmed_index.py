from inverted_index_gcp import InvertedIndex
from nltk.stem.porter import *

stemmer = PorterStemmer()
index = InvertedIndex.read_index("small_body_index", "postings_gcp_index")
stemmedIndex = InvertedIndex()
stemmedIndex.AVGDL = index.AVGDL
for word in index.df:
    stemmed_word = stemmer.stem(word)
    stemmedIndex.df[stemmed_word]=stemmedIndex.df.get(stemmed_word, 0 ) + index.df[word]
    stemmedIndex.posting_locs[stemmed_word].extend(index.posting_locs[word])
stemmedIndex.write_index("small_body_index", "small_body_index_stemmed")




#/postings_gcp_103_000.bin'