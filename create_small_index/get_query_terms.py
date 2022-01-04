import json
from nltk.stem.porter import *

with open("../experiments_on_small_index/queries_train.json") as f:
    queries = json.load(f)
stemmer = PorterStemmer()
query_terms = [stemmer.stem(term) for query in queries for term in query.split(" ")]
# query_terms = [term for query in queries for term in query.split(" ")]
print(query_terms)