import json

with open("queries_train.json") as f:
    queries = json.load(f)

query_terms = [term for query in queries for term in query.split(" ")]
print(query_terms)