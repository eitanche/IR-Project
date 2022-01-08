import json

with open("queries_train.json") as f:
    queries = json.load(f)

for index in [0, 7, 11, 16, 26]:
    print(list(queries.keys())[index])

# query_terms = [term for query in queries for term in query.split(" ")]
# print(query_terms)


