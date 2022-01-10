import json

import requests
from time import time

import os
url = 'http://35.222.182.191:8080'
# place the domain you got from ngrok or GCP IP below.
# url = 'http://34.134.225.55:8080'
with open(f'{os.pardir}{os.sep}experiments_on_small_index{os.sep}queries_train.json',"r") as f:
    queries = json.load(f)

def average_precision(true_list, predicted_list, k=40):
    true_set = frozenset(true_list)
    predicted_list = predicted_list[:k]
    precisions = []
    for i, doc_id in enumerate(predicted_list):
        if doc_id in true_set:
            prec = (len(precisions) + 1) / (i + 1)
            precisions.append(prec)
    if len(precisions) == 0:
        return 0.0
    return round(sum(precisions) / len(precisions), 3)

def map_at_k(relevant_docs, doc_id_oredered_bm_25_scores,k=40):
    sum = 0
    hit = 0
    doc_id_oredered_bm_25_scores = doc_id_oredered_bm_25_scores[:k]
    for i in range(len(doc_id_oredered_bm_25_scores)):#we already cuted the list to size 40 in order_scores funciton
        if doc_id_oredered_bm_25_scores[i] in relevant_docs:
            hit += 1
            sum += hit / (i + 1)
    if hit == 0:
        return 0
    return sum/hit

qs_res = []

for q, true_wids in queries.items():
    # if q!="how to make pasta":
    #     continue
    duration, ap = None, None
    t_start = time()
    try:
        res = requests.get(url + '/search', {'query': q,'case':'4'}, timeout=35)
        duration = time() - t_start
        if res.status_code == 200:
            pred_wids, _ = zip(*res.json())
            ap = average_precision(true_wids, pred_wids)
    except Exception as e:
        print(e)
        print(q)

    qs_res.append((q, duration, ap))
count = 0
total_prec = 0
total_time = 0
for query, duration, map in qs_res:
    if duration!=None and map!=None:
        count+=1
        total_prec+=map
        total_time+=duration
if count!=0:
    print(f"average time is: {total_time/count}")
    print(f"average precision is: {total_prec/count}")
print(qs_res)
#print(sorted(qs_res,key=lambda x:x[1], reverse=True))
stopper = time.time()

# #response = requests.post("http://34.134.225.55:8080/get_pageview",json=[128397, 1283791, 45908, 239048, 129367, 32478928])
#
# stopper = time.time() - stopper
# print(len(response.json()))
# print([tup[1] for tup in response.json()])
# print(stopper)
# #
# response = requests.post("http://127.0.0.1:8080/search_anchor",params={'query':'how to make pasta'})
# print(len(response.json()))
# print(response.json())

#black friday
#best ice cream flavour
#what is critical race theory