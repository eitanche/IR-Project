import json
import pickle
from math import log
with open("page_rank.json","r") as f:
    page_rank_dict = json.load(f)

with open("normalized_page_views_of_each_doc.pkl","rb") as f:
    normalized_page_views_dict = pickle.load(f)

with open("dict_of_id_title.pkl", "rb") as f:
    id_2_title_dict = pickle.load(f)

final_pages_score_dict_without_index = dict()

for doc_id in id_2_title_dict.keys():
    final_pages_score_dict_without_index[doc_id]=normalized_page_views_dict.get(doc_id,0)*0.03 + log((float(page_rank_dict.get(str(doc_id), 0))+1),5) * 0.19

with open("combined_page_rank_page_view_score_weighted_019_003","wb") as f:
    pickle.dump(final_pages_score_dict_without_index,f)

# with open("combined_page_rank_page_view_score_weighted_019_003","rb") as f:
#     final_pages_score_dict_without_index = pickle.load(f)
#
# print(sorted(final_pages_score_dict_without_index.items(), key = lambda x:x[1],reverse=True)[:10])