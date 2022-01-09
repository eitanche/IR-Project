import csv
import json
from collections import defaultdict
import os
from inverted_index_gcp import InvertedIndex


need_words = ['python', 'data', 'science', 'migraine', 'chocolate', 'how', 'to', 'make', 'pasta', 'Does', 'pasta', 'have', 'preserv', 'how', 'google', 'works', 'what', 'is', 'information', 'retrieval', 'NBA', 'yoga', 'how', 'to', 'not', 'kill', 'plants', 'masks', 'black', 'friday', 'why', 'do', 'men', 'have', 'nipples', 'rubber', 'duck', 'michelin', 'what', 'to', 'watch', 'best', 'marvel', 'movie', 'how', 'tall', 'is', 'the', 'eiffel', 'tower', 'where', 'does', 'vanilla', 'flavoring', 'come', 'from', 'best', 'ice', 'cream', 'flavour', 'how', 'to', 'tie', 'a', 'tie', 'how', 'to', 'earn', 'money', 'online', 'what', 'is', 'critical', 'race', 'theory', 'what', 'space', 'movie', 'was', 'made', 'in', '1992', 'how', 'to', 'vote', 'google', 'trends', 'dim', 'sum', 'ted', 'fairy', 'tale']
all_page_ranks_dict = {}

body_index = InvertedIndex.read_index("small_indexes_pickles_and_bins/postings_small_body_gcp", "index")
title_index = InvertedIndex.read_index("small_indexes_pickles_and_bins/postings_small_title_gcp", "index")
anchor_index = InvertedIndex.read_index("small_indexes_pickles_and_bins/postings_small_anchor_text_gcp_2", "index")


def update_dict_of_needed_page_ranks(index, term ,base_dir, all_page_ranks_dict):
    terms_from_pls = index.read_posting_list(term, base_dir)
    for pls in terms_from_pls:
        # print(pls[0])
        if str(pls[0]) in all_page_ranks_dict.keys():
            # print("HERE")
            new_dict[pls[0]] = all_page_ranks_dict[str(pls[0])]


with open(f'{os.pardir}/Page_rank_big.csv', mode='r') as infile:
    reader = csv.reader(infile)
    all_page_ranks_dict = {rows[0]:rows[1] for rows in reader if rows[0]}
    # print(list(mydict.items())[0])
    new_dict = defaultdict(list)
    for term in need_words:
        if term in body_index.df:
            update_dict_of_needed_page_ranks(body_index, term, "small_indexes_pickles_and_bins/postings_small_body_gcp", all_page_ranks_dict)
        if term in title_index.df:
            update_dict_of_needed_page_ranks(title_index, term,
                                             "small_indexes_pickles_and_bins/postings_small_title_gcp", all_page_ranks_dict)
        if term in anchor_index.df:
            update_dict_of_needed_page_ranks(anchor_index, term,
                                             "small_indexes_pickles_and_bins/postings_small_anchor_text_gcp_2", all_page_ranks_dict)


    print(list(new_dict.items())[:10])






    with open('jsons/page_rank/small_page_rank.json', mode='w') as outfile:
        json.dump(new_dict,outfile)



