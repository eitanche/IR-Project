import csv
import json
from collections import defaultdict

from inverted_index_gcp import InvertedIndex


# need_words = ['python', 'data', 'science', 'migraine', 'chocolate', 'how', 'to', 'make', 'pasta', 'Does', 'pasta', 'have', 'preservatives?', 'how', 'google', 'works', 'what', 'is', 'information', 'retrieval', 'NBA', 'yoga', 'how', 'to', 'not', 'kill', 'plants', 'masks', 'black', 'friday', 'why', 'do', 'men', 'have', 'nipples', 'rubber', 'duck', 'michelin', 'what', 'to', 'watch', 'best', 'marvel', 'movie', 'how', 'tall', 'is', 'the', 'eiffel', 'tower', 'where', 'does', 'vanilla', 'flavoring', 'come', 'from', 'best', 'ice', 'cream', 'flavour', 'how', 'to', 'tie', 'a', 'tie', 'how', 'to', 'earn', 'money', 'online', 'what', 'is', 'critical', 'race', 'theory', 'what', 'space', 'movie', 'was', 'made', 'in', '1992', 'how', 'to', 'vote', 'google', 'trends', 'dim', 'sum', 'ted', 'fairy', 'tale']
mydict = {}

index = InvertedIndex.read_index("small_indexes/small_body_index", "postings_gcp_index")
with open('/Page_rank_big.csv', mode='r') as infile:
    reader = csv.reader(infile)
    mydict = {rows[0]:rows[1] for rows in reader if rows[0]}
    # print(list(mydict.items())[0])
    new_dict = defaultdict(list)
    for term in need_words:
        # try:
        if term in index.df:
            terms_from_pls = index.read_posting_list(term)
            for pls in terms_from_pls:
                # print(pls[0])
                if str(pls[0]) in mydict.keys():
                    # print("HERE")
                    new_dict[pls[0]] = mydict[str(pls[0])]
            # except Exception as e:
            #     print(e)
            #     # continue

    print(list(new_dict.items())[:10])






    with open('small_page_rank.json', mode='w') as outfile:
        json.dump(mydict,outfile)



