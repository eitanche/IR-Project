from inverted_index_gcp import InvertedIndex

index = InvertedIndex.read_index("postings_small_body_gcp","index")
print(index.posting_locs["python"])