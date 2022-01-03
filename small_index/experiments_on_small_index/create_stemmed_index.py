from inverted_index_changes import InvertedIndex

index = InvertedIndex().read_index("small_body_index", "postings_gcp_index")
# print(len(index.read_posting_list["python"]))
# print(index.read_posting_list["python"])