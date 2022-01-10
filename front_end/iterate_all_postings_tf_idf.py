from body_tf_idf_inverted_index_gcp_without_stemming_2 import InvertedIndex


def run_all_pls_in_index(base_dir, bucket_name):
    index = InvertedIndex.read_index(bucket_name,base_dir,"index")
    for word in index.df:
        for pls in index.read_posting_list(word, "anchor_binary_index"):
            x=1

    print(f"index: {base_dir} is ok")