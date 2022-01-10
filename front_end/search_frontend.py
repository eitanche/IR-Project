import json
import pickle
from flask import Flask, request, jsonify
import os
from anchor_title_simple_search import get_achor_title_all_docs
from body_tf_idf_search import get_top_100_tf_idf_scores
from best_search_function_for_frontend import *
from nltk.stem.porter import *
from merged_all_inexes_with_bm_25_score import InvertedIndex
import nltk
from nltk.corpus import stopwords
class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        with open(DOC_ID_TO_TITLE, "rb") as f:
            self.doc_id_to_title = pickle.load(f)

        with open(PAGE_VIEWS_PAGE_RANK_DICT, "rb") as f:
            self.page_views_page_rank_dict = pickle.load(f)

        # if there is a ram problems, this ones need to sit in their own function
        with open(DOC_ID_TO_PAGE_VIEWS, "rb") as f:
            self.doc_id_to_page_views = pickle.load(f)

        with open(PAGE_RANK_FILE_NAME, 'r') as f:
            self.page_ranks_dict = json.load(f)
        # if there is a ram problems, this ones need to sit in their own function
        nltk.download('stopwords')
        english_stopwords = frozenset(stopwords.words('english'))
        corpus_stopwords = ["category", "references", "also", "external", "links",
                            "may", "first", "see", "history", "people", "one", "two",
                            "part", "thumb", "including", "second", "following",
                            "many", "however", "would", "became"]

        self.all_stopwords = english_stopwords.union(corpus_stopwords)
        #self.best_final_merged_index = InvertedIndex.read_index(BUCKET_NAME, BEST_FINAL_MERGED_INDEX, "index") ################# load the final index here
        self.best_final_merged_index = InvertedIndex.read_index_from_local_storage(BEST_FINAL_MERGED_INDEX_LOCAL,
                                                                "index")  ################# load the final index here

        self.stemmer = PorterStemmer()
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)

app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

####global names####

#update TITLE AND ANCHOR FOLDERS TO BE READ FROM THE BUCKET

BUCKET_NAME = "idx316179928316366087idx"
PAGE_RANK_FILE_NAME = f'{os.pardir}{os.sep}final_indexes_and_files{os.sep}page_rank.json'
DOC_ID_TO_TITLE = f"{os.pardir}{os.sep}final_indexes_and_files{os.sep}dict_of_id_title.pkl"
DOC_ID_TO_PAGE_VIEWS = f"{os.pardir}{os.sep}final_indexes_and_files{os.sep}page_views_of_each_doc.pkl"
PAGE_VIEWS_PAGE_RANK_DICT = f"{os.pardir}{os.sep}final_indexes_and_files{os.sep}combined_page_rank_page_view_score_weighted"
TITLE_INDEX_FOLDER = f"simple_indexes_for_frontend/title_binary_index"
ANCHOR_INDEX_FOLDER = f"simple_indexes_for_frontend/anchor_binary_index"
BEST_FINAL_MERGED_INDEX = "merged_corpus_index"
BEST_FINAL_MERGED_INDEX_LOCAL = f"{os.pardir}{os.sep}final_indexes_and_files{os.sep}merged_corpus_index_two_words_sorted"
####global names####
@app.route("/search")
def search():
    ''' Returns up to a 100 of your best search results for the query. This is 
        the place to put forward your best search engine, and you are free to
        implement the retrieval whoever you'd like within the bound of the 
        project requirements (efficiency, quality, etc.). That means it is up to
        you to decide on whether to use stemming, remove stopwords, use 
        PageRank, query expansion, etc.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each 
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    case = request.args.get('case', '')
    if len(query) == 0:
      return jsonify(res)
    #####
    #word2vec?
    #####
    # BEGIN SOLUTION
    if case == '1':
        best_top_doc_ids_and_scores = get_top_100_best_search_1(query,app.best_final_merged_index,app.page_views_page_rank_dict, app.stemmer,app.all_stopwords)
        res = [[doc_id, app.doc_id_to_title.get(doc_id,'Relevant document without title')] for doc_id, score in best_top_doc_ids_and_scores]
    elif case == '2':
        best_top_doc_ids_and_scores = get_top_100_best_search_2(query, app.best_final_merged_index,
                                                              app.page_views_page_rank_dict, app.stemmer,
                                                              app.all_stopwords)
        res = [[doc_id, app.doc_id_to_title.get(doc_id, 'Relevant document without title')] for doc_id, score in
               best_top_doc_ids_and_scores]
    elif case == '3':
        best_top_doc_ids_and_scores = get_top_100_best_search_3(query, app.best_final_merged_index,
                                                              app.page_views_page_rank_dict, app.stemmer,
                                                              app.all_stopwords)
        res = [[doc_id, app.doc_id_to_title.get(doc_id, 'Relevant document without title')] for doc_id, score in
               best_top_doc_ids_and_scores]
    elif case == '4':
        best_top_doc_ids_and_scores = get_top_100_best_search_4(query, app.best_final_merged_index,
                                                              app.page_views_page_rank_dict, app.stemmer,
                                                              app.all_stopwords)
        res = [[doc_id, app.doc_id_to_title.get(doc_id, 'Relevant document without title')] for doc_id, score in
               best_top_doc_ids_and_scores]
    # END SOLUTION
    x = jsonify(res)
    return x

@app.route("/search_body")
def search_body():
    ''' Returns up to a 100 search results for the query using TFIDF AND COSINE
        SIMILARITY OF THE BODY OF ARTICLES ONLY. DO NOT use stemming. DO USE the 
        staff-provided tokenizer from Assignment 3 (GCP part) to do the 
        tokenization and remove stopwords. 

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search_body?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each 
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    top_doc_ids_and_scores = get_top_100_tf_idf_scores(query)
    res = [[doc_id, app.doc_id_to_title.get(doc_id,'Relevant document without title')] for doc_id, score in top_doc_ids_and_scores]
    # END SOLUTION
    return jsonify(res)

@app.route("/search_title")
def search_title():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE TITLE of articles, ordered in descending order of the NUMBER OF 
        QUERY WORDS that appear in the title. For example, a document with a 
        title that matches two of the query words will be ranked before a 
        document with a title that matches only one query term. 

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_title?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    doc_ids_and_scores = get_achor_title_all_docs(query, TITLE_INDEX_FOLDER)
    res = [[doc_id[0],app.doc_id_to_title.get(doc_id[0],'Relevant document without title')] for doc_id,score in doc_ids_and_scores]
    # END SOLUTION
    return jsonify(res)


@app.route("/search_anchor")
def search_anchor():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD
        IN THE ANCHOR TEXT of articles, ordered in descending order of the
        NUMBER OF QUERY WORDS that appear in anchor text linking to the page.
        For example, a document with a anchor text that matches two of the
        query words will be ranked before a document with anchor text that
        matches only one query term.

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_anchor?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    doc_ids_and_scores = get_achor_title_all_docs(query, ANCHOR_INDEX_FOLDER)
    res = [[doc_id[0],app.doc_id_to_title.get(doc_id[0],'Relevant document without title')] for doc_id,score in doc_ids_and_scores]
    # END SOLUTION
    return jsonify(res)




@app.route("/get_pagerank", methods=['POST'])
def get_pagerank():
    ''' Returns PageRank values for a list of provided wiki article IDs. 

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pagerank
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pagerank', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of floats:
          list of PageRank scores that correrspond to the provided article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    for id in wiki_ids:
        if str(id) in app.page_ranks_dict:
            res.append(app.page_ranks_dict[str(id)])
    # END SOLUTION
    return jsonify(res)

@app.route("/get_pageview", methods=['POST'])
def get_pageview():
    ''' Returns the number of page views that each of the provide wiki articles
        had in August 2021.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pageview
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pageview', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ints:
          list of page view numbers from August 2021 that correrspond to the 
          provided list article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    for id in wiki_ids:
        if id in app.doc_id_to_page_views:
            res.append(app.doc_id_to_page_views[id])
    # END SOLUTION
    return jsonify(res)


if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080, debug=False)