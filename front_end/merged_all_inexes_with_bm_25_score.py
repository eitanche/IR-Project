import pyspark
import sys
from collections import Counter, OrderedDict
import itertools
from itertools import islice, count, groupby
import pandas as pd
import os
import re
from operator import itemgetter
from time import time
from pathlib import Path
import pickle
import struct
from google.cloud import storage
from collections import defaultdict
from contextlib import closing

# Let's start with a small block size of 30 bytes just to test things out.
BLOCK_SIZE = 1999998


class MultiFileWriter:
    """ Sequential binary writer to multiple files of up to BLOCK_SIZE each. """

    def __init__(self, base_dir, name, bucket_name):
        self._base_dir = Path(base_dir)
        self._name = name
        self._file_gen = (open(self._base_dir / f'{name}_{i:03}.bin', 'wb')
                          for i in itertools.count())
        self._f = next(self._file_gen)
        # Connecting to google storage bucket.
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)

    def write(self, b, folder_name):
        locs = []
        while len(b) > 0:
            pos = self._f.tell()
            remaining = BLOCK_SIZE - pos
            # if the current file is full, close and open a new one.
            if remaining == 0:
                self.upload_to_gcp(folder_name)
                self._f = next(self._file_gen)
                pos, remaining = 0, BLOCK_SIZE
            self._f.write(b[:remaining])
            locs.append((self._f.name, pos))
            b = b[remaining:]
        return locs

    def close(self):
        self._f.close()

    def upload_to_gcp(self, folder_name):
        '''
            The function saves the posting files into the right bucket in google storage.
        '''
        file_name = self._f.name
        self._f.close()
        blob = self.bucket.blob(f"{folder_name}/{file_name}")
        with open(self._base_dir / f'{file_name}', 'rb') as f:
            blob.upload_from_file(f)

    @staticmethod
    def do_it():
        print("i am fuckingggggg up to date man! sstfuuu")


class MultiFileReader:
    """ Sequential binary reader of multiple files of up to BLOCK_SIZE each. """

    def __init__(self, base_dir, bucket_name):
        self._open_files = {}
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)
        self.base_dir = base_dir +"/"

    def read(self, locs, n_bytes):
        b = []
        for f_name, offset in locs:
            if f_name not in self._open_files:
                self._open_files[f_name] = self.bucket.get_blob(self.base_dir + f_name).open('rb')
            f = self._open_files[f_name]
            f.seek(offset)
            n_read = min(n_bytes, BLOCK_SIZE - offset)
            b.append(f.read(n_read))
            n_bytes -= n_read
        return b''.join(b)

    def read_from_local_storage(self, locs, n_bytes):
        b = []
        for f_name, offset in locs:
            if f_name not in self._open_files:
                self._open_files[f_name] = open(f_name, 'rb')
            f = self._open_files[f_name]
            f.seek(offset)
            n_read = min(n_bytes, BLOCK_SIZE - offset)
            b.append(f.read(n_read))
            n_bytes -= n_read
        return b''.join(b)

    def close(self):
        for f in self._open_files.values():
            f.close()

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False


from collections import defaultdict
from contextlib import closing

TUPLE_SIZE = 16  # We're going to pack the doc_id and tf values in this
# many bytes.
TF_MASK = 2 ** 16 - 1  # Masking the 16 low bits of an integer


class InvertedIndex:
    def __init__(self, docs={}):
        """ Initializes the inverted index and add documents to it (if provided).
        Parameters:
        -----------
          docs: dict mapping doc_id to list of tokens
        """
        # stores document frequency per term
        self.df = Counter()
        # AVERAGE corpus document length
        self.AVGDL = 0
        # stores total frequency per term
        self.term_total = Counter()
        # stores posting list per term while building the index (internally),
        # otherwise too big to store in memory.
        self._posting_list = defaultdict(list)
        # mapping a term to posting file locations, which is a list of
        # (file_name, offset) pairs. Since posting lists are big we are going to
        # write them to disk and just save their location in this list. We are
        # using the MultiFileWriter helper class to write fixed-size files and store
        # for each term/posting list its list of locations. The offset represents
        # the number of bytes from the beginning of the file where the posting list
        # starts.
        self.posting_locs = defaultdict(list)

        for doc_id, tokens in docs.items():
            self.add_doc(doc_id, tokens)

    def add_doc(self, doc_id, tokens):
        """ Adds a document to the index with a given `doc_id` and tokens. It counts
            the tf of tokens, then update the index (in memory, no storage
            side-effects).
        """
        w2cnt = Counter(tokens)
        self.term_total.update(w2cnt)
        for w, cnt in w2cnt.items():
            self.df[w] = self.df.get(w, 0) + 1
            self._posting_list[w].append((doc_id, cnt))

    def write_index(self, base_dir, name):
        """ Write the in-memory index to disk. Results in the file:
            (1) `name`.pkl containing the global term stats (e.g. df).
        """
        #### GLOBAL DICTIONARIES ####
        self._write_globals(base_dir, name)

    def _write_globals(self, base_dir, name):
        with open(Path(base_dir) / f'{name}.pkl', 'wb') as f:
            pickle.dump(self, f)

    def __getstate__(self):
        """ Modify how the object is pickled by removing the internal posting lists
            from the object's state dictionary.
        """
        state = self.__dict__.copy()
        del state['_posting_list']
        return state

    def read_posting_list(self, w, base_dir, bucket_name):
        with closing(MultiFileReader(base_dir, bucket_name)) as reader:
            locs = self.posting_locs[w]
            b = reader.read(locs, self.df[w] * TUPLE_SIZE)
            posting_list = []
            for i in range(self.df[w]):
                ##################### EDITED TO NEW POSTING LIST#######################

                doc_id,tf_idf_score = struct.unpack("Id", b[i * TUPLE_SIZE:(i + 1) * TUPLE_SIZE])
                posting_list.append((doc_id, tf_idf_score))

            return posting_list

    def read_posting_list_from_local_storage(self, w, base_dir):
        with closing(MultiFileReader(base_dir, "")) as reader:
            locs = self.posting_locs[w]
            b = reader.read_from_local_storage(locs, self.df[w] * TUPLE_SIZE)
            posting_list = []
            for i in range(self.df[w]):
                ##################### EDITED TO NEW POSTING LIST#######################

                doc_id,tf_idf_score = struct.unpack("Id", b[i * TUPLE_SIZE:(i + 1) * TUPLE_SIZE])
                posting_list.append((doc_id, tf_idf_score))

            return posting_list

    @staticmethod
    def read_index_from_local_storage(base_dir, name):
        with open(Path(base_dir) / f'{name}.pkl', 'rb') as f:
            return pickle.load(f)

    def posting_lists_iter(self):
        """ A generator that reads one posting list from disk and yields
            a (word:str, [(doc_id:int, tf:int), ...]) tuple.
        """
        with closing(MultiFileReader()) as reader:
            for w, locs in self.posting_locs.items():
                b = reader.read(locs[0], self.df[w] * TUPLE_SIZE)
                posting_list = []
                for i in range(self.df[w]):
                    ###########################MODIFIED HERE #####################################
                    doc_id,tf_idf_score = struct.unpack("Id", b[i * TUPLE_SIZE:(i + 1) * TUPLE_SIZE])
                    posting_list.append((doc_id,tf_idf_score))
                yield w, posting_list

    @staticmethod
    def read_index(bucket_name, base_dir, name):
        client = storage.Client()
        bucket = client.get_bucket(bucket_name)
        with bucket.get_blob(f'{base_dir}/{name}.pkl').open("rb") as f:
            return pickle.load(f)

    @staticmethod
    def delete_index(base_dir, name):
        path_globals = Path(base_dir) / f'{name}.pkl'
        path_globals.unlink()
        for p in Path(base_dir).rglob(f'{name}_*.bin'):
            p.unlink()

    @staticmethod
    def write_a_posting_list(b_w_pl, bucket_name, folder_name):
        posting_locs = defaultdict(list)
        bucket_id, list_w_pl = b_w_pl

        with closing(MultiFileWriter(".", bucket_id, bucket_name)) as writer:
            for w, pl in list_w_pl:
                # convert to bytes
                ###########################HERE MODIFIED######################################
                b = b''.join([struct.pack("Id", doc_id, tf_idf_score) for doc_id,tf_idf_score in pl])
                # write to file(s)
                locs = writer.write(b, folder_name)
                # save file locations to index
                posting_locs[w].extend(locs)
            writer.upload_to_gcp(folder_name)
            InvertedIndex._upload_posting_locs(bucket_id, posting_locs, bucket_name, folder_name)
        return bucket_id

    @staticmethod
    def _upload_posting_locs(bucket_id, posting_locs, bucket_name, folder_name):
        with open(f"{bucket_id}_posting_locs.pickle", "wb") as f:
            pickle.dump(posting_locs, f)
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob_posting_locs = bucket.blob(f"{folder_name}/{bucket_id}_posting_locs.pickle")
        blob_posting_locs.upload_from_filename(f"{bucket_id}_posting_locs.pickle")

  # tokens = [token.group() for token in RE_WORD.finditer(text.lower())]
  # #ADDED stopwords removal
  #
  # one_word_tokens = [stemmer.stem(token) for token in tokens if token not in all_stopwords]
  # two_word_tokens = [one_word_tokens[i]+" "+one_word_tokens[i+1] for i in range(len(one_word_tokens)-1) if one_word_tokens[i] in query_terms and one_word_tokens[i+1] in query_terms]
  # # YOUR CODE HERE
  # count = Counter(two_word_tokens)
  # if len(count)==0:
  #     return []
  # #### ADDED doc_len and max_tf to posting lists
  # doc_len = len(one_word_tokens)
  # max_tf = 0#count.most_common(1)[0][1]
  # #return [(word,(id,tf,max_tf, doc_len)) for word,tf in count.items() if word not in all_stopwords]
  # return [(word,(id,tf*2,max_tf, doc_len)) for word,tf in count.items()]
