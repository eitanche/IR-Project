from collections import Counter, OrderedDict, defaultdict
from pathlib import Path
import pickle
import struct
from contextlib import closing
from reader_writer import MultiFileReader, MultiFileWriter
import sys

BLOCK_SIZE = 1999998
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
        self.bug_files = []
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
        self._write_globals(base_dir, name)

    def _write_globals(self, base_dir, name):
        with open(Path(base_dir) / f'{name}.pkl', 'wb') as f:
            pickle.dump(self, f)

    def __getstate__(self):
        """ Modify how the object is pickled by removing the internal posting lists
            from the object's state dictionary.
        """
        state = self.__dict__.copy()
        #del state['_posting_list']
        return state

    def posting_lists_iter(self,base_dir):
        """ A generator that reads one posting list from disk and yields
            a (word:str, [(doc_id:int, tf:int), ...]) tuple.
        """
        with closing(MultiFileReader(base_dir)) as reader:
            for w, locs in self.posting_locs.items():
                b = reader.read(locs, self.df[w] * TUPLE_SIZE)
                posting_list = []
                for i in range(self.df[w]):
                    #   doc_id = int.from_bytes(b[i*TUPLE_SIZE:i*TUPLE_SIZE+4], 'big')
                    #   tf = int.from_bytes(b[i*TUPLE_SIZE+4:(i+1)*TUPLE_SIZE], 'big')
                    ###########################MODIFIED HERE #####################################

                    doc_id, tf, max_tf, doc_len = struct.unpack("IIII", b[i * TUPLE_SIZE:(i + 1) * TUPLE_SIZE])
                    posting_list.append((doc_id, tf, max_tf, doc_len))
                yield w, posting_list

    def read_posting_list(self, w, base_dir):
        # print(self.df[w])
        # print(self.df["vanilla"])
        with closing(MultiFileReader(base_dir)) as reader:
            locs = self.posting_locs[w]
            b = reader.read(locs, self.df[w] * TUPLE_SIZE)
            posting_list = []
            for i in range(self.df[w]):
                ##################### EDITED TO NEW POSTING LIST#######################
                try:
                    doc_id, tf, max_tf, doc_len = struct.unpack("IIII", b[i * TUPLE_SIZE:(i + 1) * TUPLE_SIZE])
                    posting_list.append((doc_id, tf, max_tf, doc_len))
                except:
                    #self.bug_files.append(w)
                    print(self.posting_locs[w])
                # doc_id = int.from_bytes(b[i*TUPLE_SIZE:i*TUPLE_SIZE+4], 'big')
                # tf = int.from_bytes(b[i*TUPLE_SIZE+4:(i+1)*TUPLE_SIZE], 'big')
                # posting_list.append((doc_id, tf))
            return posting_list

    @staticmethod
    def read_index(base_dir, name):
        with open(Path(base_dir) / f'{name}.pkl', 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def delete_index(base_dir, name):
        path_globals = Path(base_dir) / f'{name}.pkl'
        path_globals.unlink()
        for p in Path(base_dir).rglob(f'{name}_*.bin'):
            p.unlink()

    @staticmethod
    def write_a_posting_list(b_w_pl):
        ''' Takes a (bucket_id, [(w0, posting_list_0), (w1, posting_list_1), ...])
        and writes it out to disk as files named {bucket_id}_XXX.bin under the
        current directory. Returns a posting locations dictionary that maps each
        word to the list of files and offsets that contain its posting list.
        Parameters:
        -----------
          b_w_pl: tuple
            Containing a bucket id and all (word, posting list) pairs in that bucket
            (bucket_id, [(w0, posting_list_0), (w1, posting_list_1), ...])
        Return:
          posting_locs: dict
            Posting locations for each of the words written out in this bucket.
        '''
        posting_locs = defaultdict(list)
        bucket, list_w_pl = b_w_pl

        with closing(MultiFileWriter('', bucket)) as writer:
            for w, pl in list_w_pl:
                # convert to bytes

                ###########################HERE MODIFIED######################################
                b = b''.join([struct.pack("IIII", doc_id, tf, max_tf, doc_len) for doc_id, tf, max_tf, doc_len in pl])

                # b = b''.join([(doc_id << 16 | (tf & TF_MASK)).to_bytes(TUPLE_SIZE, 'big')
                #               for doc_id, tf in pl])
                # write to file(s)
                locs = writer.write(b)
                # save file locations to index
                posting_locs[w].extend(locs)
        return posting_locs