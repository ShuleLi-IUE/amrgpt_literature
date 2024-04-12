from embedding_utils import get_embedding_bge
import hnswlib
import numpy as np
import time
import pickle
from datetime import datetime

DIM = 1024
MAX_ELEMENTS = 2000000
THREADS = 4

class PaperVectorDB:
    def __init__(self, space = "l2", batch_size = 12):
        self.index_hnsw = hnswlib.Index(space=space, dim=DIM)
        self.index_hnsw.init_index(max_elements=MAX_ELEMENTS, ef_construction=100, M=32)
        self.index_hnsw.set_num_threads(THREADS)
        self.cnt = 0
        self.data_dict = {}
        self.batch_size = batch_size
        self.space = space
        '''
        "embeddings": ,
        "title": ,
        "abstract",
        "journal",
        "author",
        "citation",
        "DOI" ,
        "link",
        "year"
        '''

    def add_documents_dense_paper(self, ids, title, abstract, journal, author, citation, DOI, link, year):
        # ids = np.array([f"id_{i}" for i in np.arange(self.cnt, self.cnt + n)])
        n = len(ids)
        train_texts = [title[i] + '. ' + abstract[i].strip("'") for i in range(n)]
        t0 = time.perf_counter()
        embeddings = get_embedding_bge(train_texts,
                                       batch_size=self.batch_size)
        t1 = time.perf_counter()
        print("one embedding time ", (t1 - t0) / n)
        self.index_hnsw.add_items(embeddings, ids=ids)
        t2 = time.perf_counter()
        print("one add index time ", (t2 - t1) / n)
        # store documents to dict
        for i in range(n):
            self.data_dict[ids[i]] = {"embeddings": embeddings[i],
                                      "title": title[i],
                                      "abstract": abstract[i],
                                      "journal": journal[i],
                                      "author": author[i],
                                      "citation": citation[i],
                                      "year": year[i],
                                      "link": link[i],
                                      "DOI": DOI[i]}
        self.cnt += n
        print(f"#{type}# Adding batch of {n} elements, now total index contains {self.cnt} elements")

    def search_bge(self, query, top_n, verbose=True):
        t0 = time.time()
        embedding = get_embedding_bge(query,
                                      batch_size=self.batch_size)

        if verbose:
            t1 = time.time()
            print("get_embedding_bge costs ", t1 - t0)

        labels, distances = self.index_hnsw.knn_query(embedding, k=top_n)

        if verbose:
            t2 = time.time()
            print("index_hnsw.knn_query costs", t2 - t1)
        return labels[0].tolist()

    def get_context_by_labels(self, labels):
        title = [self.data_dict[key]["title"] for key in labels]
        abstract = [self.data_dict[key]["abstract"] for key in labels]
        journal = [self.data_dict[key]["journal"] for key in labels]
        author = [self.data_dict[key]["author"] for key in labels]
        citation = [self.data_dict[key]["citation"] for key in labels]
        year = [self.data_dict[key]["year"] for key in labels]
        link = [self.data_dict[key]["link"] for key in labels]
        DOI = [self.data_dict[key]["DOI"] for key in labels]
        
        return title, abstract, journal, author, citation, year, link, DOI

    def get_cnt(self):
        return self.cnt

    def get_embeddings_by_labels(self, labels):
        embeddings = [self.data_dict[key]["embeddings"] for key in labels]
        return embeddings

    def get_space(self):
        return self.space

    def dump(self, file_dump=None):
        if file_dump == None: file_dump = 'index_' + 'n'+ str(self.cnt) + '_' + datetime.now().strftime('%m%d%H%M') + '.pickle'
        with open(file_dump, 'wb') as f:
            pickle.dump(self, f)
        print("Dump index successfully, path: " + file_dump)