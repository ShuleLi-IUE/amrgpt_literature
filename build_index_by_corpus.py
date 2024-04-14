#!/usr/bin/env python
# coding=utf-8
from vectordb_utils_paper import PaperVectorDB
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) 
import os
import pandas as pd
import time
import sys
import errno
import pickle
import re
import math
# import warnings
# warnings.filterwarnings("ignore")

distance = "l2"
batch_size = 40
num_workers = None
search_strategy = "hnsw"
vec_db_paper = PaperVectorDB(space=distance,
                             batch_size=batch_size)
# def init_db_pdf(file):
#     print("---init database---")
#     paragraphs, pages = extract_text_from_pdf_pdfplumber_with_pages(file.name)
#     # larger intersect
#     documents = split_text_with_pages(paragraphs, pages, 300, 150)
#     print(len(documents))
#     vec_db.add_documents_bge(documents)

# load data from ./corpus
def init_db_local():
    print("---init local database begin---")
    corpus_df = "CORPUS_PATH"
        
    # report pdf
    print("local database of report initing...") 
    
    paper_df = pd.read_csv(corpus_df)
    n = paper_df.shape[0]
    for i in range(math.ceil(n / batch_size)):
        t0 = time.perf_counter()
        indexes = list(range(i*batch_size, min((i+1)*batch_size, n)))
        df_batch = paper_df.loc[indexes]
        vec_db_paper.add_documents_dense_paper(ids = indexes,
                                title=df_batch['Title'].tolist(), 
                                abstract=df_batch['Abstract'].tolist(), 
                                journal=df_batch['Journal'].tolist(), 
                                author=df_batch['Author'].tolist(), 
                                citation=df_batch['Citations'].tolist(), 
                                DOI=df_batch['DOI'].tolist(), 
                                link=df_batch['Link'].tolist(), 
                                year=df_batch['Year'].tolist(), 
                                )
        print(f"{indexes} end, cost time: {time.perf_counter()-t0}")
        
    print("---init database end---")
    print("---PaperVectorDB dump begin---")
    t2 = time.perf_counter()
    vec_db_paper.dump()
    print("---PasperVectorDB dump end---")
    print(f"dump costs time: {time.perf_counter()-t2}")

if __name__ == "__main__":
    init_db_local()
