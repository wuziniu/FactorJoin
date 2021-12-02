#!/usr/bin/env python
# -*- coding: utf-8 -*-

import psycopg2
import time
import sys, os

conn=psycopg2.connect(host="localhost", database="imdb")
cur = conn.cursor()
fname_path = ""
print(fname_path)
query_file = open(fname_path)
result_fname = os.path.basename(fname_path)
print(result_fname)
output_file = open("/data01/hanyuxing/psql_result_"+result_fname, 'w')

for no, query in enumerate(query_file.readlines()):
    start = time.time()
    cur.execute(query.split("||")[0])
    res = cur.fetchone()[0]
    output_file.write(f"{no}:{res}:{time.time()-start}")
    # output_file.write(str(cur.fetchone()[0]))
    output_file.write("\n")
    output_file.flush()

query_file.close()
output_file.close()


cur.close()
conn.close()