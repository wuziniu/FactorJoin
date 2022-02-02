import psycopg2
import os

conn = psycopg2.connect(database="imdb", host="localhost", user="postgres", password="postgres")
cursor = conn.cursor()

query_file = "/home/ubuntu/data_CE/job/all_queries.sql"
with open(query_file, "r") as f:
    queries = f.readlines()

for no, query in enumerate(queries):
    cursor.execute("EXPLAIN ANALYZE" + query)
    res = cursor.fetchall()
    print(res)
    print("%d-th query finished." % no)

cursor.close()
conn.close()


