import psycopg2
import time
import os
import numpy as np

conn = psycopg2.connect(database="stats", user="postgres", password="postgres", host="127.0.0.1", port=5436,)
cursor = conn.cursor()


with open("stats_CEB.sql", "r") as f:
    queries = f.readlines()

# cursor.execute('SET debug_card_est=true')
# cursor.execute('SET print_sub_queries=true')
# cursor.execute('SET print_single_tbl_queries=true')
cursor.execute("SET ml_joinest_enabled=true;")
cursor.execute("SET join_est_no=0;")
cursor.execute("SET ml_joinest_fname='stats_CEB_CE_scheme_200_greedy.txt';")


planning_time = [] 
execution_time = []
for no, query in enumerate(queries):
    print(f"Executing query {no}")
    start = time.time()
    cursor.execute("EXPLAIN ANALYZE " + query)
    res = cursor.fetchall()
    planning_time.append(float(res[-2][0].split(":")[-1].split("ms")[0].strip()))
    execution_time.append(float(res[-1][0].split(":")[-1].split("ms")[0].strip()))
    end = time.time()
    print(f"{no}-th query finished in {end-start}, with planning_time {planning_time[no]} ms and execution_time {execution_time[no]} ms" )

cursor.close()
conn.close()
np.save("planning_time_CE_scheme_200_greedy_stats", np.asarray(planning_time))
np.save("query_time_CE_scheme_200_greedy_stats", np.asarray(execution_time))
