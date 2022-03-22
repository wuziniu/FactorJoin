import psycopg2
import time
import os
import argparse
import numpy as np


def send_query(dataset, method_name, query_file):
    conn = psycopg2.connect(database=dataset, user="postgres", password="postgres", host="127.0.0.1", port=5436,)
    cursor = conn.cursor()


    with open(query_file, "r") as f:
        queries = f.readlines()

    # cursor.execute('SET debug_card_est=true')
    # cursor.execute('SET print_sub_queries=true')
    # cursor.execute('SET print_single_tbl_queries=true')
    cursor.execute("SET ml_joinest_enabled=true;")
    cursor.execute("SET join_est_no=0;")
    cursor.execute(f"SET ml_joinest_fname='{method_name}';")


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
    save_file_name = method_name.split(".txt")[0]
    np.save(f"plan_time_{save_file_name}", np.asarray(planning_time))
    np.save(f"exec_time_{save_file_name}", np.asarray(execution_time))
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='stats', help='Which dataset to be used')
    parser.add_argument('--method_name', default='stats_CEB_sub_queries_model_stats_greedy_50.txt', help='save estimates')
    parser.add_argument('--query_file', default='/home/ubuntu/data_CE/stats_CEB/stats_CEB.sql', help='Query file location')
    args = parser.parse_args()
    
    send_query(args.dataset, args.method_name, args.query_file)
    
    