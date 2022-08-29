import pickle5 as pickle
import time
import os


def get_job_sub_plan_queires(query_folder):
	"""
	This is a helper function for extracting the sub-plan query string from the postgres analyzed results.
	More details on how to derive the "job_sub_plan_queries.txt" can be found at:
	https://github.com/Nathaniel-Han/End-to-End-CardEst-Benchmark#how-to-generate-sub-plan-queries
	"""
	with open(query_folder + "job_sub_plan_queries.txt", "r") as f:
		sub_plan_queries = f.read()
	psql_raw = sub_plan_queries.split("query: 0")[1:]
	queries = []
	q_file_names = []

	for file in os.listdir(query_folder):
		if file.endswith(".sql") and file[0].isnumeric():
			q_file_names.append(file.split(".sql")[0] + ".pkl")
			with open(query_folder + file, "r") as f:
				q = f.readline()
				queries.append(q)

	psql_raw = sub_plan_queries.split("query: 0")[1:]
	sub_plan_queries_str_all = []
	for per_query in psql_raw:
		sub_plan_queries = []
		sub_plan_queries_str = []
		num_sub_plan_queries = len(per_query.split("query: "))
		all_info = per_query.split("RELOPTINFO (")[1:]
		assert num_sub_plan_queries * 2 == len(all_info)
		for i in range(num_sub_plan_queries):
			idx = i * 2
			table1 = all_info[idx].split("): rows=")[0]
			table2 = all_info[idx + 1].split("): rows=")[0]
			table_str = (table1, table2)
			sub_plan_queries_str.append(table_str)
		sub_plan_queries_str_all.append(sub_plan_queries_str)

	all_queries = dict()
	all_sub_plan_queries_str = dict()
	for i in range(len(q_file_names)):
		name = q_file_names[i].split(".pkl")[0]
		all_queries[name] = queries[i]
		all_sub_plan_queries_str[name] = sub_plan_queries_str_all[i]

	return all_queries, all_sub_plan_queries_str


def test_on_imdb(model_path, query_file, query_sub_plan_file, SPERCENTAGE=None, query_sample_location=None,
				 save_res=None):
	"""
	Evaluate the trained FactorJoin model on the IMDB-JOB workload.
	:param model_path: the trained model
	:param query_file: a dictionary of queries, e.g. '1a': SQL query string for query '1a'
	:param query_sub_plan_file: a dictionary of all subplans of a query,
	:param SPERCENTAGE: the sampling rate for doing base table cardinality estimation
	:param query_sample_location: if there exist a materialized sample that we can directly load from.
	"""
	with open(model_path, "rb") as f:
		bound_ensemble = pickle.load(f)
	if SPERCENTAGE:
		bound_ensemble.SPERCENTAGE = SPERCENTAGE
	if query_sample_location:
		bound_ensemble.query_sample_location = query_sample_location

	with open(query_file, "rb") as f:
		all_queries = pickle.load(f)
	with open(query_sub_plan_file, "rb") as f:
		all_sub_plan_queries_str = pickle.load(f)

	res = dict()
	t = time.time()
	for q_name in all_queries:
		# print(q_file_id, q_file_names[q_file_id])
		temp = bound_ensemble.get_cardinality_bound_all(all_queries[q_name], all_sub_plan_queries_str[q_name],
														q_name + ".pkl")
		res[q_name] = temp
	print("total estimation latency is: ", time.time() - t)

	if save_res:
		# save the sub-plan estimates according to the query execution order (1a, 1b, ..., 33c)
		f = open(save_res, "w")
		for query_no in range(1, 34):
			for suffix in ['a', 'b', 'c', 'd', 'e', 'f', 'g']:
				q_name = f"{query_no}{suffix}"
				if q_name in res:
					for pred in res[q_name]:
						f.write(str(pred) + "\n")
		f.close()

