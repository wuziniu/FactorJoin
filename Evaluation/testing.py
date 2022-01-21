import pickle
import time
import numpy as np


def test_on_stats(model_path, query_file, save_res=None):
	with open(model_path, "rb") as f:
		bound_ensemble = pickle.load(f)

	with open(query_file, "r") as f:
		queries = f.readlines()

	qerror = []
	latency = []
	pred = []
	for i, query_str in enumerate(queries):
		query = query_str.split("||")[0][:-1]
		true_card = int(query_str.split("||")[-1])
		t = time.time()
		res = bound_ensemble.get_cardinality_bound(query)
		pred.append(res)
		latency.append(time.time() - t)
		qerror.append(max(res/true_card, true_card/res))

	qerror = np.asarray(qerror)
	for i in [50, 90, 95, 99, 100]:
		print(f"q-error {i}% percentile is {np.percentile(qerror, i)}")
	print(f"average latency per query is {np.mean(latency)}")

	if save_res:
		np.save("prediction", np.asarray(pred))

