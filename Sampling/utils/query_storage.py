import copy
import pickle
import networkx as nx
from networkx.readwrite import json_graph


def load_sql_rep(fn, dummy=None):
    assert ".pkl" in fn
    try:
        with open(fn, "rb") as f:
            query = pickle.load(f)
    except:
        print(fn + " failed to load...")
        exit(-1)

    query["subset_graph"] = \
            nx.OrderedDiGraph(json_graph.adjacency_graph(query["subset_graph"]))
    query["join_graph"] = json_graph.adjacency_graph(query["join_graph"])
    if "subset_graph_paths" in query:
        query["subset_graph_paths"] = \
                nx.OrderedDiGraph(json_graph.adjacency_graph(query["subset_graph_paths"]))

    return query

def save_sql_rep(fn, cur_qrep):
    assert ".pkl" in fn
    qrep = copy.deepcopy(cur_qrep)
    qrep["join_graph"] = nx.adjacency_data(qrep["join_graph"])
    qrep["subset_graph"] = nx.adjacency_data(qrep["subset_graph"])

    with open(fn, "wb") as f:
        pickle.dump(qrep, f)
