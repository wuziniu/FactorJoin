import sys
sys.path.append(".")

import collections.abc
#hyper needs the four following aliases to be done manually.
collections.Iterable = collections.abc.Iterable
collections.Mapping = collections.abc.Mapping
collections.MutableSet = collections.abc.MutableSet
collections.MutableMapping = collections.abc.MutableMapping

import argparse
import psycopg2 as pg
# from db_utils.utils import *
from utils.utils import *
from db_utils.query_storage import *
import pdb
import random
import klepto
from multiprocessing import Pool, cpu_count
import json
import pickle
from sql_rep.utils import execute_query
from networkx.readwrite import json_graph
import re
from sql_rep.query import parse_sql
# from wanderjoin import WanderJoin
import math
# from progressbar import progressbar as bar
import scipy.stats as st
import pickle
import itertools

CACHE_CARD_TYPES = ["actual"]

FILTER_TMP = """AND {COL} IN ({VALS})"""
COL_TEMPLATE = "{COL}_bin"
GB_TMP = "SELECT {COL}, COUNT(*) FROM {TABLE} AS {ALIAS} WHERE {WHERE} GROUP BY {COL}"

def read_flags():
    parser = argparse.ArgumentParser()

    parser.add_argument("--db_name", type=str, required=False,
            default="imdb")
    parser.add_argument("--card_cache_dir", type=str, required=False,
            default="./cardinality_cache")
    parser.add_argument("--db_host", type=str, required=False,
            default="localhost")
    parser.add_argument("--user", type=str, required=False,
            default="pari")
    parser.add_argument("--pwd", type=str, required=False,
            default="")
    parser.add_argument("--port", type=str, required=False,
            default=5433)

    parser.add_argument("--query_dir", type=str, required=False,
            default=None)
    parser.add_argument("-n", "--num_queries", type=int,
            required=False, default=-1)

    parser.add_argument("--no_parallel", type=int,
            required=False, default=1)
    parser.add_argument("--card_type", type=str, required=False,
            default=None)
    parser.add_argument("--key_name", type=str, required=False,
            default=None)
    parser.add_argument("--true_timeout", type=int,
            required=False, default=1800000*5)
    parser.add_argument("--pg_total", type=int,
            required=False, default=1)
    parser.add_argument("--num_proc", type=int,
            required=False, default=-1)
    parser.add_argument("--seed", type=int,
            required=False, default=1234)
    parser.add_argument("--sampling_percentage", type=float,
            required=False, default=None)
    parser.add_argument("--sampling_type", type=str,
            required=False, default=None)
    parser.add_argument("--db_year", type=int,
            required=False, default=None)

    return parser.parse_args()

def update_bad_qrep(qrep):
    qrep = parse_sql(qrep["sql"], None, None, None, None, None,
            compute_ground_truth=False)
    qrep["subset_graph"] = \
            nx.OrderedDiGraph(json_graph.adjacency_graph(qrep["subset_graph"]))
    qrep["join_graph"] = json_graph.adjacency_graph(qrep["join_graph"])
    return qrep

def is_cross_join(sg):
    '''
    enforces the constraint that the graph should be connected.
    '''
    if len(sg.nodes()) < 2:
        # FIXME: should be return False
        return False
    sg2 = nx.Graph(sg)
    to_remove = []

    # do this in case of stackexchange database; because of the weird query
    # structure, if the graph is connected only through the `site` table, it
    # still behaves like a cross-join. Check the appendix of the Flow-Loss
    # paper for more details
    for node, data in sg2.nodes(data=True):
        if data["real_name"] == "site":
            to_remove.append(node)

    for node in to_remove:
        sg2.remove_node(node)
    if nx.is_connected(sg2):
        return False
    return True

def get_binned_sqls(qrep, card_type, key_name, db_host, db_name, user, pwd,
        port, true_timeout, pg_total, cache_dir, fn, idx,
        sampling_percentage, sampling_type, skip_zero_queries, db_year):
    '''
    updates qrep's fields with the needed cardinality estimates, and returns
    the qrep.
    '''
    with open("bins.pkl", "rb") as f:
        bins = pickle.load(f)

    with open("equivalent_keys.pkl", "rb") as f:
        equivalent_keys = pickle.load(f)

    table_cols = {}
    binids = {}

    for t,cols in equivalent_keys.items():
        for col in cols:
            tname = col[0:col.find(".")]
            colname = col[col.find(".")+1:]
            if tname not in table_cols:
                table_cols[tname] = set([colname])
            else:
                table_cols[tname].add(colname)

            if col not in binids:
                binids[col] = t

    node_list = list(qrep["subset_graph"].nodes())
    node_list.sort(reverse=True, key = lambda x: len(x))

    qrep_tables = [qrep["join_graph"].nodes()[node]["real_name"] for node in
            qrep["join_graph"].nodes()]

    alltabs = []
    allcols = []
    allsqls = []

    for subqi, subset in enumerate(node_list):
        if len(subset) > 1:
            continue

        cursqls = {}
        info = qrep["subset_graph"].nodes()[subset]

        ## want to get the per bin cardinality for each column w/o filters too
        # if len(qrep["join_graph"].nodes()[subset[0]]["pred_vals"]) == 0:
            # continue

        sg = qrep["join_graph"].subgraph(subset)
        subsql = nx_graph_to_query(sg)
        where_clause = " AND ".join(sg.nodes()[subset[0]]["predicates"])

        tname = qrep["join_graph"].nodes()[subset[0]]["real_name"]

        # will need to loop over each potential bin value in these columns
        if tname not in table_cols:
            print("MISSING!!")
            print(tname)
            # pdb.set_trace()
            continue

        if sampling_percentage is not None:
            sample_tname = tname + "_ss" + str(args.sampling_percentage)
            # sample_tname = sample_tname.replace(".","")
            sample_tname = sample_tname.replace(".","d")
        else:
            sample_tname = tname

        curtcols = [c for c in table_cols[tname]]

        querycols = []
        binvals = []

        for curcol in curtcols:
            # find the newly created column for this
            newcolname = COL_TEMPLATE.format(COL=curcol)
            gsql = GB_TMP.format(COL = newcolname,
                          TABLE = sample_tname,
                          ALIAS = subset[0],
                          WHERE = where_clause)

            if where_clause.strip() == "":
                gsql = gsql.replace("WHERE", "")

            print(gsql)
            alltabs.append(subset)
            allcols.append(curcol)
            allsqls.append(gsql)

    return alltabs,allcols,allsqls

def exec_sql(sql, dummy):
    start = time.time()
    con = pg.connect(user=args.user, host=args.db_host, port=args.port,
            password=args.pwd, database=args.db_name)
    cursor = con.cursor()
    cursor.execute(sql)
    res = cursor.fetchall()

    exec_time = time.time()-start
    return res,exec_time

def main():
    fns = list(glob.glob(args.query_dir + "/*"))
    fns.sort()
    par_args = []

    for i, fn in enumerate(fns):
        print(fn)
        if i >= args.num_queries and args.num_queries != -1:
            break

        if (".pkl" not in fn and ".sql" not in fn):
            continue

        if ".pkl" in fn:
            qrep = load_sql_rep(fn)
        else:
            with open(fn, "r") as f:
                sql = f.read()
            qrep = parse_sql(sql, None, None, None, None, None,
                    compute_ground_truth=False)
            qrep["subset_graph"] = \
                    nx.OrderedDiGraph(json_graph.adjacency_graph(qrep["subset_graph"]))
            qrep["join_graph"] = json_graph.adjacency_graph(qrep["join_graph"])
            fn = fn.replace(".sql", ".pkl")
            save_sql_rep(fn, qrep)
            print("updated sql rep!")

        alltabs, allcols, allsqls = get_binned_sqls(qrep, args.card_type, args.key_name, args.db_host,
                args.db_name, args.user, args.pwd, args.port,
                args.true_timeout, args.pg_total, args.card_cache_dir, fn,
                i, args.sampling_percentage,
                args.sampling_type, True, args.db_year)

        par_args = []
        for sql in allsqls:
            par_args.append((sql, None))

        with Pool(processes = 8) as pool:
            res = pool.starmap(exec_sql, par_args)

        times = [r[1] for r in res]
        print(max(times), min(times))

        binfn = fn.replace("queries", "./binned_cards/" +
                str(args.sampling_percentage) + "/")
        print(binfn)
        newdir = os.path.dirname(binfn)
        make_dir(newdir)

        data = {}
        data["all_aliases"] = alltabs
        data["all_columns"] = allcols
        data["all_sqls"] = allsqls
        data["results"] = res

        with open(binfn, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

args = read_flags()
main()
