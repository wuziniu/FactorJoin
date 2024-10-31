import ast
import time
import pandas as pd
import numpy as np
import logging
from time import perf_counter
from BayesCard.Evaluation.utils import parse_query, save_csv

logger = logging.getLogger(__name__)


OPS = {
    '>': np.greater,
    '<': np.less,
    '>=': np.greater_equal,
    '<=': np.less_equal,
    '=': np.equal,
    '==': np.equal
}


def str_pattern_matching(s):
    # split the string "attr==value" to ('attr', '=', 'value')
    op_start = 0
    if len(s.split(' IN ')) != 1:
        s = s.split(' IN ')
        attr = s[0].strip()
        try:
            value = list(ast.literal_eval(s[1].strip()))
        except:
            temp_value = s[1].strip()[1:][:-1].split(',')
            value = []
            for v in temp_value:
                value.append(v.strip())
        return attr, 'in', value

    for i in range(len(s)):
        if s[i] in OPS:
            op_start = i
            if i + 1 < len(s) and s[i + 1] in OPS:
                op_end = i + 1
            else:
                op_end = i
            break
    attr = s[:op_start]
    value = s[(op_end + 1):].strip()
    ops = s[op_start:(op_end + 1)]
    try:
        value = list(ast.literal_eval(s[1].strip()))
    except:
        try:
            value = int(value)
        except:
            try:
                value = float(value)
            except:
                value = value
    return attr.strip(), ops.strip(), value

def construct_table_query(BN, table_query, attr, ops, val, epsilon=1e-6):
    if BN is None or attr not in BN.attr_type:
        return None
    if BN.attr_type[attr] == 'continuous':
        if ops == ">=":
            query_domain = (val, np.infty)
        elif ops == ">":
            query_domain = (val + epsilon, np.infty)
        elif ops == "<=":
            query_domain = (-np.infty, val)
        elif ops == "<":
            query_domain = (-np.infty, val - epsilon)
        elif ops == "=" or ops == "==":
            query_domain = val
        else:
            assert False, f"operation {ops} is invalid for continous domain"

        if attr not in table_query:
            table_query[attr] = query_domain
        else:
            prev_l = table_query[attr][0]
            prev_r = table_query[attr][1]
            query_domain = (max(prev_l, query_domain[0]), min(prev_r, query_domain[1]))
            table_query[attr] = query_domain

    else:
        attr_domain = BN.domain[attr]
        if type(attr_domain[0]) != str:
            attr_domain = np.asarray(attr_domain)
        if ops == "in" or ops == "IN":
            assert type(val) == list, "use list for in query"
            query_domain = val
        elif ops == "=" or ops == "==":
            if type(val) == list:
                query_domain = val
            else:
                query_domain = [val]
        else:
            if type(val) == list:
                assert len(val) == 1
                val = val[0]
                assert (type(val) == int or type(val) == float)
            operater = OPS[ops]
            query_domain = list(attr_domain[operater(attr_domain, val)])
            if len(query_domain) == 0:
                # nothing satisfies this query
                return None

        if attr not in table_query:
            table_query[attr] = query_domain
        else:
            query_domain = [i for i in query_domain if i in table_query[attr]]
            table_query[attr] = query_domain

    return table_query


def timestamp_transorform(time_string, start_date="2010-07-19 00:00:00"):
    start_date_int = time.strptime(start_date, "%Y-%m-%d %H:%M:%S")
    time_array = time.strptime(time_string, "'%Y-%m-%d %H:%M:%S'")
    return int(time.mktime(time_array)) - int(time.mktime(start_date_int))


def parse_query_single_table(query, BN):
    result = dict()
    if ' WHERE ' not in query:
        return result
    useful = query.split(' WHERE ')[-1].strip(';').strip()
    if len(useful) == 0:
        return result
    for sub_query in useful.split(' AND '):
        attr, ops, value = str_pattern_matching(sub_query.strip())
        if "Date" in attr:
            assert "::timestamp" in value
            value = timestamp_transorform(value.strip().split("::timestamp")[0])
        construct_table_query(BN, result, attr, ops, value)
    return result


def evaluate_card(bn, query_filename='/home/ziniu.wzn/deepdb-public/benchmarks/imdb_single/query.sql'):
    with open(query_filename) as f:
        queries = f.readlines()
    latencies = []
    error = []
    for query_no, query in enumerate(queries):
        query = query.strip().split('||')
        query_str = query[0]
        true_card = int(query[1])
        tic = time.time()
        est = bn.infer_query(parse_query_single_table(query_str))
        latencies.append(time.time() - tic)
        error = max(est / true_card, true_card / est)
        print(true_card, est)
    return latencies, error

def single_table_experiment():
    from Models.Bayescard_BN import Bayescard_BN
    df = pd.read_hdf("/home/ziniu.wzn/imdb-benchmark/gen_single_light/title.hdf")
    new_cols = []
    for col in df.columns:
        new_cols.append(col.replace('.', '__'))
    df.columns = new_cols
    BN = Bayescard_BN('title')
    BN.build_from_data(df, algorithm="greedy", max_parents=1, n_mcv=30, n_bins=30, ignore_cols=['title_id'],
                       sample_size=500000)
    gd_latency, gd_error = evaluate_card(BN)
    np.save('gd_latency', np.asarray(gd_latency))
    np.save('gd_error', np.asarray(gd_error))

    BN = Bayescard_BN('title')
    BN.build_from_data(df, algorithm="chow-liu", max_parents=1, n_mcv=30, n_bins=30, ignore_cols=['title_id'],
                       sample_size=500000)
    cl_latency, cl_error = evaluate_card(BN)
    np.save('cl_latency', np.asarray(cl_latency))
    np.save('cl_error', np.asarray(cl_error))

    BN.model = BN.model.to_junction_tree()
    BN.algorithm = "junction"
    BN.init_inference_method()
    jt_latency, jt_error = evaluate_card(BN)
    np.save('jt_latency', np.asarray(jt_latency))
    np.save('jt_error', np.asarray(jt_error))
    
def evaluate_cardinality(BN_e, schema, query_path, true_cardinalities_path):
    # read all queries
    with open(query_path) as f:
        queries = f.readlines()
    df_true_card = pd.read_csv(true_cardinalities_path)
    latencies = []
    q_errors = []
    for query_no, query_str in enumerate(queries):

        print(f"Predicting cardinality for query {query_no}: {query_str}")

        query = parse_query(query_str.strip(), schema)
        cardinality_true = df_true_card.loc[df_true_card['query_no'] == query_no, ['cardinality_true']].values[0][0]
        card_start_t = perf_counter()
        cardinality_predict = BN_e.naive_cardinality(query)
        if cardinality_predict is None:
            continue
        card_end_t = perf_counter()
        latency_ms = (card_end_t - card_start_t) * 1000
        if cardinality_predict == 0 and cardinality_true == 0:
            q_error = 1.0
        elif cardinality_predict == 0:
            cardinality_predict = 1
        elif cardinality_true == 0:
            cardinality_true = 1

        q_error = max(cardinality_predict / cardinality_true, cardinality_true / cardinality_predict)
        print(f"latency: {latency_ms} and error: {q_error}")
        latencies.append(latency_ms)
        q_errors.append(q_error)
    return latencies, q_errors


if __name__ == "main":
    single_table_experiment()
