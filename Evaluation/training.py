import pickle
import os
import numpy as np
from Join_scheme.data_prepare import process_stats_data, process_imdb_data
from Join_scheme.bound import Bound_ensemble
from Join_scheme.tools import get_n_bins_from_query
from BayesCard.Models.Bayescard_BN import Bayescard_BN
from BayesCard.Evaluation.cardinality_estimation import parse_query_single_table



def test_trained_BN_on_stats(bn, t_name):
    queries = {
        "posts": "SELECT COUNT(*) FROM posts as p WHERE posts.CommentCount<=18 AND posts.CreationDate>='2010-07-23 07:27:31'::timestamp AND posts.CreationDate<='2014-09-09 01:43:00'::timestamp",
        "comments": "SELECT COUNT(*) FROM comments as c WHERE comments.CreationDate>='2010-08-05 00:36:02'::timestamp AND comments.CreationDate<='2014-09-08 16:50:49'::timestamp",
        "postHistory": "SELECT COUNT(*) FROM postHistory as ph WHERE postHistory.PostHistoryTypeId=1 AND postHistory.CreationDate>='2010-09-14 11:59:07'::timestamp",
        "votes": "SELECT COUNT(*) FROM votes as v WHERE votes.VoteTypeId=2 AND votes.CreationDate<='2014-09-10 00:00:00'::timestamp",
        "postLinks": "SELECT COUNT(*) FROM postLinks as pl WHERE postLinks.LinkTypeId=1 AND postLinks.CreationDate>='2011-09-03 21:00:10'::timestamp AND postLinks.CreationDate<='2014-07-30 21:29:52'::timestamp",
        "users": "SELECT COUNT(*) FROM users as u WHERE users.DownVotes>=0 AND users.DownVotes<=0 AND users.UpVotes>=0 AND users.UpVotes<=31 AND users.CreationDate<='2014-08-06 20:38:52'::timestamp",
        "badges": "SELECT COUNT(*) FROM badges as b WHERE badges.Date>='2010-09-26 12:17:14'::timestamp",
        "tags": "SELECT COUNT(*) FROM tags"
    }

    true_cards = {
        "posts": 90764,
        "comments": 172156,
        "postHistory": 42308,
        "votes": 261476,
        "postLinks": 8776,
        "users": 37062,
        "badges": 77704,
        "tags": 1032
    }

    bn.init_inference_method()
    bn.infer_algo = "exact-jit"
    query = parse_query_single_table(queries[t_name], bn)
    pred = bn.query(query)
    assert min(pred, true_cards[t_name]) / max(pred, true_cards[t_name]) <= 1.5, f"Qerror too large, we have predition" \
                                                                        f"{pred} for true card {true_cards[t_name]}"

    query = parse_query_single_table(queries[t_name], bn)
    _, id_probs = bn.query_id_prob(query, bn.id_attributes)
    if t_name not in ['votes', 'tags']:
        assert min(pred, np.sum(id_probs)) / max(pred, np.sum(id_probs)) <= 1.5, "query_id_prob is incorrect"


def train_one_stats(dataset, data_path, model_folder, n_dim_dist=2, n_bins=200, bucket_method="greedy",
                    save_bucket_bins=False, seed=0, validate=True, actual_data=None):
    np.random.seed(seed)
    data, null_values, key_attrs, table_buckets, equivalent_keys, schema, bin_size = process_stats_data(data_path,
                                        model_folder, n_bins, bucket_method, save_bucket_bins, actual_data=actual_data)
    all_bns = dict()
    for table in schema.tables:
        t_name = table.table_name
        print(t_name)
        bn = Bayescard_BN(t_name, key_attrs[t_name], bin_size[t_name], null_values=null_values[t_name])
        bn.build_from_data(data[t_name])
        if validate:
            test_trained_BN_on_stats(bn, t_name)
        all_bns[t_name] = bn

    be = Bound_ensemble(table_buckets, schema, n_dim_dist, bns=all_bns, null_value=null_values)

    if not os.path.exists(model_folder):
        os.mkdir(model_folder)
    model_path = model_folder + f"model_{dataset}_{bucket_method}_{n_bins}.pkl"
    pickle.dump(be, open(model_path, 'wb'), pickle.HIGHEST_PROTOCOL)
    print(f"models save at {model_path}")


def train_one_imdb(data_path, model_folder, n_dim_dist=1, bin_size=None, bucket_method="fixed_start_key",
                   query_workload_file=None, save_bucket_bins=False, seed=0):
    """
    Training one FactorJoin model on IMDB dataset.
    :param data_path: The path to IMDB dataset
    :param model_folder: The folder where we would like to save the trained models
    :param bin_size: The total number of bins we would like to assign to all keys.
           If set to None, we provide our hardcoded bin size derived by analyzing a similar workload to IMDB-JOB.
    :param query_workload_file: If there exists a query workload, we can use it to plan our binning budget.
    :param save_bucket_bins:
    :return:
    """
    np.random.seed(seed)
    if bin_size is None:
        # The following is determined by analyzing a similar workload to the IMDB-JOB workload
        n_bins = {
            'title.id': 800,
            'info_type.id': 100,
            'keyword.id': 100,
            'company_name.id': 100,
            'name.id': 100,
            'company_type.id': 100,
            'comp_cast_type.id': 50,
            'kind_type.id': 50,
            'char_name.id': 50,
            'role_type.id': 50,
            'link_type.id': 50
        }
    elif type(bin_size) == int:
        # need to provide a workload file to automatically generate number of bins
        n_bins = get_n_bins_from_query(bin_size, data_path, query_workload_file)
    else:
        assert type(bin_size) == dict, "bin_size must of type int or dictionary mapping id attributes to int"
        n_bins = bin_size

    if not os.path.exists(model_folder):
        os.mkdir(model_folder)
    schema, table_buckets, ground_truth_factors_no_filter = process_imdb_data(data_path, model_folder, n_bins,
                                                                              bucket_method, save_bucket_bins, seed)
    be = Bound_ensemble(table_buckets, schema, n_dim_dist, ground_truth_factors_no_filter)
    if bin_size is None:
        bin_size = "default"
    elif type(bin_size) == dict:
        bin_size = "costumized"
    model_path = model_folder + f"model_imdb_{bin_size}.pkl"
    pickle.dump(be, open(model_path, 'wb'), pickle.HIGHEST_PROTOCOL)
    print(f"models save at {model_path}")


