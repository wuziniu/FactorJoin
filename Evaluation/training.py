import pickle
import os
import numpy as np
from Join_scheme.data_prepare import process_stats_data
from Join_scheme.bound import Bound_ensemble
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


def train_one_stats(dataset, data_path, model_folder, n_bins=200, bucket_method="greedy", save_bucket_bins=False,
                    validate=True, actual_data=None):
    data, null_values, key_attrs, table_buckets, equivalent_keys, schema, bin_size = process_stats_data(data_path,
                                        model_folder, n_bins, bucket_method, save_bucket_bins, data=actual_data)
    all_bns = dict()
    for table in schema.tables:
        t_name = table.table_name
        print(t_name)
        bn = Bayescard_BN(t_name, key_attrs[t_name], bin_size[t_name], null_values=null_values[t_name])
        bn.build_from_data(data[t_name])
        if validate:
            test_trained_BN_on_stats(bn, t_name)
        all_bns[t_name] = bn

    be = Bound_ensemble(all_bns, table_buckets, schema, null_values)

    if not os.path.exists(model_folder):
        os.mkdir(model_folder)
    model_path = model_folder + f"model_{dataset}_{bucket_method}_{n_bins}.pkl"
    pickle.dump(be, open(model_path, 'wb'), pickle.HIGHEST_PROTOCOL)
    print(f"models save at {model_path}")

