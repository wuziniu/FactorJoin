import numpy as np
import pickle
import time
import os
from Schemas.stats.schema import gen_stats_light_schema
from Evaluation.training import train_one_stats, test_trained_BN_on_stats
from Join_scheme.data_prepare import read_table_csv, update_stats_data
from BayesCard.Models.Bayescard_BN import Bayescard_BN


def timestamp_transorform(time_string, start_date="2010-07-19 00:00:00"):
    start_date_int = time.strptime(start_date, "%Y-%m-%d %H:%M:%S")
    time_array = time.strptime(time_string, "%Y-%m-%d %H:%M:%S")
    return int(time.mktime(time_array)) - int(time.mktime(start_date_int))


def get_data_by_date(data_path, time_date="2014-01-01 00:00:00"):
    time_value = timestamp_transorform(time_date)
    if not data_path.endswith(".csv"):
        data_path += "/{}.csv"
    schema = gen_stats_light_schema(data_path)
    before_data = dict()
    after_data = dict()
    for table_obj in schema.tables:
        table_name = table_obj.table_name
        df_rows = read_table_csv(table_obj, db_name="stats")
        idx = len(df_rows)
        for attribute in df_rows.columns:
            if "Date" in attribute:
                idx = np.searchsorted(df_rows[attribute].values, time_value)
                break

        before_data[table_name] = df_rows[:idx] if idx > 0 else None
        after_data[table_name] = df_rows[idx:] if idx < len(df_rows) else None
    return before_data, after_data


def update_one_stats(FJmodel, buckets, table_buckets, data_path, save_model_folder, save_bucket_bins=False,
                     update_BN=True, retrain_BN=False, old_data=None, validate=False):
    """
    Incrementally update the FactorJoin model
    """
    data, table_buckets, null_values = update_stats_data(data_path, save_model_folder, buckets, table_buckets,
                                                         save_bucket_bins)
    FJmodel.table_buckets = table_buckets
    if update_BN:
        # updating the single table estimator
        if retrain_BN:
            # retrain the BN based on the new and old data
            for table in FJmodel.schema.tables:
                t_name = table.table_name
                if t_name in data and data[t_name] is not None:
                    bn = Bayescard_BN(t_name, table_buckets[t_name].id_attributes, table_buckets[t_name].bin_sizes,
                                      null_values=null_values[t_name])
                    new_data = old_data[t_name].append(data[t_name], ignore_index=True)
                    bn.build_from_data(new_data)
                    if validate:
                        test_trained_BN_on_stats(bn, t_name)
                    FJmodel.bns[t_name] = bn
        else:
            # incrementally update BN
            for table in FJmodel.schema.tables:
                t_name = table.table_name
                if t_name in data and data[t_name] is not None:
                    bn = FJmodel.bns[t_name]
                    bn.null_values = null_values[t_name]
                    bn.update_from_data(data)

    model_path = os.path.join(save_model_folder, f"update_model.pkl")
    pickle.dump(FJmodel, open(model_path, 'wb'), pickle.HIGHEST_PROTOCOL)
    print(f"models save at {model_path}")


def update_one_imdb():
    "TODO: implement the update on IMDB, should be straight-forward as it uses the sampling for base-table"
    return

def eval_update(data_folder, model_path, n_dim_dist, bin_size, bucket_method, split_date="2014-01-01 00:00:00", seed=0):
    np.random.seed(seed)
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    before_data, after_data = get_data_by_date(data_folder, split_date)
    print("************************************************************")
    print(f"Training the model with data before {split_date}")
    start_time = time.time()
    train_one_stats("stats", data_folder, model_path, n_dim_dist, bin_size, bucket_method, True, actual_data=before_data)
    print(f"training completed, took {time.time() - start_time} sec")

    # loading the trained model and buckets
    with open(os.path.join(model_path, "stats_buckets.pkl"), "rb") as f:
        buckets = pickle.load(f)
    with open(os.path.join(model_path, f"model_stats_{bucket_method}_{bin_size}.pkl"), "rb") as f:
        FJmodel = pickle.load(f)
    print("************************************************************")
    print(f"Updating the model with data after {split_date}")
    start_time = time.time()
    table_buckets = FJmodel.table_buckets
    null_values = FJmodel.null_value
    data, table_buckets, null_values = update_stats_data(data_folder, model_path, buckets, table_buckets,
                                                         null_values, False, after_data)
    for table in FJmodel.schema.tables:
        t_name = table.table_name
        if t_name in data and data[t_name] is not None:
            bn = FJmodel.bns[t_name]
            bn.null_values = null_values[t_name]
            bn.update_from_data(data[t_name])
    print(f"updating completed, took {time.time() - start_time} sec")
    model_path = os.path.join(model_path, f"updated_model_stats_{bucket_method}_{bin_size}.pkl")
    pickle.dump(FJmodel, open(model_path, 'wb'), pickle.HIGHEST_PROTOCOL)
    print(f"updated models save at {model_path}")


