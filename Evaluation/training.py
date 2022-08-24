import pickle
import time
import numpy as np
import os
from Join_scheme.data_prepare import process_imdb_data
from Join_scheme.bound import Bound_ensemble
from Join_scheme.tools import get_n_bins_from_query



def train_one_imdb(data_path, model_folder, bin_size=None, query_workload=None, save_bucket_bins=False):
    """
    Training one FactorJoin model on IMDB dataset.
    :param data_path: The path to IMDB dataset
    :param model_folder: The folder where we would like to save the trained models
    :param bin_size: The total number of bins we would like to assign to all keys.
           If set to None, we provide our hardcoded bin size derived by analyzing a similar workload to IMDB-JOB.
    :param query_workload: If there exists a query workload, we can use it to plan our binning budget.
    :param save_bucket_bins:
    :return:
    """
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
    else:
        #TODO: implement auto bin size generation
        n_bins = get_n_bins_from_query(bin_size, data_path, query_workload)
    if not os.path.exists(model_folder):
        os.mkdir(model_folder)
    schema, table_buckets, ground_truth_factors_no_filter = process_imdb_data(data_path, model_folder, n_bins,
                                                                              save_bucket_bins)
    be = Bound_ensemble(table_buckets, schema, ground_truth_factors_no_filter)
    if bin_size is None:
        bin_size = "default"
    model_path = model_folder + f"model_imdb_{bin_size}.pkl"
    pickle.dump(be, open(model_path, 'wb'), pickle.HIGHEST_PROTOCOL)
    print(f"models save at {model_path}")



