import pickle
import time
import numpy as np
from Join_scheme.data_prepare import process_imdb_data
from Join_scheme.bound import Bound_ensemble


def train_one_imdb(data_path, model_folder, bin_size=None, save_bucket_bins=False):
    if bin_size is None:
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
            'role_type.id': 50
        }
    else:
        #TODO: implement auto bin size generation
        n_bins = dict()
    schema, table_buckets = process_imdb_data(data_path, model_folder, n_bins, save_bucket_bins)
    return table_buckets



