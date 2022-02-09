import copy
import logging
import pickle
import numpy as np
import pandas as pd
import time

from Schemas.imdb.schema import gen_imdb_schema
from Schemas.stats.schema import gen_stats_light_schema
from Join_scheme.binning import identify_key_values, sub_optimal_bucketize, Table_bucket
from Join_scheme.binning import apply_binning_to_data_value_count

logger = logging.getLogger(__name__)


def timestamp_transorform(time_string, start_date="2010-07-19 00:00:00"):
    start_date_int = time.strptime(start_date, "%Y-%m-%d %H:%M:%S")
    time_array = time.strptime(time_string, "%Y-%m-%d %H:%M:%S")
    return int(time.mktime(time_array)) - int(time.mktime(start_date_int))


def read_table_hdf(table_obj):
    """
    Reads hdf from path, renames columns and drops unnecessary columns
    """
    df_rows = pd.read_hdf(table_obj.csv_file_location)
    df_rows.columns = [table_obj.table_name + '.' + attr for attr in table_obj.attributes]

    for attribute in table_obj.irrelevant_attributes:
        df_rows = df_rows.drop(table_obj.table_name + '.' + attribute, axis=1)

    return df_rows.apply(pd.to_numeric, errors="ignore")


def read_table_csv(table_obj, csv_seperator=',', stats=True):
    """
    Reads csv from path, renames columns and drops unnecessary columns
    """
    if stats:
        df_rows = pd.read_csv(table_obj.csv_file_location)
    else:
        df_rows = pd.read_csv(table_obj.csv_file_location, header=None, escapechar='\\', encoding='utf-8',
                              quotechar='"',
                              sep=csv_seperator)
    df_rows.columns = [table_obj.table_name + '.' + attr for attr in table_obj.attributes]

    for attribute in table_obj.irrelevant_attributes:
        df_rows = df_rows.drop(table_obj.table_name + '.' + attribute, axis=1)

    return df_rows.apply(pd.to_numeric, errors="ignore")


def make_sample(np_data, nrows=1000000, seed=0):
    np.random.seed(seed)
    if len(np_data) <= nrows:
        return np_data, 1.0
    else:
        selected = np.random.choice(len(np_data), size=nrows, replace=False)
        return np_data[selected], nrows/len(np_data)


def stats_analysis(sample, data, sample_rate, show=10):
    n, c = np.unique(sample, return_counts=True)
    idx = np.argsort(c)[::-1]
    for i in range(min(show, len(idx))):
        print(c[idx[i]], c[idx[i]]/sample_rate, len(data[data == n[idx[i]]]))


def get_ground_truth_no_filter(equivalent_keys, data, bins, table_lens, na_values):
    all_factor_pdfs = dict()
    for PK in equivalent_keys:
        bin_value = bins[PK]
        for key in equivalent_keys[PK]:
            table = key.split(".")[0]
            temp = apply_binning_to_data_value_count(bin_value, data[key])
            if table not in all_factor_pdfs:
                all_factor_pdfs[table] = dict()
            all_factor_pdfs[table][key] = temp / np.sum(temp)

    all_factors = dict()
    for table in all_factor_pdfs:
        all_factors[table] = Factor(table, table_lens[table], list(all_factor_pdfs[table].keys()),
                                    all_factor_pdfs[table], na_values[table])
    return all_factors



def process_imdb_data(data_path, model_folder, n_bins, sample_size=100000, save_bucket_bins=False):
    schema = gen_imdb_schema(data_path)
    all_keys, equivalent_keys = identify_key_values(schema)
    data = dict()
    table_lens = dict()
    na_values = dict()
    primary_keys = []
    for table_obj in schema.tables:
        df_rows = pd.read_csv(table_obj.csv_file_location, header=None, escapechar='\\', encoding='utf-8',
                              quotechar='"',
                              sep=",")

        df_rows.columns = [table_obj.table_name + '.' + attr for attr in table_obj.attributes]

        for attribute in table_obj.irrelevant_attributes:
            df_rows = df_rows.drop(table_obj.table_name + '.' + attribute, axis=1)

        df_rows.apply(pd.to_numeric, errors="ignore")
        table_lens[table_obj.table_name] = len(df_rows)
        if table_obj.table_name not in na_values:
            na_values[table_obj.table_name] = dict()
        for attr in df_rows.columns:
            if attr in all_keys:
                data[attr] = df_rows[attr].values
                data[attr][np.isnan(data[attr])] = -1
                data[attr][data[attr] < 0] = -1
                na_values[table_obj.table_name][attr] = len(data[attr][data[attr] != -1]) / table_lens[
                    table_obj.table_name]
                data[attr] = copy.deepcopy(data[attr])[data[attr] >= 0]
                if len(np.unique(data[attr])) >= len(data[attr]) - 10:
                    primary_keys.append(attr)

    sample_rate = dict()
    sampled_data = dict()
    for k in data:
        temp = make_sample(data[k], sample_size)
        sampled_data[k] = temp[0]
        sample_rate[k] = temp[1]

    optimal_buckets = dict()
    bin_size = dict()
    all_bin_modes = dict()
    for PK in equivalent_keys:
        group_data = {}
        group_sample_rate = {}
        for K in equivalent_keys[PK]:
            group_data[K] = sampled_data[K]
            group_sample_rate[K] = sample_rate[K]
        _, optimal_bucket = sub_optimal_bucketize(group_data, group_sample_rate, n_bins=n_bins[PK], primary_keys=primary_keys)
        for K in equivalent_keys[PK]:
            optimal_buckets[K] = optimal_bucket
            temp_table_name = K.split(".")[0]
            if temp_table_name not in bin_size:
                bin_size[temp_table_name] = dict()
                all_bin_modes[temp_table_name] = dict()
            bin_size[temp_table_name][K] = len(optimal_bucket.bins)
            all_bin_modes[temp_table_name][K] = optimal_bucket.buckets[K].bin_modes

    table_buckets = dict()
    for table_name in bin_size:
        table_buckets[table_name] = Table_bucket(table_name, list(bin_size[table_name].keys()), bin_size[table_name],
                                                 all_bin_modes[table_name])

    all_bins = dict()
    for key in optimal_buckets:
        all_bins[key] = optimal_buckets[key].bins

    ground_truth_factors_no_filter = get_ground_truth_no_filter(equivalent_keys, data, all_bins, table_lens, na_values)

    if save_bucket_bins:
        with open(model_folder + f"/buckets.pkl") as f:
            pickle.dump(optimal_buckets, f, pickle.HIGHEST_PROTOCOL)

    return schema, table_buckets, ground_truth_factors_no_filter

