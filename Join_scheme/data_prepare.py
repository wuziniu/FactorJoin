import copy
import logging
import pickle
import numpy as np
import pandas as pd
import time
import sys
import os

sys.path.append("/Users/ziniuw/Desktop/research/Learned_QO/CE_scheme/")

from Schemas.stats.schema import gen_stats_light_schema
from Join_scheme.binning import identify_key_values, sub_optimal_bucketize, greedy_bucketize, \
                                naive_bucketize, Table_bucket

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


def convert_time_to_int(data_folder):
    for file in os.listdir(data_folder):
        if file.endswith(".csv"):
            csv_file_location = data_folder + file
            df_rows = pd.read_csv(csv_file_location)
            for attribute in df_rows.columns:
                if "Date" in attribute:
                    if df_rows[attribute].values.dtype == 'object':
                        new_value = []
                        for value in df_rows[attribute].values:
                            new_value.append(timestamp_transorform(value))
                        df_rows[attribute] = new_value
            df_rows.to_csv(csv_file_location, index=False)


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


def generate_table_buckets(data, key_attrs, bin_sizes, bin_modes, optimal_buckets):
    table_buckets = dict()
    for table in data:
        table_data = data[table]
        table_bucket = Table_bucket(table, key_attrs[table], bin_sizes[table])
        for key in key_attrs[table]:
            if key in bin_modes and len(bin_modes[key]) != 0:
                table_bucket.oned_bin_modes[key] = bin_modes[key]
            else:
                # this is a primary key
                table_bucket.oned_bin_modes[key] = np.ones(table_bucket.bin_sizes[key])
        # getting mode for 2D bins
        if len(key_attrs[table]) == 2:
            key1 = key_attrs[table][0]
            key2 = key_attrs[table][1]
            res1 = np.zeros((table_bucket.bin_sizes[key1], table_bucket.bin_sizes[key2]))
            res2 = np.zeros((table_bucket.bin_sizes[key1], table_bucket.bin_sizes[key2]))
            key_data = np.stack((table_data[key1], table_data[key2]), axis=1)
            assert table_bucket.bin_sizes[key1] == len(optimal_buckets[key1].bins)
            assert table_bucket.bin_sizes[key2] == len(optimal_buckets[key2].bins)
            for v1, b1 in enumerate(optimal_buckets[key1].bins):
                temp_data = key_data[key_data[:, 0] == v1]
                if len(temp_data) == 0:
                    continue
                for v2, b2 in enumerate(optimal_buckets[key2].bins):
                    temp_data2 = copy.deepcopy(temp_data[temp_data[:, 1] == v2])
                    if len(temp_data2) == 0:
                        continue
                    res1[v1, v2] = np.max(np.unique(temp_data2[:, 0], return_counts=True)[-1])
                    res2[v1, v2] = np.max(np.unique(temp_data2[:, 1], return_counts=True)[-1])
            table_bucket.twod_bin_modes[key1] = res1
            table_bucket.twod_bin_modes[key2] = res2
        table_buckets[table] = table_bucket

    return table_buckets


def generate_table_buckets_means(data, key_attrs, bin_sizes, all_bin_means, optimal_buckets):
    table_buckets = dict()
    for table in data:
        table_data = data[table]
        table_bucket = Table_bucket(table, key_attrs[table], bin_sizes[table])
        for key in key_attrs[table]:
            if key in all_bin_means and len(all_bin_means[key]) != 0:
                table_bucket.oned_bin_modes[key] = all_bin_means[key]
            else:
                # this is a primary key
                table_bucket.oned_bin_modes[key] = np.ones(table_bucket.bin_sizes[key])
        # getting mode for 2D bins
        if len(key_attrs[table]) == 2:
            key1 = key_attrs[table][0]
            key2 = key_attrs[table][1]
            res1 = np.zeros((table_bucket.bin_sizes[key1], table_bucket.bin_sizes[key2]))
            res2 = np.zeros((table_bucket.bin_sizes[key1], table_bucket.bin_sizes[key2]))
            key_data = np.stack((table_data[key1], table_data[key2]), axis=1)
            assert table_bucket.bin_sizes[key1] == len(optimal_buckets[key1].bins)
            assert table_bucket.bin_sizes[key2] == len(optimal_buckets[key2].bins)
            for v1, b1 in enumerate(optimal_buckets[key1].bins):
                temp_data = key_data[key_data[:, 0] == v1]
                if len(temp_data) == 0:
                    continue
                for v2, b2 in enumerate(optimal_buckets[key2].bins):
                    temp_data2 = copy.deepcopy(temp_data[temp_data[:, 1] == v2])
                    if len(temp_data2) == 0:
                        continue
                    res1[v1, v2] = np.mean(np.unique(temp_data2[:, 0], return_counts=True)[-1])
                    res2[v1, v2] = np.mean(np.unique(temp_data2[:, 1], return_counts=True)[-1])
            table_bucket.twod_bin_modes[key1] = res1
            table_bucket.twod_bin_modes[key2] = res2
        table_buckets[table] = table_bucket

    return table_buckets


def generate_table_buckets2(data, key_attrs, bin_sizes, bin_modes, optimal_buckets):
    table_buckets = dict()
    for table in data:
        table_data = data[table]
        table_bucket = Table_bucket(table, key_attrs[table], bin_sizes[table])
        for key in key_attrs[table]:
            if key in bin_modes and len(bin_modes[key]) != 0:
                table_bucket.oned_bin_modes[key] = bin_modes[key]
            else:
                # this is a primary key
                table_bucket.oned_bin_modes[key] = np.ones(table_bucket.bin_sizes[key])
        # getting mode for 2D bins
        if len(key_attrs[table]) == 2:
            key1 = key_attrs[table][0]
            key2 = key_attrs[table][1]
            res1 = np.zeros((table_bucket.bin_sizes[key1], table_bucket.bin_sizes[key2]))
            res2 = np.zeros((table_bucket.bin_sizes[key1], table_bucket.bin_sizes[key2]))
            key_data = np.stack((table_data[key1].values, table_data[key2].values), axis=1)
            assert table_bucket.bin_sizes[key1] == len(optimal_buckets[key1].bins)
            assert table_bucket.bin_sizes[key2] == len(optimal_buckets[key2].bins)
            for v1, b1 in enumerate(optimal_buckets[key1].bins):
                temp_data = key_data[np.isin(key_data[:, 0], b1)]
                if len(temp_data) == 0:
                    continue
                #assert np.max(np.unique(temp_data[:, 0], return_counts=True)[-1]) == table_bucket.oned_bin_modes[key1][
                 #   v1], f"{key1} data error at {v1}, with " \
                  #       f"{np.max(np.unique(temp_data[:, 0], return_counts=True)[-1])} and " \
                   #      f"{table_bucket.oned_bin_modes[key1][v1]}."
                for v2, b2 in enumerate(optimal_buckets[key2].bins):
                    temp_data2 = copy.deepcopy(temp_data[np.isin(temp_data[:, 1], b2)])
                    if len(temp_data2) == 0:
                        continue
                    res1[v1, v2] = np.max(np.unique(temp_data2[:, 0], return_counts=True)[-1])
                    res2[v1, v2] = np.max(np.unique(temp_data2[:, 1], return_counts=True)[-1])
            table_bucket.twod_bin_modes[key1] = res1
            table_bucket.twod_bin_modes[key2] = res2
        table_buckets[table] = table_bucket

    return table_buckets


def process_stats_data(data_path, model_folder, n_bins=500, bucket_method="greedy", save_bucket_bins=False,
                       return_bin_means=False, get_bin_means=False):
    """
    Preprocessing stats data and generate optimal bucket
    :param data_path: path to stats data folder
    :param n_bins: number of bins (the actually number of bins returned will be smaller than this)
    :param bucket_method: choose between "sub_optimal" and "greedy". Please refer to binning.py for details.
    :param save_bucket_bins: Set to true for dynamic environment, the default is False for static environment
    :return:
    """
    if not data_path.endswith(".csv"):
        data_path += "/{}.csv"
    schema = gen_stats_light_schema(data_path)
    all_keys, equivalent_keys = identify_key_values(schema)
    data = dict()
    key_data = dict()  # store the columns of all keys
    sample_rate = dict()
    primary_keys = []
    null_values = dict()
    key_attrs = dict()
    for table_obj in schema.tables:
        table_name = table_obj.table_name
        null_values[table_name] = dict()
        key_attrs[table_name] = []
        df_rows = read_table_csv(table_obj, stats=True)
        for attr in df_rows.columns:
            if attr in all_keys:
                key_data[attr] = df_rows[attr].values
                # the nan value of id are set to -1, this is hardcoded.
                key_data[attr][np.isnan(key_data[attr])] = -1
                key_data[attr][key_data[attr] < 0] = -1
                null_values[table_name][attr] = -1
                key_data[attr] = copy.deepcopy(key_data[attr])[key_data[attr] >= 0]
                # if the all keys have exactly one appearance, we consider them primary keys
                # we set a error margin of 0.01 in case of data mis-write.
                if len(np.unique(key_data[attr])) >= len(key_data[attr]) * 0.99:
                    primary_keys.append(attr)
                sample_rate[attr] = 1.0
                key_attrs[table_name].append(attr)
            else:
                temp = df_rows[attr].values
                null_values[table_name][attr] = np.nanmin(temp) - 100
                temp[np.isnan(temp)] = null_values[table_name][attr]
        data[table_name] = df_rows

    all_bin_modes = dict()
    bin_size = dict()
    binned_data = dict()
    optimal_buckets = dict()
    all_bin_means = dict()
    all_bin_width = dict()
    for PK in equivalent_keys:
        print(f"bucketizing equivalent key group:", equivalent_keys[PK])
        group_data = {}
        group_sample_rate = {}
        for K in equivalent_keys[PK]:
            group_data[K] = key_data[K]
            group_sample_rate[K] = sample_rate[K]
        if bucket_method == "greedy":
            temp_data, optimal_bucket = greedy_bucketize(group_data, sample_rate, n_bins, primary_keys, True)
        elif bucket_method == "sub_optimal":
            temp_data, optimal_bucket = sub_optimal_bucketize(group_data, sample_rate, n_bins, primary_keys, return_bin_means)
        elif bucket_method == "naive":
            temp_data, optimal_bucket = naive_bucketize(group_data, sample_rate, n_bins, primary_keys, True)
        else:
            assert False, f"unrecognized bucketization method: {bucket_method}"

        binned_data.update(temp_data)
        for K in equivalent_keys[PK]:
            optimal_buckets[K] = optimal_bucket
            all_bin_means[K] = np.asarray(optimal_bucket.buckets[K].bin_means)
            all_bin_width[K] = np.asarray(optimal_bucket.buckets[K].bin_width)
            temp_table_name = K.split(".")[0]
            if temp_table_name not in bin_size:
                bin_size[temp_table_name] = dict()
            bin_size[temp_table_name][K] = len(optimal_bucket.bins)
            all_bin_modes[K] = np.asarray(optimal_bucket.buckets[K].bin_modes)

    for K in binned_data:
        temp_table_name = K.split(".")[0]
        temp = data[temp_table_name][K].values
        temp[temp >= 0] = binned_data[K]
        
    if get_bin_means:
        table_buckets = generate_table_buckets_means(data, key_attrs, bin_size, all_bin_means, optimal_buckets)
    else:
        table_buckets = generate_table_buckets(data, key_attrs, bin_size, all_bin_modes, optimal_buckets)

    if save_bucket_bins:
        with open(model_folder + f"/buckets.pkl") as f:
            pickle.dump(optimal_buckets, f, pickle.HIGHEST_PROTOCOL)
    if return_bin_means:
        return data, null_values, key_attrs, table_buckets, equivalent_keys, schema, bin_size, all_bin_means, all_bin_width
    return data, null_values, key_attrs, table_buckets, equivalent_keys, schema, bin_size
