import copy
import logging
import pickle
import numpy as np
import pandas as pd
import time
import os

from Schemas.stats.schema import gen_stats_light_schema
from Schemas.imdb.schema import gen_imdb_schema
from Join_scheme.binning import identify_key_values, sub_optimal_bucketize, greedy_bucketize, \
                                fixed_start_key_bucketize, get_start_key, naive_bucketize, \
                                Table_bucket, update_bins, apply_binning_to_data_value_count
from Join_scheme.bound import Factor

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


def get_data_by_date(data_path, time_date="2014-01-01 00:00:00"):
    time_value = timestamp_transorform(time_date)
    if not data_path.endswith(".csv"):
        data_path += "/{}.csv"
    schema = gen_stats_light_schema(data_path)
    before_data = dict()
    after_data = dict()
    for table_obj in schema.tables:
        table_name = table_obj.table_name
        df_rows = read_table_csv(table_obj)
        idx = len(df_rows)
        for attribute in df_rows.columns:
            if "Date" in attribute:
                idx = np.searchsorted(df_rows[attribute].values, time_value)
                break

        before_data[table_name] = df_rows[:idx] if idx > 0 else None
        after_data[table_name] = df_rows[idx:] if idx < len(df_rows) else None
    return before_data, after_data


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


def generate_table_buckets(data, binned_data, key_attrs, bin_sizes, bin_modes, optimal_buckets):
    """
    Bin all join keys in a DB schema and calculate the bin value count and mode
    :param data: [dict] table : data [pandas dataframe]
    :param key_attrs: [dict] table: list of all join key attributes on this table
    :param bin_sizes: [dict] table: the bin size for this table
    :param bin_modes: one dimensional bin mode
    :param optimal_buckets: the optimal bucketization for this DB schema
    :return:
    """
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
            key_binned_data = np.stack((binned_data[key1], binned_data[key2]), axis=1)
            key_data = np.stack((table_data[key1], table_data[key2]), axis=1)
            assert table_bucket.bin_sizes[key1] == len(optimal_buckets[key1].bins)
            assert table_bucket.bin_sizes[key2] == len(optimal_buckets[key2].bins)
            for v1 in range(len(optimal_buckets[key1].bins)):
                temp_binned_data = key_binned_data[key_binned_data[:, 0] == v1]
                temp_data = key_data[key_binned_data[:, 0] == v1]
                if len(temp_data) == 0:
                    continue
                for v2, b2 in enumerate(optimal_buckets[key2].bins):
                    temp_data2 = copy.deepcopy(temp_data[temp_binned_data[:, 1] == v2])
                    if len(temp_data2) == 0:
                        continue
                    res1[v1, v2] = np.max(np.unique(temp_data2[:, 0], return_counts=True)[-1])
                    res2[v1, v2] = np.max(np.unique(temp_data2[:, 1], return_counts=True)[-1])
            table_bucket.twod_bin_modes[key1] = res1
            table_bucket.twod_bin_modes[key2] = res2
        table_buckets[table] = table_bucket

    return table_buckets


def generate_table_bucket_means(data, binned_data, key_attrs, bin_sizes, all_bin_means, optimal_buckets):
    """
    Similar to function "generate_table_buckets" but instead of using bin mode we use bin mean as histogram summary.
    """
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
            key_binned_data = np.stack((binned_data[key1], binned_data[key2]), axis=1)
            key_data = np.stack((table_data[key1], table_data[key2]), axis=1)
            assert table_bucket.bin_sizes[key1] == len(optimal_buckets[key1].bins)
            assert table_bucket.bin_sizes[key2] == len(optimal_buckets[key2].bins)
            for v1 in range(len(optimal_buckets[key1].bins)):
                temp_binned_data = key_binned_data[key_binned_data[:, 0] == v1]
                temp_data = key_data[key_binned_data[:, 0] == v1]
                if len(temp_data) == 0:
                    continue
                for v2, b2 in range(len(optimal_buckets[key2].bins)):
                    temp_data2 = copy.deepcopy(temp_data[temp_binned_data[:, 1] == v2])
                    if len(temp_data2) == 0:
                        continue
                    res1[v1, v2] = np.mean(np.unique(temp_data2[:, 0], return_counts=True)[-1])
                    res2[v1, v2] = np.mean(np.unique(temp_data2[:, 1], return_counts=True)[-1])
            table_bucket.twod_bin_modes[key1] = res1
            table_bucket.twod_bin_modes[key2] = res2
        table_buckets[table] = table_bucket

    return table_buckets


def update_table_buckets(buckets, data, binned_data, all_bin_modes, table_buckets):
    """
    Incrementally update the bin mode
    """
    for table in data:
        table_data = data[table]
        if table_data is None:
            continue
        key_attrs = table_buckets[table].id_attributes

        for key in key_attrs:
            if key in all_bin_modes and len(all_bin_modes[key]) != 0:
                prev_bin_mode = table_buckets[table].oned_bin_modes[key]
                for i, nbm in enumerate(all_bin_modes[key]):
                    bm = prev_bin_mode[i]
                    if bm <= 10:
                        table_buckets[table].oned_bin_modes[key][i] = max(nbm, bm)
                    else:
                        table_buckets[table].oned_bin_modes[key][i] = bm + 0.5 * nbm

        # getting mode for 2D bins
        if len(key_attrs) == 2:
            key1 = key_attrs[0]
            key2 = key_attrs[1]
            if key1 not in binned_data or key2 not in binned_data:
                print(key1, key2, binned_data)
                continue
            key_binned_data = np.stack((binned_data[key1], binned_data[key2]), axis=1)
            key_data = np.stack((table_data[key1], table_data[key2]), axis=1)
            old_twod_bin_modes1 = table_buckets[table].twod_bin_modes[key1]
            old_twod_bin_modes2 = table_buckets[table].twod_bin_modes[key2]
            for v1 in range(len(buckets[key1].bins)):
                temp_binned_data = key_binned_data[key_binned_data[:, 0] == v1]
                temp_data = key_data[key_binned_data[:, 0] == v1]
                if len(temp_data) == 0:
                    continue
                for v2 in range(len(buckets[key2].bins)):
                    temp_data2 = copy.deepcopy(temp_data[temp_binned_data[:, 1] == v2])
                    if len(temp_data2) == 0:
                        continue
                    nbm1 = np.max(np.unique(temp_data2[:, 0], return_counts=True)[-1])
                    bm1 = old_twod_bin_modes1[v1, v2]
                    if bm1 <= 5:
                        table_buckets[table].twod_bin_modes[key1][v1, v2] = max(bm1, nbm1)
                    else:
                        table_buckets[table].twod_bin_modes[key1][v1, v2] = bm1 + 0.5 * nbm1
                    nbm2 = np.max(np.unique(temp_data2[:, 1], return_counts=True)[-1])
                    bm2 = old_twod_bin_modes2[v1, v2]
                    if bm2 <= 5:
                        table_buckets[table].twod_bin_modes[key2][v1, v2] = max(bm2, nbm2)
                    else:
                        table_buckets[table].twod_bin_modes[key2][v1, v2] = bm2 + 0.5 * nbm2
                        
    return table_buckets


def process_stats_data(data_path, model_folder, n_bins=500, bucket_method="greedy", save_bucket_bins=False,
                       return_bin_means=False, get_bin_means=False, actual_data=None):
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
    if actual_data is None:
        actual_data = dict()
    data = dict()
    key_data = dict()  # store the columns of all keys
    table_key_lens = dict()
    sample_rate = dict()
    primary_keys = []
    null_values = dict()
    key_attrs = dict()
    for table_obj in schema.tables:
        table_name = table_obj.table_name
        null_values[table_name] = dict()
        key_attrs[table_name] = []
        if table_name in actual_data:
            df_rows = copy.deepcopy(actual_data[table_name])
        else:
            df_rows = read_table_csv(table_obj, stats=True)
        for attr in df_rows.columns:
            if attr in all_keys:
                table_key_lens[attr] = len(df_rows)
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
            temp_data, optimal_bucket = sub_optimal_bucketize(group_data, sample_rate, n_bins, primary_keys,
                                                              return_bin_means, True)
        elif bucket_method == "naive":
            temp_data, optimal_bucket = naive_bucketize(group_data, sample_rate, n_bins, primary_keys, True)
        elif bucket_method == "fixed_start_key":
            start_key = get_start_key(list(group_data.keys()), table_key_lens, primary_keys)
            temp_data, optimal_bucket = fixed_start_key_bucketize(start_key, group_data, group_sample_rate,
                                                                  n_bins=n_bins[PK], primary_keys=primary_keys,
                                                                  return_data=True)
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
        temp = copy.deepcopy(data[temp_table_name][K].values)
        temp[temp >= 0] = binned_data[K]
        binned_data[K] = temp
        
    if get_bin_means:
        table_buckets = generate_table_bucket_means(data, binned_data, key_attrs, bin_size, all_bin_means, optimal_buckets)
    else:
        table_buckets = generate_table_buckets(data, binned_data, key_attrs, bin_size, all_bin_modes, optimal_buckets)

    for K in binned_data:
        temp_table_name = K.split(".")[0]
        data[temp_table_name][K] = binned_data[K]

    if save_bucket_bins:
        with open(model_folder + "buckets.pkl", "wb") as f:
            pickle.dump(optimal_buckets, f, pickle.HIGHEST_PROTOCOL)
    if return_bin_means:
        return data, null_values, key_attrs, table_buckets, equivalent_keys, schema, bin_size, all_bin_means, all_bin_width
    return data, null_values, key_attrs, table_buckets, equivalent_keys, schema, bin_size


def update_stats_data(data_path, model_folder, buckets, table_buckets, null_values=None,
                      save_bucket_bins=False, data=None):
    """
    updating stats data according to an existing bucket
    """
    if not data_path.endswith(".csv"):
        data_path += "/{}.csv"
    schema = gen_stats_light_schema(data_path)
    all_keys, equivalent_keys = identify_key_values(schema)
    if data is None:
        data = dict()
    key_data = dict()  # store the columns of all keys
    sample_rate = dict()
    primary_keys = []
    if null_values is None:
        null_values = dict()
    key_attrs = dict()
    for table_obj in schema.tables:
        table_name = table_obj.table_name
        if table_name not in null_values:
            null_values[table_name] = dict()
        key_attrs[table_name] = []
        if table_name in data:
            df_rows = data[table_name]
            if df_rows is None or len(df_rows) == 0:
                print(f"{table_name} does not have data to update")
                continue
        else:
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
                if attr not in null_values[table_name]:
                    null_values[table_name][attr] = np.nanmin(temp) - 100
                temp[np.isnan(temp)] = null_values[table_name][attr]
        data[table_name] = df_rows

    all_bin_modes = dict()
    all_binned_data = dict()
    for PK in equivalent_keys:
        print(f"updating equivalent key group:", equivalent_keys[PK])
        binned_data, new_bin_modes = update_bins(buckets[PK], key_data, equivalent_keys[PK])
        all_binned_data.update(binned_data)
        all_bin_modes.update(new_bin_modes)

    for K in all_binned_data:
        temp_table_name = K.split(".")[0]
        temp = copy.deepcopy(data[temp_table_name][K].values)
        temp[temp >= 0] = all_binned_data[K]
        all_binned_data[K] = temp

    new_table_buckets = update_table_buckets(buckets, data, all_binned_data, all_bin_modes, table_buckets)

    for K in all_binned_data:
        temp_table_name = K.split(".")[0]
        data[temp_table_name][K] = all_binned_data[K]

    if save_bucket_bins:
        with open(model_folder + f"buckets.pkl", "wb") as f:
            pickle.dump(buckets, f, pickle.HIGHEST_PROTOCOL)

    return data, new_table_buckets, null_values


def make_sample(np_data, nrows=1000000, seed=0):
    np.random.seed(seed)
    samp_data = np_data[np_data != -1]
    if len(samp_data) <= nrows:
        return samp_data, 1.0
    else:
        selected = np.random.choice(len(samp_data), size=int(nrows), replace=False)
        return samp_data[selected], nrows / len(samp_data)


def stats_analysis(sample, data, sample_rate, show=10):
    n, c = np.unique(sample, return_counts=True)
    idx = np.argsort(c)[::-1]
    for i in range(min(show, len(idx))):
        print(c[idx[i]], c[idx[i]] / sample_rate, len(data[data == n[idx[i]]]))


def get_ground_truth_no_filter(equivalent_keys, data, bins, table_lens, na_values, save_path=None):
    """
    Get the ground-truth data distribution on each single table without filter predicates.
    """
    if save_path and os.path.exists(save_path + f"/gt_no_filter.pkl"):
        with open(save_path + f"/gt_no_filter.pkl", "rb") as f:
            all_factors = pickle.load(f)
        return all_factors

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
                                    all_factor_pdfs[table], na_values=na_values[table])
    if save_path:
        with open(save_path + f"/gt_no_filter.pkl", "wb") as f:
            pickle.dump(all_factors, f, pickle.HIGHEST_PROTOCOL)

    return all_factors


def process_imdb_data(data_path, model_folder, n_bins, bucket_method, sample_size=1000000, save_bucket_bins=False,
                      seed=0):
    schema = gen_imdb_schema(data_path)
    all_keys, equivalent_keys = identify_key_values(schema)
    data = dict()
    table_lens = dict()
    table_key_count_var = dict()
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
                attr_value_count = np.unique(data[attr], return_counts=True)[-1]
                table_key_count_var[attr] = np.var(attr_value_count)

    sample_rate = dict()
    sampled_data = dict()
    for k in data:
        temp = make_sample(data[k], sample_size, seed)
        sampled_data[k] = temp[0]
        sample_rate[k] = temp[1]

    optimal_buckets = dict()
    bin_size = dict()
    all_bin_modes = dict()
    for PK in equivalent_keys:
        #print("===============================================")
        #print(PK)
        # if PK != 'kind_type.id':
        #   continue
        group_data = {}
        group_sample_rate = {}
        for K in equivalent_keys[PK]:
            group_data[K] = sampled_data[K]
            group_sample_rate[K] = sample_rate[K]
        if bucket_method == "greedy":
            _, optimal_bucket = greedy_bucketize(group_data, sample_rate, n_bins, primary_keys, True)
        elif bucket_method == "sub_optimal":
            optimal_bucket = sub_optimal_bucketize(group_data, group_sample_rate, n_bins=n_bins[PK],
                                                      primary_keys=primary_keys, return_data=False)
        elif bucket_method == "naive":
            optimal_bucket = naive_bucketize(group_data, sample_rate, n_bins, primary_keys, False)
        elif bucket_method == "fixed_start_key":
            start_key = get_start_key(list(group_data.keys()), table_key_count_var, primary_keys)
            print(start_key)
            optimal_bucket = fixed_start_key_bucketize(start_key, group_data, group_sample_rate, n_bins=n_bins[PK],
                                                      primary_keys=primary_keys, return_data=False)
        else:
            assert False, f"unrecognized bucketization method: {bucket_method}"

        optimal_buckets[PK] = optimal_bucket
        for K in equivalent_keys[PK]:
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
        print(key, len(all_bins[key]))

    ground_truth_factors_no_filter = get_ground_truth_no_filter(equivalent_keys, data, all_bins, table_lens, na_values,
                                                                model_folder)

    if save_bucket_bins:
        with open(model_folder + f"/imdb_buckets.pkl", "wb") as f:
            pickle.dump(optimal_buckets, f, pickle.HIGHEST_PROTOCOL)

    return schema, table_buckets, ground_truth_factors_no_filter


