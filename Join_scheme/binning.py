import numpy as np
import copy


class Bucket:
    """
    The class of bucketization of a key attribute
    """

    def __init__(self, name, bins=[], bin_modes=[], bin_vars=[], bin_means=[], rest_bins_remaining=None):
        self.name = name
        self.bins = bins
        self.bin_modes = bin_modes
        self.bin_vars = bin_vars
        self.bin_means = bin_means
        self.rest_bins_remaining = rest_bins_remaining
        if len(bins) != 0:
            assert len(bins) == len(bin_modes)


class Table_bucket:
    """
    The class of bucketization for all key attributes in a table.
    Supporting more than three dimensional bin modes requires simplifying the causal structure, which is left as a
    future work.
    """

    def __init__(self, table_name, id_attributes, bin_sizes):
        self.table_name = table_name
        self.id_attributes = id_attributes
        self.bin_sizes = bin_sizes
        self.oned_bin_modes = dict()
        self.twod_bin_modes = dict()


class Bucket_group:
    """
    The class of bucketization for a group of equivalent join keys
    """

    def __init__(self, buckets, start_key, sample_rate, bins=None, primary_keys=[]):
        self.buckets = buckets
        self.start_key = start_key
        self.sample_rate = sample_rate
        self.bins = bins
        self.primary_keys = primary_keys

    def bucketize(self, data):
        """
        Discretize data based on the bucket
        """
        res = dict()
        seen_remain_key = np.array([])
        cumulative_bin = copy.deepcopy(self.buckets[self.start_key].bins)
        start_means = np.asarray(self.buckets[self.start_key].bin_means)

        for key in data:
            if key in self.primary_keys:
                continue
            res[key] = copy.deepcopy(data[key])
            if key != self.start_key:
                unique_remain = np.setdiff1d(self.buckets[key].rest_bins_remaining, seen_remain_key)
                assert sum([np.sum(np.isin(unique_remain, b) == 1) for b in cumulative_bin]) == 0

                if len(unique_remain) != 0:
                    remaining_data = data[key][np.isin(data[key], unique_remain)]
                    unique_remain, count_remain = np.unique(remaining_data, return_counts=True)
                    unique_counts = np.unique(count_remain)
                    for u in unique_counts:
                        temp_idx = np.searchsorted(start_means, u)
                        if temp_idx == len(cumulative_bin):
                            idx = -1
                            if u > self.buckets[key].bin_modes[-1]:
                                self.buckets[key].bin_modes[-1] = u
                        elif temp_idx == 0:
                            idx = 0
                        else:
                            if (u - start_means[temp_idx - 1]) >= (start_means[temp_idx] - u):
                                idx = temp_idx - 1
                            else:
                                idx = temp_idx
                        temp_unique = unique_remain[count_remain == u]
                        cumulative_bin[idx] = np.concatenate((cumulative_bin[idx], temp_unique))
                        seen_remain_key = np.concatenate((seen_remain_key, temp_unique))
                        if u > self.buckets[key].bin_modes[idx]:
                            self.buckets[key].bin_modes[idx] = u
            res[key] = copy.deepcopy(data[key])
            count = 0
            for i, b in enumerate(cumulative_bin):
                count += len(data[key][np.isin(data[key], b)])
                res[key][np.isin(data[key], b)] = i

            if self.sample_rate[key] < 1.0:
                bin_modes = self.buckets[key].bin_modes
                bin_modes[bin_modes != 1] = bin_modes[bin_modes != 1] / self.sample_rate[key]
                self.buckets[key].bin_modes = bin_modes

        self.bins = cumulative_bin

        for key in data:
            if key in self.primary_keys:
                res[key] = self.bucketize_PK(data[key])
                self.buckets[key] = Bucket(key)
        return res

    def bucketize_PK(self, data):
        res = copy.deepcopy(data)
        remaining_data = np.unique(data)
        for i, b in enumerate(self.bins):
            res[np.isin(data, b)] = i
            remaining_data = np.setdiff1d(remaining_data, b)
        if len(remaining_data) != 0:
            self.bins.append(list(remaining_data))
            for key in self.buckets:
                if key not in self.primary_keys:
                    self.buckets[key].bin_modes = np.append(self.buckets[key].bin_modes, 0)
        res[np.isin(data, remaining_data)] = len(self.bins)
        return res


def identify_key_values(schema):
    """
    identify all the key attributes from the schema of a DB, currently we assume all possible joins are known
    It is also easy to support unseen joins, which we left as a future work.
    :param schema: the schema of a DB
    :return: a dict of all keys, {table: [keys]};
             a dict of set, each indicating which keys on different tables are considered the same key.
    """
    all_keys = set()
    equivalent_keys = dict()
    for i, join in enumerate(schema.relationships):
        keys = join.identifier.split(" = ")
        all_keys.add(keys[0])
        all_keys.add(keys[1])
        seen = False
        for k in equivalent_keys:
            if keys[0] in equivalent_keys[k]:
                equivalent_keys[k].add(keys[1])
                seen = True
                break
            elif keys[1] in equivalent_keys[k]:
                equivalent_keys[k].add(keys[0])
                seen = True
                break
        if not seen:
            # set the keys[-1] as the identifier of this equivalent join key group for convenience.
            equivalent_keys[keys[-1]] = set(keys)

    assert len(all_keys) == sum([len(equivalent_keys[k]) for k in equivalent_keys])
    return all_keys, equivalent_keys


def equal_freq_binning(name, data, n_bins, data_len):
    uniques, counts = data
    if len(uniques) <= n_bins:
        bins = []
        bin_modes = []
        bin_vars = []
        bin_means = []

        for i, uni in enumerate(uniques):
            bins.append([uni])
            bin_modes.append(counts[i])
            bin_vars.append(0)
            bin_means.append(counts[i])
        return Bucket(name, bins, bin_modes, bin_vars, bin_means)

    unique_counts, count_counts = np.unique(counts, return_counts=True)
    idx = np.argsort(unique_counts)
    unique_counts = unique_counts[idx]
    count_counts = count_counts[idx]

    bins = []
    bin_modes = []
    bin_vars = []
    bin_means = []

    bin_freq = data_len / n_bins
    cur_freq = 0
    cur_bin = []
    cur_bin_count = []
    for i, uni_c in enumerate(unique_counts):
        cur_freq += count_counts[i] * uni_c
        cur_bin.append(uniques[np.where(counts == uni_c)[0]])
        cur_bin_count.extend([uni_c] * count_counts[i])
        if (cur_freq > bin_freq) or (i == (len(unique_counts) - 1)):
            bins.append(np.concatenate(cur_bin))
            cur_bin_count = np.asarray(cur_bin_count)
            bin_modes.append(uni_c)
            bin_means.append(np.mean(cur_bin_count))
            bin_vars.append(np.var(cur_bin_count))
            cur_freq = 0
            cur_bin = []
            cur_bin_count = []
    assert len(uniques) == sum([len(b) for b in bins]), f"some unique values missed or duplicated"
    return Bucket(name, bins, bin_modes, bin_vars, bin_means)


def compute_variance_score(buckets):
    """
    compute the variance of products of random variables
    """
    all_mean = np.asarray([buckets[k].bin_means for k in buckets])
    all_var = np.asarray([buckets[k].bin_vars for k in buckets])
    return np.sum(np.prod(all_var + all_mean ** 2, axis=0) - np.prod(all_mean, axis=0) ** 2)


def sub_optimal_bucketize(data, sample_rate, n_bins=30, primary_keys=[]):
    """
    Perform sub-optimal bucketization on a group of equivalent join keys.
    :param data: a dict of (potentially sampled) table data of the keys
                 the keys of this dict are one group of equivalent join keys
    :param sample_rate: the sampling rate the data, could be all 1 if no sampling is performed
    :param n_bins: how many bins can we allocate
    :param primary_keys: the primary keys in the equivalent group since we don't need to bucketize PK.
    :return: new data, where the keys are bucketized
             the mode of each bucket
    """
    unique_values = dict()
    for key in data:
        if key not in primary_keys:
            unique_values[key] = np.unique(data[key], return_counts=True)

    best_variance_score = np.infty
    best_bin_len = 0
    best_start_key = None
    best_buckets = None
    for start_key in data:
        if start_key in primary_keys:
            continue
        start_bucket = equal_freq_binning(start_key, unique_values[start_key], n_bins, len(data[start_key]))
        rest_buckets = dict()
        for key in data:
            if key == start_key or key in primary_keys:
                continue
            uniques = unique_values[key][0]
            counts = unique_values[key][1]
            rest_buckets[key] = Bucket(key, [], [0] * len(start_bucket.bins), [0] * len(start_bucket.bins),
                                       [0] * len(start_bucket.bins), uniques)
            for i, bin in enumerate(start_bucket.bins):
                idx = np.where(np.isin(uniques, bin) == 1)[0]
                if len(idx) != 0:
                    bin_count = counts[idx]
                    unique_bin_keys = uniques[idx]
                    # unique_bin_count = np.unique(bin_count)
                    # bin_count = np.concatenate([counts[counts == j] for j in unique_bin_count])
                    # unique_bin_keys = np.concatenate([uniques[counts == j] for j in unique_bin_count])
                    rest_buckets[key].rest_bins_remaining = np.setdiff1d(rest_buckets[key].rest_bins_remaining,
                                                                         unique_bin_keys)
                    rest_buckets[key].bin_modes[i] = np.max(bin_count)
                    rest_buckets[key].bin_vars[i] = np.var(bin_count)
                    rest_buckets[key].bin_means[i] = np.mean(bin_count)

        rest_buckets[start_key] = start_bucket
        var_score = compute_variance_score(rest_buckets)
        if len(start_bucket.bins) > best_bin_len:
            best_variance_score = var_score
            best_start_key = start_key
            best_buckets = rest_buckets
            best_bin_len = len(start_bucket.bins)
        elif len(start_bucket.bins) >= best_bin_len * 0.9 and var_score < best_variance_score:
            best_variance_score = var_score
            best_start_key = start_key
            best_buckets = rest_buckets
            best_bin_len = len(start_bucket.bins)

    best_buckets = Bucket_group(best_buckets, best_start_key, sample_rate, primary_keys=primary_keys)
    new_data = best_buckets.bucketize(data)
    return new_data, best_buckets


def fixed_start_key_bucketize(start_key, data, sample_rate, n_bins=30, primary_keys=[]):
    """
    Perform sub-optimal bucketization on a group of equivalent join keys based on the pre-defined start_key.
    :param data: a dict of (potentially sampled) table data of the keys
                 the keys of this dict are one group of equivalent join keys
    :param sample_rate: the sampling rate the data, could be all 1 if no sampling is performed
    :param n_bins: how many bins can we allocate
    :param primary_keys: the primary keys in the equivalent group since we don't need to bucketize PK.
    :return: new data, where the keys are bucketized
             the mode of each bucket
    """
    unique_values = dict()
    for key in data:
        if key not in primary_keys:
            unique_values[key] = np.unique(data[key], return_counts=True)

    start_bucket = equal_freq_binning(start_key, unique_values[start_key], n_bins, len(data[start_key]))
    rest_buckets = dict()
    for key in data:
        if key == start_key or key in primary_keys:
            continue
        uniques = unique_values[key][0]
        counts = unique_values[key][1]
        rest_buckets[key] = Bucket(key, [], [0] * len(start_bucket.bins), [0] * len(start_bucket.bins),
                                   [0] * len(start_bucket.bins), uniques)
        for i, bin in enumerate(start_bucket.bins):
            idx = np.where(np.isin(uniques, bin) == 1)[0]
            if len(idx) != 0:
                bin_count = counts[idx]
                unique_bin_keys = uniques[idx]
                rest_buckets[key].rest_bins_remaining = np.setdiff1d(rest_buckets[key].rest_bins_remaining,
                                                                     unique_bin_keys)
                rest_buckets[key].bin_modes[i] = np.max(bin_count)
                rest_buckets[key].bin_means[i] = np.mean(bin_count)

    best_buckets = Bucket_group(rest_buckets, start_key, sample_rate, primary_keys=primary_keys)
    new_data = best_buckets.bucketize(data, n_bins)
    return new_data, best_buckets


