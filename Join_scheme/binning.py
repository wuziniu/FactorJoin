import numpy as np
import copy
from scipy import stats
import jenkspy


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
        self.bin_width = [0] * len(bin_means)
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
        res[np.isin(data, remaining_data)] = -1
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


def equal_freq_binning(name, data, n_bins, data_len, return_bucket=True, return_bin_means=False):
    uniques, counts = data
    unique_counts, count_counts = np.unique(counts, return_counts=True)
    idx = np.argsort(unique_counts)
    unique_counts = unique_counts[idx]
    count_counts = count_counts[idx]

    bins = []
    bin_modes = []
    bin_vars = []
    bin_means = []
    bin_sizes = []

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
            bin_sizes.append(len(cur_bin_count))
            cur_freq = 0
            cur_bin = []
            cur_bin_count = []
    assert len(uniques) == sum([len(b) for b in bins]), f"some unique values missed or duplicated"
    if return_bucket:
        bucket = Bucket(name, bins, bin_modes, bin_vars, bin_means)
        if return_bin_means:
            bucket.bin_width = bin_sizes
        return bucket
    else:
        return bins, bin_means


def apply_binning_to_data(bins, bin_means, data, start_key_data, n_bins, uniques, counts):
    # apply one greedy binning step based on existing bins
    unique_remains = np.setdiff1d(uniques, np.concatenate(bins))
    if len(unique_remains) != 0:
        remaining_data = data[np.isin(data, unique_remains)]
        unique_remain, count_remain = np.unique(remaining_data, return_counts=True)
        unique_counts = np.unique(count_remain)
        for u in unique_counts:
            temp_idx = np.searchsorted(bin_means, u)
            if temp_idx == len(bins):
                idx = -1
            elif temp_idx == 0:
                idx = 0
            else:
                if (u - bin_means[temp_idx - 1]) >= (bin_means[temp_idx] - u):
                    idx = temp_idx - 1
                else:
                    idx = temp_idx
            temp_unique = unique_remain[count_remain == u]
            bins[idx] = np.concatenate((bins[idx], temp_unique))  # modifying bins in place

    bin_vars = []
    temp_bin_means = []
    for i, bin in enumerate(bins):
        idx = np.where(np.isin(uniques, bin) == 1)[0]
        if len(idx) != 0:
            bin_vars.append(np.var(counts[idx]))
            temp_bin_means.append(np.mean(counts[idx]))
        else:
            bin_vars.append(0)
            temp_bin_means.append(1)

    assign_nbins = assign_bins_by_var(n_bins, bin_vars, temp_bin_means)

    new_bins = []
    new_bin_means = []
    for i, bin in enumerate(bins):
        if assign_nbins[i] == 0:
            new_bins.append(bin)
            new_bin_means.append(bin_means[i])
        else:
            curr_bin_data = data[np.isin(data, bin)]
            curr_start_key_data = start_key_data[np.isin(start_key_data, bin)]
            curr_bins, curr_bin_means = divide_bin(bin, curr_bin_data, assign_nbins[i] + 1, curr_start_key_data)
            new_bins.extend(curr_bins)
            new_bin_means.extend(curr_bin_means)

    return new_bins, new_bin_means


def assign_bins_by_var(n_bins, bin_vars, bin_means, small_threshold=0.2, large_threshold=2):
    assign_nbins = np.zeros(len(bin_vars))
    remaining_nbins = n_bins
    idx = np.argsort(bin_vars)[::-1]
    if bin_vars[idx[0]] / bin_means[idx[0]] <= small_threshold:
        return assign_nbins

    while remaining_nbins > 0:
        for i in range(len(assign_nbins)):
            normalized_var = bin_vars[idx[i]] / bin_means[idx[i]]
            if normalized_var >= large_threshold:
                assign_nbins[i] += min(remaining_nbins, 2)
                remaining_nbins -= min(remaining_nbins, 2)
            elif normalized_var > small_threshold:
                assign_nbins[i] += 1
                remaining_nbins -= 1
            if remaining_nbins <= 0:
                break
    return assign_nbins


def divide_bin(bin, curr_bin_data, n_bins, start_key_data):
    # divide one bin into multiple bins to minimize the variance of curr_bin_data
    uniques, counts = np.unique(curr_bin_data, return_counts=True)
    if len(uniques) == 0:
        return [], []

    if len(uniques) <= n_bins:
        new_bins = []
        bin_means = []
        remaining_values = bin

        for i, uni in enumerate(uniques):
            new_bins.append([uni])
            remaining_values = np.setdiff1d(remaining_values, np.asarray([uni]))

        # randomly assign the remaining index to some bins
        if len(remaining_values) > 0:
            assign_idx = np.random.randint(0, len(new_bins), size=len(remaining_values))
            for i in range(len(new_bins)):
                new_bins[i].extend(list(remaining_values[assign_idx == i]))
                new_bins[i] = np.asarray(new_bins[i])

        for bin in new_bins:
            curr_bin_data = start_key_data[np.isin(start_key_data, bin)]
            if len(curr_bin_data) == 0:
                bin_means.append(0)
            else:
                _, count = np.unique(curr_bin_data, return_counts=True)
                bin_means.append(np.mean(count))
        return new_bins, bin_means

    idx = np.argsort(counts)
    counts = counts[idx]
    uniques = uniques[idx]

    # Natural breaks optimization using Fisher-Jenks Algorithms
    breaks = jenkspy.jenks_breaks(counts, nb_class=n_bins)
    breaks[-1] += 0.01
    new_bins = []
    bin_means = []
    remaining_values = np.asarray(bin)
    for i in range(1, len(breaks)):
        idx = np.where((breaks[i - 1] <= counts) & (counts < breaks[i]))[0]
        new_bins.append(uniques[idx])
        remaining_values = np.setdiff1d(remaining_values, uniques[idx])

    if len(remaining_values) > 0:
        assign_idx = np.random.randint(0, len(new_bins), size=len(remaining_values))
        for i in range(len(new_bins)):
            new_bins[i] = np.concatenate((new_bins[i], remaining_values[assign_idx == i]))

    for bin in new_bins:
        curr_bin_data = start_key_data[np.isin(start_key_data, bin)]
        if len(curr_bin_data) == 0:
            bin_means.append(0)
        else:
            _, count = np.unique(curr_bin_data, return_counts=True)
            bin_means.append(np.mean(count))
    return new_bins, bin_means


def compute_variance_score(buckets):
    """
    compute the variance of products of random variables
    """
    all_mean = np.asarray([buckets[k].bin_means for k in buckets])
    all_var = np.asarray([buckets[k].bin_vars for k in buckets])
    return np.sum(np.prod(all_var + all_mean ** 2, axis=0) - np.prod(all_mean, axis=0) ** 2)


def greedy_bucketize(data, sample_rate, n_bins=30, primary_keys=[], return_data=False):
    """
    Perform sub-optimal bucketization on a group of equivalent join keys.
    A greedy algorithm that assigns half of the bins to one key at a time.
    :param data: a dict of (potentially sampled) table data of the keys
                 the keys of this dict are one group of equivalent join keys
    :param sample_rate: the sampling rate the data, could be all 1 if no sampling is performed
    :param n_bins: how many bins can we allocate
    :param primary_keys: the primary keys in the equivalent group since we don't need to bucketize PK.
    :return: new data, where the keys are bucketized
             the mode of each bucket
    """
    unique_values = dict()
    key_orders = []
    data_lens = []
    curr_pk = []
    for key in data:
        if key not in primary_keys:
            unique_values[key] = np.unique(data[key], return_counts=True)
            key_orders.append(key)
            data_lens.append(len(data[key]))
        else:
            curr_pk.append(key)
    key_orders = [key_orders[i] for i in np.argsort(data_lens)[::-1]]
    remaining_bins = n_bins
    start_key = key_orders[0]
    curr_bins = None
    curr_bin_means = None
    for key in key_orders:
        if key == key_orders[-1]:
            # least key value use up all remaining bins, otherwise use half of it
            assign_bins = remaining_bins
        else:
            assign_bins = remaining_bins // 2
        if key == start_key:
            curr_bins, curr_bin_means = equal_freq_binning(key, unique_values[key], assign_bins, len(data[key]), False)
        else:
            curr_bins, curr_bin_means = apply_binning_to_data(curr_bins, curr_bin_means, data[key],
                                                              data[start_key], assign_bins,
                                                              unique_values[key][0], unique_values[key][1])
        remaining_bins = n_bins - len(curr_bins)

    new_data, best_buckets, curr_bins = bin_all_data_with_existing_binning(curr_bins, data, sample_rate, curr_pk,
                                                                           return_data)
    best_buckets = Bucket_group(best_buckets, start_key, sample_rate, curr_bins, primary_keys=curr_pk)
    return new_data, best_buckets


def sub_optimal_bucketize(data, sample_rate, n_bins=30, primary_keys=[], return_bin_means=False):
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
    all_bin_width = dict()
    for start_key in data:
        if start_key in primary_keys:
            continue
        start_bucket = equal_freq_binning(start_key, unique_values[start_key], n_bins, len(data[start_key]), 
                                          True, return_bin_means)
        
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
                    rest_buckets[key].bin_width[i] = len(bin_count)
        rest_buckets[start_key] = start_bucket
        var_score = compute_variance_score(rest_buckets)
        if len(start_bucket.bins) >= best_bin_len * 1.1:
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


def naive_bucketize(data, sample_rate, n_bins=30, primary_keys=[], return_data=True):
    """
        Perform naive bucketization on a group of equivalent join keys (i.e. equal width binning)
        :param data: a dict of (potentially sampled) table data of the keys
                     the keys of this dict are one group of equivalent join keys
        :param sample_rate: the sampling rate the data, could be all 1 if no sampling is performed
        :param n_bins: how many bins can we allocate
        :param primary_keys: the primary keys in the equivalent group since we don't need to bucketize PK.
        :param return_data: return discretized data.
        :return: new data, where the keys are bucketized
                 the mode of each bucket
    """
    key_orders = []
    data_lens = []
    new_data = dict()
    for key in data:
        key_orders.append(key)
        data_lens.append(len(data[key]))
    key_orders = [key_orders[i] for i in np.argsort(data_lens)[::-1]]
    best_buckets = dict()
    start_key = key_orders[0]
    start_key_bin_mode = []
    data_start_key = np.sort(data[start_key])
    _, curr_bins = np.histogram(data_start_key, bins=n_bins)

    for key in key_orders:
        key_bin_mode = []
        data_key = np.sort(data[key])
        temp_data_key = copy.deepcopy(data[key])
        curr_bins[0] = min(data_key[0]-0.1, curr_bins[0])
        curr_bins[-1] = max(data_key[-1] + 0.1, curr_bins[-1])
        for i in range(len(curr_bins) - 1):
            start = curr_bins[i]
            end = curr_bins[i + 1]
            idx = np.where((data_key >= start) & (data_key < end))[0]
            if len(idx) == 0:
                bin_mode = 0
            else:
                bin_mode = stats.mode(data_key[idx]).count[0]
                temp_data_key[idx] = i
            key_bin_mode.append(bin_mode/sample_rate[key])
        best_buckets[key] = Bucket(key, [], key_bin_mode)
        new_data[key] = temp_data_key
    best_buckets = Bucket_group(best_buckets, start_key, sample_rate, curr_bins[:-1], primary_keys=primary_keys)
    if return_data:
        return new_data, best_buckets
    else:
        return best_buckets


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

    start_bucket = equal_freq_binning(start_key, unique_values[start_key], n_bins, len(data[start_key]), True)
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
    new_data = best_buckets.bucketize(data)
    return new_data, best_buckets


def bin_all_data_with_existing_binning(bins, data, sample_rate, curr_pk, return_data):
    buckets = dict()
    new_data = dict()
    if return_data:
        new_data = copy.deepcopy(data)

    for key in curr_pk:
        bin_modes = [1 for i in range(len(bins))]
        remaining_data = np.unique(data[key])
        for i, bin in enumerate(bins):
            if return_data:
                new_data[key][np.isin(data[key], bin)] = i
            remaining_data = np.setdiff1d(remaining_data, bin)
        if len(remaining_data) != 0:
            if return_data:
                # assigning all remaining key values to the first bin
                new_data[key][np.isin(data[key], remaining_data)] = 0
            bins[0] = np.concatenate((bins[0], remaining_data))
        buckets[key] = Bucket(key, bin_modes=bin_modes)

    for key in data:
        bin_modes = []
        for i, bin in enumerate(bins):
            curr_data = data[key][np.isin(data[key], bin)]
            if len(curr_data) == 0:
                bin_modes.append(0)
            else:
                bin_mode = stats.mode(curr_data).count[0]
                if bin_mode > 1:
                    bin_mode /= sample_rate[key]
                bin_modes.append(bin_mode)
                if return_data:
                    new_data[key][np.isin(data[key], bin)] = i
        buckets[key] = Bucket(key, bin_modes=bin_modes)

    return new_data, buckets, bins


def apply_binning_to_data_value_count(bins, data):
    res = np.zeros(len(bins))
    unique_remain = np.unique(data)
    for i, bin in enumerate(bins):
        res[i] = np.sum(np.isin(data, bin))
        unique_remain = np.setdiff1d(unique_remain, bin)

    res[0] += np.sum(np.isin(data, unique_remain))
    return res
