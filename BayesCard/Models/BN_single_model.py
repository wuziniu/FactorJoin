import pomegranate
import time
import bz2
import pickle
import logging
import numpy as np
import pandas as pd

from BayesCard.Models.tools import categorical_qcut, discretize_series

logger = logging.getLogger(__name__)

class BN_Single():
    """
    Build a single Bayesian Network for a single table.
    Initialize with an appropriate table_name.
    """

    def __init__(self, table_name, null_values, method='Pome', debug=True):
        self.table_name = table_name
        self.null_values = null_values
        self.n_in_bin = dict()
        self.bin_width = dict()
        self.encoding = dict()
        self.mapping = dict()
        self.domain = dict()
        self.id_exist_null = dict()
        self.id_value_position = dict()
        self.max_value = dict()
        self.method = method
        self.model = None
        self.structure = None
        self.debug = debug

    def __str__(self):
        return f"bn{self.table_name}.{self.algorithm}-{self.max_parents}-{self.root}-{self.n_mcv}-{self.n_bins}"

    def build_discrete_table(self, data, id_attributes, n_mcv, n_bins, drop_na=True, ignore_cols=[]):
        """
        Discretize the entire table use bining (This is using histogram method for continuous data)
        ::Param:: table: original table
                  n_mcv: for categorical data we keep the top n most common values and bin the rest
                  n_bins: number of bins for histogram, larger n_bins will provide more accuracy but less efficiency
                  drop_na: if True, we drop all rows with nan in it
                  ignore_cols: drop the unnessary columns for example id attribute
        """
        table = data.copy()
        if drop_na:
            table = table.dropna()
        for col in table.columns:
            if col in ignore_cols:
                table = table.drop(col, axis=1)
            elif col in id_attributes:
                self.encoding[col] = np.sort(np.unique(table[col].values)).astype(int)
                if self.encoding[col][0] == -1:
                    self.id_exist_null[col] = True
                    self.id_value_position[col] = self.encoding[col][1:]
                else:
                    self.id_exist_null[col] = False
                    self.id_value_position[col] = self.encoding[col]
                self.mapping[col] = None
            else:
                table[col], self.n_in_bin[col], self.bin_width[col], self.encoding[col], self.mapping[col], \
                    self.domain[col] = discretize_series(
                    table[col],
                    n_mcv=n_mcv,
                    n_bins=n_bins,
                    is_continous=self.attr_type[col] == "continuous",
                    drop_na=not drop_na,
                    )
                self.max_value[col] = int(table[col].max()) + 1
        self.node_names = list(table.columns)
        return table

    def process_update_dataset(self, data, n_bins=30, drop_na=True, ignore_cols=[]):
        """
        This function can only be called when updating models.
        It works similar to build_discrete_table function.
        It featurizes the newly inserted data the same way as how the original data is processed.
        Parameters
        ----------
        data: input raw dataset of pd.dataframe

        Returns: discretized dataset in the same way the model is trained on
        -------
        """
        table = data.copy()
        if drop_na:
            table = table.dropna()
        for col in table.columns:
            if col in ignore_cols:
                table = table.drop(col, axis=1)
            else:
                f = 0
                if col in self.fanout_attr_inverse:
                    f = 2
                elif col in self.fanout_attr_positive:
                    f = 1
                table[col], self.n_in_bin_update[col], self.encoding_update[col], mapping\
                    = self.discretize_series_based_on_existing(
                    table[col],
                    col,
                    n_bins=n_bins,
                    is_continuous=self.attr_type[col] == "continuous",
                    drop_na=not drop_na,
                    fanout=f
                )
                self.max_value[col] = int(table[col].max()) + 1
                if self.mapping_update[col] and mapping:
                    self.mapping_update[col].update(mapping)
                    # sorted it by key
                    self.mapping_update[col] = {k: self.mapping_update[col][k] for k in sorted(self.mapping_update[col])}
        return table

    def discretize_series_based_on_existing(self, series, col, n_bins, is_continuous=False,
                                            drop_na=True, fanout=0):
        """
        Map every value to category, binning the small categories if there are more than n_mcv categories.
        Map intervals to categories for efficient model learning
        return:
        s: discretized series
        n_distinct: number of distinct values in a mapped category (could be empty)
        encoding: encode the original value to new category (will be empty for continous attribute)
        mapping: map the new category to pd.Interval (for continuous attribute only)
        """
        s = series.copy()
        encoding = self.encoding_update[col]
        mapping = dict()

        if is_continuous:
            assert self.mapping[col] is not None, f"column {col} is not previously recognized as continuous"

            old_mapping = self.mapping[col]
            # handling things that are out of range
            oof_left = False
            oof_right = False
            continuous_bins = [old_mapping[0].left] + [old_mapping[k].right for k in old_mapping]
            if s.min() < self.domain[col][0]:
                oof_left = True
                continuous_bins = [s.min()-0.0001] + continuous_bins
                val = -1
            else:
                val = max(list(old_mapping.keys())) + 1
            if s.max() > self.domain[col][1]:
                oof_right = True
                continuous_bins = continuous_bins + [s.max()]
            self.domain[col] = (min(self.domain[col][0], s.min()), max(self.domain[col][1], s.max()))
            temp = pd.cut(s, bins=continuous_bins, duplicates='drop')
            categ = dict()

            old_mapping_reversed = {old_mapping[v]: v for v in old_mapping}
            # map interval object to int category for efficient training
            for interval in sorted(list(temp.unique()), key=lambda x: x.left):
                if interval in old_mapping_reversed:
                    categ[interval] = old_mapping_reversed[interval]
                else:
                    categ[interval] = val
                    mapping[val] = interval
                    if val == -1:
                        val = max(list(old_mapping_reversed.values())) + 1

                if fanout != 0:
                    #update stored fanout values
                    curr_values = np.asarray(temp[temp == interval].index)
                    if val == -1:
                        if fanout == 1:
                            first_fanout = np.nanmean(curr_values)
                        elif fanout == 2:
                            curr_values[curr_values == 0] = 1
                            first_fanout = np.nanmean(1 / curr_values)
                        first_fanout_sum = len(curr_values)
                    elif val >= len(self.fanout_sum[col][val]):
                        if fanout == 1:
                            last_fanout = np.nanmean(curr_values)
                        elif fanout == 2:
                            curr_values[curr_values == 0] = 1
                            last_fanout = np.nanmean(1 / curr_values)
                        self.fanouts[col] = np.concatenate((self.fanouts[col], [last_fanout]))
                        self.fanout_sum[col] = np.concatenate((self.fanout_sum[col], [len(curr_values)]))
                    else:
                        prev_sum = self.fanout_sum[col][val] * self.fanouts[col][val]
                        self.fanout_sum[col][val] += len(curr_values)
                        if fanout == 1:
                            curr_sum = np.nansum(curr_values)
                        elif fanout == 2:
                            curr_values[curr_values == 0] = 1
                            curr_sum = np.nansum(1/curr_values)
                        self.fanouts[col][val] = (curr_sum + prev_sum) / self.fanout_sum[col][val]
                val += 1

            if oof_left and fanout != 0:
                self.fanouts[col] = np.concatenate(([first_fanout], self.fanouts[col]))
                self.fanout_sum[col] = np.concatenate(([first_fanout_sum], self.fanout_sum[col]))

            s = temp.cat.rename_categories(categ)

            if drop_na:
                s = s.cat.add_categories(int(val))
                s = s.fillna(val)  # Replace np.nan with some integer that is not in encoding
            return s, None, encoding, mapping


        # Remove trailing whitespace
        if s.dtype == 'object':
            s = s.str.strip()
        domains = list(s.unique())
        self.domain[col] = list(set(domains) | set(self.domain[col]))

        # map the original value to encoded value
        value_counts = s.value_counts()
        start_val = np.max(np.unique(np.asarray(list(encoding.values()))))+1
        max_val = start_val + n_bins
        val = start_val
        temp = series.copy()
        fanout_values = dict()
        fanout_sums = dict()
        n_distinct = dict()
        for i in value_counts.index:
            if i in encoding:
                temp[s == i] = encoding[i]
                if encoding[i] in self.n_in_bin[col]:
                    if encoding[i] in n_distinct:
                        n_distinct[encoding[i]][i] = value_counts[i]
                    else:
                        n_distinct[encoding[i]] = {i: value_counts[i]}
                if fanout != 0:
                    # adding fanout values
                    if fanout == 1:
                        curr_fanout_sums = i * value_counts[i]
                    elif fanout == 2:
                        if i == 0:
                            curr_fanout_sums = value_counts[i]
                        else:
                            curr_fanout_sums = value_counts[i]/i
                    if encoding[i] in fanout_values:
                        fanout_values[encoding[i]] += curr_fanout_sums
                        fanout_sums[encoding[i]] += value_counts[i]
                    else:
                        fanout_values[encoding[i]] = curr_fanout_sums
                        fanout_sums[encoding[i]] = value_counts[i]

            else:
                if val in n_distinct:
                    temp[s == i] = val
                    encoding[i] = val
                    n_distinct[val][i] = value_counts[i]
                else:
                    temp[s == i] = val
                    encoding[i] = val
                    n_distinct[val] = {i: value_counts[i]}
                if fanout != 0:
                    # adding fanout values
                    if fanout == 1:
                        curr_fanout_sums = i * value_counts[i]
                    elif fanout == 2:
                        if i == 0:
                            curr_fanout_sums = value_counts[i]
                        else:
                            curr_fanout_sums = value_counts[i] / i
                    if val in fanout_values:
                        fanout_values[val] += curr_fanout_sums
                        fanout_sums[val] += value_counts[i]
                    else:
                        fanout_values[val] = curr_fanout_sums
                        fanout_sums[val] += value_counts[i]

                val += 1
                if val >= max_val:
                    val = start_val

        del s
        if drop_na:
            # temp = temp.cat.add_categories(int(n_mcv+n_bins+1))
            temp = temp.fillna(max_val)  # Replace np.nan with some integer that is not in encoding

        n_distinct = self.update_n_distinct_fanout(n_distinct, fanout_values, fanout_sums, fanout, col)  #update the bin frequency

        return temp, n_distinct, encoding, None


    def update_n_distinct_fanout(self, n_distinct, fanout_values, fanout_sums, fanout, col):
        result = dict()
        # updating the old bin frequency in self.n_in_bin
        for enc in self.n_in_bin[col]:
            if enc in n_distinct:
                bin_freq = 0
                result[enc] = dict()
                for i in self.n_in_bin[col][enc]:
                    if i in n_distinct[enc]:
                        n_distinct[enc][i] += self.bin_width[col][enc] * self.n_in_bin[col][enc][i]
                    else:
                        n_distinct[enc][i] = self.bin_width[col][enc] * self.n_in_bin[col][enc][i]
                    bin_freq += n_distinct[enc][i]
                p_val = 0
                for i in n_distinct[enc]:
                    p = n_distinct[enc][i]/bin_freq
                    result[enc][i] = p
                    p_val += p
                assert np.isclose(p_val, 1), f"invalid probability distribution with sum {p_val}"

                if fanout != 0:
                    prev_sum = self.fanout_sum[col][enc] * self.fanouts[col][enc]
                    curr_sum = fanout_values[enc]
                    self.fanouts[col][enc] = (prev_sum+curr_sum) / (self.fanout_sum[col][enc]+fanout_sums[enc])
                    self.fanout_sum[col][enc] = self.fanout_sum[col][enc] + fanout_sums[enc]

            else:
                result[enc] = self.n_in_bin[col][enc]

        # adding the new bin frequency in n_distinct
        for enc in n_distinct:
            if enc not in self.n_in_bin[col]:
                result[enc] = dict()
                bin_freq = 0
                for i in n_distinct[enc]:
                    bin_freq += n_distinct[enc][i]
                p_val = 0
                for i in n_distinct[enc]:
                    p = n_distinct[enc][i] / bin_freq
                    result[enc][i] = p
                    p_val += p
                assert np.isclose(p_val, 1), f"invalid probability distribution with sum {p_val}"

                if fanout != 0:
                    curr_value = fanout_values[enc]/fanout_sums[enc]
                    self.fanouts[col] = np.concatenate((self.fanouts[col], [curr_value]))
                    self.fanout_sum[col] = np.concatenate((self.fanout_sum[col], [fanout_sums[enc]]))
            else:
                assert enc in result, f"invalid encoding {enc}"
        return result


    def is_numeric(self, val):
        if isinstance(val, int):
            return True
        if isinstance(val, float):
            return True

    def get_attr_type(self, dataset, id_attributes=[], threshold=3000):
        attr_type = dict()
        for col in dataset.columns:
            if col in id_attributes:
                attr_type[col] = "categorical"
                continue
            n_unique = dataset[col].nunique()
            if n_unique == 2:
                attr_type[col] = 'boolean'
            elif n_unique >= len(dataset)/20 or (self.is_numeric(dataset[col].iloc[0]) and n_unique > threshold):
                attr_type[col] = 'continuous'
            else:
                attr_type[col] = 'categorical'
        return attr_type

    def apply_encoding_to_value(self, value, col):
        """ Given the original value in the corresponding column and return its encoded value
            Note that every value of all col in encoded.
        """
        if col not in self.encoding:
            return None
        else:
            if type(value) == list:
                enc_value = []
                for val in value:
                    if val not in self.encoding[col]:
                        enc_value.append(None)
                    else:
                        enc_value.append(self.encoding[col][val])
                return enc_value
            else:
                if value not in self.encoding[col]:
                    return None
                else:
                    return self.encoding[col][value]

    def apply_ndistinct_to_value(self, enc_value, value, col):
        # return the number of distinct value in the bin
        if col not in self.n_in_bin:
            return 1
        else:
            if type(enc_value) != list:
                enc_value = [enc_value]
                value = [value]
            else:
                assert len(enc_value) == len(value), "incorrect number of values"
            n_distinct = []
            for i, enc_val in enumerate(enc_value):
                if enc_val not in self.n_in_bin[col]:
                    n_distinct.append(1)
                elif type(self.n_in_bin[col][enc_val]) == int:
                    n_distinct.append(1 / self.n_in_bin[col][enc_val])
                elif value[i] not in self.n_in_bin[col][enc_val]:
                    n_distinct.append(1)
                else:
                    n_distinct.append(self.n_in_bin[col][enc_val][value[i]])
            return np.asarray(n_distinct)

    def learn_model_structure(self, dataset, nrows=None, id_attributes=[], attr_type=None, rows_to_use=500000,
                              n_mcv=30, n_bins=60, ignore_cols=['id'], algorithm="greedy", drop_na=True, max_parents=2,
                              root=None, n_jobs=8, return_model=False, return_dataset=False, discretized=False):
        """ Build the Pomegranate model from data, including structure learning and paramter learning
            ::Param:: dataset: pandas.dataframe
                      attr_type: type of attributes (binary, discrete or continuous)
                      rows_to_use: subsample the number of rows to use to learn structure
                      n_mcv: for categorical data we keep the top n most common values and bin the rest
                      n_bins: number of bins for histogram, larger n_bins will provide more accuracy but less efficiency
            for other parameters, pomegranate gives a detailed explaination:
            https://pomegranate.readthedocs.io/en/latest/BayesianNetwork.html
        """
        if nrows is None:
            self.nrows = len(dataset)
        else:
            self.nrows = nrows
        self.algorithm = algorithm
        self.max_parents = max_parents
        self.n_mcv = n_mcv
        self.n_bins = n_bins
        self.root = root

        if attr_type is None:
            self.attr_type = self.get_attr_type(dataset, id_attributes)
        else:
            self.attr_type = attr_type
        t = time.time()
        if not discretized:
            discrete_table = self.build_discrete_table(dataset, id_attributes, n_mcv, n_bins, drop_na, ignore_cols)
            logger.info(f'Discretizing table takes {time.time() - t} secs')
            logger.info(f'Learning BN optimal structure from data with {self.nrows} rows and'
                        f' {len(self.node_names)} cols')
            print(f'Discretizing table takes {time.time() - t} secs')
        t = time.time()
        if len(discrete_table) <= rows_to_use:
            model = pomegranate.BayesianNetwork.from_samples(discrete_table,
                                                         algorithm=algorithm,
                                                         state_names=self.node_names,
                                                         max_parents=max_parents,
                                                         n_jobs=n_jobs,
                                                         root=self.root)
        else:
            model = pomegranate.BayesianNetwork.from_samples(discrete_table.sample(n=rows_to_use),
                                                         algorithm=algorithm,
                                                         state_names=self.node_names,
                                                         max_parents=max_parents,
                                                         n_jobs=n_jobs,
                                                         root=self.root)
        logger.info(f'Structure learning took {time.time() - t} secs.')
        print(f'Structure learning took {time.time() - t} secs.')

        self.structure = model.structure

        if return_model:
            if return_dataset:
                return model, discrete_table
            else:
                return model
        elif return_dataset:
            return discrete_table

        return None


    def save(self, path, compress=False):
        if compress:
            with bz2.BZ2File(path, 'wb') as f:
                pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
        else:
            with open(path, 'wb') as f:
                pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    def query(self, query):
        raise NotImplemented


def load_BN_single(path):
    """Load BN ensembles from pickle file"""
    with open(path, 'rb') as handle:
        bn = pickle.load(handle)
    return bn
