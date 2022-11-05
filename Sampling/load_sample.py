import pickle
import numpy as np
import os
from Join_scheme.binning import apply_binning_to_data_value_count


class Factor:
    """
    This the class defines a multidimensional conditional probability.
    """
    def __init__(self, table, table_len, variables, pdfs, equivalent_variables=[], na_values=None):
        self.table = table
        self.table_len = table_len
        self.variables = variables
        self.equivalent_variables = equivalent_variables
        self.pdfs = pdfs
        self.cardinalities = dict()
        for i, var in enumerate(self.variables):
            self.cardinalities[var] = pdfs.shape[i]
            if len(equivalent_variables) != 0:
                self.cardinalities[equivalent_variables[i]] = pdfs.shape[i]
        self.na_values = na_values  # the percentage of data, which is not nan, so the variable name is misleading.



def load_sample_imdb(table_buckets, tables_alias, query_file_orders, join_keys, table_key_equivalent_group,
                     SPERCENTAGE=1.0, qdir="/home/ubuntu/data_CE/saved_models/binned_cards/{}/job/all_job/"):
    qdir = qdir.format(SPERCENTAGE)
    all_sample_factors = []
    for fn in query_file_orders:
        conditional_factors = load_sample_imdb_one_query(table_buckets, tables_alias, fn, join_keys,
                                                         table_key_equivalent_group, SPERCENTAGE, qdir)
        all_sample_factors.append(conditional_factors)
    return all_sample_factors


def load_sample_imdb_one_query(table_buckets, tables_alias, query_file_name, join_keys, table_key_equivalent_group,
                               SPERCENTAGE=1.0, qdir="/home/ubuntu/data_CE/saved_models/binned_cards/{}/job/all_job/"):
    qdir = qdir.format(SPERCENTAGE)
    fpath = os.path.join(qdir, query_file_name)
    with open(fpath, "rb") as f:
        data = pickle.load(f)

    conditional_factors = dict()
    table_pdfs = dict()
    filter_size = dict()
    for i, alias in enumerate(data["all_aliases"]):
        column = data["all_columns"][i]
        alias = alias[0]
        key = tables_alias[alias] + "." + column
        cards = data["results"][i][0]
        n_bins = table_buckets[tables_alias[alias]].bin_sizes[key]
        pdfs = np.zeros(n_bins)
        for (j, val) in cards:
            if j is None:
                j = 0
            pdfs[j] += val
        table_len = np.sum(pdfs)
        if table_len == 0:
            # no sample satisfy the filter, set it with a small value
            table_len = 1
            pdfs = table_key_equivalent_group[tables_alias[alias]].pdfs[key]
        else:
            pdfs /= table_len
        if alias not in table_pdfs:
            table_pdfs[alias] = dict()
            filter_size[alias] = table_len
        table_pdfs[alias][key] = pdfs

    for alias in tables_alias:
        if alias in table_pdfs:
            table_len = min(table_key_equivalent_group[tables_alias[alias]].table_len,
                            filter_size[alias]/(SPERCENTAGE/100))
            na_values = table_key_equivalent_group[tables_alias[alias]].na_values
            conditional_factors[alias] = Factor(tables_alias[alias], table_len, list(table_pdfs[alias].keys()),
                                                table_pdfs[alias], na_values=na_values)
        else:
            #TODO: ground-truth distribution
            conditional_factors[alias] = table_key_equivalent_group[tables_alias[alias]]
    return conditional_factors


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
                                    all_factor_pdfs[table], na_values=na_values[table])
    return all_factors



