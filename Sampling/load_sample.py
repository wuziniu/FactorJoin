import pickle5 as pickle
import numpy as np
import os
from Join_scheme.bound import Factor


def load_sample_imdb(table_buckets, tables_alias, query_file_orders, SPERCENTAGE=1.0,
                     qdir="/home/ubuntu/data_CE/saved_models/binned_cards/{}/job/all_job/"):
    qdir = qdir.format(SPERCENTAGE)
    all_sample_factors = []
    for fn in query_file_orders:
        conditional_factors = load_sample_imdb_one_query(table_buckets, tables_alias, fn, SPERCENTAGE, qdir)
        all_sample_factors.append(conditional_factors)
    return all_sample_factors


def load_sample_imdb_one_query(table_buckets, tables_alias, query_file_name, SPERCENTAGE=1.0,
                     qdir="/home/ubuntu/data_CE/saved_models/binned_cards/{}/job/all_job/",):
    qdir = qdir.format(SPERCENTAGE)
    fpath = os.path.join(qdir, query_file_name)
    with open(fpath, "rb") as f:
        data = pickle.load(f)

    conditional_factors = dict()
    table_pdfs = dict()
    table_lens = dict()
    for i, alias in enumerate(data["all_aliases"]):
        column = data["all_columns"][i]
        key = tables_alias[alias] + "." + column
        cards = data["results"][i][0]
        n_bins = table_buckets[tables_alias[alias]].bin_sizes[key]
        pdfs = np.zeros(n_bins)
        for (j, val) in cards:
            if j is None:
                j = 0
            pdfs[j] += val
        table_len = np.sum(pdfs)
        pdfs /= table_len
        if alias not in table_pdfs:
            table_pdfs[alias] = dict()
            table_lens[alias] = table_len
        else:
            assert table_len == table_lens[alias]
        table_pdfs[alias][key] = pdfs

    for alias in table_pdfs:
        conditional_factors[alias] = Factor(tables_alias[alias], table_lens[alias]/(SPERCENTAGE*100),
                                            list(table_pdfs[alias].keys()), table_pdfs[alias])
    return conditional_factors


