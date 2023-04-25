import numpy as np
from multiprocessing import Pool
from Join_scheme.factor import Factor
from Sampling.get_query_binned_cards import get_binned_sqls, exec_sql


def sample_on_the_fly(sql, table_buckets, tables_alias, ground_truth_factors_no_filter, sampling_percentage,
                      equivalent_keys, db_conn_kwargs):
    alltabs, allcols, allsqls = get_binned_sqls(sql, equivalent_keys, sampling_percentage)
    par_args = []
    for sql in allsqls:
        par_args.append((sql, db_conn_kwargs))

    with Pool(processes=8) as pool:
        res = pool.starmap(exec_sql, par_args)

    conditional_factors = dict()
    table_pdfs = dict()
    filter_size = dict()

    for i, alias in enumerate(alltabs):
        cards = res[i][0]
        if cards is None:
            continue
        column = allcols[i]
        alias = alias[0]
        key = tables_alias[alias] + "." + column
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
            pdfs = ground_truth_factors_no_filter[tables_alias[alias]].pdfs[key]
        else:
            pdfs /= table_len
        if alias not in table_pdfs:
            table_pdfs[alias] = dict()
            filter_size[alias] = table_len
        table_pdfs[alias][key] = pdfs

    for alias in tables_alias:
        if alias in table_pdfs:
            table_len = min(ground_truth_factors_no_filter[tables_alias[alias]].table_len,
                            filter_size[alias]/(sampling_percentage/100))
            na_values = ground_truth_factors_no_filter[tables_alias[alias]].na_values
            conditional_factors[alias] = Factor(tables_alias[alias], table_len, list(table_pdfs[alias].keys()),
                                                table_pdfs[alias], na_values=na_values)
        else:
            #ground-truth distribution
            conditional_factors[alias] = ground_truth_factors_no_filter[tables_alias[alias]]
    return conditional_factors

