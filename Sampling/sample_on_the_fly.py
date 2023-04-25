import psycopg2 as pg
from Sampling.utils.parse_sql import parse_sql
from multiprocessing import Pool
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

    return None
