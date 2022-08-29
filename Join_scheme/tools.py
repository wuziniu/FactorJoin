import numpy as np
from Schemas.imdb.schema import gen_imdb_schema
from Join_scheme.binning import identify_key_values
from Join_scheme.join_graph import parse_query_all_join


def count_join_key_appearance(queries, equivalent_keys):
    """
    analyze the workload and count how many times each join key group appears
    """
    all_join_keys_stats = dict()
    total_num_appearance = 0
    for q in queries:
        res = parse_query_all_join(q)
        for table in res[-1]:
            for join_key in list(res[-1][table]):
                for PK in equivalent_keys:
                    if join_key in equivalent_keys[PK]:
                        total_num_appearance += 1
                        if PK in all_join_keys_stats:
                            all_join_keys_stats[PK] += 1
                        else:
                            all_join_keys_stats[PK] = 1
                        break
    return all_join_keys_stats, total_num_appearance


def get_n_bins_from_query(bin_size, data_path, query_file):
    """
    Derive the optimal number of bins to use for each join key group
    :param bin_size: average number of bins to use for each join key group
    :param data_path:
    :param query_file:
    :return:
    """
    schema = gen_imdb_schema(data_path)
    all_keys, equivalent_keys = identify_key_values(schema)
    n_bins = dict()
    if query_file is None:
        for key in equivalent_keys:
            n_bins[key] = bin_size
    else:
        with open(query_file, "r") as f:
            queries = f.readlines()
        all_join_keys_stats, total_num_appearance = count_join_key_appearance(queries, equivalent_keys)

        total_bins = bin_size * len(equivalent_keys)

    return n_bins
