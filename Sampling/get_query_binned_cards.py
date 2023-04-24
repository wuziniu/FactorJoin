import time
import glob
import os
import pickle
from multiprocessing import Pool
import psycopg2 as pg
from Sampling.utils.parse_sql import parse_sql
import collections.abc
#hyper needs the four following aliases to be done manually.
collections.Iterable = collections.abc.Iterable
collections.Mapping = collections.abc.Mapping
collections.MutableSet = collections.abc.MutableSet
collections.MutableMapping = collections.abc.MutableMapping


CACHE_CARD_TYPES = ["actual"]

FILTER_TMP = """AND {COL} IN ({VALS})"""
COL_TEMPLATE = "{COL}_bin"
GB_TMP = "SELECT {COL}, COUNT(*) FROM {TABLE} AS {ALIAS} WHERE {WHERE} GROUP BY {COL}"


def get_binned_sqls(sql, equivalent_keys, sampling_percentage=1.0):
    '''
    updates qrep's fields with the needed cardinality estimates, and returns
    the qrep.
    '''
    #with open(args.equivalent_keys_dir, "rb") as f:
     #   equivalent_keys = pickle.load(f)

    table_cols = {}
    binids = {}

    for t, cols in equivalent_keys.items():
        for col in cols:
            tname = col[0:col.find(".")]
            colname = col[col.find(".")+1:]
            if tname not in table_cols:
                table_cols[tname] = set([colname])
            else:
                table_cols[tname].add(colname)

            if col not in binids:
                binids[col] = t

    tables, where_clause = parse_sql(sql)

    alltabs = []
    allcols = []
    allsqls = []

    for table in tables:
        table_alias = tables[table]
        if table_alias is None:
            table_alias = table
        where_clause = " AND ".join(where_clause[table_alias])

        # will need to loop over each potential bin value in these columns
        if table not in table_cols:
            print("MISSING!!")
            print(table)
            continue

        if sampling_percentage is not None:
            sample_tname = table + "_ss" + str(sampling_percentage)
            sample_tname = sample_tname.replace(".", "d")
        else:
            sample_tname = table

        curtcols = [c for c in table_cols[table]]

        for curcol in curtcols:
            # find the newly created column for this
            newcolname = COL_TEMPLATE.format(COL=curcol)
            gsql = GB_TMP.format(COL=newcolname,
                                 TABLE=sample_tname,
                                 ALIAS=table_alias,
                                 WHERE=where_clause)

            if where_clause.strip() == "":
                gsql = gsql.replace("WHERE", "")

            print(gsql)
            alltabs.append((table_alias, ))
            allcols.append(curcol)
            allsqls.append(gsql)

    return alltabs, allcols, allsqls


def exec_sql(sql, db_conn_kwargs):
    start = time.time()
    con = pg.connect(db_conn_kwargs)
    cursor = con.cursor()
    cursor.execute(sql)
    res = cursor.fetchall()

    exec_time = time.time()-start
    return res, exec_time


def get_query_binned_cards(query_dir, db_conn_kwargs, equivalent_keys, sampling_percentage, save_dir="checkpoints/"):
    files = list(glob.glob(query_dir + "/*"))
    files.sort()

    for i, file in enumerate(files):
        if file.endswith(".sql") and file != "schema.sql" and file != "fkindexes.sql":
            with open(file, "r") as f:
                sql = f.read()
            query_name = file.split("/")[-1].split(".sql")[0]
            alltabs, allcols, allsqls = get_binned_sqls(sql, equivalent_keys, sampling_percentage)

            par_args = []
            for sql in allsqls:
                par_args.append((sql, db_conn_kwargs))

            with Pool(processes=8) as pool:
                res = pool.starmap(exec_sql, par_args)

            times = [r[1] for r in res]
            print(max(times), min(times))

            new_dir = os.path.join(save_dir, f"binned_cards_{sampling_percentage}")
            new_file = os.path.join(new_dir, f"{query_name}.pkl")
            os.makedirs(new_dir, exist_ok=True)

            data = {}
            data["all_aliases"] = alltabs
            data["all_columns"] = allcols
            data["all_sqls"] = allsqls
            data["results"] = res

            with open(new_file, "wb") as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
