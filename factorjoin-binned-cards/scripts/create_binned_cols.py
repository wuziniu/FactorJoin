import sys
sys.path.append(".")
import argparse
import psycopg2 as pg
from utils.utils import *
import pdb
import random
from multiprocessing import Pool, cpu_count
import json
import pickle
from collections import defaultdict
import pdb

SEL_TEMPLATE = "SELECT {COLS} FROM {TABLE} WHERE random() < {FRAC}"
CREATE_TEMPLATE = "CREATE TABLE IF NOT EXISTS {TABLE_NAME} AS {SEL_SQL}"
DROP_TEMPLATE = "DROP TABLE IF EXISTS {TABLE_NAME}"
DROP_COL_TEMPLATE = "ALTER TABLE {TAB} DROP COLUMN IF EXISTS {COL};"

NEW_TABLE_TEMPLATE = "{TABLE}_{SS}{PERCENTAGE}"
COL_TEMPLATE = "{COL}_bin"
# update info_type set debug='a' where id IN ('1', '2', '3', '4', '5');

CREATE_COL_TMP = "ALTER TABLE {TABLE} ADD COLUMN {COL} int;"
UPDATE_TMP = "update {TABLE} set {COL}={VAL} where {KEY} IN ({BINVALS});"

def read_flags():
    parser = argparse.ArgumentParser()

    parser.add_argument("--db_name", type=str, required=False,
            default="imdb")
    parser.add_argument("--db_host", type=str, required=False,
            default="localhost")
    parser.add_argument("--user", type=str, required=False,
            default="pari")
    parser.add_argument("--pwd", type=str, required=False,
            default="")
    parser.add_argument("--port", type=str, required=False,
            default=5433)
    parser.add_argument("--bin_dir", type=str, required=False,
                        default="bins.pkl")

    parser.add_argument("--sampling_percentage", type=float, required=False,
            default=1.0)
    parser.add_argument("--sampling_type", type=str, required=False,
            default="ss")

    return parser.parse_args()

def main():
    con = pg.connect(user=args.user, host=args.db_host, port=args.port,
            password=args.pwd, database=args.db_name)
    cursor = con.cursor()

    with open("bins.pkl", "rb") as f:
        bins = pickle.load(f)

    for k in bins:
        curvals = []
        for v in bins[k].bins:
            # curvals.append(["'" + str(int(v2)) + "'" for v2 in v])
            curvals.append([str(int(v2)) for v2 in v])
        bins[k] = curvals

    with open("equivalent_keys.pkl", "rb") as f:
        equivalent_keys = pickle.load(f)

    sampling_frac = float(args.sampling_percentage) / 100.00
    for bkey, binvals in bins.items():
        for key in equivalent_keys[bkey]:
            # lets create the sampled table
            # let's build all the tables on primary keys first
            table = key[0:key.find(".")]
            new_table = NEW_TABLE_TEMPLATE.format(TABLE = table,
                                SS = args.sampling_type,
                                PERCENTAGE = str(args.sampling_percentage))
            count_sql = "SELECT COUNT(*) FROM {}".format(table)
            cursor.execute(count_sql)
            output = cursor.fetchall()[0][0]
            if output < 1000:
                cur_sampling_frac = 1.0
            else:
                cur_sampling_frac = sampling_frac

            new_table = new_table.replace(".","d")
            print(new_table)

            # drop_sql = DROP_TEMPLATE.format(TABLE_NAME = new_table)
            # cursor.execute(drop_sql)

            sel_sql = "SELECT * FROM {} WHERE random() < {}".format(\
                    table, str(cur_sampling_frac))
            create_sql = CREATE_TEMPLATE.format(TABLE_NAME = new_table,
                                                    SEL_SQL=sel_sql)
            print(create_sql)

            cursor.execute(create_sql)
            con.commit()

            # lets create a new column for this table
            orig_col = key[key.find(".")+1:]
            newcolname = COL_TEMPLATE.format(COL=orig_col)

            drop_col_sql = DROP_COL_TEMPLATE.format(TAB = new_table,
                                                    COL = newcolname)
            print(drop_col_sql)
            cursor.execute(drop_col_sql)
            con.commit()

            create_col_sql = CREATE_COL_TMP.format(TABLE=new_table,
                                  COL = newcolname)
            print(create_col_sql)

            cursor.execute(create_col_sql)
            con.commit()

            newkey = key[key.find(".")+1:]
            for bi, vals in enumerate(binvals):
                curbinvals = ','.join(vals)
                # UPDATE_TMP = "update {TABLE} set {COL}={VAL} where {KEY} IN ({BINVALS});"
                updatesql = UPDATE_TMP.format(TABLE = new_table,
                                  COL = newcolname,
                                  VAL = bi,
                                  KEY = newkey,
                                  BINVALS  = curbinvals)
                # print(updatesql)
                print("updating bin: ", bi)
                cursor.execute(updatesql)
                con.commit()

    pdb.set_trace()


args = read_flags()
main()
