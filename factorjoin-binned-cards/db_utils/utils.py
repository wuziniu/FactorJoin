import numpy as np
import glob
import string
from moz_sql_parser import parse
import json
import pdb
import re
import sqlparse
import itertools
import psycopg2 as pg
from utils.utils import *
import networkx as nx
import klepto
import getpass
import os
import subprocess as sp
import time
from sqlparse.sql import IdentifierList, Identifier
from sqlparse.tokens import Keyword, DML

import networkx as nx
from networkx.drawing.nx_agraph import write_dot,graphviz_layout
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
# from matplotlib.backends.backend_pdf import PdfPages
from networkx.drawing.nx_agraph import graphviz_layout, to_agraph
from networkx.algorithms import bipartite
from IPython.display import Image, display

from sql_rep.utils import extract_from_clause, extract_join_clause, \
        extract_join_graph, get_pg_join_order, nx_graph_to_query

# FIXME: shouldn't need these...
from sql_rep.utils import join_types
import platform
from ctypes import *
# TIMEOUT_COUNT_CONSTANT = 150001001
# TIMEOUT_COUNT_CONSTANT = 15000100001
# CROSS_JOIN_CONSTANT = 15000100000
# EXCEPTION_COUNT_CONSTANT = 15000100002

SAMPLE_TABLES = ["title", "name", "aka_name", "keyword", "movie_info",
        "movie_companies", "company_type", "kind_type", "info_type",
        "role_type", "company_name"]

SOURCE_NODE = tuple("s")

# TODO: FIXME
# SOURCE_NODE2 = tuple("SOURCE")

SOURCE_NODE_CONST = 100000
OLD_TIMEOUT_COUNT_CONSTANT = 150001001
OLD_CROSS_JOIN_CONSTANT = 150001000
OLD_EXCEPTION_COUNT_CONSTANT = 150001002

TIMEOUT_COUNT_CONSTANT = 150001000001
CROSS_JOIN_CONSTANT = 150001000000
EXCEPTION_COUNT_CONSTANT = 150001000002

CROSS_JOIN_CARD = 19329323

CREATE_TABLE_TEMPLATE = "CREATE TABLE {name} (id SERIAL, {columns})"
INSERT_TEMPLATE = "INSERT INTO {name} ({columns}) VALUES %s"

NTILE_CLAUSE = "ntile({BINS}) OVER (ORDER BY {COLUMN}) AS {ALIAS}"
GROUPBY_TEMPLATE = "SELECT {COLS}, COUNT(*) FROM {FROM_CLAUSE} GROUP BY {COLS}"
COUNT_SIZE_TEMPLATE = "SELECT COUNT(*) FROM {FROM_CLAUSE}"

SELECT_ALL_COL_TEMPLATE = "SELECT {COL} FROM {TABLE} WHERE {COL} IS NOT NULL"
ALIAS_FORMAT = "{TABLE} AS {ALIAS}"
MIN_TEMPLATE = "SELECT {COL} FROM {TABLE} WHERE {COL} IS NOT NULL ORDER BY {COL} ASC LIMIT 1"
MAX_TEMPLATE = "SELECT {COL} FROM {TABLE} WHERE {COL} IS NOT NULL ORDER BY {COL} DESC LIMIT 1"
UNIQUE_VALS_TEMPLATE = "SELECT DISTINCT {COL} FROM {FROM_CLAUSE}"
UNIQUE_COUNT_TEMPLATE = "SELECT COUNT(*) FROM (SELECT DISTINCT {COL} from {FROM_CLAUSE}) AS t"

INDEX_LIST_CMD = """
select
    t.relname as table_name,
    a.attname as column_name,
    i.relname as index_name
from
    pg_class t,
    pg_class i,
    pg_index ix,
    pg_attribute a
where
    t.oid = ix.indrelid
    and i.oid = ix.indexrelid
    and a.attrelid = t.oid
    and a.attnum = ANY(ix.indkey)
    and t.relkind = 'r'
order by
    t.relname,
    i.relname;"""


RANGE_PREDS = ["gt", "gte", "lt", "lte"]

CREATE_INDEX_TMP = '''CREATE INDEX IF NOT EXISTS {INDEX_NAME} ON {TABLE} ({COLUMN});'''

NODE_COLORS = {}
# NODE_COLORS["Hash Join"] = 'b'
# NODE_COLORS["Merge Join"] = 'r'
# NODE_COLORS["Nested Loop"] = 'c'

NODE_COLORS["Index Scan"] = 'b'
NODE_COLORS["Seq Scan"] = 'r'
NODE_COLORS["Bitmap Heap Scan"] = 'c'

# figure this out...
NODE_COLORS["Hash"] = 'b'
NODE_COLORS["Materialize"] = 'w'
NODE_COLORS["Sort"] = 'b'

# for signifying whether the join was a left join or right join
EDGE_COLORS = {}
EDGE_COLORS["left"] = "b"
EDGE_COLORS["right"] = "r"

NILJ_CONSTANT = 0.001
NILJ_CONSTANT2 = 2.0
SEQ_CONST = 20.0
RATIO_MUL_CONST = 1.0
NILJ_MIN_CARD = 5.0
CARD_DIVIDER = 0.001
INDEX_COST_CONSTANT = 10000
INDEX_PENALTY_MULTIPLE = 10.0

# def add_single_node_edges(subset_graph):

    # source = SOURCE_NODE

    # subset_graph.add_node(source)
    # subset_graph.nodes()[source]["cardinality"] = {}
    # subset_graph.nodes()[source]["cardinality"]["actual"] = 1.0
    # subset_graph.nodes()[source]["cardinality"]["total"] = 1.0

    # for node in subset_graph.nodes():
        # if len(node) != 1:
            # continue
        # if node[0] == source[0]:
            # continue

        # # print("going to add edge from source to node: ", node)
        # # subset_graph.add_edge(node, source, cost=0.0)
        # subset_graph.add_edge(node, source)
        # in_edges = subset_graph.in_edges(node)
        # out_edges = subset_graph.out_edges(node)
        # # print("in edges: ", in_edges)
        # # print("out edges: ", out_edges)

        # # if we need to add edges between single table nodes and rest
        # for node2 in subset_graph.nodes():
            # if len(node2) != 2:
                # continue
            # if node[0] in node2:
                # subset_graph.add_edge(node2, node)

def get_default_con_creds():
    if "user" in os.environ:
        user = os.environ["LC_PG_USER"]
    else:
        user = getpass.getuser()

    if "LC_PG_PWD" in os.environ:
        pwd = os.environ["LC_PG_PWD"]
    else:
        pwd = ""

    if "LC_PG_DB" in os.environ:
        db = os.environ["LC_PG_DB"]
    else:
        db = "imdb"

    if "LC_PG_HOST" in os.environ:
        db_host = os.environ["LC_PG_HOST"]
    else:
        db_host = "localhost"

    if "LC_PG_PORT" in os.environ:
        port = int(os.environ["LC_PG_PORT"])
    else:
        port = 5432

    return user, pwd, db, db_host, port

def _find_all_tables(plan):
    '''
    '''
    # find all the scan nodes under the current level, and return those
    table_names = extract_values(plan, "Relation Name")
    alias_names = extract_values(plan, "Alias")
    table_names.sort()
    alias_names.sort()

    return table_names, alias_names

def extract_aliases2(plan):
    aliases = extract_values(plan, "Alias")
    return aliases

def get_leading_hint(explain):
    '''
    '''
    def __extract_jo(plan):
        if len(plan["Plans"]) == 2:
            left = list(extract_aliases2(plan["Plans"][0]))
            right = list(extract_aliases2(plan["Plans"][1]))

            if len(left) == 1 and len(right) == 1:
                return left[0] + " " + right[0]

            if len(left) == 1:
                left_alias = left[0]
                return left_alias + " (" + __extract_jo(plan["Plans"][1]) + ")"

            if len(right) == 1:
                right_alias = right[0]
                return "(" + __extract_jo(plan["Plans"][0]) + ") " + right_alias

            return ("(" + __extract_jo(plan["Plans"][0])
                    + ") ("
                    + __extract_jo(plan["Plans"][1]) + ")")

        return __extract_jo(plan["Plans"][0])

    jo = __extract_jo(explain[0][0][0]["Plan"])
    jo = "(" + jo + ")"
    return jo

def _plot_join_order_graph(G, base_table_nodes, join_nodes, pdf, title):

    def format_ints(num):
        # returns the number formatted to closest 1000 + K
        return str(round(num, -3)).replace("000","") + "K"

    def _plot_labels(xdiff, ydiff, key, font_color, font_size):
        labels = {}
        label_pos = {}
        for k, v in pos.items():
            label_pos[k] = (v[0]+xdiff, v[1]+ydiff)
            if key in G.nodes[k]:
                if is_float(G.nodes[k][key]):
                    labels[k] = format_ints(G.nodes[k][key])
                else:
                    labels[k] = G.nodes[k][key]
            else:
                est_labels[k] = -1

        nx.draw_networkx_labels(G, label_pos, labels,
                font_size=font_size, font_color=font_color)

    NODE_SIZE = 600
    pos = graphviz_layout(G, prog='dot')
    plt.title(title)
    color_intensity = [G.nodes[n]["cur_cost"] for n in G.nodes()]
    vmin = min(color_intensity)
    vmax = max(color_intensity)
    cmap = 'viridis_r'
    # pos = graphviz_layout(G, prog='dot',
            # args='-Gnodesep=0.05')
    nx.draw_networkx_nodes(G, pos,
               node_size=NODE_SIZE,
               node_color = color_intensity,
               cmap = cmap,
               alpha=0.2,
               vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin = vmin,
        vmax=vmax))
    sm._A = []
    plt.colorbar(sm, alpha=0.2)

    _plot_labels(0, -25, "est_card", "r", 8)
    _plot_labels(0, +25, "true_card", "g", 8)
    _plot_labels(0, 0, "node_label", "b", 14)

    # TODO: shape of node based on scan types
    # _plot_labels(+25, +5, "scan_type", "b", 10)

    x_values, y_values = zip(*pos.values())
    x_max = max(x_values)
    x_min = min(x_values)
    x_margin = (x_max - x_min) * 0.10
    plt.xlim(x_min - x_margin, x_max + x_margin)

    edge_colors = []
    for edge in G.edges():
        edge_colors.append(EDGE_COLORS[G.edges[edge]["join_direction"]])

    nx.draw_networkx_edges(G, pos, width=1.0,
            alpha=1.0,with_labels=False, arrows=False,
            edge_color=edge_colors)

    plt.tight_layout()
    # plt.savefig(fn)
    # pdf.savefig()
    # plt.close()

def explain_to_nx(explain):
    '''
    '''
    # JOIN_KEYS = ["Hash Join", "Nested Loop", "Join"]
    base_table_nodes = []
    join_nodes = []

    def _get_node_name(tables):
        name = ""
        if len(tables) > 1:
            name = str(deterministic_hash(str(tables)))[0:5]
            join_nodes.append(name)
        else:
            name = tables[0]
            if len(name) >= 6:
                # no aliases, shorten it
                name = "".join([n[0] for n in name.split("_")])
                if name in base_table_nodes:
                    name = name + "2"
            base_table_nodes.append(name)
        return name

    def _add_node_stats(node, plan):
        # add stats for the join
        G.nodes[node]["Plan Rows"] = plan["Plan Rows"]
        if "Actual Rows" in plan:
            G.nodes[node]["Actual Rows"] = plan["Actual Rows"]
        else:
            G.nodes[node]["Actual Rows"] = -1.0
        if "Actual Total Time" in plan:
            G.nodes[node]["total_time"] = plan["Actual Total Time"]

            if "Plans" not in plan:
                children_time = 0.0
            elif len(plan["Plans"]) == 2:
                children_time = plan["Plans"][0]["Actual Total Time"] \
                        + plan["Plans"][1]["Actual Total Time"]
            elif len(plan["Plans"]) == 1:
                children_time = plan["Plans"][0]["Actual Total Time"]
            else:
                assert False

            G.nodes[node]["cur_time"] = plan["Actual Total Time"]-children_time

        else:
            G.nodes[node]["Actual Total Time"] = -1.0

        if "Node Type" in plan:
            G.nodes[node]["Node Type"] = plan["Node Type"]

        total_cost = plan["Total Cost"]
        G.nodes[node]["Total Cost"] = total_cost
        aliases = G.nodes[node]["aliases"]
        if len(G.nodes[node]["tables"]) > 1:
            children_cost = plan["Plans"][0]["Total Cost"] \
                    + plan["Plans"][1]["Total Cost"]

            # +1 to avoid cases which are very close
            if not total_cost+1 >= children_cost:
                print("aliases: {} children cost: {}, total cost: {}".format(\
                        aliases, children_cost, total_cost))
                # pdb.set_trace()
            G.nodes[node]["cur_cost"] = total_cost - children_cost
            G.nodes[node]["node_label"] = plan["Node Type"][0]
            G.nodes[node]["scan_type"] = ""
        else:
            G.nodes[node]["cur_cost"] = total_cost
            G.nodes[node]["node_label"] = node
            # what type of scan was this?
            node_types = extract_values(plan, "Node Type")
            for i, full_n in enumerate(node_types):
                shortn = ""
                for n in full_n.split(" "):
                    shortn += n[0]
                node_types[i] = shortn

            scan_type = "\n".join(node_types)
            G.nodes[node]["scan_type"] = scan_type

    def traverse(obj):
        if isinstance(obj, dict):
            if "Plans" in obj:
                if len(obj["Plans"]) == 2:
                    # these are all the joins
                    left_tables, left_aliases = _find_all_tables(obj["Plans"][0])
                    right_tables, right_aliases = _find_all_tables(obj["Plans"][1])
                    if len(left_tables) == 0 or len(right_tables) == 0:
                        return
                    all_tables = left_tables + right_tables
                    all_aliases = left_aliases + right_aliases
                    all_aliases.sort()
                    all_tables.sort()

                    if len(left_aliases) > 0:
                        node0 = _get_node_name(left_aliases)
                        node1 = _get_node_name(right_aliases)
                        node_new = _get_node_name(all_aliases)
                    else:
                        node0 = _get_node_name(left_tables)
                        node1 = _get_node_name(right_tables)
                        node_new = _get_node_name(all_tables)

                    # update graph
                    # G.add_edge(node0, node_new)
                    # G.add_edge(node1, node_new)
                    G.add_edge(node_new, node0)
                    G.add_edge(node_new, node1)
                    G.edges[(node_new, node0)]["join_direction"] = "left"
                    G.edges[(node_new, node1)]["join_direction"] = "right"

                    # add other parameters on the nodes
                    G.nodes[node0]["tables"] = left_tables
                    G.nodes[node1]["tables"] = right_tables
                    G.nodes[node0]["aliases"] = left_aliases
                    G.nodes[node1]["aliases"] = right_aliases
                    G.nodes[node_new]["tables"] = all_tables
                    G.nodes[node_new]["aliases"] = all_aliases

                    # TODO: if either the left, or right were a scan, then add
                    # scan stats
                    _add_node_stats(node_new, obj)

                    if len(left_tables) == 1:
                        _add_node_stats(node0, obj["Plans"][0])
                    if len(right_tables) == 1:
                        _add_node_stats(node1, obj["Plans"][1])

            for k, v in obj.items():
                if isinstance(v, (dict, list)):
                    traverse(v)

        elif isinstance(obj, list) or isinstance(obj,tuple):
            for item in obj:
                traverse(item)

    G = nx.DiGraph()
    traverse(explain)
    G.base_table_nodes = base_table_nodes
    G.join_nodes = join_nodes
    return G

def plot_explain_join_order(explain, true_cardinalities,
        est_cardinalities, pdf, title):
    '''
    @true_cardinalities: dict for this particular explain
    '''
    G = explain_to_nx(explain)
    for node in G.nodes():
        aliases = G.nodes[node]["aliases"]
        aliases.sort()
        card_key = " ".join(aliases)
        if true_cardinalities is None:
            G.nodes[node]["est_card"] = G.nodes[node]["Plan Rows"]
            G.nodes[node]["true_card"] = G.nodes[node]["Actual Rows"]
        elif card_key in true_cardinalities:
            G.nodes[node]["est_card"] = est_cardinalities[card_key]
            G.nodes[node]["true_card"] = true_cardinalities[card_key]
        elif tuple(aliases) in true_cardinalities:
            G.nodes[node]["est_card"] = est_cardinalities[tuple(aliases)]
            G.nodes[node]["true_card"] = true_cardinalities[tuple(aliases)]
        else:
            # unknown, might be a cross-join?
            G.nodes[node]["est_card"] = CROSS_JOIN_CARD
            G.nodes[node]["true_card"] = CROSS_JOIN_CARD
            print("did not find alias in cards, is this cross join?")
            print(aliases)
            pdb.set_trace()

        # if G.nodes[node]["Plan Rows"] != G.nodes[node]["true_card"]:
            # if len(aliases) != 1 and \
                # G.nodes[node]["true_card"] != TIMEOUT_COUNT_CONSTANT:
            # if len(aliases) != 1:
                # print("aliases: {}, true: {}, est: {}, plan rows: {}".format(\
                        # aliases,
                        # G.nodes[node]["true_card"], G.nodes[node]["est_card"],
                        # G.nodes[node]["Plan Rows"]))

    _plot_join_order_graph(G, G.base_table_nodes, G.join_nodes, pdf, title)
    return G

def benchmark_sql(sql, user, db_host, port, pwd, db_name,
        join_collapse_limit):
    '''
    TODO: should we be doing anything smarter?
    '''
    query_tmp = "SET join_collapse_limit={jcl}; {sql}"
    sql = query_tmp.format(jcl=join_collapse_limit, sql=sql)
    # first drop cache
    # FIXME: choose the right file automatically
    drop_cache_cmd = "./drop_cache.sh"
    p = sp.Popen(drop_cache_cmd, shell=True)
    p.wait()
    time.sleep(2)

    os_user = getpass.getuser()
    # con = pg.connect(user=user, port=port,
            # password=pwd, database=db_name, host=db_host)

    if os_user == "ubuntu":
        # for aws
        # con = pg.connect(user=user, port=port,
                # password=pwd, database=db_name)
        print(user, db_host, port, pwd, db_name)
        con = pg.connect(user=user, host=db_host, port=port,
                password=pwd, database=db_name)
    else:
        # for chunky
        con = pg.connect(user=user, host=db_host, port=port,
                password=pwd, database=db_name)

    cursor = con.cursor()
    start = time.time()
    cursor.execute(sql)

    exec_time = time.time() - start

    output = cursor.fetchall()
    cursor.close()
    con.close()

    return output, exec_time

def visualize_query_plan(sql, db_name, out_name_suffix):
    '''
    '''
    if "EXPLAIN" not in sql:
        sql = "EXPLAIN (ANALYZE, COSTS, VERBOSE, BUFFERS, FORMAT JSON) " + sql
    # first drop cache
    drop_cache_cmd = "./drop_cache.sh"
    p = sp.Popen(drop_cache_cmd, shell=True)
    p.wait()
    time.sleep(2)

    tmp_fn = "./explain/test_" + out_name_suffix + ".sql"
    with open(tmp_fn, "w") as f:
        f.write(sql)
    json_out = "./explain/analyze_" + out_name_suffix + ".json"
    psql_cmd = "psql -d {} -qAt -f {} > {}".format(db_name, tmp_fn, json_out)

    p = sp.Popen(psql_cmd, shell=True)
    p.wait()

def pg_est_from_explain(output):
    '''
    '''
    est_vals = None
    for line in output:
        line = line[0]
        if "Seq Scan" in line or "Loop" in line or "Join" in line \
                or "Index Scan" in line or "Scan" in line:
            for w in line.split():
                if "rows" in w and est_vals is None:
                    est_vals = int(re.findall("\d+", w)[0])
                    return est_vals

    print("pg est failed!")
    print(output)
    pdb.set_trace()
    return 1.00

def get_all_wheres(parsed_query):
    pred_vals = []
    if "where" not in parsed_query:
        pass
    elif "and" not in parsed_query["where"]:
        pred_vals = [parsed_query["where"]]
    else:
        pred_vals = parsed_query["where"]["and"]
    return pred_vals

def extract_predicates2(query):
    '''
    @ret:
        - column names with predicate conditions in WHERE.
        - predicate operator type (e.g., "in", "lte" etc.)
        - predicate value
    Note: join conditions don't count as predicate conditions.

    FIXME: temporary hack. For range queries, always returning key
    "lt", and vals for both the lower and upper bound
    '''
    predicate_cols = []
    predicate_types = []
    predicate_vals = []
    if "::float" in query:
        query = query.replace("::float", "")
    elif "::int" in query:
        query = query.replace("::int", "")
    # really fucking dumb
    bad_str1 = "mii2.info ~ '^(?:[1-9]\d*|0)?(?:\.\d+)?$' AND"
    bad_str2 = "mii1.info ~ '^(?:[1-9]\d*|0)?(?:\.\d+)?$' AND"
    if bad_str1 in query:
        query = query.replace(bad_str1, "")

    if bad_str2 in query:
        query = query.replace(bad_str2, "")

    parsed = sqlparse.parse(query)[0]
    # let us go over all the where clauses
    start = time.time()
    where_clauses = None
    for token in parsed.tokens:
        if (type(token) == sqlparse.sql.Where):
            where_clauses = token
    if where_clauses is None:
        assert False
        return [], [], []

    froms, aliases, table_names = extract_from_clause(query)
    if len(aliases) > 0:
        tables = [k for k in aliases]
    else:
        tables = table_names
    matches = find_all_clauses(tables, where_clauses)

    print(matches)
    print(where_clauses)
    pdb.set_trace()

def extract_predicates(query):
    '''
    @ret:
        - column names with predicate conditions in WHERE.
        - predicate operator type (e.g., "in", "lte" etc.)
        - predicate value
    Note: join conditions don't count as predicate conditions.

    FIXME: temporary hack. For range queries, always returning key
    "lt", and vals for both the lower and upper bound
    '''
    def parse_column(pred, cur_pred_type):
        '''
        gets the name of the column, and whether column location is on the left
        (0) or right (1)
        '''
        for i, obj in enumerate(pred[cur_pred_type]):
            assert i <= 1
            if isinstance(obj, str) and "." in obj:
                # assert "." in obj
                column = obj
            elif isinstance(obj, dict):
                assert "literal" in obj
                val = obj["literal"]
                val_loc = i
            else:
                val = obj
                val_loc = i

        assert column is not None
        assert val is not None
        return column, val_loc, val

    def _parse_predicate(pred, pred_type):
        if pred_type == "eq":
            columns = pred[pred_type]
            if len(columns) <= 1:
                return None
            # FIXME: more robust handling?
            if "." in str(columns[1]):
                # should be a join, skip this.
                # Note: joins only happen in "eq" predicates
                return None
            predicate_types.append(pred_type)
            predicate_cols.append(columns[0])
            predicate_vals.append(columns[1])

        elif pred_type in RANGE_PREDS:
            vals = [None, None]
            col_name, val_loc, val = parse_column(pred, pred_type)
            vals[val_loc] = val

            # this loop may find no matching predicate for the other side, in
            # which case, we just leave the val as None
            for pred2 in pred_vals:
                pred2_type = list(pred2.keys())[0]
                if pred2_type in RANGE_PREDS:
                    col_name2, val_loc2, val2 = parse_column(pred2, pred2_type)
                    if col_name2 == col_name:
                        # assert val_loc2 != val_loc
                        if val_loc2 == val_loc:
                            # same predicate as pred
                            continue
                        vals[val_loc2] = val2
                        break

            predicate_types.append("lt")
            predicate_cols.append(col_name)
            if "g" in pred_type:
                # reverse vals, since left hand side now means upper bound
                vals.reverse()
            predicate_vals.append(vals)

        elif pred_type == "between":
            # we just treat it as a range query
            col = pred[pred_type][0]
            val1 = pred[pred_type][1]
            val2 = pred[pred_type][2]
            vals = [val1, val2]
            predicate_types.append("lt")
            predicate_cols.append(col)
            predicate_vals.append(vals)
        elif pred_type == "in" \
                or "like" in pred_type:
            # includes preds like, ilike, nlike etc.
            column = pred[pred_type][0]
            # what if column has been seen before? Will just be added again to
            # the list of predicates, which is the correct behaviour
            vals = pred[pred_type][1]
            if isinstance(vals, dict):
                vals = vals["literal"]
            if not isinstance(vals, list):
                vals = [vals]
            predicate_types.append(pred_type)
            predicate_cols.append(column)
            predicate_vals.append(vals)
        elif pred_type == "or":
            for pred2 in pred[pred_type]:
                # print(pred2)
                assert len(pred2.keys()) == 1
                pred_type2 = list(pred2.keys())[0]
                _parse_predicate(pred2, pred_type2)

        elif pred_type == "missing":
            column = pred[pred_type]
            val = ["NULL"]
            predicate_types.append("in")
            predicate_cols.append(column)
            predicate_vals.append(val)
        else:
            # assert False
            # TODO: need to support "OR" statements
            return None
            # assert False, "unsupported predicate type"

    start = time.time()
    predicate_cols = []
    predicate_types = []
    predicate_vals = []
    if "::float" in query:
        query = query.replace("::float", "")
    elif "::int" in query:
        query = query.replace("::int", "")
    # really fucking dumb
    bad_str1 = "mii2.info ~ '^(?:[1-9]\d*|0)?(?:\.\d+)?$' AND"
    bad_str2 = "mii1.info ~ '^(?:[1-9]\d*|0)?(?:\.\d+)?$' AND"
    if bad_str1 in query:
        query = query.replace(bad_str1, "")

    if bad_str2 in query:
        query = query.replace(bad_str2, "")

    # FIXME: temporary workaround moz_sql_parser...
    query = query.replace("ILIKE", "LIKE")
    try:
        parsed_query = parse(query)
    except:
        print(query)
        print("moz sql parser failed to parse this!")
        pdb.set_trace()
    pred_vals = get_all_wheres(parsed_query)

    for i, pred in enumerate(pred_vals):
        try:
            assert len(pred.keys()) == 1
        except:
            print(pred)
            pdb.set_trace()
        pred_type = list(pred.keys())[0]
        # if pred == "or" or pred == "OR":
            # continue
        _parse_predicate(pred, pred_type)

    # print("extract predicate cols done!")
    # print("extract predicates took ", time.time() - start)
    return predicate_cols, predicate_types, predicate_vals

# def extract_from_clause(query):
    # '''
    # Optimized version using sqlparse.
    # Extracts the from statement, and the relevant joins when there are multiple
    # tables.
    # @ret: froms:
          # froms: [alias1, alias2, ...] OR [table1, table2,...]
          # aliases:{alias1: table1, alias2: table2} (OR [] if no aliases present)
          # tables: [table1, table2, ...]
    # '''
    # def handle_table(identifier):
        # table_name = identifier.get_real_name()
        # alias = identifier.get_alias()
        # tables.append(table_name)
        # if alias is not None:
            # from_clause = ALIAS_FORMAT.format(TABLE = table_name,
                                # ALIAS = alias)
            # froms.append(from_clause)
            # aliases[alias] = table_name
        # else:
            # froms.append(table_name)

    # start = time.time()
    # froms = []
    # # key: alias, val: table name
    # aliases = {}
    # # just table names
    # tables = []

    # start = time.time()
    # parsed = sqlparse.parse(query)[0]
    # # let us go over all the where clauses
    # from_token = None
    # from_seen = False
    # for token in parsed.tokens:
        # # print(type(token))
        # # print(token)
        # if from_seen:
            # if isinstance(token, IdentifierList) or isinstance(token,
                    # Identifier):
                # from_token = token
        # if token.ttype is Keyword and token.value.upper() == 'FROM':
            # from_seen = True

    # assert from_token is not None
    # if isinstance(from_token, IdentifierList):
        # for identifier in from_token.get_identifiers():
            # handle_table(identifier)
    # elif isinstance(from_token, Identifier):
        # handle_table(from_token)
    # else:
        # assert False

    # # print("extract froms parse took: ", time.time() - start)

    # return froms, aliases, tables

def check_table_exists(cur, table_name):
    cur.execute("select exists(select * from information_schema.tables where\
            table_name=%s)", (table_name,))
    return cur.fetchone()[0]

def db_vacuum(conn, cur):
    old_isolation_level = conn.isolation_level
    conn.set_isolation_level(0)
    query = "VACUUM ANALYZE"
    cur.execute(query)
    conn.set_isolation_level(old_isolation_level)
    conn.commit()

def to_bitset(num_attrs, arr):
    ret = [i for i, val in enumerate(arr) if val == 1.0]
    for i, r in enumerate(ret):
        ret[i] = r % num_attrs
    return ret

def bitset_to_features(bitset, num_attrs):
    '''
    @bitset set of ints, which are the index of elements that should be in 1.
    Converts this into an array of size self.attr_count, with the appropriate
    elements 1.
    '''
    features = []
    for i in range(num_attrs):
        if i in bitset:
            features.append(1.00)
        else:
            features.append(0.00)
    return features

def find_all_tables_till_keyword(token):
    tables = []
    # print("fattk: ", token)
    index = 0
    while (True):
        if (type(token) == sqlparse.sql.Comparison):
            left = token.left
            right = token.right
            if (type(left) == sqlparse.sql.Identifier):
                tables.append(left.get_parent_name())
            if (type(right) == sqlparse.sql.Identifier):
                tables.append(right.get_parent_name())
            break
        elif (type(token) == sqlparse.sql.Identifier):
            tables.append(token.get_parent_name())
            break
        try:
            index, token = token.token_next(index)
            if ("Literal" in str(token.ttype)) or token.is_keyword:
                break
        except:
            break

    return tables

def find_next_match(tables, wheres, index):
    '''
    ignore everything till next
    '''
    match = ""
    _, token = wheres.token_next(index)
    if token is None:
        return None, None
    # FIXME: is this right?
    if token.is_keyword:
        index, token = wheres.token_next(index)

    tables_in_pred = find_all_tables_till_keyword(token)
    assert len(tables_in_pred) <= 2

    token_list = sqlparse.sql.TokenList(wheres)

    while True:
        index, token = token_list.token_next(index)
        if token is None:
            break
        # print("token.value: ", token.value)
        if token.value == "AND":
            break

        match += " " + token.value

        if (token.value == "BETWEEN"):
            # ugh ugliness
            index, a = token_list.token_next(index)
            index, AND = token_list.token_next(index)
            index, b = token_list.token_next(index)
            match += " " + a.value
            match += " " + AND.value
            match += " " + b.value
            # Note: important not to break here! Will break when we hit the
            # "AND" in the next iteration.

    # print("tables: ", tables)
    # print("match: ", match)
    # print("tables in pred: ", tables_in_pred)
    for table in tables_in_pred:
        if table not in tables:
            # print(tables)
            # print(table)
            # pdb.set_trace()
            # print("returning index, None")
            return index, None

    if len(tables_in_pred) == 0:
        return index, None

    return index, match

def find_all_clauses(tables, wheres):
    matched = []
    # print(tables)
    index = 0
    while True:
        index, match = find_next_match(tables, wheres, index)
        # print("got index, match: ", index)
        # print(match)
        if match is not None:
            matched.append(match)
        if index is None:
            break

    return matched

def get_join_graph(joins, tables=None):
    join_graph = nx.Graph()
    for j in joins:
        j1 = j.split("=")[0]
        j2 = j.split("=")[1]
        t1 = j1[0:j1.find(".")].strip()
        t2 = j2[0:j2.find(".")].strip()
        if tables is not None:
            try:
                assert t1 in tables
                assert t2 in tables
            except:
                print(t1, t2)
                print(tables)
                print(joins)
                print("table not in tables!")
                pdb.set_trace()
        join_graph.add_edge(t1, t2)
    return join_graph

def _gen_subqueries(all_tables, wheres, aliases):
    '''
    my old shitty sqlparse code that should be updated...
    @tables: list
    @wheres: sqlparse object
    '''
    # FIXME: nicer setup
    if len(aliases) > 0:
        all_tables = [a for a in aliases]

    all_subqueries = []
    combs = []
    for i in range(1, len(all_tables)+1):
        combs += itertools.combinations(list(range(len(all_tables))), i)
    # print("num combs: ", len(combs))
    for comb in combs:
        cur_tables = []
        for i, idx in enumerate(comb):
            cur_tables.append(all_tables[idx])

        matches = find_all_clauses(cur_tables, wheres)
        cond_string = " AND ".join(matches)
        if cond_string != "":
            cond_string = " WHERE " + cond_string

        # need to handle joins: if there are more than 1 table in tables, then
        # the predicates must include a join in between them
        if len(cur_tables) > 1:
            all_joins = True
            for ctable in cur_tables:
                joined = False
                for match in matches:
                    # FIXME: so hacky ugh. more band-aid
                    if match.count(".") == 2 \
                            and "=" in match:
                        if (" " + ctable + "." in " " + match):
                            joined = True
                if not joined:
                    all_joins = False
                    break
            if not all_joins:
                continue
        if len(aliases) > 0:
            aliased_tables = [ALIAS_FORMAT.format(TABLE=aliases[a], ALIAS=a) for a in cur_tables]
            from_clause = " , ".join(aliased_tables)
            # print(from_clause)
        else:
            from_clause = " , ".join(cur_tables)

        query = COUNT_SIZE_TEMPLATE.format(FROM_CLAUSE=from_clause) + cond_string
        # final sanity checks
        joins = extract_join_clause(query)
        _,_, tables = extract_from_clause(query)

        # TODO: maybe this should be done somewhere earlier in the pipeline?
        join_graph = nx.Graph()
        for j in joins:
            j1 = j.split("=")[0]
            j2 = j.split("=")[1]
            t1 = j1[0:j1.find(".")].strip()
            t2 = j2[0:j2.find(".")].strip()
            try:
                assert t1 in tables or t1 in aliases
                assert t2 in tables or t2 in aliases
            except:
                print(t1, t2)
                print(tables)
                print(joins)
                print("table not in tables!")
                pdb.set_trace()
            join_graph.add_edge(t1, t2)
        if len(joins) > 0 and not nx.is_connected(join_graph):
            # print("skipping query!")
            # print(tables)
            # print(joins)
            # pdb.set_trace()
            continue
        all_subqueries.append(query)

    return all_subqueries

# def nx_graph_to_query(G):
    # froms = []
    # conds = []
    # for nd in G.nodes(data=True):
        # node = nd[0]
        # data = nd[1]
        # if "real_name" in data:
            # froms.append(ALIAS_FORMAT.format(TABLE=data["real_name"],
                                             # ALIAS=node))
        # else:
            # froms.append(node)

        # for pred in data["predicates"]:
            # conds.append(pred)

    # for edge in G.edges(data=True):
        # conds.append(edge[2]['join_condition'])

    # # preserve order for caching
    # froms.sort()
    # conds.sort()
    # from_clause = " , ".join(froms)
    # if len(conds) > 0:
        # wheres = ' AND '.join(conds)
        # from_clause += " WHERE " + wheres
    # count_query = COUNT_SIZE_TEMPLATE.format(FROM_CLAUSE=from_clause)
    # return count_query

def _gen_subqueries_nx(query):
    start = time.time()
    froms,aliases,tables = extract_from_clause(query)
    joins = extract_join_clause(query)
    # pred_columns, pred_types, pred_vals = extract_predicates(query)
    join_graph = nx.Graph()
    for j in joins:
        j1 = j.split("=")[0]
        j2 = j.split("=")[1]
        t1 = j1[0:j1.find(".")].strip()
        t2 = j2[0:j2.find(".")].strip()
        try:
            assert t1 in tables or t1 in aliases
            assert t2 in tables or t2 in aliases
        except:
            print(t1, t2)
            print(tables)
            print(joins)
            print("table not in tables!")
            pdb.set_trace()

        join_graph.add_edge(t1, t2)
        join_graph[t1][t2]["join_condition"] = j
        if t1 in aliases:
            table1 = aliases[t1]
            table2 = aliases[t2]

            join_graph.nodes()[t1]["real_name"] = table1
            join_graph.nodes()[t2]["real_name"] = table2

    parsed = sqlparse.parse(query)[0]
    # let us go over all the where clauses
    where_clauses = None
    for token in parsed.tokens:
        if (type(token) == sqlparse.sql.Where):
            where_clauses = token
    assert where_clauses is not None

    for t1 in join_graph.nodes():
        tables = [t1]
        matches = find_all_clauses(tables, where_clauses)
        join_graph.nodes()[t1]["predicates"] = matches

    # TODO: Next, need an efficient way to generate all connected subgraphs, and
    # then convert each of them to a sql queries
    all_subqueries = []

    # find all possible subsets of the nodes
    combs = []
    all_nodes = list(join_graph.nodes())
    for i in range(1, len(all_nodes)+1):
        combs += itertools.combinations(list(range(len(all_nodes))), i)

    for node_idxs in combs:
        nodes = [all_nodes[idx] for idx in node_idxs]
        subg = join_graph.subgraph(nodes)
        if nx.is_connected(subg):
            sql_str = nx_graph_to_query(subg)
            all_subqueries.append(sql_str)

    print("num subqueries: ", len(all_subqueries))
    print("took: ", time.time() - start)

    return all_subqueries

def gen_all_subqueries(query):
    '''
    @query: sql string.
    @ret: [sql strings], that represent all subqueries excluding cross-joins.
    FIXME: mix-match of moz_sql_parser AND sqlparse...
    '''
    start = time.time()
    all_subqueries = _gen_subqueries_nx(query)
    return all_subqueries

def cached_execute_query(sql, user, db_host, port, pwd, db_name,
        execution_cache_threshold, sql_cache_dir=None, timeout=120000):
    '''
    @timeout:
    @db_host: going to ignore it so default localhost is used.
    executes the given sql on the DB, and caches the results in a
    persistent store if it took longer than self.execution_cache_threshold.
    '''
    sql_cache = None
    if sql_cache_dir is not None:
        assert isinstance(sql_cache_dir, str)
        sql_cache = klepto.archives.dir_archive(sql_cache_dir,
                cached=True, serialized=True)

    hashed_sql = deterministic_hash(sql)

    # archive only considers the stuff stored in disk
    if sql_cache is not None and hashed_sql in sql_cache.archive:
        return sql_cache.archive[hashed_sql], False

    start = time.time()

    os_user = getpass.getuser()
    if os_user == "ubuntu":
        # for aws
        con = pg.connect(user=user, port=port,
                password=pwd, database=db_name)
    else:
        # for chunky
        con = pg.connect(user=user, host=db_host, port=port,
                password=pwd, database=db_name)

    cursor = con.cursor()
    if timeout is not None:
        cursor.execute("SET statement_timeout = {}".format(timeout))
    try:
        cursor.execute(sql)
    except Exception as e:
        # print("query failed to execute: ", sql)
        # FIXME: better way to do this.
        cursor.execute("ROLLBACK")
        con.commit()
        cursor.close()
        con.close()

        if not "timeout" in str(e):
            print("failed to execute for reason other than timeout")
            print(e)
            print(sql)
            pdb.set_trace()
        else:
            print("failed because of timeout!")
            return None, True

        return None, False

    exp_output = cursor.fetchall()
    cursor.close()
    con.close()
    end = time.time()
    if (end - start > execution_cache_threshold) \
            and sql_cache is not None:
        sql_cache.archive[hashed_sql] = exp_output
    return exp_output, False

def get_total_count_query(sql):
    '''
    @ret: sql query.
    '''
    froms, _, _ = extract_from_clause(sql)
    # FIXME: should be able to store this somewhere and not waste
    # re-executing it always
    from_clause = " , ".join(froms)
    joins = extract_join_clause(sql)
    if len(joins) < len(froms)-1:
        print("joins < len(froms)-1")
        print(sql)
        print(joins)
        print(len(joins))
        print(froms)
        print(len(froms))
        # pdb.set_trace()
    join_clause = ' AND '.join(joins)
    if len(join_clause) > 0:
        from_clause += " WHERE " + join_clause
    count_query = COUNT_SIZE_TEMPLATE.format(FROM_CLAUSE=from_clause)
    # print("COUNT QUERY:\n", count_query)
    # pdb.set_trace()
    return count_query

def sql_to_query_object(sql, user, db_host, port, pwd, db_name,
        total_count=None,execution_cache_threshold=None,
        sql_cache=None, timeout=None, num_query=1):
    '''
    @sql: string sql.
    @ret: Query object with all fields appropriately initialized.
          If it fails anywhere, then return None.
    @execution_cache_threshold: In seconds, if query takes beyond this, then
    cache it.
    '''
    if num_query % 100 == 0:
        print("sql_to_query_object num query: ", num_query)

    if execution_cache_threshold is None:
        execution_cache_threshold = 60

    if "SELECT COUNT" not in sql:
        print("no SELECT COUNT in sql!")
        exit(-1)

    output, tout = cached_execute_query(sql, user, db_host, port, pwd, db_name,
            execution_cache_threshold, sql_cache, timeout)

    if tout:
        # just fix all vals to be same
        true_val = TIMEOUT_COUNT_CONSTANT
        pg_est = TIMEOUT_COUNT_CONSTANT
        total_count = TIMEOUT_COUNT_CONSTANT
        # pred_columns, pred_types, pred_vals = extract_predicates(sql)
        pred_columns, pred_types, pred_vals = None, None, None

        from cardinality_estimation.query import Query
        query = Query(sql, pred_columns, pred_vals, pred_types,
                true_val, total_count, pg_est)
        return query
    else:
        if output is None:
            print("cached execute query returned None!!")
            exit(-1)
            # return None
        # from query string, to Query object
        true_val = output[0][0]

    exp_query = "EXPLAIN " + sql
    exp_output, tout  = cached_execute_query(exp_query, user, db_host, port, pwd, db_name,
            execution_cache_threshold, sql_cache, timeout)

    assert not tout

    if exp_output is None:
        return None
    pg_est = pg_est_from_explain(exp_output)

    # FIXME: start caching the true total count values
    if total_count is None:
        total_count_query = get_total_count_query(sql)

        # if we should just update value based on pg' estimate for total count
        # v/s finding true count
        TRUE_TOTAL_COUNT = False
        total_timeout = 180000
        if TRUE_TOTAL_COUNT:
            exp_output, _ = cached_execute_query(total_count_query, user,
                    db_host, port, pwd, db_name, execution_cache_threshold,
                    sql_cache, total_timeout)
            if exp_output is None:
                # print("total count query timed out")
                # print(total_count_query)
                # execute it with explain
                exp_query = "EXPLAIN " + total_count_query
                exp_output, _ = cached_execute_query(exp_query, user, db_host, port, pwd, db_name,
                        execution_cache_threshold, sql_cache, total_timeout)
                if exp_output is None:
                    print("pg est was None for ")
                    print(exp_query)
                    pdb.set_trace()
                total_count = pg_est_from_explain(exp_output)
                print("pg total count est: ", total_count)
            else:
                total_count = exp_output[0][0]
                # print("total count: ", total_count)
        else:
            exp_query = "EXPLAIN " + total_count_query
            exp_output, _ = cached_execute_query(exp_query, user, db_host, port, pwd, db_name,
                    execution_cache_threshold, sql_cache, total_timeout)
            if exp_output is None:
                print("pg est was None for ")
                print(exp_query)
                pdb.set_trace()
            total_count = pg_est_from_explain(exp_output)
            # print("pg total count est: ", total_count)

    # need to extract predicate columns, predicate operators, and predicate
    # values now.
    # pred_columns, pred_types, pred_vals = extract_predicates(sql)
    pred_columns, pred_types, pred_vals = None, None, None

    from cardinality_estimation.query import Query
    query = Query(sql, pred_columns, pred_vals, pred_types,
            true_val, total_count, pg_est)
    return query

def draw_graph(g, highlight_nodes=set(), color_nodes={}, bold_edges=[],
        save_to=None, node_attrs=[], node_shape="oval", edge_widths=None):
    '''
    ryan's version.
    '''
    g = g.copy()
    if highlight_nodes:
        for n in g.nodes:
            if n in highlight_nodes:
                g.nodes[n]["style"] = "filled"
                g.nodes[n]["fillcolor"] = "#FAED27"
    elif color_nodes:
        for n in g.nodes:
            g.nodes[n]["style"] = "filled"
            g.nodes[n]["fillcolor"] = color_nodes.get(n, "#FFFFFF")

    if node_shape != "oval":
        for n in g.nodes:
            g.nodes[n]["shape"] = node_shape

    if node_attrs:
        for n in g.nodes:
            # label = f"{n}\n"
            label = "{}\n".format(n)
            if len(node_attrs) == 1:
                # label += f"({g.nodes[n][node_attrs[0]]})"
                label += "({})".format(g.nodes[n][node_attrs[0]])
            else:
                for attr in node_attrs:
                    # label += f"{attr}: {g.nodes[n][attr]}"
                    label += "{}: {}".format(attr,g.nodes[n][attr])
            g.nodes[n]["label"] = label

    if bold_edges:
        for e in g.edges:
            if e in bold_edges:
                g.edges[e]["penwidth"] = "3"

    if edge_widths:
        for e in g.edges:
            if e in edge_widths:
                weight = edge_widths[e]
                pen_width = max(10*weight, 0.05)
                g.edges[e]["penwidth"] = pen_width

    A = to_agraph(g)
    if save_to:
        A.draw(save_to, prog="dot")

    display(Image(A.draw(format="png", prog="dot")))

def add_single_node_edges(subset_graph, source=None):

    global SOURCE_NODE
    if source is None:
        source = tuple("s")
    else:
        SOURCE_NODE = source

    # source = SOURCE_NODE
    # print(SOURCE_NODE)

    subset_graph.add_node(source)
    subset_graph.nodes()[source]["cardinality"] = {}
    subset_graph.nodes()[source]["cardinality"]["actual"] = 1.0

    for node in subset_graph.nodes():
        if len(node) != 1:
            continue
        if node[0] == source[0]:
            continue

        # print("going to add edge from source to node: ", node)
        subset_graph.add_edge(node, source, cost=0.0)
        in_edges = subset_graph.in_edges(node)
        out_edges = subset_graph.out_edges(node)
        # print("in edges: ", in_edges)
        # print("out edges: ", out_edges)

        # if we need to add edges between single table nodes and rest
        for node2 in subset_graph.nodes():
            if len(node2) != 2:
                continue
            if node[0] in node2:
                subset_graph.add_edge(node2, node)

def compute_costs(subset_graph, cost_model,
        cardinality_key, cost_key="cost", ests=None):
    '''
    @computes costs based on the MM1 cost model.
    '''
    total_cost = 0.0
    cost_key = cost_model + cost_key
    for edge in subset_graph.edges():
        if len(edge[0]) == len(edge[1]):
            # assert edge[1] == SOURCE_NODE
            subset_graph[edge[0]][edge[1]][cost_key] = 1.0
            continue
        assert len(edge[1]) < len(edge[0])
        # print(edge[0])
        # print(edge[1][0])
        # assert edge[1][0] in edge[0]
        ## FIXME:
        node1 = edge[1]
        diff = set(edge[0]) - set(edge[1])
        node2 = list(diff)
        node2.sort()
        node2 = tuple(node2)
        assert node2 in subset_graph.nodes()
        # joined node
        node3 = edge[0]
        cards1 = subset_graph.nodes()[node1][cardinality_key]
        cards2 = subset_graph.nodes()[node2][cardinality_key]
        cards3 = subset_graph.nodes()[edge[0]][cardinality_key]
        # if cards2["actual"] == 0:
            # print(cards2)
            # pdb.set_trace()

        if "total" in cards1 and "total" in cards2:
            total1 = cards1["total"]
            total2 = cards2["total"]
        else:
            total1 = None
            total2 = None

        if isinstance(ests, str):
            try:
                card1 = cards1[ests]
            except:
                assert node1 == tuple("s")
                card1 = 1.0

            try:
                card2 = cards2[ests]
            except:
                assert node2 == tuple("s")
                card2 = 1.0

            try:
                card3 = cards3[ests]
            except:
                print(cards3)
                pdb.set_trace()

        elif ests is None:
            card1 = cards1["actual"]
            card2 = cards2["actual"]
            card3 = cards3["actual"]
        else:
            assert isinstance(ests, dict)
            if node1 in ests:
                card1 = ests[node1]
                card2 = ests[node2]
                card3 = ests[node3]
            else:
                card1 = ests[" ".join(node1)]
                card2 = ests[" ".join(node2)]
                card3 = ests[" ".join(node3)]

        cost, edges_kind = get_costs(subset_graph, card1, card2, card3, node1, node2,
                cost_model, total1, total2)
        assert cost != 0.0

        subset_graph[edge[0]][edge[1]][cost_key] = cost
        # print(cost_key + "scan_type")
        subset_graph[edge[0]][edge[1]][cost_key + "scan_type"] = edges_kind

        total_cost += cost
    return total_cost

def get_costs(subset_graph, card1, card2, card3, node1, node2,
        cost_model, total1=None, total2=None):
    def update_edges_kind_with_seq(edges_kind, nilj_cost, cost2):
        if cost2 is not None and cost2 < nilj_cost:
            cost = cost2
            if len(node1) == 1:
                edges_kind["".join(node1)] = "Seq Scan"
            if len(node2) == 1:
                edges_kind["".join(node2)] = "Seq Scan"
        else:
            cost = nilj_cost
            if len(node1) == 1:
                edges_kind["".join(node1)] = "Index Scan"
                if len(node2) == 1:
                    edges_kind["".join(node2)] = "Seq Scan"
            elif len(node2) == 1:
                edges_kind["".join(node2)] = "Index Scan"
                if len(node1) == 1:
                    edges_kind["".join(node1)] = "Seq Scan"

    # assert card1 != 0.0
    # assert card2 != 0.0
    if card1 == 0:
        print(node1)
        pdb.set_trace()
    if card2 == 0:
        print(node2)
        pdb.set_trace()

    edges_kind = {}
    if cost_model == "cm1":
        if len(node1) == 1:
            nilj_cost = card2 + NILJ_CONSTANT*card1
        elif len(node2) == 1:
            nilj_cost = card1 + NILJ_CONSTANT*card2

        cost = nilj_cost

    elif cost_model == "cm2":
        cost = CARD_DIVIDER*card1 + CARD_DIVIDER*card2
    elif cost_model == "nested_loop_index":
        # TODO: calculate second multiple
        # joined_total = subset_graph.nodes()[joined_node]["cardinality"]["total"]
        if len(node1) == 1:
            # using index on node1
            ratio_mul = max(card3 / card2, 1.0)
            cost = NILJ_CONSTANT2*card2*ratio_mul
        elif len(node2) == 1:
            # using index on node2
            ratio_mul = max(card3 / card1, 1.0)
            cost = NILJ_CONSTANT2*card1*ratio_mul
        else:
            assert False
        update_edges_kind_with_seq(edges_kind, cost, 1e20)
        assert cost >= 1.0
    elif cost_model == "nested_loop_index2":
        if len(node1) == 1:
            # using index on node1
            cost = NILJ_CONSTANT2*card2
        elif len(node2) == 1:
            # using index on node2
            cost = NILJ_CONSTANT2*card1
        else:
            assert False
            # cost = card1*card2
        update_edges_kind_with_seq(edges_kind, cost, 1e20)
        assert cost >= 1.0
    elif cost_model == "nested_loop_index3":
        if len(node1) == 1:
            nilj_cost = card2 + NILJ_CONSTANT*card1 + card3
        elif len(node2) == 1:
            nilj_cost = card1 + NILJ_CONSTANT*card2 + card3
        else:
            assert False
        cost = nilj_cost
        update_edges_kind_with_seq(edges_kind, cost, 1e20)

    elif cost_model == "nested_loop_index4":
        # same as nested_loop_index, but also considering just joining the two
        # tables

        # because card3 for ratio_mul should be calculated without applying the
        # predicate on the table with the index, and we don't have that value,
        # we replace it with a const
        if len(node1) == 1:
            # using index on node1
            ratio_mul = max(card3 / card2, 1.0)
            # ratio_mul = ratio_mul*(total1 / card1)
            ratio_mul = ratio_mul*RATIO_MUL_CONST
            cost = NILJ_CONSTANT2*card2*ratio_mul
        elif len(node2) == 1:
            # using index on node2
            ratio_mul = max(card3 / card1, 1.0)
            # ratio_mul = ratio_mul*(total2 / card2)
            ratio_mul = ratio_mul*RATIO_MUL_CONST
            cost = NILJ_CONSTANT2*card1*ratio_mul
        else:
            assert False
            # cost = card1*card2

        # w/o indexes
        cost2 = card1*card2
        if cost2 < cost:
            cost = cost2
        update_edges_kind_with_seq(edges_kind, cost, cost2)

        assert cost >= 1.0
    elif cost_model == "nested_loop_index5":
        # same as nested_li4, BUT disallowing index joins if either of
        # the nodes have cardinality less than NILJ_MIN_CARD

        if card1 < NILJ_MIN_CARD or card2 < NILJ_MIN_CARD:
            cost = 1e10
        else:
            if len(node1) == 1:
                # using index on node1
                ratio_mul = max(card3 / card2, 1.0)
                # ratio_mul = ratio_mul*(total1 / card1)
                ratio_mul = ratio_mul*RATIO_MUL_CONST
                cost = NILJ_CONSTANT2*card2*ratio_mul
            elif len(node2) == 1:
                # using index on node2
                ratio_mul = max(card3 / card1, 1.0)
                # ratio_mul = ratio_mul*(total2 / card2)
                ratio_mul = ratio_mul*RATIO_MUL_CONST
                cost = NILJ_CONSTANT2*card1*ratio_mul
            else:
                assert False

        # w/o indexes
        cost2 = card1*card2
        if cost2 < cost or (card1 < NILJ_MIN_CARD or card2 < NILJ_MIN_CARD):
            cost = cost2
        update_edges_kind_with_seq(edges_kind, cost, cost2)

    elif cost_model == "nested_loop_index6":
        # want it like nested_loop5, BUT not depend on node3
        if card1 < NILJ_MIN_CARD or card2 < NILJ_MIN_CARD:
            cost = 1e10
        else:
            if card1 > card2:
                cost = NILJ_CONSTANT2*card1*RATIO_MUL_CONST
            else:
                cost = NILJ_CONSTANT2*card2*RATIO_MUL_CONST

        # w/o indexes
        cost2 = card1*card2
        if cost2 < cost or (card1 < NILJ_MIN_CARD or card2 < NILJ_MIN_CARD):
            cost = cost2

        update_edges_kind_with_seq(edges_kind, cost, cost2)

    elif cost_model == "nested_loop_index7" or \
            cost_model == "nested_loop_index8b":
        if len(node1) == 1:
            nilj_cost = card2 + NILJ_CONSTANT*card1
        elif len(node2) == 1:
            nilj_cost = card1 + NILJ_CONSTANT*card2
        else:
            assert False

        cost2 = card1*card2
        if cost2 < nilj_cost:
            cost = cost2
        else:
            cost = nilj_cost
        update_edges_kind_with_seq(edges_kind, nilj_cost, cost2)

    elif cost_model == "nested_loop_index7b":
        if len(node1) == 1:
            nilj_cost = card2 + NILJ_CONSTANT*card1
        elif len(node2) == 1:
            nilj_cost = card1 + NILJ_CONSTANT*card2
        else:
            assert False

        cost = nilj_cost
        update_edges_kind_with_seq(edges_kind, nilj_cost, None)

    elif cost_model == "nested_loop_index8":
            # or cost_model == "nested_loop_index8b":
        # same as nli7 --> but consider the fact the right side of an index
        # nested loop join WILL not have predicates pushed down
        # also, remove the term for index entirely
        if len(node1) == 1:
            # using index on node1
            # nilj_cost = card2 + NILJ_CONSTANT*card1
            nilj_cost = card2
            # expected output size, if node 1 did not have predicate pushed
            # down
            node1_selectivity = total1 / card1
            joined_node_est = card3 * node1_selectivity
            # nilj_cost += joined_node_est
            nilj_cost = max(joined_node_est, nilj_cost)
        elif len(node2) == 1:
            # using index on node2
            # nilj_cost = card1 + NILJ_CONSTANT*card2
            nilj_cost = card1
            node2_selectivity = total2 / card2
            joined_node_est = card3 * node2_selectivity
            # nilj_cost += joined_node_est
            nilj_cost = max(joined_node_est, nilj_cost)
        else:
            assert False
        cost2 = SEQ_CONST*card1*card2
        if cost2 < nilj_cost:
            cost = cost2
        else:
            cost = nilj_cost
        update_edges_kind_with_seq(edges_kind, nilj_cost, cost2)

    elif cost_model == "nested_loop_index8_debug":
        # same as nli7 --> but consider the fact the right side of an index
        # nested loop join WILL not have predicates pushed down
        # also, remove the term for index entirely
        if len(node1) == 1:
            # using index on node1
            # nilj_cost = card2 + NILJ_CONSTANT*card1
            nilj_cost = card2
            # expected output size, if node 1 did not have predicate pushed
            # down
            node1_selectivity = total1 / card1
            joined_node_est = card3 * node1_selectivity
            nilj_cost += joined_node_est
        elif len(node2) == 1:
            # using index on node2
            # nilj_cost = card1 + NILJ_CONSTANT*card2
            nilj_cost = card1
            node2_selectivity = total2 / card2
            joined_node_est = card3 * node2_selectivity
            nilj_cost += joined_node_est
        else:
            assert False

        if card1 < 10 or card2 < 10:
            cost2 = card1*card2
        else:
            cost2 = 10000.0*card1*card2
        # cost2 = SEQ_CONST*card1*card2
        if cost2 < nilj_cost:
            cost = cost2
            if len(node1) == 1:
                edges_kind["".join(node1)] = "Seq Scan"
            if len(node2) == 1:
                edges_kind["".join(node2)] = "Seq Scan"
        else:
            if len(node1) == 1:
                edges_kind["".join(node1)] = "Index Scan"
                if len(node2) == 1:
                    # edges_kind["".join(node2)] = "Seq Scan"
                    edges_kind["".join(node2)] = "Index Scan"
            elif len(node2) == 1:
                edges_kind["".join(node2)] = "Index Scan"
                if len(node1) == 1:
                    edges_kind["".join(node1)] = "Index Scan"
            cost = nilj_cost

    elif cost_model == "nested_loop_index9":
        # nli8, but no cost1*cost2 comparison
        if len(node1) == 1:
            # using index on node1
            # nilj_cost = card2 + NILJ_CONSTANT*card1
            nilj_cost = card2
            # expected output size, if node 1 did not have predicate pushed
            # down
            node1_selectivity = total1 / card1
            joined_node_est = card3 * node1_selectivity
            nilj_cost += joined_node_est
        elif len(node2) == 1:
            # using index on node2
            # nilj_cost = card1 + NILJ_CONSTANT*card2
            nilj_cost = card1
            node2_selectivity = total2 / card2
            joined_node_est = card3 * node2_selectivity
            nilj_cost += joined_node_est
        else:
            assert False
        cost = nilj_cost
        if len(node1) == 1:
            edges_kind["".join(node1)] = "Index Scan"
            if len(node2) == 1:
                edges_kind["".join(node2)] = "Seq Scan"
        elif len(node2) == 1:
            edges_kind["".join(node2)] = "Index Scan"

    elif cost_model == "nested_loop_index10":
        # debug one
        if len(node1) == 1:
            node1_selectivity = total1 / card1
            joined_node_est = card3 * node1_selectivity
            if joined_node_est > card2:
                nilj_cost = joined_node_est
            else:
                nilj_cost = card2

        elif len(node2) == 1:
            node2_selectivity = total2 / card2
            joined_node_est = card3 * node2_selectivity
            if joined_node_est > card1:
                nilj_cost = joined_node_est
            else:
                nilj_cost = card1
        else:
            assert False

        cost = nilj_cost

    elif cost_model == "nested_loop_index11":
        # ni10, but adds index_cost value
        if len(node1) == 1:
            index_cost = total1 / INDEX_COST_CONSTANT
            node1_selectivity = total1 / card1
            joined_node_est = card3 * node1_selectivity
            if joined_node_est > card2:
                nilj_cost = joined_node_est
            else:
                nilj_cost = card2
            nilj_cost *= index_cost

        elif len(node2) == 1:
            index_cost = total2 / INDEX_COST_CONSTANT
            node2_selectivity = total2 / card2
            joined_node_est = card3 * node2_selectivity
            if joined_node_est > card1:
                nilj_cost = joined_node_est
            else:
                nilj_cost = card1
            nilj_cost *= index_cost
        else:
            assert False

    elif cost_model == "nested_loop_index12":
        # mix nli8, nli3
        if len(node1) == 1:
            # using index on node1
            nilj_cost = card2 + NILJ_CONSTANT*card1
            node1_selectivity = total1 / card1
            joined_node_est = card3 * node1_selectivity
            nilj_cost += joined_node_est

        elif len(node2) == 1:
            # using index on node2
            nilj_cost = card1 + NILJ_CONSTANT*card2
            nilj_cost = card1
            node2_selectivity = total2 / card2
            joined_node_est = card3 * node2_selectivity
            nilj_cost += joined_node_est
        else:
            assert False

        # TODO: we may be doing fine without this one
        cost2 = card1*card2
        if cost2 < nilj_cost:
            cost = cost2
        else:
            cost = nilj_cost

    elif cost_model == "nested_loop_index13":
        # same as nli8, but additional conditions

        if len(node1) == 1:
            # using index on node1
            # nilj_cost = card2 + NILJ_CONSTANT*card1
            nilj_cost = card2
            node1_selectivity = total1 / card1
            joined_node_est = card3 * node1_selectivity
            nilj_cost += joined_node_est
            # expected output size, if node 1 did not have predicate pushed
            # down
            if "ci" in node1:
                if not ("t" in node2 or "mc" in node2 or "mi1" in node2 \
                        or "mi2" in node2 or "mii1" in node2):
                    nilj_cost *= INDEX_PENALTY_MULTIPLE

            # Non-primary key penalty
            if not ("t" in node1 \
                    or "n" in node1 \
                    or "k" in node1):
                nilj_cost *= INDEX_PENALTY_MULTIPLE

            # if "it" in node1[0] \
                # or "rt" in node1 \
                # or "kt" in node1:
                    # nilj_cost *= 10.0

        elif len(node2) == 1:
            # using index on node2
            # nilj_cost = card1 + NILJ_CONSTANT*card2
            nilj_cost = card1
            node2_selectivity = total2 / card2
            joined_node_est = card3 * node2_selectivity
            nilj_cost += joined_node_est
            if "ci" in node2:
                if not ("t" in node1 or "mc" in node1 or "mi1" in node1 \
                        or "mi2" in node1 or "mii1" in node1):
                    nilj_cost *= INDEX_PENALTY_MULTIPLE

            if not ("t" in node2 \
                    or "n" in node2 \
                    or "k" in node2):
                nilj_cost *= INDEX_PENALTY_MULTIPLE

        else:
            assert False

        cost2 = card1*card2
        if cost2 < nilj_cost:
            cost = cost2
        else:
            cost = nilj_cost
        # cost = nilj_cost

    elif cost_model == "nested_loop_index14":
        # same as nli7 --> but added term for index node as well
        cost2 = 0.0
        if len(node1) == 1:
            # using index on node1
            nilj_cost = card2 + NILJ_CONSTANT*card1
            cost2 += NILJ_CONSTANT*card1
            # expected output size, if node 1 did not have predicate pushed
            # down
            node1_selectivity = total1 / card1
            joined_node_est = card3 * node1_selectivity
            nilj_cost += joined_node_est

        elif len(node2) == 1:
            # using index on node2
            nilj_cost = card1 + NILJ_CONSTANT*card2
            cost2 += NILJ_CONSTANT*card1
            node2_selectivity = total2 / card2
            joined_node_est = card3 * node2_selectivity
            nilj_cost += joined_node_est
        else:
            assert False

        # TODO: we may be doing fine without this one
        cost2 += card1*card2
        if cost2 < nilj_cost:
            cost = cost2
        else:
            cost = nilj_cost

    elif cost_model == "nested_loop":
        cost = card1*card2
    elif cost_model == "hash_join":
        # skip multiplying with constants
        cost = CARD_DIVIDER*card1 + CARD_DIVIDER*card2
    else:
        assert False
    return cost, edges_kind

def constructG(subsetg, cost_key="cost"):
    '''
    @ret:
        G:
        i:
    '''
    N = len(subsetg.nodes())
    M = len(subsetg.edges())
    G = np.zeros((N,N))
    Q = np.zeros((M,N))
    Gv = np.zeros(N)

    node_dict = {}
    edge_dict = {}

    nodes = list(subsetg.nodes())
    nodes.sort()

    edges = list(subsetg.edges())
    edges.sort()
    for i, edge in enumerate(edges):
        edge_dict[edge] = i

    # node with all tables is source, node with no tables is target
    source_node = nodes[0]
    for i, node in enumerate(nodes):
        node_dict[node] = i
        if len(node) > len(source_node):
            source_node = node
    Gv[node_dict[source_node]] = 1
    target_node = tuple("s")
    Gv[node_dict[target_node]] = -1

    for i, node in enumerate(nodes):
        # going to set G[i,:]
        in_edges = subsetg.in_edges(node)
        out_edges = subsetg.out_edges(node)
        for edge in in_edges:
            assert edge[1] == node
            cost = subsetg[edge[0]][edge[1]][cost_key] / 10000.0
            # cost = subsetg[edge[0]][edge[1]]["cost"]
            cost = 1 / cost
            cur_node_idx = node_dict[edge[1]]
            other_node_idx = node_dict[edge[0]]
            G[i,cur_node_idx] += cost
            G[i,other_node_idx] -= cost

        for edge in out_edges:
            assert edge[0] == node
            cost = subsetg[edge[0]][edge[1]][cost_key] / 10000.0
            # cost = subsetg[edge[0]][edge[1]]["cost"]
            cost = 1 / cost
            cur_node_idx = node_dict[edge[0]]
            other_node_idx = node_dict[edge[1]]
            G[i,cur_node_idx] += cost
            G[i,other_node_idx] -= cost

    for i, edge in enumerate(edges):
        head_node = edge[0]
        tail_node = edge[1]
        hidx = node_dict[head_node]
        tidx = node_dict[tail_node]
        cost = subsetg[edge[0]][edge[1]][cost_key] / 10000.0
        cost = 1 / cost
        Q[i,hidx] = cost
        Q[i,tidx] = -cost

    return edges, G, Gv, Q

def construct_lp(subsetg, cost_key="cost"):
    '''
    @ret:
        list of node names
        node_names : idx
        edge_names : idx for the LP
        A: |V| x |E| matrix
        b: |V| matrix
        where the edges
    '''
    node_dict = {}
    edge_dict = {}
    b = np.zeros(len(subsetg.nodes()))
    A = np.zeros((len(subsetg.nodes()), len(subsetg.edges())))

    nodes = list(subsetg.nodes())
    nodes.sort()

    # node with all tables is source, node with no tables is target
    source_node = nodes[0]
    for i, node in enumerate(nodes):
        node_dict[node] = i
        if len(node) > len(source_node):
            source_node = node
    target_node = tuple("s")
    b[node_dict[source_node]] = 1
    b[node_dict[target_node]] = -1

    edges = list(subsetg.edges())
    edges.sort()
    for i, edge in enumerate(edges):
        edge_dict[edge] = i

    for ni, node in enumerate(nodes):
        in_edges = subsetg.in_edges(node)
        out_edges = subsetg.out_edges(node)
        for edge in in_edges:
            idx = edge_dict[edge]
            assert A[ni,idx] == 0.00
            A[ni,idx] = -1

        for edge in out_edges:
            idx = edge_dict[edge]
            assert A[ni,idx] == 0.00
            A[ni,idx] = 1

    G = np.eye(len(edges))
    G = -G
    h = np.zeros(len(edges))
    c = np.zeros(len(edges))
    # find cost of each edge
    for i, edge in enumerate(edges):
        c[i] = subsetg[edge[0]][edge[1]][cost_key]

    return edges, c, A, b, G, h

def get_subsetg_vectors(sample, cost_model, source_node=None):
    start = time.time()
    node_dict = {}
    # edge_dict = {}
    nodes = list(sample["subset_graph"].nodes())

    # if source_node is not None:
        # SOURCE_NODE = source_node

    if SOURCE_NODE in nodes:
        nodes.remove(SOURCE_NODE)

    nodes.sort()

    subsetg = sample["subset_graph"]
    join_graph = sample["join_graph"]
    edges = list(sample["subset_graph"].edges())
    edges.sort()

    N = len(nodes)
    num_edges = len(edges)
    if cost_model == "nested_loop_index8b":
        # separate edges for nested_loop v/s nested_loop_index
        M = len(edges)*2
    else:
        M = len(edges)

    totals = np.zeros(N, dtype=np.float32)
    edges_head = [0]*M
    edges_tail = [0]*M
    edges_cost_node1 = [0]*M
    edges_cost_node2 = [0]*M
    edges_penalties = [1]*M

    nilj = [0]*M
    nilj2 = [0]*M
    final_node = 0
    max_len_nodes = 0

    for nodei, node in enumerate(nodes):
        node_dict[node] = nodei
        # totals[nodei] = subsetg.nodes()[node]["cardinality"]["total"]
        if len(node) > max_len_nodes:
            max_len_nodes = len(node)
            final_node = nodei

    for edgei, edge in enumerate(edges):
        if len(edge[0]) == len(edge[1]):
            assert edge[1] == SOURCE_NODE
            edges_head[edgei] = node_dict[edge[0]]
            edges_tail[edgei] = SOURCE_NODE_CONST
            edges_cost_node1[edgei] = SOURCE_NODE_CONST
            edges_cost_node2[edgei] = SOURCE_NODE_CONST
            if cost_model == "nested_loop_index8b":
                edges_head[edgei+num_edges] = node_dict[edge[0]]
                edges_tail[edgei+num_edges] = SOURCE_NODE_CONST
                edges_cost_node1[edgei+num_edges] = SOURCE_NODE_CONST
                edges_cost_node2[edgei+num_edges] = SOURCE_NODE_CONST
            continue

        edges_head[edgei] = node_dict[edge[0]]
        edges_tail[edgei] = node_dict[edge[1]]

        if cost_model == "nested_loop_index8b":
            edges_head[edgei+num_edges] = node_dict[edge[0]]
            edges_tail[edgei+num_edges] = node_dict[edge[1]]

        assert len(edge[1]) < len(edge[0])
        assert edge[1][0] in edge[0]
        ## FIXME:
        node1 = edge[1]
        diff = set(edge[0]) - set(edge[1])
        node2 = list(diff)
        # node2.sort()
        node2 = tuple(node2)
        assert node2 in subsetg.nodes()

        edges_cost_node1[edgei] = node_dict[node1]
        edges_cost_node2[edgei] = node_dict[node2]
        if cost_model == "nested_loop_index8b":
            edges_cost_node1[edgei+num_edges] = node_dict[node1]
            edges_cost_node2[edgei+num_edges] = node_dict[node2]

        if len(node1) == 1:
            fkey_join = True

            join_edge_data = join_graph[node1[0]]
            for other_node in node2:
                if other_node not in join_edge_data:
                    continue
                jc = join_edge_data[other_node]["join_condition"]
                if "!=" in jc:
                    fkey_join = False
                    break

            if fkey_join:
                nilj[edgei] = 2
            else:
                nilj[edgei] = 3

        elif len(node2) == 1:
            fkey_join = True
            join_edge_data = join_graph[node2[0]]
            for other_node in node1:
                if other_node not in join_edge_data:
                    continue
                jc = join_edge_data[other_node]["join_condition"]
                if "!=" in jc:
                    fkey_join = False
                    break

            if fkey_join:
                nilj[edgei] = 2
            else:
                nilj[edgei] = 3

        if cost_model == "nested_loop_index8b":
            nilj[edgei+num_edges] = 3

        # penalties
        if cost_model == "nested_loop_index13":
            penalty = 1.0
            if len(node1) == 1:
                if "ci" in node1:
                    if not ("t" in node2 or "mc" in node2 or "mi1" in node2 \
                            or "mi2" in node2 or "mii1" in node2):
                        penalty *= INDEX_PENALTY_MULTIPLE

                # Non-primary key penalty
                if not ("t" in node1 \
                        or "n" in node1 \
                        or "k" in node1):
                    penalty *= INDEX_PENALTY_MULTIPLE

            elif len(node2) == 1:
                if "ci" in node2:
                    if not ("t" in node1 or "mc" in node1 or "mi1" in node1 \
                            or "mi2" in node1 or "mii1" in node1):
                        penalty *= INDEX_PENALTY_MULTIPLE

                if not ("t" in node2 \
                        or "n" in node2 \
                        or "k" in node2):
                    penalty *= INDEX_PENALTY_MULTIPLE
            edges_penalties[edgei] = penalty

    edges_head = np.array(edges_head, dtype=np.int32)
    edges_tail = np.array(edges_tail, dtype=np.int32)
    edges_cost_node1 = np.array(edges_cost_node1, dtype=np.int32)
    edges_cost_node2 = np.array(edges_cost_node2, dtype=np.int32)
    nilj = np.array(nilj, dtype=np.int32)
    edges_penalties = np.array(edges_penalties, dtype=np.float32)

    # print("get subsetg vectors took: ", time.time()-start)
    return totals, edges_head, edges_tail, nilj, \
            edges_cost_node1, edges_cost_node2, final_node, \
            edges_penalties

def get_subq_flows(qrep, cost_key):
    # TODO: save or not?
    start = time.time()
    flow_cache = klepto.archives.dir_archive("./flow_cache",
            cached=True, serialized=True)
    if cost_key != "cost":
        key = deterministic_hash(cost_key + qrep["sql"])
    else:
        key = deterministic_hash(qrep["sql"])

    if key in flow_cache.archive:
        return flow_cache.archive[key]

    # subsetg = qrep["subset_graph_paths"]
    subsetg = qrep["subset_graph"]
    edges, c, A, b, G, h = construct_lp(subsetg, cost_key)

    n = len(edges)
    P = np.zeros((len(edges),len(edges)))
    for i,c in enumerate(c):
        P[i,i] = c

    q = np.zeros(len(edges))
    x = cp.Variable(n)
    prob = cp.Problem(cp.Minimize((1/2)*cp.quad_form(x, P) + q.T @ x),
                     [G @ x <= h,
                      A @ x == b])
    prob.solve()
    qsolx = np.array(x.value)

    edge_dict = {}
    for i, e in enumerate(edges):
        edge_dict[e] = i

    flow_cache.archive[key] = (qsolx, edge_dict)
    return qsolx, edge_dict

def debug_flow_loss(sample, source_node, cost_key,
        cost_model, all_ests=None):
    start = time.time()
    subsetg_vectors = list(get_subsetg_vectors(sample))
    assert len(subsetg_vectors) == 7

    totals, edges_head, edges_tail, nilj, edges_cost_node1, \
            edges_cost_node2, final_node, edges_penalties = subsetg_vectors
    nodes = list(sample["subset_graph"].nodes())
    if SOURCE_NODE_CONST in nodes:
        nodes.remove(SOURCE_NODE)
    nodes.sort()

    assert all_ests is None
    true_cards = np.zeros(len(subsetg_vectors[0]),
            dtype=np.float32)

    for ni, node in enumerate(nodes):
        true_cards[ni] = \
                sample["subset_graph"].nodes()[node]["cardinality"]["actual"]

    trueC_vec, _, G2, Q2 = get_optimization_variables(true_cards, totals,
            0.0, 24.0, None, edges_cost_node1,
            edges_cost_node2, nilj, edges_head, edges_tail, cost_model,
            edges_penalties)

    Gv2 = np.zeros(len(totals), dtype=np.float32)
    Gv2[final_node] = 1.0
    Gv2 = to_variable(Gv2).float()
    G2 = to_variable(G2).float()
    invG = torch.inverse(G2)
    v = invG @ Gv2 # vshape: Nx1
    v = v.detach().numpy()
    invG = invG.detach().numpy()
    Gv2 = Gv2.detach().numpy()
    G2 = G2.detach().numpy()
    loss2 = np.zeros(1, dtype=np.float32)
    assert Q2.dtype == np.float32
    assert v.dtype == np.float32
    if isinstance(trueC_vec, torch.Tensor):
        trueC_vec = trueC_vec.detach().numpy()
    assert trueC_vec.dtype == np.float32
    fl_cpp.get_qvtqv(
            c_int(len(edges_head)),
            c_int(len(v)),
            edges_head.ctypes.data_as(c_void_p),
            edges_tail.ctypes.data_as(c_void_p),
            Q2.ctypes.data_as(c_void_p),
            v.ctypes.data_as(c_void_p),
            trueC_vec.ctypes.data_as(c_void_p),
            loss2.ctypes.data_as(c_void_p)
            )

    return loss2[0], trueC_vec, Q2, G2, Gv2, v, invG

def get_optimization_variables(ests, totals, min_val, max_val,
        normalization_type, edges_cost_node1, edges_cost_node2,
        nilj, edges_head, edges_tail, cost_model, edges_penalties):
    '''
    @ests: these are actual values for each estimate. totals,min_val,max_val
    are only required for the derivatives.
    '''
    start = time.time()

    # TODO: speed up this init stuff?
    if normalization_type is None:
        norm_type = 0
    elif normalization_type == "mscn":
        norm_type = 2
    else:
        # temporarily, screw this
        assert False
        norm_type = 1

    if cost_model == "cm1":
        cost_model_num = 1
    elif cost_model == "nested_loop_index":
        cost_model_num = 2
    elif cost_model == "nested_loop_index2":
        cost_model_num = 3
    elif cost_model == "nested_loop_index3":
        cost_model_num = 4
    elif cost_model == "nested_loop_index4":
        cost_model_num = 5
    elif cost_model == "nested_loop_index5":
        cost_model_num = 6
    elif cost_model == "nested_loop":
        cost_model_num = 7
    elif cost_model == "hash_join":
        cost_model_num = 8
    elif cost_model == "cm2":
        cost_model_num = 9
    elif cost_model == "nested_loop_index6":
        cost_model_num = 10
    elif cost_model == "nested_loop_index7":
        cost_model_num = 11
    elif cost_model == "nested_loop_index7b":
        cost_model_num = 1
    elif cost_model == "nested_loop_index8":
        cost_model_num = 12
    elif cost_model == "nested_loop_index8_debug":
        # temporary
        cost_model_num = 12
    elif cost_model == "nested_loop_index9":
        cost_model_num = 13
    elif cost_model == "nested_loop_index10":
        # debug cost model to calculate cost model loss
        # 1.9, 1.6; 1.14M, 0.96M
        cost_model_num = 14
        # print("C nested loop index 10 not implemented yet!")
    elif cost_model == "nested_loop_index11":
        # 2.3, 2.8; 2M, 4M;
        cost_model_num = 14
        # print("C nested loop index 10 not implemented yet!")
    elif cost_model == "nested_loop_index12":
        # 1.5, ; 0.86M, ;
        cost_model_num = 14
        # print("C nested loop index 10 not implemented yet!")
    elif cost_model == "nested_loop_index13":
        cost_model_num = 15
        # print("c nested loop index 10 not implemented yet!")
    elif cost_model == "nested_loop_index14":
        cost_model_num = 16
    elif cost_model == "nested_loop_index8b":
        cost_model_num = 17
    else:
        assert False

    # TODO: make sure everything is the correct type beforehand
    if min_val is None:
        min_val = 0.0
        max_val = 0.0

    if not isinstance(ests, np.ndarray):
        ests = ests.detach().cpu().numpy()

    costs2 = np.zeros(len(edges_cost_node1), dtype=np.float32)
    dgdxT2 = np.zeros((len(ests), len(edges_cost_node1)), dtype=np.float32)
    G2 = np.zeros((len(ests),len(ests)), dtype=np.float32)
    Q2 = np.zeros((len(edges_cost_node1),len(ests)), dtype=np.float32)

    assert ests.dtype == np.float32
    # if np.min(ests) < 1.0:
        # print("ests was < 1")
    ests = np.maximum(ests, 1.0)

    start = time.time()
    fl_cpp.get_optimization_variables(ests.ctypes.data_as(c_void_p),
            totals.ctypes.data_as(c_void_p),
            c_double(min_val),
            c_double(max_val),
            c_int(norm_type),
            edges_cost_node1.ctypes.data_as(c_void_p),
            edges_cost_node2.ctypes.data_as(c_void_p),
            edges_head.ctypes.data_as(c_void_p),
            edges_tail.ctypes.data_as(c_void_p),
            nilj.ctypes.data_as(c_void_p),
            edges_penalties.ctypes.data_as(c_void_p),
            c_int(len(ests)),
            c_int(len(costs2)),
            costs2.ctypes.data_as(c_void_p),
            dgdxT2.ctypes.data_as(c_void_p),
            G2.ctypes.data_as(c_void_p),
            Q2.ctypes.data_as(c_void_p),
            c_int(cost_model_num))

    return costs2, dgdxT2, G2, Q2

