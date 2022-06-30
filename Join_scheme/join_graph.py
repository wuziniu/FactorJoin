import sqlparse
from sqlparse.sql import IdentifierList, Identifier
from sqlparse.tokens import Keyword, DML
import time
import networkx as nx
import pdb
import ast
import numpy as np


ALIAS_FORMAT = "{TABLE} AS {ALIAS}"

OPS = {
    '>': np.greater,
    '<': np.less,
    '>=': np.greater_equal,
    '<=': np.less_equal,
    '=': np.equal,
    '==': np.equal
}


def process_condition(cond, tables_all=None):
    # parse a condition, either filter predicate or join operation
    start = None
    join = False
    join_keys = {}
    cond = cond.replace(" in ", " IN ")
    cond = cond.replace(" not in ", " NOT IN ")
    cond = cond.replace(" like ", " LIKE ")
    cond = cond.replace(" not like ", " NOT LIKE ")
    cond = cond.replace(" between ", " BETWEEN ")
    s = None
    ops = None

    if ' IN ' in cond:
        s = cond.split(' IN ')
        ops = "in"
    elif " NOT IN " in cond:
        s = cond.split(' NOT IN ')
        ops = "not in"
    elif " LIKE " in cond:
        s = cond.split(' LIKE ')
        ops = "like"
    elif " NOT LIKE " in cond:
        s = cond.split(' NOT LIKE ')
        ops = "not like"
    elif " BETWEEN " in cond:
        s = cond.split(' BETWEEN ')
        ops = "between"
    elif " IS " in cond:
        s = cond.split(' IS ')
        ops = OPS["="]

    if ' IN ' in cond or " NOT IN " in cond:
        attr = s[0].strip()
        try:
            value = list(ast.literal_eval(s[1].strip()))
        except:
            temp_value = s[1].strip()[1:][:-1].split(',')
            value = []
            for v in temp_value:
                value.append(v.strip())
        if tables_all:
            table = tables_all[attr.split(".")[0].strip()]
            attr = table + "." + attr.split(".")[-1].strip()
        else:
            table = attr.split(".")[0].strip()
        return table, [attr, ops, value], join, join_keys

    elif s is not None:
        attr = s[0].strip()
        value = s[1].strip()
        if tables_all:
            table = tables_all[attr.split(".")[0].strip()]
            attr = table + "." + attr.split(".")[-1].strip()
        else:
            table = attr.split(".")[0].strip()
        return table, [attr, ops, value], join, join_keys

    for i in range(len(cond)):
        s = cond[i]
        if s in OPS:
            start = i
            if cond[i + 1] in OPS:
                end = i + 2
            else:
                end = i + 1
            break

    if start is None:
        return None, [None, None, None], join, join_keys
    assert start is not None
    left = cond[:start].strip()
    ops = cond[start:end].strip()
    right = cond[end:].strip()
    table1 = left.split(".")[0].strip().lower()
    if tables_all:
        cond = cond.replace(table1 + ".", tables_all[table1] + ".")
        table1 = tables_all[table1]
        left = table1 + "." + left.split(".")[-1].strip()
    if "." in right:
        table2 = right.split(".")[0].strip().lower()
        if table2 in tables_all:
            cond = cond.replace(table2 + ".", tables_all[table2] + ".")
            table2 = tables_all[table2]
            right = table2 + "." + right.split(".")[-1].strip()
            join = True
            join_keys[table1] = left
            join_keys[table2] = right
            return table1 + " " + table2, cond, join, join_keys

    value = right.strip()
    if value[0] == "'" and value[-1] == "'":
        value = value[1:-1]
    try:
        value = list(ast.literal_eval(value.strip()))
    except:
        try:
            value = int(value)
        except:
            try:
                value = float(value)
            except:
                value = value

    return table1, [left, ops, value], join, join_keys


def process_condition_join(cond, tables_all):
    start = None
    join = False
    join_keys = {}
    for i in range(len(cond)):
        s = cond[i]
        if s == "=":
            start = i
            if cond[i + 1] == "=":
                end = i + 2
            else:
                end = i + 1
            break

    if start is None:
        return None, None, False, None

    left = cond[:start].strip()
    ops = cond[start:end].strip()
    right = cond[end:].strip()
    table1 = left.split(".")[0].strip().lower()
    if table1 in tables_all:
        left = tables_all[table1] + "." + left.split(".")[-1].strip()
    else:
        return None, None, False, None
    if "." in right:
        table2 = right.split(".")[0].strip().lower()
        if table2 in tables_all:
            right = tables_all[table2] + "." + right.split(".")[-1].strip()
            join = True
            join_keys[table1] = left
            join_keys[table2] = right
            return table1 + " " + table2, cond, join, join_keys
    return None, None, False, None


def parse_query_all_join(query):
    """
    This function will parse out all join conditions from the query.
    """
    query = query.replace(" where ", " WHERE ")
    query = query.replace(" from ", " FROM ")
    # query = query.replace(" and ", " AND ")
    query = query.split(";")[0]
    query = query.strip()
    tables_all = {}
    join_cond = {}
    join_keys = {}
    tables_str = query.split(" WHERE ")[0].split(" FROM ")[-1]
    for table_str in tables_str.split(","):
        table_str = table_str.strip()
        if " as " in table_str:
            tables_all[table_str.split(" as ")[-1]] = table_str.split(" as ")[0]
        else:
            tables_all[table_str.split(" ")[-1]] = table_str.split(" ")[0]
    # processing conditions
    conditions = query.split(" WHERE ")[-1].split(" AND ")
    for cond in conditions:
        cond = cond.strip()
        if cond[0] == "(" and cond[-1] == ")":
            cond = cond[1:-1]
        table, cond, join, join_key = process_condition_join(cond, tables_all)
        if join:
            for tab in join_key:
                if tab in join_keys:
                    join_keys[tab].add(join_key[tab])
                    join_cond[tab].add(cond)
                else:
                    join_keys[tab] = set([join_key[tab]])
                    join_cond[tab] = set([cond])

    return tables_all, join_cond, join_keys


def get_join_hyper_graph(join_keys, equivalent_keys):
    equivalent_group = dict()
    table_equivalent_group = dict()
    table_key_equivalent_group = dict()
    table_key_group_map = dict()
    for table in join_keys:
        for key in join_keys[table]:
            seen = False
            for indicator in equivalent_keys:
                if key in equivalent_keys[indicator]:
                    if seen:
                        assert False, f"{key} appears in multiple equivalent groups."
                    if indicator not in equivalent_group:
                        equivalent_group[indicator] = [key]
                    else:
                        equivalent_group[indicator].append(key)
                    if table not in table_key_equivalent_group:
                        table_key_equivalent_group[table] = dict()
                        table_equivalent_group[table] = set([indicator])
                        table_key_group_map[table] = dict()
                        table_key_group_map[table][key] = indicator
                    else:
                        table_equivalent_group[table].add(indicator)
                        table_key_group_map[table][key] = indicator
                    if indicator not in table_key_equivalent_group[table]:
                        table_key_equivalent_group[table][indicator] = [key]
                    else:
                        table_key_equivalent_group[table][indicator].append(key)

                    seen = True
            if not seen:
                assert False, f"no equivalent groups found for {key}."
    return equivalent_group, table_equivalent_group, table_key_equivalent_group, table_key_group_map

def parse_query_all_single_table(query):
    return


def parse_sub_plan_queries(psql_raw_file):
    with open(psql_raw_file, "r") as f:
        psql_raw = f.read()
    sub_plan_queries_raw = psql_raw.split("query: 0")[1:]
    sub_plan_queries_str_all = []
    for per_query in sub_plan_queries_raw:
        sub_plan_queries_str = []
        num_sub_plan_queries = len(per_query.split("query: "))
        all_info = per_query.split("RELOPTINFO (")[1:]
        assert num_sub_plan_queries * 2 == len(all_info)
        for i in range(num_sub_plan_queries):
            idx = i * 2
            table1 = all_info[idx].split("): rows=")[0]
            table2 = all_info[idx + 1].split("): rows=")[0]
            table_str = table1 + " " + table2
            sub_plan_queries_str.append(table_str)
        sub_plan_queries_str_all.append(sub_plan_queries_str)
    return sub_plan_queries_str_all


def extract_join_clause(query):
    '''
    FIXME: this can be optimized further / or made to handle more cases
    '''
    parsed = sqlparse.parse(query)[0]
    # let us go over all the where clauses
    start = time.time()
    where_clauses = None
    for token in parsed.tokens:
        if (type(token) == sqlparse.sql.Where):
            where_clauses = token
    if where_clauses is None:
        return []
    join_clauses = []

    froms, aliases, table_names = extract_from_clause(query)
    if len(aliases) > 0:
        tables = [k for k in aliases]
    else:
        tables = table_names
    matches = find_all_clauses(tables, where_clauses)

    for match in matches:
        if "=" not in match or match.count("=") > 1:
            continue
        if "<=" in match or ">=" in match:
            continue
        match = match.replace(";", "")
        if "!" in match:
            left, right = match.split("!=")
            if "." in right:
                # must be a join, so add it.
                join_clauses.append(left.strip() + " != " + right.strip())
            continue
        left, right = match.split("=")
        # ugh dumb hack
        if "." in right:
            # must be a join, so add it.
            join_clauses.append(left.strip() + " = " + right.strip())

    return join_clauses


def extract_from_clause(query):
    '''
    Optimized version using sqlparse.
    Extracts the from statement, and the relevant joins when there are multiple
    tables.
    @ret: froms:
          froms: [alias1, alias2, ...] OR [table1, table2,...]
          aliases:{alias1: table1, alias2: table2} (OR [] if no aliases present)
          tables: [table1, table2, ...]
    '''
    def handle_table(identifier):
        table_name = identifier.get_real_name()
        alias = identifier.get_alias()
        tables.append(table_name)
        if alias is not None:
            from_clause = ALIAS_FORMAT.format(TABLE = table_name,
                                ALIAS = alias)
            froms.append(from_clause)
            aliases[alias] = table_name
        else:
            froms.append(table_name)

    start = time.time()
    froms = []
    # key: alias, val: table name
    aliases = {}
    # just table names
    tables = []

    start = time.time()
    parsed = sqlparse.parse(query)[0]
    # let us go over all the where clauses
    from_token = None
    from_seen = False
    for token in parsed.tokens:
        if from_seen:
            if isinstance(token, IdentifierList) or isinstance(token,
                    Identifier):
                from_token = token
                break
        if token.ttype is Keyword and token.value.upper() == 'FROM':
            from_seen = True
    assert from_token is not None
    if isinstance(from_token, IdentifierList):
        for identifier in from_token.get_identifiers():
            handle_table(identifier)
    elif isinstance(from_token, Identifier):
        handle_table(from_token)
    else:
        assert False

    return froms, aliases, tables


def extract_join_graph(sql):
    '''
    @sql: string
    '''
    froms, aliases, tables = extract_from_clause(sql)
    joins = extract_join_clause(sql)
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

    parsed = sqlparse.parse(sql)[0]
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

    return join_graph


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
        if token.value.upper() == "AND":
            break

        match += " " + token.value

        if (token.value.upper() == "BETWEEN"):
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




