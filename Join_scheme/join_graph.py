import numpy as np
import ast
import re


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
    end = None
    join = False
    join_keys = {}
    if ' IN ' in cond:
        s = cond.split(' IN ')
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
        return table, [attr, 'in', value], join, join_keys

    for i in range(len(cond)):
        s = cond[i]
        if s in OPS:
            start = i
            if cond[i + 1] in OPS:
                end = i + 2
            else:
                end = i + 1
            break
    assert start is not None and end is not None
    left = cond[:start].strip()
    ops = cond[start:end].strip()
    right = cond[end:].strip()
    table1 = left.split(".")[0].strip().lower()
    if tables_all:
        cond = cond.replace(table1 + ".", tables_all[table1] + ".")
        table1 = tables_all[table1]
        left = table1 + "." + left.split(".")[-1].strip()
    pattern = r"\b[a-zA-Z_]+\.[a-zA-Z_]+\b"
    if re.match(pattern, right):
        table2 = right.split(".")[0].strip().lower()
        if tables_all:
            cond = cond.replace(table2 + ".", tables_all[table2] + ".")
            table2 = tables_all[table2]
            right = table2 + "." + right.split(".")[-1].strip()
        join = True
        join_keys[table1] = left
        join_keys[table2] = right
        return table1 + " " + table2, cond, join, join_keys

    value = right
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


def parse_query_simple(query):
    """
    If your selection query contains no aggregation and nested sub-queries, you can use this function to parse a
    join query. Otherwise, use parse_query function.
    """
    query = query.replace(" where ", " WHERE ")
    query = query.replace(" from ", " FROM ")
    query = query.replace(" and ", " AND ")
    query = query.split(";")[0]
    query = query.strip()
    tables_all = {}
    join_cond = []
    table_cond = {}
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
        table, cond, join, join_key = process_condition(cond, tables_all)
        if not join:
            if table not in table_cond:
                table_cond[table] = [cond]
            else:
                table_cond[table].append(cond)
        else:
            join_cond.append(cond)
            for tab in join_key:
                if tab in join_keys:
                    join_keys[tab].add(join_key[tab])
                else:
                    join_keys[tab] = set([join_key[tab]])

    return tables_all, table_cond, join_cond, join_keys


def get_join_hyper_graph(join_keys, equivalent_keys):
    equivalent_group = dict()
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
                    seen = True
            if not seen:
                assert False, f"no equivalent groups found for {key}."
    return equivalent_group


def get_equivalent_key_group(join_keys, equivalent_keys):
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

