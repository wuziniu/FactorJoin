import ast
import numpy as np

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
    # only support equality join
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
        cond = cond.replace(table1 + ".", tables_all[table1] + ".")
        table1 = tables_all[table1]
        left = table1 + "." + left.split(".")[-1].strip()
    else:
        return None, None, False, None
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

