import sqlparse


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


def find_filter_clauses(table, wheres):
    matched = []
    # print(tables)
    index = 0
    while True:
        index, match = find_next_match(table, wheres, index)
        # print("got index, match: ", index)
        # print(match)
        if match is not None:
            matched.append(match)
        if index is None:
            break

    return matched


def parse_sql(sql):
    parsed = sqlparse.parse(sql)[0]
    from_clause = None
    from_seen = False
    where_clauses = None
    for token in parsed.tokens:
        if isinstance(token, sqlparse.sql.Where):
            where_clauses = token
    for token in parsed.tokens:
        if from_seen:
            if isinstance(token, sqlparse.sql.IdentifierList) or isinstance(token,
                                                                            sqlparse.sql.Identifier):
                from_clause = token
                break
        if token.value.upper() == 'FROM':
            from_seen = True

    tables = dict()
    for identifier in from_clause.get_identifiers():
        table_name = identifier.get_real_name()
        table_alias = identifier.get_alias()
        tables[table_name] = table_alias

    filter_conditions = dict()
    for table in tables:
        table_alias = tables[table]
        if table_alias is not None:
            matches = find_filter_clauses([table_alias], where_clauses)
            filter_conditions[table_alias] = matches
        else:
            matches = find_filter_clauses([table], where_clauses)
            filter_conditions[table] = matches
    return tables, filter_conditions
