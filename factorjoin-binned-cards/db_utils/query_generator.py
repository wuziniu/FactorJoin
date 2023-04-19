from db_utils.utils import *

class QueryGenerator():
    '''
    Generates sql queries based on a template.
    TODO: explain rules etc.
    '''
    def __init__(self, query_template, user, db_host, port,
            pwd, db_name):
        self.user = user
        self.pwd = pwd
        self.db_host = db_host
        self.port = port
        self.db_name = db_name
        self.query_template = query_template
        # key: column_name, val: [vals]
        self.valid_pred_vals = {}

        # tune-able params
        self.max_in_vals = 15

    def gen_queries(self, num_samples, column_stats=None):
        '''
        @ret: [sql queries]
        '''
        start = time.time()
        # TODO: make these instance variables initialized in __init__
        pred_columns, pred_types, pred_strs = extract_predicates(self.query_template)
        from_clauses, aliases, tables = extract_from_clause(self.query_template)
        joins = extract_join_clause(self.query_template)
        all_query_strs = []

        while len(all_query_strs) < num_samples:
            gen_query = self.query_template
            # now, replace each predicate value 1 by 1
            for i, col in enumerate(pred_columns):
                pred_str = pred_strs[i]
                if pred_types[i] == "eq":
                    pass
                elif pred_types[i] == "in":
                    if not "SELECT" in pred_str[0]:
                        # leave this as is.
                        continue
                    pred_sql = pred_str[0]
                    if col not in self.valid_pred_vals:
                        # pred_sql should be a sql that we can execute
                        output = cached_execute_query(pred_sql, self.user,
                                self.db_host, self.port, self.pwd, self.db_name,
                                100, None, None)
                        self.valid_pred_vals[col] = output

                    min_val = 1
                    # replace pred_sql by a value from the valid ones
                    num_pred_vals = random.randint(min_val, self.max_in_vals)

                    # # find this many values randomly from the given col, and
                    # # update col_vals with it.
                    vals = []
                    for k in range(num_pred_vals):
                        val = random.choice(self.valid_pred_vals[col])
                        if val is not None:
                            # pdb.set_trace()
                            # vals.append("'{}'".format(val[0].replace("'","")))
                            vals.append("'{}'".format(str(val[0]).replace("'","")))
                    # "'{}'".format(random.choice(self.valid_pred_vals[col])[0].replace("'",""))
                    vals = [s for s in set(vals)]
                    vals.sort()
                    new_pred_str = ",".join(vals)
                    gen_query = gen_query.replace("'" + pred_sql + "'", new_pred_str)

                elif pred_types[i] == "lte" or pred_types[i] == "lt":
                    # print("going to sample for range query")
                    # pdb.set_trace()
                    # if not "SELECT" in str(pred_str[0]):
                        # # leave this as is.
                        # continue
                    assert len(pred_str) == 2
                    if col not in self.valid_pred_vals:
                        table = col[0:col.find(".")]
                        if table in aliases:
                            table = ALIAS_FORMAT.format(TABLE = aliases[table],
                                                ALIAS = table)

                        sel_query = SELECT_ALL_COL_TEMPLATE.format(COL = col,
                                                TABLE = table)
                        output = cached_execute_query(sel_query, self.user,
                                self.db_host, self.port, self.pwd, self.db_name,
                                100, None, None)
                        self.valid_pred_vals[col] = output

                    val1 = random.choice(self.valid_pred_vals[col])[0]
                    val2 = random.choice(self.valid_pred_vals[col])[0]
                    low_pred = "X" + col[col.find(".")+1:]
                    high_pred = "Y" + col[col.find(".")+1:]
                    low_val = str(min(val1, val2))
                    high_val = str(max(val1, val2))
                    gen_query = gen_query.replace(low_pred, low_val)
                    gen_query = gen_query.replace(high_pred, high_val)

            # print(gen_query)
            all_query_strs.append(gen_query)
        print("{} query strings took {} seconds to generate".format(len(all_query_strs),
            time.time()-start))
        return all_query_strs

