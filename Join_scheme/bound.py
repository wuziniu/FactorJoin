import numpy as np
import copy

from Join_scheme.join_graph import process_condition, get_join_hyper_graph, parse_query_all_join
from Join_scheme.data_prepare import identify_key_values
from BayesCard.Evaluation.cardinality_estimation import timestamp_transorform, construct_table_query
from Sampling.load_sample import load_sample_imdb_one_query



class Factor:
    """
    This the class defines a multidimensional conditional probability.
    """
    def __init__(self, table=None, table_len=None, variables=None, pdfs=None, equivalent_variables=None, na_values=None):
        self.table = table
        self.table_len = table_len
        self.variables = variables
        self.equivalent_variables = equivalent_variables
        self.pdfs = pdfs
        self.cardinalities = dict()
        for i, var in enumerate(self.variables):
            if type(pdfs) == dict:
                self.cardinalities[var] = len(pdfs[var])
                if equivalent_variables and len(equivalent_variables) != 0:
                    self.cardinalities[equivalent_variables[i]] = pdfs[var]
            else:
                self.cardinalities[var] = pdfs.shape[i]
                if equivalent_variables and len(equivalent_variables) != 0:
                    self.cardinalities[equivalent_variables[i]] = pdfs.shape[i]
        self.na_values = na_values  # the percentage of data, which is not nan, so the variable name is misleading.


class Group_Factor:
    """
        This the class defines a multidimensional conditional probability on a group of tables.
    """
    def __init__(self, tables, tables_size, variables, pdfs, bin_modes, equivalent_groups=None,
                 table_key_equivalent_group=None, na_values=None, join_cond=None):
        self.table = tables
        self.tables_size = tables_size
        self.variables = variables
        self.pdfs = pdfs
        self.bin_modes = bin_modes
        self.equivalent_groups = equivalent_groups
        self.table_key_equivalent_group = table_key_equivalent_group
        self.na_values = na_values
        self.join_cond = join_cond

class Bound_ensemble:
    """
    This the class where we store all the trained models and perform inference on the bound.
    """
    def __init__(self, table_buckets, schema, n_dim_dist=1, ground_truth_factors_no_filter=None, SPERCENTAGE=None,
                 query_sample_location=None, bns=None, null_value=None):
        """
        The current implementation is a bit hacky, that some features are hardcoded to the
            STATS-CEB and IMDB-JOB workload.
        :param table_buckets: [of type dict of Table_bucket] contain all MVF information
                              (see data_prepare.generate_table_buckets for more information)
        :param schema: the schema of the database (see Schemas)
        :param n_dim_dist: corresponding to section 5.1 of the paper (the Bayesian decomposition of multidimensional
                           data distribution.)
                           Currently, this only supports 1 (independent) or 2 (tree-structure Bayesian factorization).
        :param ground_truth_factors_no_filter: In the case where a table has no filter (which is very common),
                                               we can just load the pre-computed data distribution.
        :param SPERCENTAGE: The sampling rate the training data.
        :param query_sample_location: The location of stored sample: this is a bit hacky, we should make it
                                      online sampling. Set to None if you are using BN.
        :param bns: The diction of bayesian network. Set to None if you are using sampling.
        :param null_value: the percentage of na values.
        """
        self.table_buckets = table_buckets
        self.schema = schema
        self.n_dim_dist = n_dim_dist
        assert n_dim_dist in [1, 2], f"got n_dim_dist to be {n_dim_dist}, we can only support it to be 1 or 2"
        self.all_join_conds = None
        self.ground_truth_factors_no_filter = ground_truth_factors_no_filter
        self.SPERCENTAGE = SPERCENTAGE
        self.query_sample_location = query_sample_location
        self.all_keys, self.equivalent_keys = identify_key_values(schema)
        self.bns = bns
        self.null_value = null_value

    def parse_query_simple(self, query):
        """
        If your selection query contains no aggregation and nested sub-queries, you can use this function to parse a
        join query. Otherwise, use parse_query function.
        """
        if self.bns is None:
            # this is a hack, since we currently only implemented the bayescard and sampling for single table estimate
            tables_all, join_cond, join_keys = parse_query_all_join(query)
            # TODO: implement functions on parsing filter conditions.
            table_filters = dict()
            return tables_all, table_filters, join_cond, join_keys

        query = query.replace(" where ", " WHERE ")
        query = query.replace(" from ", " FROM ")
        query = query.replace(" and ", " AND ")
        query = query.split(";")[0]
        query = query.strip()
        tables_all = {}
        join_cond = []
        table_query = {}
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
                attr = cond[0]
                op = cond[1]
                value = cond[2]
                if "Date" in attr:
                    assert "::timestamp" in value  # this is hardcoded for STATS-CEB workload
                    value = timestamp_transorform(value.strip().split("::timestamp")[0])
                if table not in table_query:
                    table_query[table] = dict()
                construct_table_query(self.bns[table], table_query[table], attr, op, value)
            else:
                join_cond.append(cond)
                for tab in join_key:
                    if tab in join_keys:
                        join_keys[tab].add(join_key[tab])
                    else:
                        join_keys[tab] = set([join_key[tab]])

        return tables_all, table_query, join_cond, join_keys


    def get_all_id_conidtional_distribution_sample(self, query_file_name, tables_alias, join_keys):
        if self.query_sample_location is not None:
            return load_sample_imdb_one_query(self.table_buckets, tables_alias, query_file_name, join_keys,
                                              self.ground_truth_factors_no_filter, self.SPERCENTAGE,
                                              self.query_sample_location)
        else:
            # TODO: sample on the fly
            return

    def get_all_id_conidtional_distribution_bn(self, table_queries, join_keys, equivalent_group):
        res = dict()
        for table in join_keys:
            key_attrs = list(join_keys[table])
            if table in table_queries:
                table_query = table_queries[table]
            else:
                table_query = {}
            id_attrs, probs = self.bns[table].query_id_prob(table_query, key_attrs)
            new_id_attrs = []
            for K in id_attrs:
                for PK in equivalent_group:
                    if K in equivalent_group[PK]:
                        new_id_attrs.append(PK)
            assert len(new_id_attrs) == len(id_attrs)
            res[table] = Factor(variables=id_attrs, pdfs=probs, equivalent_variables=new_id_attrs)
        return res

    def eliminate_one_key_group(self, tables, key_group, factors, relevant_keys):
        """This version only supports 2D distributions (i.e. the distribution learned with tree-structured PGM)"""
        rest_group = None
        rest_group_cardinalty = 0
        eliminated_tables = []
        rest_group_tables = []
        for table in tables:
            assert key_group in factors[table].equivalent_variables
            temp = copy.deepcopy(factors[table].equivalent_variables)
            temp.remove(key_group)
            if len(temp) == 0:
                eliminated_tables.append(table)
            for key in temp:
                if rest_group:
                    assert factors[table].cardinalities[key] == rest_group_cardinalty
                    rest_group_tables.append(table)
                else:
                    rest_group = key
                    rest_group_cardinalty = factors[table].cardinalities[key]
                    rest_group_tables = [table]

        all_probs_eliminated = []
        all_modes_eliminated = []
        for table in eliminated_tables:
            bin_modes = self.table_buckets[table].oned_bin_modes[relevant_keys[key_group][table]]
            all_probs_eliminated.append(factors[table].pdfs)
            all_modes_eliminated.append(np.minimum(bin_modes, factors[table].pdfs))
        if rest_group:
            new_factor_pdf = np.zeros(rest_group_cardinalty)
        else:
            return self.compute_bound_oned(all_probs_eliminated, all_modes_eliminated)

        for i in range(rest_group_cardinalty):
            rest_group_probs_eliminated = []
            rest_group_modes_eliminated = []
            for table in rest_group_tables:

                idx_f = factors[table].equivalent_variables.index(key_group)
                idx_b = self.table_buckets[table].id_attributes.index(relevant_keys[key_group][table])
                bin_modes = self.table_buckets[table].twod_bin_modes[relevant_keys[key_group][table]]
                if idx_f == 0 and idx_b == 0:
                    rest_group_probs_eliminated.append(factors[table].pdfs[:, i])
                    rest_group_modes_eliminated.append(np.minimum(bin_modes[:, i], factors[table].pdfs[:, i]))
                elif idx_f == 0 and idx_b == 1:
                    rest_group_probs_eliminated.append(factors[table].pdfs[:, i])
                    rest_group_modes_eliminated.append(np.minimum(bin_modes[i, :], factors[table].pdfs[:, i]))
                elif idx_f == 1 and idx_b == 0:
                    rest_group_probs_eliminated.append(factors[table].pdfs[i, :])
                    rest_group_modes_eliminated.append(np.minimum(bin_modes[:, i], factors[table].pdfs[i, :]))
                else:
                    rest_group_probs_eliminated.append(factors[table].pdfs[i, :])
                    rest_group_modes_eliminated.append(np.minimum(bin_modes[i, :], factors[table].pdfs[i, :]))
            new_factor_pdf[i] = self.compute_bound_oned(all_probs_eliminated + rest_group_probs_eliminated,
                                                        all_modes_eliminated + rest_group_modes_eliminated)

        for table in rest_group_tables:
            factors[table] = Factor(variables=[rest_group], pdfs=new_factor_pdf, equivalent_variables=[rest_group])

        return None

    def compute_bound_oned(self, all_probs, all_modes, return_factor=False):
        temp_all_modes = []
        for i in range(len(all_modes)):
            temp_all_modes.append(np.minimum(all_probs[i], all_modes[i]))
        all_probs = np.stack(all_probs, axis=0)
        temp_all_modes = np.stack(temp_all_modes, axis=0)
        multiplier = np.prod(all_modes, axis=0)
        non_zero_idx = np.where(multiplier != 0)[0]
        min_number = np.amin(all_probs[:, non_zero_idx] / temp_all_modes[:, non_zero_idx], axis=0)
        if return_factor:
            new_probs = np.zeros(multiplier.shape)
            new_probs[non_zero_idx] = multiplier[non_zero_idx] * min_number
            return new_probs, multiplier
        else:
            multiplier[non_zero_idx] = multiplier[non_zero_idx] * min_number
            return np.sum(multiplier)

    def get_optimal_elimination_order(self, equivalent_group, join_keys, factors):
        cardinalities = dict()
        lengths = dict()
        tables_involved = dict()
        relevant_keys = dict()
        for group in equivalent_group:
            relevant_keys[group] = dict()
            lengths[group] = len(equivalent_group[group])
            cardinalities[group] = []
            tables_involved[group] = set([])
            for keys in equivalent_group[group]:
                for table in join_keys:
                    if keys in join_keys[table]:
                        cardinalities[group].append(len(join_keys[table]))
                        tables_involved[group].add(table)
                        variables = factors[table].variables
                        variables[variables.index(keys)] = group
                        factors[table].variables = variables
                        relevant_keys[group][table] = keys
                        break
            cardinalities[group] = np.asarray(cardinalities[group])

        optimal_order = list(equivalent_group.keys())
        for i in range(len(optimal_order)):
            min_idx = i
            for j in range(i+1, len(optimal_order)):
                min_group = optimal_order[min_idx]
                curr_group = optimal_order[j]
                if np.max(cardinalities[curr_group]) < np.max(cardinalities[min_group]):
                    min_idx = j
                else:
                    min_max_tables = np.max(cardinalities[min_group])
                    min_num_max_tables = len(np.where(cardinalities[min_group] == min_max_tables)[0])
                    curr_max_tables = np.max(cardinalities[curr_group])
                    curr_num_max_tables = len(np.where(cardinalities[curr_group] == curr_max_tables)[0])
                    if curr_num_max_tables < min_num_max_tables:
                        min_idx = j
                    elif lengths[curr_group] < lengths[min_group]:
                        min_idx = j
            optimal_order[i], optimal_order[min_idx] = optimal_order[min_idx], optimal_order[i]
        return optimal_order, tables_involved, relevant_keys

    def get_cardinality_bound_one(self, query_str, query_name=None):
        tables_all, table_queries, join_cond, join_keys = self.parse_query_simple(query_str)
        equivalent_group = get_join_hyper_graph(join_keys, self.equivalent_keys)
        if self.bns is not None:
            conditional_factors = self.get_all_id_conidtional_distribution_bn(table_queries, join_keys, equivalent_group)
        else:
            conditional_factors = self.get_all_id_conidtional_distribution_sample(query_name, tables_all, join_keys)
        optimal_order, tables_involved, relevant_keys = self.get_optimal_elimination_order(equivalent_group, join_keys,
                                                                                           conditional_factors)
        res = None
        for key_group in optimal_order:
            tables = tables_involved[key_group]
            res = self.eliminate_one_key_group(tables, key_group, conditional_factors, relevant_keys)
        return res

    def get_sub_plan_queries_sql(self, query_str, sub_plan_query_str_all, query_name=None):
        tables_all, table_queries, join_cond, join_keys = self.parse_query_simple(query_str)
        equivalent_group, table_equivalent_group, table_key_equivalent_group = get_join_hyper_graph(join_keys,
                                                                                                    self.equivalent_keys)
        cached_sub_queries_sql = dict()
        cached_union_key_group = dict()
        res_sql = []
        for (left_tables, right_tables) in sub_plan_query_str_all:
            assert " " not in left_tables, f"{left_tables} contains more than one tables, violating left deep plan"
            sub_plan_query_list = right_tables.split(" ") + [left_tables]
            sub_plan_query_list.sort()
            sub_plan_query_str = " ".join(sub_plan_query_list)  # get the string name of the sub plan query
            sql_header = "SELECT COUNT(*) FROM "
            for alias in sub_plan_query_list:
                sql_header += (tables_all[alias] + " AS " + alias + ", ")
            sql_header = sql_header[:-2] + " WHERE "
            if " " in right_tables:
                assert right_tables in cached_sub_queries_sql, f"{right_tables} not in cache, input is not ordered"
                right_sql = cached_sub_queries_sql[right_tables]
                right_union_key_group = cached_union_key_group[right_tables]
                if left_tables in table_queries:
                    left_sql = table_queries[left_tables]
                    curr_sql = right_sql + " AND (" + left_sql + ")"
                else:
                    curr_sql = right_sql
                additional_joins, union_key_group = self.get_additional_join_with_table_group(left_tables,
                                                                                              right_union_key_group,
                                                                                              table_equivalent_group,
                                                                                              table_key_equivalent_group)
                for join in additional_joins:
                    curr_sql = curr_sql + " AND " + join
            else:
                curr_sql = ""
                if left_tables in table_queries:
                    curr_sql += ("(" + table_queries[left_tables] + ")")
                if right_tables in table_queries:
                    if curr_sql != "":
                        curr_sql += " AND "
                    curr_sql += ("(" + table_queries[right_tables] + ")")

                additional_joins, union_key_group = self.get_additional_joins_two_tables(left_tables, right_tables,
                                                                                         table_equivalent_group,
                                                                                         table_key_equivalent_group)
                for join in additional_joins:
                    if curr_sql == "":
                        curr_sql += join
                    else:
                        curr_sql = curr_sql + " AND " + join
            cached_sub_queries_sql[sub_plan_query_str] = curr_sql
            cached_union_key_group[sub_plan_query_str] = union_key_group
            res_sql.append(sql_header + curr_sql + ";")
        return res_sql

    def get_cardinality_bound_all(self, query_str, sub_plan_query_str_all, query_name=None, debug=False,
                                  true_card=None):
        """
        Get the cardinality bounds for all sub_plan_queires of a query.
        Note: Due to efficiency, this current version only support left_deep plans (like the one generated by postgres),
              but it can easily support right deep or bushy plans.
        :param query_str: the target query
        :param sub_plan_query_str_all: all sub_plan_queries of the target query,
               it should be sorted by number of the tables in the sub_plan_query
        """
        tables_all, table_queries, join_cond, join_keys = self.parse_query_simple(query_str)
        equivalent_group, table_equivalent_group, table_key_equivalent_group, table_key_group_map = \
            get_join_hyper_graph(join_keys, self.equivalent_keys)
        if self.bns is not None:
            conditional_factors = self.get_all_id_conidtional_distribution_bn(table_queries, join_keys, equivalent_group)
        else:
            conditional_factors = self.get_all_id_conidtional_distribution_sample(query_name, tables_all, join_keys)
        # self.reverse_table_alias = {v: k for k, v in tables_all.items()}
        cached_sub_queries = dict()
        cardinality_bounds = []
        for i, (left_tables, right_tables) in enumerate(sub_plan_query_str_all):
            assert " " not in left_tables, f"{left_tables} contains more than one tables, violating left deep plan"
            sub_plan_query_list = right_tables.split(" ") + [left_tables]
            sub_plan_query_list.sort()
            sub_plan_query_str = " ".join(sub_plan_query_list)  # get the string name of the sub plan query
            if " " in right_tables:
                assert right_tables in cached_sub_queries, f"{right_tables} not in cache, input is not ordered"
                right_bound_factor = cached_sub_queries[right_tables]
                curr_bound_factor, res = self.join_with_one_table(sub_plan_query_str,
                                                                  left_tables,
                                                                  tables_all,
                                                                  right_bound_factor,
                                                                  conditional_factors[left_tables],
                                                                  table_equivalent_group,
                                                                  table_key_equivalent_group,
                                                                  table_key_group_map,
                                                                  join_cond)
            else:
                curr_bound_factor, res = self.join_two_tables(sub_plan_query_str,
                                                              left_tables,
                                                              right_tables,
                                                              tables_all,
                                                              conditional_factors,
                                                              join_keys,
                                                              table_equivalent_group,
                                                              table_key_equivalent_group,
                                                              table_key_group_map,
                                                              join_cond)
            cached_sub_queries[sub_plan_query_str] = curr_bound_factor
            res = max(res, 1)
            if debug:
                if true_card[i] == -1:
                    error = "NA"
                else:
                    error = max(res / true_card[i], true_card[i] / res)
                print(error)
            cardinality_bounds.append(res)
        return cardinality_bounds

    def join_with_one_table(self, sub_plan_query_str, left_table, tables_all, right_bound_factor, cond_factor_left,
                            table_equivalent_group, table_key_equivalent_group, table_key_group_map, join_cond):
        """
        Get the cardinality bound by joining the left_table with the seen right_tables
        """
        equivalent_key_group, union_key_group_set, union_key_group, new_join_cond = \
            self.get_join_keys_with_table_group(left_table, right_bound_factor, tables_all, table_equivalent_group,
                                                table_key_equivalent_group, table_key_group_map, join_cond)
        bin_mode_left = self.table_buckets[tables_all[left_table]].oned_bin_modes
        bin_mode_right = right_bound_factor.bin_modes
        key_group_pdf = dict()
        key_group_bin_mode = dict()
        new_union_key_group = dict()
        new_na_values = dict()
        right_variables = right_bound_factor.variables
        new_variables = copy.deepcopy(right_variables)
        res = right_bound_factor.tables_size
        for key_group in equivalent_key_group:
            all_pdfs = [cond_factor_left.pdfs[key] * cond_factor_left.table_len * cond_factor_left.na_values[key]
                        for key in equivalent_key_group[key_group]["left"]]
            all_bin_modes = [bin_mode_left[key] for key in equivalent_key_group[key_group]["left"]]
            for key in equivalent_key_group[key_group]["left"]:
                new_variables[key] = key_group
            for key in equivalent_key_group[key_group]["right"]:
                if key in right_bound_factor.pdfs:
                    new_variables[key] = key_group
                    all_pdfs.append(right_bound_factor.pdfs[key] * res * right_bound_factor.na_values[key])
                    all_bin_modes.append(bin_mode_right[key])
                else:
                    key = right_variables[key]
                    all_pdfs.append(right_bound_factor.pdfs[key] * res * right_bound_factor.na_values[key])
                    all_bin_modes.append(bin_mode_right[key])

            new_pdf, new_bin_mode = self.compute_bound_oned(all_pdfs, all_bin_modes, return_factor=True)
            res = np.sum(new_pdf)
            if res == 0:
                res = 10.0
                new_pdf[-1] = 1
                key_group_pdf[key_group] = new_pdf
            else:
                key_group_pdf[key_group] = new_pdf / res
            key_group_bin_mode[key_group] = new_bin_mode
            new_union_key_group[key_group] = [key_group]
            new_na_values[key_group] = 1

        for group in union_key_group:
            if group not in new_union_key_group:
                new_union_key_group[group] = []
            for table, keys in union_key_group[group]:
                for key in keys:
                    new_union_key_group[group].append(key)
                    if table == "left":
                        key_group_pdf[key] = cond_factor_left.pdfs[key]
                        key_group_bin_mode[key] = self.table_buckets[tables_all[left_table]].oned_bin_modes[key]
                        new_na_values[key] = cond_factor_left.na_values[key]
                    else:
                        key_group_pdf[key] = right_bound_factor.pdfs[key]
                        key_group_bin_mode[key] = right_bound_factor.bin_modes[key]
                        new_na_values[key] = right_bound_factor.na_values[key]

        new_factor = Group_Factor(sub_plan_query_str, res, new_variables, key_group_pdf, key_group_bin_mode,
                                  union_key_group_set, new_union_key_group, new_na_values, new_join_cond)
        return new_factor, res

    def get_join_keys_with_table_group(self, left_table, right_bound_factor, tables_all, table_equivalent_group,
                                       table_key_equivalent_group, table_key_group_map, join_cond):
        """
            Get the join keys between two tables
        """

        actual_join_cond = []
        for cond in join_cond[left_table]:
            if cond in right_bound_factor.join_cond:
                actual_join_cond.append(cond)
        equivalent_key_group = dict()
        union_key_group_set = table_equivalent_group[left_table].union(right_bound_factor.equivalent_groups)
        union_key_group = dict()
        new_join_cond = right_bound_factor.join_cond.union(join_cond[left_table])
        if len(actual_join_cond) != 0:
            for cond in actual_join_cond:
                key1 = cond.split("=")[0].strip()
                key2 = cond.split("=")[1].strip()
                if key1.split(".")[0] == left_table:
                    key_left = tables_all[left_table] + "." + key1.split(".")[-1]
                    key_group = table_key_group_map[left_table][key_left]
                    if key_group not in equivalent_key_group:
                        equivalent_key_group[key_group] = dict()
                    if left_table in equivalent_key_group[key_group]:
                        equivalent_key_group[key_group]["left"].append(key_left)
                    else:
                        equivalent_key_group[key_group]["left"] = [key_left]
                    right_table = key2.split(".")[0]
                    key_right = tables_all[right_table] + "." + key2.split(".")[-1]
                    key_group_t = table_key_group_map[right_table][key_right]
                    assert key_group_t == key_group, f"key group mismatch for join {cond}"
                    if "right" in equivalent_key_group[key_group]:
                        equivalent_key_group[key_group]["right"].append(key_right)
                    else:
                        equivalent_key_group[key_group]["right"] = [key_right]
                else:
                    assert key2.split(".")[0] == left_table, f"unrecognized table alias"
                    key_left = tables_all[left_table] + "." + key2.split(".")[-1]
                    key_group = table_key_group_map[left_table][key_left]
                    if key_group not in equivalent_key_group:
                        equivalent_key_group[key_group] = dict()
                    if left_table in equivalent_key_group[key_group]:
                        equivalent_key_group[key_group]["left"].append(key_left)
                    else:
                        equivalent_key_group[key_group]["left"] = [key_left]
                    right_table = key1.split(".")[0]
                    key_right = tables_all[right_table] + "." + key1.split(".")[-1]
                    key_group_t = table_key_group_map[right_table][key_right]
                    assert key_group_t == key_group, f"key group mismatch for join {cond}"
                    if "right" in equivalent_key_group[key_group]:
                        equivalent_key_group[key_group]["right"].append(key_right)
                    else:
                        equivalent_key_group[key_group]["right"] = [key_right]

            for group in union_key_group_set:
                if group in equivalent_key_group:
                    new_left_key = []
                    for key in table_key_equivalent_group[left_table][group]:
                        if key not in equivalent_key_group[group]["left"]:
                            new_left_key.append(key)
                    if len(new_left_key) != 0:
                        union_key_group[group] = [("left", new_left_key)]
                    new_right_key = []
                    for key in right_bound_factor.table_key_equivalent_group[group]:
                        if key not in equivalent_key_group[group]["right"]:
                            new_right_key.append(key)
                    if len(new_right_key) != 0:
                        if group in union_key_group:
                            union_key_group[group].append(("right", new_right_key))
                        else:
                            union_key_group[group] = [("right", new_right_key)]
                else:
                    if group in table_key_equivalent_group[left_table]:
                        if group in union_key_group:
                            union_key_group[group].append(("left", table_key_equivalent_group[left_table][group]))
                        else:
                            union_key_group[group] = [("left", table_key_equivalent_group[left_table][group])]
                    if group in right_bound_factor.table_key_equivalent_group:
                        if group in union_key_group:
                            union_key_group[group].append(
                                ("right", right_bound_factor.table_key_equivalent_group[group]))
                        else:
                            union_key_group[group] = [("right", right_bound_factor.table_key_equivalent_group[group])]

        else:
            common_key_group = table_equivalent_group[left_table].intersection(right_bound_factor.equivalent_groups)
            common_key_group = list(common_key_group)[0]
            for group in union_key_group_set:
                if group == common_key_group:
                    equivalent_key_group[group] = dict()
                    equivalent_key_group[group]["left"] = table_key_equivalent_group[left_table][group]
                    equivalent_key_group[group]["right"] = right_bound_factor.table_key_equivalent_group[group]
                else:
                    if group in table_key_equivalent_group[left_table]:
                        if group in union_key_group:
                            union_key_group[group].append(("left", table_key_equivalent_group[left_table][group]))
                        else:
                            union_key_group[group] = [("left", table_key_equivalent_group[left_table][group])]
                    if group in right_bound_factor.table_key_equivalent_group:
                        if group in union_key_group:
                            union_key_group[group].append(
                                ("right", right_bound_factor.table_key_equivalent_group[group]))
                        else:
                            union_key_group[group] = [("right", right_bound_factor.table_key_equivalent_group[group])]

        return equivalent_key_group, union_key_group_set, union_key_group, new_join_cond

    def get_additional_join_with_table_group(self, left_table, right_union_key_group, table_equivalent_group,
                                             table_key_equivalent_group):
        common_key_group = table_equivalent_group[left_table].intersection(set(right_union_key_group.keys()))
        union_key_group_set = table_equivalent_group[left_table].union(set(right_union_key_group.keys()))
        union_key_group = copy.deepcopy(right_union_key_group)
        all_join_predicates = []
        for group in union_key_group_set:
            if group in common_key_group:
                left_key = table_key_equivalent_group[left_table][group][0]
                left_key = left_table + "." + left_key.split(".")[-1]
                right_key = right_union_key_group[group]
                join_predicate = left_key + " = " + right_key
                all_join_predicates.append(join_predicate)
            if group not in union_key_group:
                assert group in table_key_equivalent_group[left_table]
                left_key = table_key_equivalent_group[left_table][group][0]
                left_key = left_table + "." + left_key.split(".")[-1]
                union_key_group[group] = left_key

        return all_join_predicates, union_key_group

    def join_two_tables(self, sub_plan_query_str, left_table, right_table, tables_all, conditional_factors, join_keys,
                        table_equivalent_group, table_key_equivalent_group, table_key_group_map, join_cond):
        """
            Get the cardinality bound by joining the left_table with the right_table
            :param left_table:
            :param right_table:
        """
        equivalent_key_group, union_key_group_set, union_key_group, new_join_cond = \
            self.get_join_keys_two_tables(left_table, right_table, table_equivalent_group, table_key_equivalent_group,
                                          table_key_group_map, join_cond, join_keys, tables_all)
        cond_factor_left = conditional_factors[left_table]
        cond_factor_right = conditional_factors[right_table]
        bin_mode_left = self.table_buckets[tables_all[left_table]].oned_bin_modes
        bin_mode_right = self.table_buckets[tables_all[right_table]].oned_bin_modes
        key_group_pdf = dict()
        key_group_bin_mode = dict()
        new_union_key_group = dict()
        res = cond_factor_right.table_len
        new_na_values = dict()
        new_variables = dict()
        for key_group in equivalent_key_group:
            # if len(equivalent_key_group[key_group][left_table]) > 1:
            #   print(len(equivalent_key_group[key_group][left_table]), sub_plan_query_str)
            # if len(equivalent_key_group[key_group][right_table]) > 1:
            #   print(len(equivalent_key_group[key_group][right_table]), sub_plan_query_str)
            all_pdfs = [cond_factor_left.pdfs[key] * cond_factor_left.table_len * cond_factor_left.na_values[key]
                        for key in equivalent_key_group[key_group][left_table]] + \
                       [cond_factor_right.pdfs[key] * res * cond_factor_right.na_values[key]
                        for key in equivalent_key_group[key_group][right_table]]
            all_bin_modes = [bin_mode_left[key] for key in equivalent_key_group[key_group][left_table]] + \
                            [bin_mode_right[key] for key in equivalent_key_group[key_group][right_table]]
            for key in equivalent_key_group[key_group][left_table] + equivalent_key_group[key_group][right_table]:
                new_variables[key] = key_group
            new_pdf, new_bin_mode = self.compute_bound_oned(all_pdfs, all_bin_modes, return_factor=True)
            res = np.sum(new_pdf)
            key_group_pdf[key_group] = new_pdf / res
            key_group_bin_mode[key_group] = new_bin_mode
            new_union_key_group[key_group] = [key_group]
            new_na_values[key_group] = 1.0

        for group in union_key_group:
            if group not in new_union_key_group:
                new_union_key_group[group] = []
            for table, keys in union_key_group[group]:
                for key in keys:
                    new_union_key_group[group].append(key)
                    key_group_pdf[key] = conditional_factors[table].pdfs[key]
                    key_group_bin_mode[key] = self.table_buckets[tables_all[table]].oned_bin_modes[key]
                    new_na_values[key] = conditional_factors[table].na_values[key]

        new_factor = Group_Factor(sub_plan_query_str, res, new_variables, key_group_pdf, key_group_bin_mode,
                                  union_key_group_set, new_union_key_group, new_na_values, new_join_cond)
        return new_factor, res

    def get_join_keys_two_tables(self, left_table, right_table, table_equivalent_group, table_key_equivalent_group,
                                 table_key_group_map, join_cond, join_keys, tables_all):
        """
            Get the join keys between two tables
        """
        actual_join_cond = []
        for cond in join_cond[left_table]:
            if cond in join_cond[right_table]:
                actual_join_cond.append(cond)
        equivalent_key_group = dict()
        union_key_group_set = table_equivalent_group[left_table].union(table_equivalent_group[right_table])
        union_key_group = dict()
        new_join_cond = join_cond[left_table].union(join_cond[right_table])
        if len(actual_join_cond) != 0:
            for cond in actual_join_cond:
                key1 = cond.split("=")[0].strip()
                key2 = cond.split("=")[1].strip()
                if key1.split(".")[0] == left_table:
                    key_left = tables_all[left_table] + "." + key1.split(".")[-1]
                    key_group = table_key_group_map[left_table][key_left]
                    if key_group not in equivalent_key_group:
                        equivalent_key_group[key_group] = dict()
                    if left_table in equivalent_key_group[key_group]:
                        equivalent_key_group[key_group][left_table].append(key_left)
                    else:
                        equivalent_key_group[key_group][left_table] = [key_left]
                    assert key2.split(".")[0] == right_table, f"unrecognized table alias"
                    key_right = tables_all[right_table] + "." + key2.split(".")[-1]
                    key_group_t = table_key_group_map[right_table][key_right]
                    assert key_group_t == key_group, f"key group mismatch for join {cond}"
                    if right_table in equivalent_key_group[key_group]:
                        equivalent_key_group[key_group][right_table].append(key_right)
                    else:
                        equivalent_key_group[key_group][right_table] = [key_right]
                else:
                    assert key2.split(".")[0] == left_table, f"unrecognized table alias"
                    key_left = tables_all[left_table] + "." + key2.split(".")[-1]
                    key_group = table_key_group_map[left_table][key_left]
                    if key_group not in equivalent_key_group:
                        equivalent_key_group[key_group] = dict()
                    if left_table in equivalent_key_group[key_group]:
                        equivalent_key_group[key_group][left_table].append(key_left)
                    else:
                        equivalent_key_group[key_group][left_table] = [key_left]
                    assert key1.split(".")[0] == right_table, f"unrecognized table alias"
                    key_right = tables_all[right_table] + "." + key1.split(".")[-1]
                    key_group_t = table_key_group_map[right_table][key_right]
                    assert key_group_t == key_group, f"key group mismatch for join {cond}"
                    if right_table in equivalent_key_group[key_group]:
                        equivalent_key_group[key_group][right_table].append(key_right)
                    else:
                        equivalent_key_group[key_group][right_table] = [key_right]

            for group in union_key_group_set:
                if group in equivalent_key_group:
                    new_left_key = []
                    for key in table_key_equivalent_group[left_table][group]:
                        if key not in equivalent_key_group[group][left_table]:
                            new_left_key.append(key)
                    if len(new_left_key) != 0:
                        union_key_group[group] = [(left_table, new_left_key)]
                    new_right_key = []
                    for key in table_key_equivalent_group[right_table][group]:
                        if key not in equivalent_key_group[group][right_table]:
                            new_right_key.append(key)
                    if len(new_right_key) != 0:
                        if group in union_key_group:
                            union_key_group[group].append((right_table, new_right_key))
                        else:
                            union_key_group[group] = [(right_table, new_right_key)]
                else:
                    if group in table_key_equivalent_group[left_table]:
                        if group in union_key_group:
                            union_key_group[group].append((left_table, table_key_equivalent_group[left_table][group]))
                        else:
                            union_key_group[group] = [(left_table, table_key_equivalent_group[left_table][group])]
                    if group in table_key_equivalent_group[right_table]:
                        if group in union_key_group:
                            union_key_group[group].append((right_table, table_key_equivalent_group[right_table][group]))
                        else:
                            union_key_group[group] = [(right_table, table_key_equivalent_group[right_table][group])]

        else:
            common_key_group = table_equivalent_group[left_table].intersection(table_equivalent_group[right_table])
            common_key_group = list(common_key_group)[0]
            for group in union_key_group_set:
                if group == common_key_group:
                    equivalent_key_group[group] = dict()
                    equivalent_key_group[group][left_table] = table_key_equivalent_group[left_table][group]
                    equivalent_key_group[group][right_table] = table_key_equivalent_group[right_table][group]
                elif group in table_key_equivalent_group[left_table]:
                    if group in union_key_group:
                        union_key_group[group].append((left_table, table_key_equivalent_group[left_table][group]))
                    else:
                        union_key_group[group] = [(left_table, table_key_equivalent_group[left_table][group])]
                else:
                    if group in union_key_group:
                        union_key_group[group].append((right_table, table_key_equivalent_group[right_table][group]))
                    else:
                        union_key_group[group] = [(right_table, table_key_equivalent_group[right_table][group])]

        return equivalent_key_group, union_key_group_set, union_key_group, new_join_cond

    def get_additional_joins_two_tables(self, left_table, right_table, table_equivalent_group,
                                        table_key_equivalent_group):
        common_key_group = table_equivalent_group[left_table].intersection(table_equivalent_group[right_table])
        union_key_group_set = table_equivalent_group[left_table].union(table_equivalent_group[right_table])
        union_key_group = dict()
        all_join_predicates = []
        for group in union_key_group_set:
            if group in common_key_group:
                left_key = table_key_equivalent_group[left_table][group][0]
                left_key = left_table + "." + left_key.split(".")[-1]
                right_key = table_key_equivalent_group[right_table][group][0]
                right_key = right_table + "." + right_key.split(".")[-1]
                join_predicate = left_key + " = " + right_key
                all_join_predicates.append(join_predicate)
            if group in table_key_equivalent_group[left_table]:
                left_key = table_key_equivalent_group[left_table][group][0]
                left_key = left_table + "." + left_key.split(".")[-1]
                union_key_group[group] = left_key
            else:
                right_key = table_key_equivalent_group[right_table][group][0]
                right_key = right_table + "." + right_key.split(".")[-1]
                union_key_group[group] = right_key

        return all_join_predicates, union_key_group

    def get_sub_plan_join_key(self, sub_plan_query, join_cond):
        # returning a subset of join_keys covered by the tables in sub_plan_query
        touched_join_cond = set()
        untouched_join_cond = set()
        for tab in join_cond:
            if tab in sub_plan_query:
                touched_join_cond = touched_join_cond.union(join_cond[tab])
            else:
                untouched_join_cond = untouched_join_cond.union(join_cond[tab])
        touched_join_cond -= untouched_join_cond

        join_keys = dict()
        for cond in touched_join_cond:
            key1 = cond.split("=")[0].strip()
            table1 = key1.split(".")[0].strip()
            if table1 not in join_keys:
                join_keys[table1] = set([key1])
            else:
                join_keys[table1].add(key1)

            key2 = cond.split("=")[1].strip()
            table2 = key2.split(".")[0].strip()
            if table2 not in join_keys:
                join_keys[table2] = set([key2])
            else:
                join_keys[table2].add(key2)

        return join_keys

