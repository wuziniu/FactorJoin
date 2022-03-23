import numpy as np
import copy
import os
import pickle5 as pickle
from Join_scheme.join_graph import get_join_hyper_graph, parse_query_all_join
from Join_scheme.data_prepare import identify_key_values


class Factor:
    """
    This the class defines a multidimensional conditional probability on one table.
    We support at most 2D conditional probability
    """

    def __init__(self, table, table_len, variables, pdfs, na_values=None):
        self.table = table
        self.table_len = table_len
        self.variables = variables
        self.pdfs = pdfs
        self.na_values = na_values  # this is the percentage of data, which is not nan.

    def get_related_pdf_one_key(self, keys, return_remaining=False, replace=None):
        all_res = []
        for key in keys:
            res = dict()
            for k in self.pdfs:
                if k not in res:
                    if key == k:
                        if replace:
                            res[replace] = self.pdfs[k]
                        else:
                            res[k] = self.pdfs[k]
                    elif (type(k) == tuple and key in k):
                        if replace:
                            a = copy.deepcopy(k)
                            a[a.index(key)] = replace
                            res[a] = self.pdfs[k]
                        else:
                            res[k] = self.pdfs[k]
        remaining_pdfs = dict()
        for k in self.pdfs:
            flag = True
            for res in all_res:
                if k in res:
                    flag = False
                    break
            if flag:
                remaining_pdfs[k] = self.pdfs[k]
        if return_remaining:
            return all_res, remaining_pdfs
        else:
            return all_res

    def get_pdf_two_keys(self, keys):
        keys = tuple(sorted(keys))
        if keys in self.pdfs:
            return self.pdfs[keys]
        else:
            return None

    def get_pdf(self, key, return_card=False):
        if key in self.pdfs:
            return self.pdfs[key] * self.table_len if return_card else self.pdfs[key]
        else:
            for k in self.pdfs:
                if type(k) == tuple and key in k:
                    pdf = self.pdfs[k]
                    if k.index(key) == 0:
                        key_pdf = np.sum(pdf, axis=1)
                    else:
                        key_pdf = np.sum(pdf, axis=0)
                    return key_pdf * self.table_len if return_card else key_pdf


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

    def __init__(self, table_buckets, schema, ground_truth_factors_no_filter=None):
        self.table_buckets = table_buckets
        self.schema = schema
        self.all_keys, self.equivalent_keys = identify_key_values(schema)
        self.all_join_conds = None
        self.ground_truth_factors_no_filter = ground_truth_factors_no_filter
        # self.reverse_table_alias = None

    def parse_query_simple(self, query):
        """
        If your selection query contains no aggregation and nested sub-queries, you can use this function to parse a
        join query. Otherwise, use parse_query function.
        """
        tables_all, join_cond, join_keys = parse_query_all_join(query)
        # TODO: implement functions on parsing filter conditions.
        table_filters = dict()
        return tables_all, table_filters, join_cond, join_keys

    def get_all_id_conidtional_distribution(self, query_file_name, tables_alias, join_keys):
        # TODO: make it work on query-driven
        return load_sample_imdb_one_query(self.table_buckets, tables_alias, query_file_name, join_keys,
                                          self.ground_truth_factors_no_filter)

    def compute_bound_oned(self, all_probs, all_modes, return_factor=False):
        """
        Compute one dimension bound
        """
        temp_all_modes = []
        for i in range(len(all_modes)):
            temp_all_modes.append(np.minimum(all_probs[i], all_modes[i]))
        all_probs = np.stack(all_probs, axis=0)
        temp_all_modes = np.stack(temp_all_modes, axis=0)
        multiplier = np.prod(temp_all_modes, axis=0)
        non_zero_idx = np.where(multiplier != 0)[0]
        min_number = np.amin(all_probs[:, non_zero_idx] / temp_all_modes[:, non_zero_idx], axis=0)
        # print(min_number, multiplier[non_zero_idx])
        if return_factor:
            new_probs = np.zeros(multiplier.shape)
            new_probs[non_zero_idx] = multiplier[non_zero_idx] * min_number
            return new_probs, multiplier
        else:
            multiplier[non_zero_idx] = multiplier[non_zero_idx] * min_number
            return np.sum(multiplier)

    def compute_bound_twod(self, left_pdfs, left_bin_modes, na_values, right_key_pdf, right_key_bin_mode, key_group):
        # TODO
        return None, None

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
        print(join_cond)
        # print(join_keys)
        equivalent_group, table_equivalent_group, table_key_equivalent_group, table_key_group_map = \
            get_join_hyper_graph(join_keys, self.equivalent_keys)
        conditional_factors = self.get_all_id_conidtional_distribution(query_name, tables_all, join_keys)
        # self.reverse_table_alias = {v: k for k, v in tables_all.items()}
        cached_sub_queries = dict()
        cardinality_bounds = []
        for i, (left_tables, right_tables) in enumerate(sub_plan_query_str_all):
            assert " " not in left_tables, f"{left_tables} contains more than one tables, violating left deep plan"
            sub_plan_query_list = right_tables.split(" ") + [left_tables]
            sub_plan_query_list.sort()
            sub_plan_query_str = " ".join(sub_plan_query_list)  # get the string name of the sub plan query
            # print(sub_plan_query_str)
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
                print(f"{left_tables}, {right_tables}|| estimate: {res}, true: {true_card[i]}, error: {error}")
            cardinality_bounds.append(res)
        return cardinality_bounds

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
        bin_mode_left = self.table_buckets[tables_all[left_table]].bin_modes
        bin_mode_right = self.table_buckets[tables_all[right_table]].bin_modes
        key_group_pdf = dict()
        key_group_bin_mode = dict()
        changed_keys = []
        new_union_key_group = dict()
        res = cond_factor_right.table_len
        new_na_values = dict()
        new_variables = dict()
        res = copy.copy(cond_factor_right.table_len)
        left_table_len = copy.copy(cond_factor_left.table_len)

        for key_group in equivalent_key_group:
            left_pdfs, left_bin_modes, left_key_pdf, left_key_bin_mode, left_table_len, left_na_value, left_elim = \
                self.prepare_one_table(equivalent_key_group[key_group][left_table],
                                       cond_factor_left,
                                       bin_mode_left,
                                       left_table_len,
                                       key_group)
            changed_keys.extend(left_elim)
            for k in left_pdfs:
                if k in key_group_pdf:
                    left_pdfs[k] = key_group_pdf[k]
                    left_bin_modes[k] = key_group_bin_mode[k]

            right_pdfs, right_bin_modes, right_key_pdf, right_key_bin_mode, res, right_na_value, right_elim = \
                self.prepare_one_table(equivalent_key_group[key_group][right_table],
                                       cond_factor_right,
                                       bin_mode_right,
                                       res,
                                       key_group)
            for k in right_pdfs:
                if k in key_group_pdf:
                    right_pdfs[k] = key_group_pdf[k]
                    right_bin_modes[k] = key_group_bin_mode[k]
            changed_keys.extend(right_elim)

            seen_oned_attr = False
            new_res = np.infty
            for left_key in left_pdfs:
                # TODO
                if type(left_key) != tuple:
                    new_pdf, new_bin_mode = self.compute_bound_oned(
                        [left_pdfs[left_key] * left_table_len * left_na_value, right_key_pdf * res * right_na_value],
                        [left_bin_modes[left_key], right_key_bin_mode],
                        return_factor=True)
                    temp_res = np.sum(new_pdf)
                    key_group_pdf[key_group] = new_pdf / temp_res
                    key_group_bin_mode[key_group] = new_bin_mode
                    seen_oned_attr = True
                else:
                    # TODO
                    new_left_pdfs, new_left_bin_mode = self.compute_bound_twod(left_pdfs[left_key] * left_table_len,
                                                                               left_bin_modes[left_key],
                                                                               cond_factor_left.na_values,
                                                                               right_key_pdf * res * right_na_value,
                                                                               right_key_bin_mode,
                                                                               key_group)
                    temp_res = np.sum(new_left_pdfs)
                    key_group_pdf[left_key] = new_left_pdfs / temp_res
                    key_group_bin_mode[left_key] = new_left_bin_mode
                if temp_res < new_res:
                    new_res = temp_res
            new_left_table_length = new_res

            for right_key in right_pdfs:
                if type(right_key) != tuple:
                    if seen_oned_attr:
                        continue
                    new_pdf, new_bin_mode = self.compute_bound_oned(
                        [right_pdfs[right_key] * res * right_na_value, left_key_pdf * left_table_len * left_na_value],
                        [right_bin_modes[right_key], left_key_bin_mode],
                        return_factor=True)
                    temp_res = np.sum(new_pdf)
                    key_group_pdf[key_group] = new_pdf / temp_res
                    key_group_bin_mode[key_group] = new_bin_mode
                else:
                    # TODO
                    new_right_pdfs, new_right_bin_mode = self.compute_bound_twod(right_pdfs[right_key] * res,
                                                                        right_bin_modes[right_key],
                                                                        cond_factor_right.na_values,
                                                                        left_key_pdf * left_table_len * left_na_value,
                                                                        left_key_bin_mode,
                                                                        key_group)
                    temp_res = np.sum(new_right_pdfs)
                    key_group_pdf[right_key] = new_right_pdfs / temp_res
                    key_group_bin_mode[key_group] = new_right_bin_mode
                if temp_res < new_res:
                    new_res = temp_res
            res = new_res

            # assigning for equivalent key of the joined table to the same name
            for key in equivalent_key_group[key_group][left_table] + equivalent_key_group[key_group][right_table]:
                new_variables[key] = key_group
            new_union_key_group[key_group] = [key_group]
            new_na_values[key_group] = 1.0

        for left_key in cond_factor_left.pdfs:
            if left_key not in changed_keys:
                key_group_pdf[left_key] = cond_factor_left.pdfs[left_key]
                key_group_bin_mode[left_key] = bin_mode_left[left_key]
                if type(left_key) == tuple:
                    for k in left_key:
                        if k not in key_group_bin_mode:
                            key_group_bin_mode[k] = bin_mode_left[k]

        for right_key in cond_factor_right.pdfs:
            if right_key not in changed_keys:
                key_group_pdf[right_key] = cond_factor_left.pdfs[right_key]
                key_group_bin_mode[right_key] = bin_mode_right[right_key]
                if type(right_key) == tuple:
                    for k in right_key:
                        if k not in key_group_bin_mode:
                            key_group_bin_mode[k] = bin_mode_right[k]

        for group in union_key_group:
            new_union_key_group[group] = []
            for table, key in union_key_group[group]:
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

    def prepare_one_table(self, equivalent_key_group, cond_factor_left, bin_mode_left, left_table_len, key_group):
        if len(equivalent_key_group) == 1:
            left_key = equivalent_key_group[0]
            left_pdfs = cond_factor_left.get_related_pdf_one_key([left_key])
            left_pdfs = left_pdfs[0]
            left_bin_modes = {k: bin_mode_left[k] for k in left_pdfs}
            left_key_pdf = cond_factor_left.get_pdf(left_key)
            left_key_bin_mode = bin_mode_left[left_key]
            left_na_value = cond_factor_left.na_values[left_key]
            left_elim = left_pdfs.keys()
            # TODO replace name
        else:
            # In presence of multiple equivalent keys on the same table, we do a self-join first.
            # This case rarely happens in STATS-CEB and IMDB-JOB
            # TODO
            left_pdfs, left_bin_modes, left_key_pdf, left_key_bin_mode, left_table_len = \
                self.conduct_self_join(equivalent_key_group, cond_factor_left,
                                       bin_mode_left, left_table_len, key_group)
            left_na_value = 1.0
            left_elim = left_pdfs.keys()

        return left_pdfs, left_bin_modes, left_key_pdf, left_key_bin_mode, left_table_len, left_na_value, left_elim

    def conduct_self_join(self, join_keys, cond_factor, bin_mode, table_len, key_group_name):
        """
           In presence of multiple equivalent keys on the same table, we do a self-join first.
           :param join_keys: the equivalent join keys on a table
        """
        new_pdfs = dict()
        new_bin_modes = dict()
        all_pdfs = cond_factor.get_related_pdf_one_key(join_keys)
        new_key_pdf = None
        new_key_bin_mode = None
        new_table_len = None

        curr_key = join_keys[0]
        curr_pdf = all_pdfs[0]
        curr_1d_pdf = cond_factor.get_pdf(curr_key)

        return new_pdfs, new_bin_modes, new_key_pdf, new_key_bin_mode, new_table_len

    def join_with_one_table(self, sub_plan_query_str, left_table, tables_all, right_bound_factor, cond_factor_left,
                            table_equivalent_group, table_key_equivalent_group, table_key_group_map, join_cond):
        """
        Get the cardinality bound by joining the left_table with the seen right_tables
        :param left_table:
        :param right_tables:
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
        changed_keys = []
        right_variables = right_bound_factor.variables
        new_variables = copy.deepcopy(right_variables)
        res = right_bound_factor.tables_size

        for key_group in equivalent_key_group:
            left_pdfs, left_bin_modes, left_key_pdf, left_key_bin_mode, left_table_len, left_na_value = \
                self.prepare_one_table(equivalent_key_group[key_group]["left"],
                                       cond_factor_left,
                                       bin_mode_left,
                                       key_group)
            changed_keys.extend(list(left_pdfs.keys()))

            all_pdfs = [cond_factor_left.pdfs[key] * cond_factor_left.table_len * cond_factor_left.na_values[key]
                        for key in equivalent_key_group[key_group]["left"]]
            all_bin_modes = [bin_mode_left[key] for key in equivalent_key_group[key_group]["left"]]
            for key in equivalent_key_group[key_group]["left"]:
                new_variables[key] = key_group
            for key in equivalent_key_group[key_group]["right"]:
                if key in right_bound_factor.pdfs[key]:
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
            new_union_key_group[group] = []
            for table, key in union_key_group[group]:
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
        print(join_cond[left_table], right_bound_factor.join_cond)
        print(actual_join_cond)
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
                elif group in table_key_equivalent_group[left_table]:
                    if group in union_key_group:
                        union_key_group[group].append(("left", table_key_equivalent_group[left_table][group]))
                    else:
                        union_key_group[group] = [("left", table_key_equivalent_group[left_table][group])]
                else:
                    if group in union_key_group:
                        union_key_group[group].append(("right", right_bound_factor.table_key_equivalent_group[group]))
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
                elif group in table_key_equivalent_group[left_table]:
                    if group in union_key_group:
                        union_key_group[group].append(("left", table_key_equivalent_group[left_table][group]))
                    else:
                        union_key_group[group] = [("left", table_key_equivalent_group[left_table][group])]
                else:
                    if group in union_key_group:
                        union_key_group[group].append(("right", right_bound_factor.table_key_equivalent_group[group]))
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


def load_sample_imdb_one_query(table_buckets, tables_alias, query_file_name, join_keys, table_key_equivalent_group,
                               SPERCENTAGE=10.0, qdir="/home/ubuntu/data_CE/saved_models/binned_cards/{}/job/all_job/"):
    qdir = qdir.format(SPERCENTAGE)
    fpath = os.path.join(qdir, query_file_name)
    with open(fpath, "rb") as f:
        data = pickle.load(f)

    conditional_factors = dict()
    table_pdfs = dict()
    filter_size = dict()
    for i, alias in enumerate(data["all_aliases"]):
        column = data["all_columns"][i]
        alias = alias[0]
        key = tables_alias[alias] + "." + column
        cards = data["results"][i][0]
        n_bins = table_buckets[tables_alias[alias]].bin_sizes[key]
        pdfs = np.zeros(n_bins)
        for (j, val) in cards:
            if j is None:
                j = 0
            pdfs[j] += val
        table_len = np.sum(pdfs)
        print(alias+"."+column, table_len, pdfs)
        if table_len == 0:
            # no sample satisfy the filter, set it with a small value
            #print("========================", alias+"."+column)
            table_len = 1
            pdfs = table_key_equivalent_group[tables_alias[alias]].pdfs[key]
        else:
            pdfs /= table_len
        if alias not in table_pdfs:
            table_pdfs[alias] = dict()
            filter_size[alias] = table_len
        table_pdfs[alias][key] = pdfs

    for alias in tables_alias:
        if alias in table_pdfs:
            table_len = min(table_key_equivalent_group[tables_alias[alias]].table_len,
                            filter_size[alias]/(SPERCENTAGE/100))
            na_values = table_key_equivalent_group[tables_alias[alias]].na_values
            conditional_factors[alias] = Factor(tables_alias[alias], table_len, list(table_pdfs[alias].keys()),
                                                table_pdfs[alias], na_values)
        else:
            #TODO: ground-truth distribution
            conditional_factors[alias] = table_key_equivalent_group[tables_alias[alias]]
    return conditional_factors

