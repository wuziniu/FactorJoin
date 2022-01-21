import numpy as np
import copy

from Join_scheme.join_graph import process_condition, get_join_hyper_graph, parse_query_all_join, \
    parse_query_all_single_table
from Join_scheme.data_prepare import identify_key_values


class Factor:
    """
    This the class defines a multidimensional conditional probability.
    """
    def __init__(self, variables, pdfs, equivalent_variables=[]):
        self.variables = variables
        self.equivalent_variables = equivalent_variables
        self.pdfs = pdfs
        self.cardinalities = dict()
        for i, var in enumerate(self.variables):
            self.cardinalities[var] = len(pdfs[i])
            if len(equivalent_variables) != 0:
                self.cardinalities[equivalent_variables[i]] = len(pdfs[i])


class Bound_ensemble:
    """
    This the class where we store all the trained models and perform inference on the bound.
    """
    def __init__(self, bns, table_buckets, schema):
        self.bns = bns
        self.table_buckets = table_buckets
        self.schema = schema
        self.all_keys, self.equivalent_keys = identify_key_values(schema)
        self.all_join_conds = None
        self.reverse_table_alias = None

    def parse_query_simple(self, query):
        """
        If your selection query contains no aggregation and nested sub-queries, you can use this function to parse a
        join query. Otherwise, use parse_query function.
        """
        tables_all, join_cond, join_keys = parse_query_all_join(query)
        #TODO: implement functions on parsing filter conditions.
        table_query = parse_query_all_single_table(query)
        return tables_all, table_query, join_cond, join_keys

    def get_all_id_conidtional_distribution(self, table_queries, join_keys, equivalent_group):
        #TODO: make it work on query-driven and sampling based
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
            res[table] = Factor(id_attrs, probs, new_id_attrs)
        return res

    def eliminate_one_key_group_general(self, tables, key_group, factors):
        rest_groups = dict()
        rest_group_tables = dict()
        for table in tables:
            assert key_group in factors[table].equivalent_variables
            temp = copy.deepcopy(factors[table].equivalent_variables)
            temp.remove(key_group)
            for keys in temp:
                if keys in rest_groups:
                    assert factors[table].cardinalities[keys] == rest_groups[keys]
                    rest_group_tables[keys].append(table)
                else:
                    rest_groups[keys] = factors[table].cardinalities[keys]
                    rest_group_tables[keys] = [table]

        new_factor_variables = []
        new_factor_cardinalities = []
        for key in rest_groups:
            new_factor_variables.append(key)
            new_factor_cardinalities.append(rest_groups[key])
        new_factor_pdf = np.zeros(tuple(new_factor_cardinalities))

    def eliminate_one_key_group(self, tables, key_group, factors, relevant_keys):
        """This version only supports 2D distributions"""
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
            for table in rest_group_tables:
                idx_f = factors[table].equivalent_variables.index(key_group)
                idx_b = self.table_buckets[table].id_attributes.index(relevant_keys[key_group][table])
                bin_modes = self.table_buckets[table].twod_bin_modes[relevant_keys[key_group][table]]
                if idx_f == 0 and idx_b == 0:
                    all_probs_eliminated.append(factors[table].pdfs[:, i])
                    all_modes_eliminated.append(np.minimum(bin_modes[:, i], factors[table].pdfs[:, i]))
                elif idx_f == 0 and idx_b == 1:
                    all_probs_eliminated.append(factors[table].pdfs[:, i])
                    all_modes_eliminated.append(np.minimum(bin_modes[i, :], factors[table].pdfs[:, i]))
                elif idx_f == 1 and idx_b == 0:
                    all_probs_eliminated.append(factors[table].pdfs[i, :])
                    all_modes_eliminated.append(np.minimum(bin_modes[:, i], factors[table].pdfs[i, :]))
                else:
                    all_probs_eliminated.append(factors[table].pdfs[i, :])
                    all_modes_eliminated.append(np.minimum(bin_modes[i, :], factors[table].pdfs[i, :]))
            new_factor_pdf[i] = self.compute_bound_oned(all_probs_eliminated, all_modes_eliminated)

        for table in rest_group_tables:
            factors[table] = Factor([rest_group], new_factor_pdf, [rest_group])

        return None

    def compute_bound_oned(self, all_probs, all_modes):
        all_probs = np.stack(all_probs, axis=0)
        all_modes = np.stack(all_modes, axis=0)
        multiplier = np.prod(all_modes, axis=0)
        non_zero_idx = np.where(multiplier != 0)[0]
        min_number = np.amin(all_probs[:, non_zero_idx]/all_modes[:, non_zero_idx], axis=0)
        multiplier[non_zero_idx] = multiplier[non_zero_idx] * min_number
        return np.sum(multiplier)

    def get_optimal_elimination_order(self, equivalent_group, join_keys, factors):
        """
        This function determines the optimial elimination order for each key group
        """
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

    def get_cardinality_bound_one(self, query_str):
        tables_all, table_queries, join_cond, join_keys = self.parse_query_simple(query_str)
        equivalent_group = get_join_hyper_graph(join_keys, self.equivalent_keys)
        conditional_factors = self.get_all_id_conidtional_distribution(table_queries, join_keys, equivalent_group)
        optimal_order, tables_involved, relevant_keys = self.get_optimal_elimination_order(equivalent_group, join_keys,
                                                                            conditional_factors)

        for key_group in optimal_order:
            tables = tables_involved[key_group]
            res = self.eliminate_one_key_group(tables, key_group, conditional_factors, relevant_keys)
        return res

    def get_cardinality_bound_all(self, query_str, sub_plan_query_str_all):
        tables_all, table_queries, join_cond, join_keys = self.parse_query_simple(query_str)
        equivalent_group = get_join_hyper_graph(join_keys, self.equivalent_keys)
        conditional_factors = self.get_all_id_conidtional_distribution(table_queries, join_keys, equivalent_group)
        self.reverse_table_alias = {v: k for k, v in tables_all.items()}
        res = []
        for sub_plan_query_str in sub_plan_query_str_all:
            sub_plan_query = set(sub_plan_query_str.split(" "))
            curr_join_keys = self.get_sub_plan_join_key(sub_plan_query, join_cond)
            curr_equivalent_group = self.get_sub_plan_equivalent_group(sub_plan_query, equivalent_group)
            curr_factors = self.get_sub_plan_conditional_factors(sub_plan_query, conditional_factors)
            optimal_order, tables_involved, relevant_keys = self.get_optimal_elimination_order(curr_equivalent_group,
                                                                                               curr_join_keys,
                                                                                               curr_factors)

            for key_group in optimal_order:
                tables = tables_involved[key_group]
                res.append(self.eliminate_one_key_group(tables, key_group, conditional_factors, relevant_keys))
        return res

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
            table1 = self.reverse_table_alias[key1.split(".")[0].strip()]
            if table1 not in join_keys:
                join_keys[table1] = set([key1])
            else:
                join_keys[table1].add(key1)

            key2 = cond.split("=")[1].strip()
            table2 = self.reverse_table_alias[key2.split(".")[0].strip()]
            if table2 not in join_keys:
                join_keys[table2] = set([key2])
            else:
                join_keys[table2].add(key2)

        return join_keys



