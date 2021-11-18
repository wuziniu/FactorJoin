import numpy as np
import copy

from Join_scheme.join_graph import process_condition, get_join_hyper_graph
from Join_scheme.data_prepare import identify_key_values
from BayesCard.Evaluation.cardinality_estimation import timestamp_transorform, construct_table_query


class Factor:
    """
    This the class defines a multidimensional conditional probability.
    """
    def __init__(self, variables, pdfs):
        self.variables = variables
        self.pdfs = pdfs
        self.cardinalities = dict()
        for i, var in enumerate(self.variables):
            self.cardinalities[var] = pdfs.shape[i]


class Bound_ensemble:
    """
    This the class where we store all the trained models and perform inference on the bound.
    """
    def __init__(self, bns, table_buckets, schema):
        self.bns = bns
        self.table_buckets = table_buckets
        self.schema = schema
        self.all_keys, self.equivalent_keys = identify_key_values(schema)

    def parse_query_simple(self, query):
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
                    assert "::timestamp" in value
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

    def get_all_id_conidtional_distribution(self, table_queries, join_keys):
        res = dict()
        for table in join_keys:
            key_attrs = list(join_keys[table])
            if table in table_queries:
                table_query = table_queries[table]
            else:
                table_query = {}
            id_attrs, probs = self.bns[table].query_id_prob(table_query, key_attrs)
            res[table] = Factor(id_attrs, probs)
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

        for i in range(len(rest_group_cardinalty)):
            for table in rest_group_tables:
                idx = factors[table].equivalent_variables.index(key_group)
                bin_modes = self.table_buckets[table].twod_bin_modes[relevant_keys[key_group][table]]
                if idx == 0:
                    all_probs_eliminated.append(factors[table].pdfs[:, i])
                    all_modes_eliminated.append(np.minimum(bin_modes[:, i], factors[table].pdfs[:, i]))
                else:
                    all_probs_eliminated.append(factors[table].pdfs[i, :])
                    all_modes_eliminated.append(np.minimum(bin_modes[i, :], factors[table].pdfs[i, :]))
            new_factor_pdf[i] = self.compute_bound_oned(all_probs_eliminated, all_modes_eliminated)

        for table in rest_group_tables:
            factors[table] = Factor(rest_group, new_factor_pdf)

        return None

    def compute_bound_oned(self, all_probs, all_modes):
        all_probs = np.stack(all_probs, axis=0)
        all_modes = np.stack(all_modes, axis=0)
        multiplier = np.prod(all_modes, axis=0)
        non_zero_idx = np.where(multiplier != 0)[0]
        min_number = np.amin(all_probs[:, non_zero_idx]/all_modes[:, non_zero_idx], axis=0)
        multiplier[non_zero_idx] = multiplier[non_zero_idx] * min_number
        return multiplier

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

    def get_cardinality_bound(self, query_str):
        tables_all, table_queries, join_cond, join_keys = self.parse_query_simple(query_str)
        equivalent_group = get_join_hyper_graph(join_keys, self.equivalent_keys)
        conditional_factors = self.get_all_id_conidtional_distribution(table_queries, join_keys, equivalent_group)
        optimal_order, tables_involved, relevant_keys = self.get_optimal_elimination_order(equivalent_group, join_keys,
                                                                            conditional_factors)
        res = 0
        for key_group in optimal_order:
            tables = tables_involved[key_group]
            res = self.eliminate_one_key_group(tables, key_group, conditional_factors, relevant_keys)
        return res
