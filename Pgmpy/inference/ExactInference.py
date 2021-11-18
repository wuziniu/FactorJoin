import numpy as np
from collections import defaultdict
import copy


def get_working_factors_by_type(working_factors, var, root_var, self_value, query_idx):
    children_value_attr = []
    children_value_id = []
    children_id = []
    # check if all children has been reduced
    for cpd in working_factors[var][1:]:
        if query_idx is not None:
            if len(cpd.values.shape) == 1:
                child_value = cpd.values[query_idx]  # M(var) = Pr(child(var)|var)
            elif len(cpd.values.shape) == 2:
                child_value = cpd.values[:, query_idx]  # M(var) = Pr(child(var)|var)
            elif len(cpd.values.shape) == 3:
                child_value = cpd.values[:, :, query_idx]  # M(var) = Pr(child(var)|var)
            else:
                # TODO: conditional independence exploration and simplification
                assert False, "for getting the probability of more than three ids, we need to " \
                              "explore their conditional independence and simplify the structure" \
                              "to avoid memory explosion. The authors are currently working on it."
        else:
            child_value = cpd.values
        if len(child_value.shape) == 1:
            children_value_attr.append(child_value)
        else:
            assert len(child_value.shape) == len(cpd.extra_info) + 1
            children_id.extend(cpd.extra_info)
            children_value_id.append(child_value)

    if len(children_value_attr) != 0:
        if len(children_value_attr) == 1:
            children_value_attr = children_value_attr[0]
        else:
            # print(children_value)
            children_value_attr = np.prod(np.stack(children_value_attr), axis=0)
        if root_var:
            self_value = self_value * children_value_attr
        else:
            self_value = np.transpose(np.transpose(self_value) * children_value_attr)
    else:
        assert len(children_value_id) != 0, "some node has no child"
    return self_value, children_value_id, children_id


class VariableEliminationJIT(object):
    def __init__(self, model, cpds, topological_order, topological_order_node, fanouts=None, probs=None, root=True):
        model.check_model()
        self.cpds = cpds
        self.topological_order = topological_order
        self.topological_order_node = topological_order_node
        self.model = model
        self.fanouts = fanouts
        if probs is not None:
            self.probs = probs
        else:
            self.probs = dict()

        self.variables = model.nodes()

        if root:
            self.root = self.get_root()

    def get_root(self):
        """Returns the network's root node."""

        def find_root(graph, node):
            predecessor = next(self.model.predecessors(node), None)
            if predecessor:
                root = find_root(graph, predecessor)
            else:
                root = node
            return root

        return find_root(self, list(self.model.nodes)[0])

    def steiner_tree(self, nodes):
        """Returns the minimal part of the tree that contains a set of nodes."""
        sub_nodes = set()

        def walk(node, path):
            if len(nodes) == 0:
                return

            if node in nodes:
                sub_nodes.update(path + [node])
                nodes.remove(node)

            for child in self.model.successors(node):
                walk(child, path + [node])

        walk(self.root, [])
        sub_graph = self.model.subgraph(sub_nodes)
        sub_graph.cardinalities = defaultdict(int)
        for node in sub_graph.nodes:
            sub_graph.cardinalities[node] = self.model.cardinalities[node]
        return sub_graph

    def _get_working_factors(self, query=None, id_attrs=[]):
        """
        Uses the evidence given to the query methods to modify the factors before running
        the variable elimination algorithm.
        Parameters
        ----------
        evidence: dict
            Dict of the form {variable: state}
        Returns
        -------
        dict: Modified working factors.
        """
        useful_var = list(query.keys()) + id_attrs
        sub_graph_model = self.steiner_tree(useful_var)

        elimination_order = []
        working_cpds = []
        working_factors = dict()
        for i, node in enumerate(self.topological_order_node[::-1]):
            ind = len(self.topological_order_node) - i - 1
            if node in sub_graph_model.nodes:
                elimination_order.append(node)
                cpd = copy.deepcopy(self.cpds[ind])
                working_cpds.append(cpd)
                working_factors[node] = [cpd]

        for node in sub_graph_model.nodes:
            for cpd in working_cpds:
                if node != cpd.variable and node in cpd.variables:
                    working_factors[node].append(cpd)

        return working_factors, sub_graph_model, elimination_order

    def query(self, query, n_distinct=None):
        """
        Compiles a ppl program into a fixed linear algebra program to speed up the inference
        ----------
        query: dict
            a dict key, value pair as {var: state_of_var_observed}
            None if no evidence
        n_distinct: dict
            a dict key, value pair as {var: probability of observed value in state}
            This is for the case, where we bin the continuous or large domain so each state now contains many observed
            value. Default to none, meaning no large domain.
        """
        working_factors, sub_graph_model, elimination_order = self._get_working_factors(query)
        for i, var in enumerate(elimination_order):
            root_var = i == (len(elimination_order) - 1)
            # print(var, len(working_factors[var]), root_var)
            if len(working_factors[var]) == 1:
                # leaf node in BN
                if var in query:
                    # print(var, query[var], n_distinct[var])
                    new_value = working_factors[var][0].values
                    if n_distinct:
                        if len(n_distinct[var]) == 1:
                            new_value = new_value[query[var]] * n_distinct[var][0]
                        else:
                            new_value = np.dot(n_distinct[var], new_value[query[var]])
                    else:
                        new_value = np.sum(new_value[query[var]], axis=0)
                    new_value = new_value.reshape(-1)
                    if root_var:
                        return new_value
                    assert len(new_value.shape) == 1, \
                        f"unreduced variable {working_factors[var][0].variables}, {new_value} with shape {new_value.shape}"
                else:
                    if root_var:
                        return 1
                    new_value = np.ones(working_factors[var][0].values.shape[-1])

                assert len(new_value.shape) == 1, f"unreduced variable {var}"
                working_factors[var][0].values = new_value
            else:
                if var in query:
                    if type(query[var]) != list:
                        assert type(query[var]) == int, f"invalid query {query[var]}"
                        query[var] = [query[var]]
                    self_value = working_factors[var][0].values[query[var]]  # Pr(var|Parent(var))
                    if n_distinct:
                        self_value = (self_value.transpose() * n_distinct[var]).transpose()
                    children_value = []
                    # check if all children has been reduced
                    for cpd in working_factors[var][1:]:
                        # print("y")
                        # print(cpd.variables)
                        child_value = cpd.values[query[var]]  # M(var) = Pr(child(var)|var)
                        assert len(child_value.shape) == 1, \
                            f"unreduced children {cpd.variables}, {child_value} with shape {child_value.shape}"
                        children_value.append(child_value)
                    if len(children_value) == 1:
                        children_value = children_value[0]
                    else:
                        # print(children_value)
                        children_value = np.prod(np.stack(children_value), axis=0)
                    if root_var:
                        new_value = np.dot(self_value, children_value)
                        return new_value
                    new_value = np.dot(np.transpose(self_value), children_value)
                else:
                    self_value = working_factors[var][0].values  # Pr(var|Parent(var))
                    children_value = []
                    # check if all children has been reduced
                    for cpd in working_factors[var][1:]:
                        child_value = cpd.values  # M(var) = Pr(child(var)|var)
                        assert len(child_value.shape) == 1, \
                            f"unreduced children {cpd.variables}, {child_value} with shape {child_value.shape}"
                        children_value.append(child_value)
                    if len(children_value) == 1:
                        children_value = children_value[0]
                    else:
                        children_value = np.prod(np.stack(children_value), axis=0)
                    if root_var:
                        new_value = np.dot(self_value, children_value)
                        return new_value
                    new_value = np.dot(np.transpose(self_value), children_value)
                assert len(new_value.shape) == 1, f"unreduced variable {var}"
                working_factors[var][0].values = new_value
        return 0

    def query_id_prob(self, query, id_attrs, n_distinct=None):
        """
        Compiles a ppl program into a fixed linear algebra program to speed up the expectation inference
        """
        working_factors, sub_graph_model, elimination_order = self._get_working_factors(query, id_attrs)
        prob_id = dict()
        for i, var in enumerate(elimination_order):
            # print(var, working_factors[var][0].values.shape)
            root_var = i == (len(elimination_order) - 1)
            # print(var, len(working_factors[var]))
            if len(working_factors[var]) == 1:
                # leaf node in BN
                if var in query:
                    new_value = working_factors[var][0].values
                    # print(new_value.shape)
                    if n_distinct:
                        # print(var, query[var], n_distinct[var])
                        if len(n_distinct[var]) == 1:
                            new_value = new_value[query[var]] * n_distinct[var][0]
                            new_value = new_value.reshape(-1)
                        else:
                            new_value = np.dot(n_distinct[var], new_value[query[var]])
                    else:
                        new_value = np.sum(new_value[query[var]], axis=0)
                    new_value = new_value.reshape(-1)
                    if root_var:
                        return None, new_value
                elif var in id_attrs:
                    new_value = working_factors[var][0].values
                    working_factors[var][0].extra_info = [var]
                    prob_id[var] = new_value
                    if root_var:
                        return [var], new_value
                    # print(new_value.shape)
                    # new_value = np.dot(self.fanouts[var], new_value)
                else:
                    if root_var:
                        assert False, "do not see id_attrs in this BN"
                    new_value = np.ones(working_factors[var][0].values.shape[-1])

                # print(new_value)
                # assert len(new_value.shape) == 1, f"unreduced variable {var} with shape {new_value.shape}"
                working_factors[var][0].values = new_value
            else:
                if var in id_attrs:
                    self_value = working_factors[var][0].values
                    self_value, children_value_id, children_id = get_working_factors_by_type(working_factors,
                                                                                             var, root_var, self_value,
                                                                                             None)
                    if len(children_value_id) == 0:
                        if root_var:
                            assert len(self_value.shape) == 1
                            return [var], self_value
                        else:
                            assert len(self_value.shape) == 2
                            new_value = self_value
                            working_factors[var][0].extra_info = [var]
                    elif len(children_value_id) == 1:
                        if len(children_value_id[0].shape) == 2:
                            n = children_value_id[0].shape[0]
                            m = children_value_id[0].shape[1]
                            assert m == self_value.shape[0]
                            if root_var:
                                return children_id + [var], children_value_id[0] * self_value
                            k = self_value.shape[1]
                            new_value = np.zeros((n, m, k))
                            for j in range(n):
                                new_value[j, :, :] = (self_value.T * children_value_id[0][j]).T
                            working_factors[var][0].extra_info = children_id + [var]
                        else:
                            # TODO: conditional independence exploration and simplification
                            assert False, "for getting the probability of more than three ids, we need to " \
                                          "explore their conditional independence and simplify the structure" \
                                          "to avoid memory explosion. The authors are currently working on it."
                    else:
                        # TODO: conditional independence exploration and simplification
                        assert False, "for getting the probability of more than three ids, we need to " \
                                      "explore their conditional independence and simplify the structure" \
                                      "to avoid memory explosion. The authors are currently working on it."
                else:
                    if var in query:
                        if type(query[var]) != list:
                            assert type(query[var]) == int, f"invalid query {query[var]}"
                            query[var] = [query[var]]
                        self_value = working_factors[var][0].values[query[var]]  # Pr(var|Parent(var))
                        if n_distinct:
                            self_value = (self_value.transpose() * n_distinct[var]).transpose()
                        self_value, children_value_id, children_id = get_working_factors_by_type(working_factors,
                                                                                                 var, root_var,
                                                                                                 self_value, query[var])
                    else:
                        self_value = working_factors[var][0].values
                        self_value, children_value_id, children_id = get_working_factors_by_type(working_factors,
                                                                                                 var, root_var,
                                                                                                 self_value, None)
                    if len(children_value_id) == 0:
                        if root_var:
                            assert False, "no id found"
                        else:
                            new_value = np.sum(self_value, axis=0)
                    else:
                        if len(children_value_id) == 1:
                            if len(children_value_id[0].shape) == 2:
                                new_value = np.dot(children_value_id[0], self_value)
                            elif len(children_value_id[0].shape) == 3:
                                m = children_value_id[0].shape[0]
                                n = children_value_id[0].shape[1]
                                k = children_value_id[0].shape[-1]
                                new_value = np.dot(children_value_id[0].reshape(m * n, k), self_value).reshape(m, n, -1)
                                if root_var:
                                    return children_id, new_value.reshape(m, n)
                            else:
                                # TODO: conditional independence exploration and simplification
                                assert False, "for getting the probability of more than three ids, we need to " \
                                              "explore their conditional independence and simplify the structure" \
                                              "to avoid memory explosion. The authors are currently working on it."
                            if root_var:
                                return children_id, new_value
                        elif len(children_value_id) == 2:
                            for v in children_value_id:
                                assert len(v.shape) == 2, "for getting the probability of more than three ids, " \
                                                          "we need to explore their conditional independence and " \
                                                          "simplify the structure to avoid memory explosion. " \
                                                          "The authors are currently working on it."
                            if root_var:
                                temp = children_value_id[1] * self_value
                                new_value = np.dot(children_value_id[0], np.transpose(temp))
                                return children_id, new_value
                            else:
                                n = children_value_id[0].shape[0]
                                e = children_value_id[0].shape[1]
                                m = children_value_id[1].shape[0]
                                k = self_value.shape[1]
                                assert e == children_value_id[1].shape[1] == self_value.shape[0]
                                new_value = np.zeros((n, m, k))
                                if n < m:
                                    for j in n:
                                        new_value[j, :, :] = np.dot(children_value_id[1] * children_value_id[0][j],
                                                                    self_value)
                                else:
                                    for j in m:
                                        new_value[:, j, :] = np.dot(children_value_id[0] * children_value_id[1][j],
                                                                    self_value)
                        else:
                            # TODO: conditional independence exploration and simplification
                            assert False, "for getting the probability of more than three ids, we need to " \
                                          "explore their conditional independence and simplify the structure" \
                                          "to avoid memory explosion. The authors are currently working on it."
                        working_factors[var][0].extra_info = children_id
                working_factors[var][0].values = new_value
        return 0

