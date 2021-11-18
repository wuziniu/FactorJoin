from collections import defaultdict
import logging
from operator import mul
from functools import reduce

import networkx as nx
import numpy as np
import pandas as pd
try:
    from tqdm import tqdm
except:
    import tqdm
from joblib import Parallel, delayed

from Pgmpy.base import DAG
from Pgmpy.factors.discrete import (
    TabularCPD,
    JointProbabilityDistribution,
    DiscreteFactor,
)
from Pgmpy.factors.continuous import ContinuousFactor
from Pgmpy.models.MarkovModel import MarkovModel


class BayesianModel(DAG):
    """
    Base class for Bayesian Models.
    """

    def __init__(self, ebunch=None):
        """
        Initializes a Bayesian Model.
        A models stores nodes and edges with conditional probability
        distribution (cpd) and other attributes.

        models hold directed edges.  Self loops are not allowed neither
        multiple (parallel) edges.

        Nodes can be any hashable python object.

        Edges are represented as links between nodes.

        Parameters
        ----------
        data : input graph
            Data to initialize graph.  If data=None (default) an empty
            graph is created.  The data can be an edge list, or any
            NetworkX graph object.
        """
        super(BayesianModel, self).__init__()
        if ebunch:
            self.add_edges_from(ebunch)
        self.cpds = []
        self.cardinalities = self.get_cardinality()
        self.probs = dict()

    def add_edge(self, u, v, **kwargs):
        """
        Add an edge between u and v.

        The nodes u and v will be automatically added if they are
        not already in the graph

        Parameters
        ----------
        u,v : nodes
              Nodes can be any hashable python object.
        """
        if u == v:
            raise ValueError("Self loops are not allowed.")
        if u in self.nodes() and v in self.nodes() and nx.has_path(self, v, u):
            raise ValueError(
                "Loops are not allowed. Adding the edge from (%s->%s) forms a loop."
                % (u, v)
            )
        else:
            super(BayesianModel, self).add_edge(u, v, **kwargs)

    def remove_node(self, node):
        """
        Remove node from the model.

        Removing a node also removes all the associated edges, removes the CPD
        of the node and marginalizes the CPDs of it's children.

        Parameters
        ----------
        node : node
            Node which is to be removed from the model.

        Returns
        -------
        None
        """
        affected_nodes = [v for u, v in self.edges() if u == node]

        for affected_node in affected_nodes:
            node_cpd = self.get_cpds(node=affected_node)
            if node_cpd:
                node_cpd.marginalize([node], inplace=True)

        if self.get_cpds(node=node):
            self.remove_cpds(node)
        super(BayesianModel, self).remove_node(node)

    def remove_nodes_from(self, nodes):
        """
        Remove multiple nodes from the model.

        Removing a node also removes all the associated edges, removes the CPD
        of the node and marginalizes the CPDs of it's children.

        Parameters
        ----------
        nodes : list, set (iterable)
            Nodes which are to be removed from the model.

        Returns
        -------
        None
        """
        for node in nodes:
            self.remove_node(node)

    def add_cpds(self, *cpds):
        """
        Add CPD (Conditional Probability Distribution) to the Bayesian Model.

        Parameters
        ----------
        cpds  :  list, set, tuple (array-like)
            List of CPDs which will be associated with the model
        """
        for cpd in cpds:
            if not isinstance(cpd, (TabularCPD, ContinuousFactor)):
                raise ValueError("Only TabularCPD or ContinuousFactor can be added.")

            if set(cpd.scope()) - set(cpd.scope()).intersection(set(self.nodes())):
                raise ValueError("CPD defined on variable not in the model", cpd)

            for prev_cpd_index in range(len(self.cpds)):
                if self.cpds[prev_cpd_index].variable == cpd.variable:
                    logging.warning(
                        "Replacing existing CPD for {var}".format(var=cpd.variable)
                    )
                    self.cpds[prev_cpd_index] = cpd
                    break
            else:
                self.cpds.append(cpd)

    def get_cpds(self, node=None):
        """
        Returns the cpd of the node. If node is not specified returns all the CPDs
        that have been added till now to the graph

        Parameters
        ----------
        node: any hashable python object (optional)
            The node whose CPD we want. If node not specified returns all the
            CPDs added to the model.

        Returns
        -------
        A list of TabularCPDs.
        """
        if node is not None:
            if node not in self.nodes():
                raise ValueError("Node not present in the Directed Graph")
            else:
                for cpd in self.cpds:
                    if cpd.variable == node:
                        return cpd
        else:
            return self.cpds

    def remove_cpds(self, *cpds):
        """
        Removes the cpds that are provided in the argument.

        Parameters
        ----------
        *cpds: TabularCPD object
            A CPD object on any subset of the variables of the model which
            is to be associated with the model.
        """
        for cpd in cpds:
            if isinstance(cpd, str):
                cpd = self.get_cpds(cpd)
            self.cpds.remove(cpd)

    def get_cardinality(self, node=None):
        """
        Returns the cardinality of the node. Throws an error if the CPD for the
        queried node hasn't been added to the network.

        Parameters
        ----------
        node: Any hashable python object(optional).
              The node whose cardinality we want. If node is not specified returns a
              dictionary with the given variable as keys and their respective cardinality
              as values.

        Returns
        -------
        int or dict : If node is specified returns the cardinality of the node.
                      If node is not specified returns a dictionary with the given
                      variable as keys and their respective cardinality as values.
        """

        if node:
            return self.cardinalities[node]
        else:
            cardinalities = defaultdict(int)
            for cpd in self.cpds:
                cardinalities[cpd.variable] = cpd.cardinality[0]
            return cardinalities

    def check_model(self):
        """
        Check the model for various errors. This method checks for the following
        errors.

        * Checks if the sum of the probabilities for each state is equal to 1 (tol=0.01).
        * Checks if the CPDs associated with nodes are consistent with their parents.

        Returns
        -------
        check: boolean
            True if all the checks are passed
        """
        for node in self.nodes():
            cpd = self.get_cpds(node=node)

            if cpd is None:
                raise ValueError("No CPD associated with {}".format(node))
            elif isinstance(cpd, (TabularCPD, ContinuousFactor)):
                evidence = cpd.get_evidence()
                parents = self.get_parents(node)
                if set(evidence if evidence else []) != set(parents if parents else []):
                    raise ValueError(
                        "CPD associated with {node} doesn't have "
                        "proper parents associated with it.".format(node=node)
                    )
                if not cpd.is_valid_cpd():
                    raise ValueError(
                        "Sum or integral of conditional probabilites for node {node}"
                        " is not equal to 1.".format(node=node)
                    )
        return True

    def to_markov_model(self):
        """
        Converts bayesian model to markov model. The markov model created would
        be the moral graph of the bayesian model.
        """
        moral_graph = self.moralize()
        mm = MarkovModel(moral_graph.edges())
        mm.add_nodes_from(moral_graph.nodes())
        mm.add_factors(*[cpd.to_factor() for cpd in self.cpds])

        return mm

    def to_junction_tree(self):
        """
        Creates a junction tree (or clique tree) for a given bayesian model.

        For converting a Bayesian Model into a Clique tree, first it is converted
        into a Markov one.

        For a given markov model (H) a junction tree (G) is a graph
        1. where each node in G corresponds to a maximal clique in H
        2. each sepset in G separates the variables strictly on one side of the
        edge to other.
        """
        mm = self.to_markov_model()
        return mm.to_junction_tree()

    def fit(
        self, data, estimator=None, state_names=[], complete_samples_only=True, **kwargs
    ):
        """
        Estimates the CPD for each variable based on a given data set.

        Parameters
        ----------
        data: pandas DataFrame object
            DataFrame object with column names identical to the variable names of the network.
            (If some values in the data are missing the data cells should be set to `numpy.NaN`.
            Note that pandas converts each column containing `numpy.NaN`s to dtype `float`.)

        estimator: Estimator class
            One of:
            - MaximumLikelihoodEstimator (default)
            - BayesianEstimator: In this case, pass 'prior_type' and either 'pseudo_counts'
            or 'equivalent_sample_size' as additional keyword arguments.
            See `BayesianEstimator.get_parameters()` for usage.

        state_names: dict (optional)
            A dict indicating, for each variable, the discrete set of states
            that the variable can take. If unspecified, the observed values
            in the data set are taken to be the only possible states.

        complete_samples_only: bool (default `True`)
            Specifies how to deal with missing data, if present. If set to `True` all rows
            that contain `np.Nan` somewhere are ignored. If `False` then, for each variable,
            every row where neither the variable nor its parents are `np.NaN` is used.
        """
        from Pgmpy.estimators import MaximumLikelihoodEstimator, BaseEstimator

        if estimator is None:
            estimator = MaximumLikelihoodEstimator
        else:
            if not issubclass(estimator, BaseEstimator):
                raise TypeError("Estimator object should be a valid pgmpy estimator.")

        self.data_length = len(data)

        _estimator = estimator(
            self,
            data,
            state_names=state_names,
            complete_samples_only=complete_samples_only,
        )
        cpds_list = _estimator.get_parameters(**kwargs)
        self.add_cpds(*cpds_list)

    def update(self, data):
        from Pgmpy.estimators import MaximumLikelihoodEstimator
        estimator = MaximumLikelihoodEstimator
        #incrementally update the model with the given data point
        old_len = self.data_length
        current_len = len(data)
        new_len = old_len + current_len

        for node in sorted(self.model.nodes()):
            state_counts = estimator.state_counts(node)
            state_counts.loc[:, (state_counts == 0).all()] = 1
            current_values = np.array(state_counts)
            assert np.all(np.isclose(np.sum(current_values, axis=0), 1.0)), \
                f"invalid condition distribution of current_values"
            for cpd in self.cpds:
                if cpd.variable == node:
                    old_values = cpd.values
                    assert np.all(np.isclose(np.sum(old_values, axis=0), 1.0)), \
                        f"invalid condition distribution of old_values"
                    assert old_values.shape == current_values.shape, \
                        f"tabular shape mismatch {old_values.shape} {current_values.shape}"
                    new_values = old_values * (old_len/new_len) + current_values * (current_len/new_len)
                    assert np.all(np.isclose(np.sum(new_values, axis=0), 1.0)), \
                        f"invalid condition distribution of new_values"
                    cpd.values = new_values
        self.data_length += current_len



    def predict(self, data, n_jobs=-1):
        """
        Predicts states of all the missing variables.

        Parameters
        ----------
        data : pandas DataFrame object
            A DataFrame object with column names same as the variables in the model.
        """
        from Pgmpy.inference import VariableElimination

        if set(data.columns) == set(self.nodes()):
            raise ValueError("No variable missing in data. Nothing to predict")

        elif set(data.columns) - set(self.nodes()):
            raise ValueError("Data has variables which are not in the model")

        data_unique = data.drop_duplicates()
        missing_variables = set(self.nodes()) - set(data_unique.columns)
        #         pred_values = defaultdict(list)
        pred_values = []

        # Send state_names dict from one of the estimated CPDs to the inference class.
        model_inference = VariableElimination(self)
        pred_values = Parallel(n_jobs=n_jobs)(
            delayed(model_inference.map_query)(
                variables=missing_variables,
                evidence=data_point.to_dict(),
                show_progress=False,
            )
            for index, data_point in tqdm(
                data_unique.iterrows(), total=data_unique.shape[0]
            )
        )

        df_results = pd.DataFrame(pred_values, index=data_unique.index)
        data_with_results = pd.concat([data_unique, df_results], axis=1)
        return data.merge(data_with_results, how="left").loc[:, missing_variables]

    def predict_probability(self, data):
        """
        Predicts probabilities of all states of the missing variables.

        Parameters
        ----------
        data : pandas DataFrame object
            A DataFrame object with column names same as the variables in the model.
        """
        from Pgmpy.inference import VariableElimination

        if set(data.columns) == set(self.nodes()):
            raise ValueError("No variable missing in data. Nothing to predict")

        elif set(data.columns) - set(self.nodes()):
            raise ValueError("Data has variables which are not in the model")

        missing_variables = set(self.nodes()) - set(data.columns)
        pred_values = defaultdict(list)

        model_inference = VariableElimination(self)
        for _, data_point in data.iterrows():
            full_distribution = model_inference.query(
                variables=list(missing_variables),
                evidence=data_point.to_dict(),
                show_progress=False,
            )
            states_dict = {}
            for var in missing_variables:
                states_dict[var] = full_distribution.marginalize(
                    missing_variables - {var}, inplace=False
                )
            for k, v in states_dict.items():
                for l in range(len(v.values)):
                    state = self.get_cpds(k).state_names[k][l]
                    pred_values[k + "_" + str(state)].append(v.values[l])
        return pd.DataFrame(pred_values, index=data.index)

    def is_imap(self, JPD):
        """
        Checks whether the bayesian model is Imap of given JointProbabilityDistribution

        Parameters
        ----------
        JPD : An instance of JointProbabilityDistribution Class, for which you want to
            check the Imap

        Returns
        -------
        boolean : True if bayesian model is Imap for given Joint Probability Distribution
                False otherwise
        """
        if not isinstance(JPD, JointProbabilityDistribution):
            raise TypeError("JPD must be an instance of JointProbabilityDistribution")
        factors = [cpd.to_factor() for cpd in self.get_cpds()]
        factor_prod = reduce(mul, factors)
        JPD_fact = DiscreteFactor(JPD.variables, JPD.cardinality, JPD.values)
        if JPD_fact == factor_prod:
            return True
        else:
            return False

    def copy(self):
        """
        Returns a copy of the model.

        Returns
        -------
        BayesianModel: Copy of the model on which the method was called.
        """
        model_copy = BayesianModel()
        model_copy.add_nodes_from(self.nodes())
        model_copy.add_edges_from(self.edges())
        if self.cpds:
            model_copy.add_cpds(*[cpd.copy() for cpd in self.cpds])
        return model_copy

    def get_markov_blanket(self, node):
        """
        Returns a markov blanket for a random variable. In the case
        of Bayesian Networks, the markov blanket is the set of
        node's parents, its children and its children's other parents.

        Returns
        -------
        list(blanket_nodes): List of nodes contained in Markov Blanket

        Parameters
        ----------
        node: string, int or any hashable python object.
              The node whose markov blanket would be returned.
        """
        children = self.get_children(node)
        parents = self.get_parents(node)
        blanket_nodes = children + parents
        for child_node in children:
            blanket_nodes.extend(self.get_parents(child_node))
        blanket_nodes = set(blanket_nodes)
        blanket_nodes.remove(node)
        return list(blanket_nodes)
