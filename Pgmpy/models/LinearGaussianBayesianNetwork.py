from __future__ import division

import numpy as np
import networkx as nx

from Pgmpy.models import BayesianModel
from Pgmpy.factors.continuous import LinearGaussianCPD
from Pgmpy.factors.distributions import GaussianDistribution
import logging


class LinearGaussianBayesianNetwork(BayesianModel):
    """
    A Linear Gaussian Bayesian Network is a Bayesian Network, all
    of whose variables are continuous, and where all of the CPDs
    are linear Gaussians.

    An important result is that the linear Gaussian Bayesian Networks
    are an alternative representation for the class of multivariate
    Gaussian distributions.

    """

    def add_cpds(self, *cpds):
        """
        Add linear Gaussian CPD (Conditional Probability Distribution)
        to the Bayesian Model.

        Parameters
        ----------
        cpds  :  instances of LinearGaussianCPD
            List of LinearGaussianCPDs which will be associated with the model
        """
        for cpd in cpds:
            if not isinstance(cpd, LinearGaussianCPD):
                raise ValueError("Only LinearGaussianCPD can be added.")

            if set(cpd.variables) - set(cpd.variables).intersection(set(self.nodes())):
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

        Parameter
        ---------
        node: any hashable python object (optional)
            The node whose CPD we want. If node not specified returns all the
            CPDs added to the model.

        Returns
        -------
        A list of linear Gaussian CPDs.
        """
        return super(LinearGaussianBayesianNetwork, self).get_cpds(node)

    def remove_cpds(self, *cpds):
        """
        Removes the cpds that are provided in the argument.

        Parameters
        ----------
        *cpds: LinearGaussianCPD object
            A LinearGaussianCPD object on any subset of the variables
            of the model which is to be associated with the model.
        """
        return super(LinearGaussianBayesianNetwork, self).remove_cpds(*cpds)

    def to_joint_gaussian(self):
        """
        The linear Gaussian Bayesian Networks are an alternative
        representation for the class of multivariate Gaussian distributions.
        This method returns an equivalent joint Gaussian distribution.

        Returns
        -------
        GaussianDistribution: An equivalent joint Gaussian
                                   distribution for the network.

        Reference
        ---------
        Section 7.2, Example 7.3,
        Probabilistic Graphical Models, Principles and Techniques
        """
        variables = list(nx.topological_sort(self))
        mean = np.zeros(len(variables))
        covariance = np.zeros((len(variables), len(variables)))

        for node_idx in range(len(variables)):
            cpd = self.get_cpds(variables[node_idx])
            mean[node_idx] = (
                sum(
                    [
                        coeff * mean[variables.index(parent)]
                        for coeff, parent in zip(cpd.mean, cpd.evidence)
                    ]
                )
                + cpd.mean[0]
            )
            covariance[node_idx, node_idx] = (
                sum(
                    [
                        coeff
                        * coeff
                        * covariance[variables.index(parent), variables.index(parent)]
                        for coeff, parent in zip(cpd.mean, cpd.evidence)
                    ]
                )
                + cpd.variance
            )

        for node_i_idx in range(len(variables)):
            for node_j_idx in range(len(variables)):
                if covariance[node_j_idx, node_i_idx] != 0:
                    covariance[node_i_idx, node_j_idx] = covariance[
                        node_j_idx, node_i_idx
                    ]
                else:
                    cpd_j = self.get_cpds(variables[node_j_idx])
                    covariance[node_i_idx, node_j_idx] = sum(
                        [
                            coeff * covariance[node_i_idx, variables.index(parent)]
                            for coeff, parent in zip(cpd_j.mean, cpd_j.evidence)
                        ]
                    )

        return GaussianDistribution(variables, mean, covariance)

    def check_model(self):
        """
        Checks the model for various errors. This method checks for the following
        error -

        * Checks if the CPDs associated with nodes are consistent with their parents.

        Returns
        -------
        check: boolean
            True if all the checks pass.

        """
        for node in self.nodes():
            cpd = self.get_cpds(node=node)

            if isinstance(cpd, LinearGaussianCPD):
                if set(cpd.evidence) != set(self.get_parents(node)):
                    raise ValueError(
                        "CPD associated with %s doesn't have "
                        "proper parents associated with it." % node
                    )
        return True

    def get_cardinality(self, node):
        """
        Cardinality is not defined for continuous variables.
        """
        raise ValueError("Cardinality is not defined for continuous variables.")

    def fit(
        self, data, estimator=None, state_names=[], complete_samples_only=True, **kwargs
    ):
        """
        For now, fit method has not been implemented for LinearGaussianBayesianNetwork.
        """

        raise NotImplementedError(
            "fit method has not been implemented for LinearGaussianBayesianNetwork."
        )

    def predict(self, data):
        """
        For now, predict method has not been implemented for LinearGaussianBayesianNetwork.
        """
        raise NotImplementedError(
            "predict method has not been implemented for LinearGaussianBayesianNetwork."
        )

    def to_markov_model(self):
        """
        For now, to_markov_model method has not been implemented for LinearGaussianBayesianNetwork.
        """
        raise NotImplementedError(
            "to_markov_model method has not been implemented for LinearGaussianBayesianNetwork."
        )

    def is_imap(self, JPD):
        """
        For now, is_imap method has not been implemented for LinearGaussianBayesianNetwork.
        """
        raise NotImplementedError(
            "is_imap method has not been implemented for LinearGaussianBayesianNetwork."
        )
