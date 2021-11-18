from collections import namedtuple
import itertools

import networkx as nx
import numpy as np
from tqdm import tqdm

from Pgmpy.inference import Inference
from Pgmpy.models import BayesianModel
from Pgmpy.utils.mathext import sample_discrete
from Pgmpy.sampling import _return_samples


State = namedtuple("State", ["var", "state"])


class BayesianModelSampling(Inference):
    """
    Class for sampling methods specific to Bayesian Models

    Parameters
    ----------
    model: instance of BayesianModel
        model on which inference queries will be computed
    """

    def __init__(self, model):
        if not isinstance(model, BayesianModel):
            raise TypeError(
                "Model expected type: BayesianModel, got type: ", type(model)
            )

        self.topological_order = list(nx.topological_sort(model))
        super(BayesianModelSampling, self).__init__(model)

    def forward_sample(self, size=1, return_type="dataframe"):
        """
        Generates sample(s) from joint distribution of the bayesian network.

        Parameters
        ----------
        size: int
            size of sample to be generated

        return_type: string (dataframe | recarray)
            Return type for samples, either of 'dataframe' or 'recarray'.
            Defaults to 'dataframe'

        Returns
        -------
        sampled: A pandas.DataFrame or a numpy.recarray object depending upon return_type argument
            the generated samples
        """
        types = [(var_name, "int") for var_name in self.topological_order]
        sampled = np.zeros(size, dtype=types).view(np.recarray)

        pbar = tqdm(self.topological_order)
        for node in pbar:
            pbar.set_description("Generating for node: {node}".format(node=node))
            cpd = self.model.get_cpds(node)
            states = range(self.cardinality[node])
            evidence = cpd.variables[:0:-1]
            if evidence:
                cached_values = self.pre_compute_reduce(variable=node)
                evidence = np.vstack([sampled[i] for i in evidence])
                weights = list(map(lambda t: cached_values[tuple(t)], evidence.T))
            else:
                weights = cpd.values
            sampled[node] = sample_discrete(states, weights, size)

        return _return_samples(return_type, sampled, self.state_names_map)

    def pre_compute_reduce(self, variable):
        variable_cpd = self.model.get_cpds(variable)
        variable_evid = variable_cpd.variables[:0:-1]
        cached_values = {}

        for state_combination in itertools.product(
            *[range(self.cardinality[var]) for var in variable_evid]
        ):
            states = list(zip(variable_evid, state_combination))
            cached_values[state_combination] = variable_cpd.reduce(
                states, inplace=False
            ).values

        return cached_values

    def rejection_sample(self, evidence=[], size=1, return_type="dataframe"):
        """
        Generates sample(s) from joint distribution of the bayesian network,
        given the evidence.

        Parameters
        ----------
        evidence: list of `pgmpy.factor.State` namedtuples
            None if no evidence
        size: int
            size of sample to be generated
        return_type: string (dataframe | recarray)
            Return type for samples, either of 'dataframe' or 'recarray'.
            Defaults to 'dataframe'

        Returns
        -------
        sampled: A pandas.DataFrame or a numpy.recarray object depending upon return_type argument
            the generated samples
        """
        # Covert evidence state names to number
        evidence = [
            (var, self.model.get_cpds(var).get_state_no(var, state))
            for var, state in evidence
        ]

        # If no evidence is given, it is equivalent to forward sampling.
        if len(evidence) == 0:
            return self.forward_sample(size)

        # Setup array to be returned
        types = [(var_name, "int") for var_name in self.topological_order]
        sampled = np.zeros(0, dtype=types).view(np.recarray)
        prob = 1
        i = 0

        # Do the sampling by generating samples from forward sampling and rejecting the
        # samples which do not match our evidence. Keep doing until we have enough
        # samples.
        pbar = tqdm(total=size)
        while i < size:
            _size = int(((size - i) / prob) * 1.5)
            _sampled = self.forward_sample(_size, "recarray")

            for evid in evidence:
                _sampled = _sampled[_sampled[evid[0]] == evid[1]]

            prob = max(len(_sampled) / _size, 0.01)
            sampled = np.append(sampled, _sampled)[:size]

            i += len(_sampled)
            pbar.update(len(_sampled))
        pbar.close()

        # Post process: Correct return type and replace state numbers with names.
        return _return_samples(return_type, sampled, self.state_names_map)

    def likelihood_weighted_sample(self, evidence=[], size=1, return_type="dataframe"):
        """
        Generates weighted sample(s) from joint distribution of the bayesian
        network, that comply with the given evidence.
        'Probabilistic Graphical Model Principles and Techniques', Koller and
        Friedman, Algorithm 12.2 pp 493.

        Parameters
        ----------
        evidence: list of `pgmpy.factor.State` namedtuples
            None if no evidence
        size: int
            size of sample to be generated
        return_type: string (dataframe | recarray)
            Return type for samples, either of 'dataframe' or 'recarray'.
            Defaults to 'dataframe'

        Returns
        -------
        sampled: A pandas.DataFrame or a numpy.recarray object depending upon return_type argument
            the generated samples with corresponding weights
        """
        # Covert evidence state names to number
        evidence = [
            (var, self.model.get_cpds(var).get_state_no(var, state))
            for var, state in evidence
        ]

        # Prepare the return array
        types = [(var_name, "int") for var_name in self.topological_order]
        types.append(("_weight", "float"))
        sampled = np.zeros(size, dtype=types).view(np.recarray)
        sampled["_weight"] = np.ones(size)
        evidence_dict = {var: st for var, st in evidence}

        # Do the sampling
        for node in self.topological_order:
            cpd = self.model.get_cpds(node)
            states = range(self.cardinality[node])
            evidence = cpd.get_evidence()

            if evidence:
                evidence_values = np.vstack([sampled[i] for i in evidence])
                cached_values = self.pre_compute_reduce(node)
                weights = list(
                    map(lambda t: cached_values[tuple(t)], evidence_values.T)
                )
                if node in evidence_dict:
                    sampled[node] = evidence_dict[node]
                    for i in range(size):
                        sampled["_weight"][i] *= weights[i][evidence_dict[node]]
                else:
                    sampled[node] = sample_discrete(states, weights)
            else:
                if node in evidence_dict:
                    sampled[node] = evidence_dict[node]
                    for i in range(size):
                        sampled["_weight"][i] *= cpd.values[evidence_dict[node]]
                else:
                    sampled[node] = sample_discrete(states, cpd.values, size)

        # Postprocess the samples: Correct return type and change state numbers to names
        return _return_samples(return_type, sampled, self.state_names_map)
