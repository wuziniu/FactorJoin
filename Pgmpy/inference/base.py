#!/usr/bin/env python3

from collections import defaultdict
from itertools import chain

from Pgmpy.models import BayesianModel
from Pgmpy.models import JunctionTree
from Pgmpy.factors.discrete import TabularCPD


class Inference(object):
    """
    Base class for all inference algorithms.

    Converts BayesianModel and MarkovModel to a uniform representation so that inference
    algorithms can be applied. Also it checks if all the associated CPDs / Factors are
    consistent with the model.

    Initialize inference for a model.

    Parameters
    ----------
    model: pgmpy.models.BayesianModel or pgmpy.models.MarkovModel or pgmpy.models.NoisyOrModel
        model for which to initialize the inference object.
    """

    def __init__(self, model):
        self.model = model
        model.check_model()

        if isinstance(model, JunctionTree):
            self.variables = set(chain(*model.nodes()))
        else:
            self.variables = model.nodes()

        self.cardinality = {}
        self.factors = defaultdict(list)

        if isinstance(model, BayesianModel):
            self.state_names_map = {}
            for node in model.nodes():
                cpd = model.get_cpds(node)
                if isinstance(cpd, TabularCPD):
                    self.cardinality[node] = cpd.variable_card
                    cpd = cpd.to_factor()
                for var in cpd.scope():
                    self.factors[var].append(cpd)
                self.state_names_map.update(cpd.no_to_name)
