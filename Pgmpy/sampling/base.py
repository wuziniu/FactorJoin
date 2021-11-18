from warnings import warn

import numpy as np

from Pgmpy import HAS_PANDAS
from Pgmpy.utils import _check_1d_array_object, _check_length_equal
import pandas


class BaseGradLogPDF(object):
    """
    Base class for evaluating gradient log of probability density function/ distribution

    Classes inheriting this base class can be passed as an argument for
    finding gradient log of probability distribution in inference algorithms

    The class should initialize  self.grad_log and self.log_pdf

    Parameters
    ----------
    variable_assignments : A 1d array like object (numpy.ndarray or list)
        Vector representing values(assignments) of variables at which we want to find gradient and log

    model : An instance of pgmpy.models
    """

    def __init__(self, variable_assignments, model):

        self.variable_assignments = _check_1d_array_object(
            variable_assignments, "variable_assignments"
        )
        _check_length_equal(
            variable_assignments,
            model.variables,
            "variable_assignments",
            "model.variables",
        )

        self.model = model

        # The gradient log of probability distribution at position
        self.grad_log = None

        # The gradient log of probability distribution at position
        self.log_pdf = None

    def get_gradient_log_pdf(self):
        """
        Returns the gradient log and log of model at given position

        Returns
        -------
        Returns a tuple of following types

        numpy.array: Representing gradient log of model at given position

        float: Representing log of model at given position
        """
        return self.grad_log, self.log_pdf


class GradLogPDFGaussian(BaseGradLogPDF):
    """
    Class for finding gradient and gradient log of Joint Gaussian Distribution
    Inherits pgmpy.inference.base_continuous.BaseGradLogPDF
    Here model must be an instance of GaussianDistribution

    Parameters
    ----------
    variable_assignments : A 1d array like object (numpy.ndarray or list)
        Vector representing values of variables at which we want to find gradient and log

    model : An instance of pgmpy.models.GaussianDistribution
    """

    def __init__(self, variable_assignments, model):
        BaseGradLogPDF.__init__(self, variable_assignments, model)
        self.grad_log, self.log_pdf = self._get_gradient_log_pdf()

    def _get_gradient_log_pdf(self):
        """
        Method that finds gradient and its log at position
        """
        sub_vec = self.variable_assignments - self.model.mean.flatten()
        grad = -np.dot(self.model.precision_matrix, sub_vec)
        log_pdf = 0.5 * np.dot(sub_vec, grad)

        return grad, log_pdf


class BaseSimulateHamiltonianDynamics(object):
    """
    Base class for proposing new values of position and momentum by simulating Hamiltonian Dynamics.

    Classes inheriting this base class can be passed as an argument for
    simulate_dynamics in inference algorithms.

    Parameters
    ----------
    model : An instance of pgmpy.models
        Model for which DiscretizeTime object is initialized

    position : A 1d array like object (numpy.ndarray or list)
        Vector representing values of parameter position( or X)

    momentum: A 1d array like object (numpy.ndarray or list)
        Vector representing the proposed value for momentum (velocity)

    stepsize: Float
        stepsize for the simulating dynamics

    grad_log_pdf : A subclass of pgmpy.inference.continuous.BaseGradLogPDF
        A class for finding gradient log and log of distribution

    grad_log_position: A 1d array like object, defaults to None
        Vector representing gradient log at given position
        If None, then will be calculated
    """

    def __init__(
        self, model, position, momentum, stepsize, grad_log_pdf, grad_log_position=None
    ):

        position = _check_1d_array_object(position, "position")

        momentum = _check_1d_array_object(momentum, "momentum")

        if not issubclass(grad_log_pdf, BaseGradLogPDF):
            raise TypeError(
                "grad_log_pdf must be an instance"
                + " of pgmpy.inference.continuous.base.BaseGradLogPDF"
            )

        _check_length_equal(position, momentum, "position", "momentum")
        _check_length_equal(position, model.variables, "position", "model.variables")

        if grad_log_position is None:
            grad_log_position, _ = grad_log_pdf(position, model).get_gradient_log_pdf()

        else:
            grad_log_positon = _check_1d_array_object(
                grad_log_position, "grad_log_position"
            )
            _check_length_equal(
                grad_log_position, position, "grad_log_position", "position"
            )

        self.position = position
        self.momentum = momentum
        self.stepsize = stepsize
        self.model = model
        self.grad_log_pdf = grad_log_pdf
        self.grad_log_position = grad_log_position

        # new_position is the new proposed position, new_momentum is the new proposed momentum, new_grad_lop
        # is the value of grad log at new_position
        self.new_position = self.new_momentum = self.new_grad_logp = None

    def get_proposed_values(self):
        """
        Returns the new proposed values of position and momentum

        Returns
        -------
        Returns a tuple of following type (in order)

        numpy.array: A 1d numpy.array representing new proposed value of position

        numpy.array: A 1d numpy.array representing new proposed value of momentum

        numpy.array: A 1d numpy.array representing gradient of log distribution at new proposed value of position
        """
        return self.new_position, self.new_momentum, self.new_grad_logp


class LeapFrog(BaseSimulateHamiltonianDynamics):
    """
    Class for simulating hamiltonian dynamics using leapfrog method
    Inherits pgmpy.inference.base_continuous.BaseSimulateHamiltonianDynamics

    Parameters
    ----------
    model : An instance of pgmpy.models
        Model for which DiscretizeTime object is initialized

    position : A 1d array like object (numpy.ndarray or list)
        Vector representing values of parameter position( or X)

    momentum: A 1d array like object (numpy.ndarray or list)
        Vector representing the proposed value for momentum (velocity)

    stepsize: Float
        stepsize for the simulating dynamics

    grad_log_pdf : A subclass of pgmpy.inference.continuous.BaseGradLogPDF, defaults to None
        A class for evaluating gradient log and log of distribution for a given assignment of variables
        If None, then model.get_gradient_log_pdf will be used

    grad_log_position: A 1d array like object, defaults to None
        Vector representing gradient log at given position
        If None, then will be calculated
    """

    def __init__(
        self, model, position, momentum, stepsize, grad_log_pdf, grad_log_position=None
    ):

        BaseSimulateHamiltonianDynamics.__init__(
            self, model, position, momentum, stepsize, grad_log_pdf, grad_log_position
        )

        (
            self.new_position,
            self.new_momentum,
            self.new_grad_logp,
        ) = self._get_proposed_values()

    def _get_proposed_values(self):
        """
        Method to perform time splitting using leapfrog
        """
        # Take half step in time for updating momentum
        momentum_bar = self.momentum + 0.5 * self.stepsize * self.grad_log_position

        # Take full step in time for updating position position
        position_bar = self.position + self.stepsize * momentum_bar

        grad_log, _ = self.grad_log_pdf(position_bar, self.model).get_gradient_log_pdf()

        # Take remaining half step in time for updating momentum
        momentum_bar = momentum_bar + 0.5 * self.stepsize * grad_log

        return position_bar, momentum_bar, grad_log


class ModifiedEuler(BaseSimulateHamiltonianDynamics):
    """
    Class for simulating Hamiltonian Dynamics using Modified euler method
    Inherits pgmpy.inference.base_continuous.BaseSimulateHamiltonianDynamics

    Parameters
    ----------
    model: An instance of pgmpy.models
        Model for which DiscretizeTime object is initialized

    position: A 1d array like object (numpy.ndarray or list)
        Vector representing values of parameter position( or X)

    momentum: A 1d array like object (numpy.ndarray or list)
        Vector representing the proposed value for momentum (velocity)

    stepsize: Float
        stepsize for the simulating dynamics

    grad_log_pdf: A subclass of pgmpy.inference.continuous.BaseGradLogPDF, defaults to None
        A class for finding gradient log and log of distribution
        If None, then will use model.get_gradient_log_pdf

    grad_log_position: A 1d array like object, defaults to None
        Vector representing gradient log at given position
        If None, then will be calculated
    """

    def __init__(
        self, model, position, momentum, stepsize, grad_log_pdf, grad_log_position=None
    ):

        BaseSimulateHamiltonianDynamics.__init__(
            self, model, position, momentum, stepsize, grad_log_pdf, grad_log_position
        )

        (
            self.new_position,
            self.new_momentum,
            self.new_grad_logp,
        ) = self._get_proposed_values()

    def _get_proposed_values(self):
        """
        Method to perform time splitting using Modified euler method
        """
        # Take full step in time and update momentum
        momentum_bar = self.momentum + self.stepsize * self.grad_log_position

        # Take full step in time and update position
        position_bar = self.position + self.stepsize * momentum_bar

        grad_log, _ = self.grad_log_pdf(position_bar, self.model).get_gradient_log_pdf()

        return position_bar, momentum_bar, grad_log


def _return_samples(return_type, samples, state_names_map=None):
    """
        A utility function to return samples according to type
    """
    if return_type.lower() == "dataframe":
        if HAS_PANDAS:
            df = pandas.DataFrame.from_records(samples)
            if state_names_map is not None:
                for var in df.columns:
                    if var != "_weight":
                        df[var] = df[var].apply(lambda t: state_names_map[var][t])
            return df
        else:
            warn("Pandas installation not found. Returning numpy.recarray object")
            return samples
    else:
        return samples
