from __future__ import division


import numpy as np

from Pgmpy.factors.base import BaseFactor
from Pgmpy.factors.distributions import GaussianDistribution, CustomDistribution


class ContinuousFactor(BaseFactor):
    """
    Base class for factors representing various multivariate
    representations.
    """

    def __init__(self, variables, pdf, *args, **kwargs):
        """
        Parameters
        ----------
        variables: list or array-like
            The variables for which the distribution is defined.

        pdf: function
            The probability density function of the distribution.
        """
        if not isinstance(variables, (list, tuple, np.ndarray)):
            raise TypeError(
                "variables: Expected type list or array-like, "
                "got type {var_type}".format(var_type=type(variables))
            )

        if len(set(variables)) != len(variables):
            raise ValueError("Variable names cannot be same.")

        variables = list(variables)

        if isinstance(pdf, str):
            if pdf == "gaussian":
                self.distribution = GaussianDistribution(
                    variables=variables,
                    mean=kwargs["mean"],
                    covariance=kwargs["covariance"],
                )
            else:
                raise NotImplementedError(
                    "{dist} distribution not supported.",
                    "Please use CustomDistribution".format(dist=pdf),
                )

        elif isinstance(pdf, CustomDistribution):
            self.distribution = pdf

        elif callable(pdf):
            self.distribution = CustomDistribution(
                variables=variables, distribution=pdf
            )

        else:
            raise ValueError(
                "pdf: Expected type: str or function, ",
                "Got: {instance}".format(instance=type(variables)),
            )

    @property
    def pdf(self):
        """
        Returns the pdf of the ContinuousFactor.
        """
        return self.distribution.pdf

    @property
    def variable(self):
        return self.scope()[0]

    def scope(self):
        """
        Returns the scope of the factor.

        Returns
        -------
        list: List of variable names in the scope of the factor.
        """
        return self.distribution.variables

    def get_evidence(self):
        return self.scope()[1:]

    def assignment(self, *args):
        """
        Returns a list of pdf assignments for the corresponding values.

        Parameters
        ----------
        *args: values
            Values whose assignment is to be computed.
        """
        return self.distribution.assignment(*args)

    def copy(self):
        """
        Return a copy of the distribution.

        Returns
        -------
        ContinuousFactor object: copy of the distribution
        """
        return ContinuousFactor(self.scope(), self.distribution.copy())

    def discretize(self, method, *args, **kwargs):
        """
        Discretizes the continuous distribution into discrete
        probability masses using various methods.

        Parameters
        ----------
        method : A Discretizer Class from pgmpy.discretize

        *args, **kwargs:
            The parameters to be given to the Discretizer Class.

        Returns
        -------
        An n-D array or a DiscreteFactor object according to the discretiztion
        method used.
        """
        return method(self, *args, **kwargs).get_discrete_values()

    def reduce(self, values, inplace=True):
        """
        Reduces the factor to the context of the given variable values.

        Parameters
        ----------
        values: list, array-like
            A list of tuples of the form (variable_name, variable_value).

        inplace: boolean
            If inplace=True it will modify the factor itself, else would return
            a new ContinuousFactor object.

        Returns
        -------
        ContinuousFactor or None: if inplace=True (default) returns None
                                  if inplace=False returns a new ContinuousFactor instance.
        """
        phi = self if inplace else self.copy()

        phi.distribution = phi.distribution.reduce(values, inplace=False)
        if not inplace:
            return phi

    def marginalize(self, variables, inplace=True):
        """
        Marginalize the factor with respect to the given variables.

        Parameters
        ----------
        variables: list, array-like
            List of variables with respect to which factor is to be maximized.

        inplace: boolean
            If inplace=True it will modify the factor itself, else would return
            a new ContinuousFactor instance.

        Returns
        -------
        DiscreteFactor or None: if inplace=True (default) returns None
                        if inplace=False returns a new ContinuousFactor instance.

        """
        phi = self if inplace else self.copy()
        phi.distribution = phi.distribution.marginalize(variables, inplace=False)

        if not inplace:
            return phi

    def normalize(self, inplace=True):
        """
        Normalizes the pdf of the continuous factor so that it integrates to
        1 over all the variables.

        Parameters
        ----------
        inplace: boolean
            If inplace=True it will modify the factor itself, else would return
            a new factor.

        Returns
        -------
        ContinuousFactor or None:
             if inplace=True (default) returns None
             if inplace=False returns a new ContinuousFactor instance.
        """
        phi = self if inplace else self.copy()
        phi.distribution.normalize(inplace=True)

        if not inplace:
            return phi

    def is_valid_cpd(self):
        return self.distribution.is_valid_cpd()

    def _operate(self, other, operation, inplace=True):
        """
        Gives the ContinuousFactor operation (product or divide) with
        the other factor.

        Parameters
        ----------
        other: ContinuousFactor
            The ContinuousFactor to be multiplied.

        operation: String
            'product' for multiplication operation and 'divide' for
            division operation.

        inplace: boolean
            If inplace=True it will modify the factor itself, else would return
            a new factor.

        Returns
        -------
        ContinuousFactor or None:
                        if inplace=True (default) returns None
                        if inplace=False returns a new `DiscreteFactor` instance.

        """
        if not isinstance(other, ContinuousFactor):
            raise TypeError(
                "ContinuousFactor objects can only be multiplied ",
                "or divided with another ContinuousFactor object. ",
                "Got {other_type}, expected: ContinuousFactor.".format(
                    other_type=type(other)
                ),
            )

        phi = self if inplace else self.copy()
        phi.distribution = phi.distribution._operate(
            other=other.distribution, operation=operation, inplace=False
        )

        if not inplace:
            return phi

    def product(self, other, inplace=True):
        """
        Gives the ContinuousFactor product with the other factor.

        Parameters
        ----------
        other: ContinuousFactor
            The ContinuousFactor to be multiplied.

        Returns
        -------
        ContinuousFactor or None:
                        if inplace=True (default) returns None
                        if inplace=False returns a new `ContinuousFactor` instance.
        """
        return self._operate(other, "product", inplace)

    def divide(self, other, inplace=True):
        """
        Gives the ContinuousFactor divide with the other factor.

        Parameters
        ----------
        other: ContinuousFactor
            The ContinuousFactor to be divided.

        Returns
        -------
        ContinuousFactor or None:
                        if inplace=True (default) returns None
                        if inplace=False returns a new `ContinuousFactor` instance.
        """
        if set(other.scope()) - set(self.scope()):
            raise ValueError("Scope of divisor should be a subset of dividend")

        return self._operate(other, "divide", inplace)

    def __mul__(self, other):
        return self.product(other, inplace=False)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return self.divide(other, inplace=False)

    __div__ = __truediv__
