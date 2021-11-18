#!/usr/bin/env python3

from functools import reduce

from Pgmpy.factors.base import BaseFactor


class FactorSet(object):
    r"""
    Base class of *DiscreteFactor Sets*.

    A factor set provides a compact representation of  higher dimensional factor
    :math:`\phi_1\cdot\phi_2\cdots\phi_n`

    For example the factor set corresponding to factor :math:`\phi_1\cdot\phi_2` would be the union of the factors
    :math:`\phi_1` and :math:`\phi_2` i.e. factor set :math:`\vec\phi = \phi_1 \cup \phi_2`.
    """

    def __init__(self, *factors_list):
        """
        Initialize the factor set class.

        Parameters
        ----------
        factors_list: Factor1, Factor2, ....
            All the factors whose product is represented by the factor set
        """
        if not all(isinstance(phi, BaseFactor) for phi in factors_list):
            raise TypeError("Input parameters must be child classes of BaseFactor")
        self.factors = set([factor.copy() for factor in factors_list])

    def add_factors(self, *factors):
        """
        Adds factors to the factor set.

        Parameters
        ----------
        factors: Factor1, Factor2, ...., Factorn
            factors to be added into the factor set
        """
        self.factors.update(factors)

    def remove_factors(self, *factors):
        """
        Removes factors from the factor set.

        Parameters
        ----------
        factors: Factor1, Factor2, ...., Factorn
            factors to be removed from the factor set
        """
        for factor in factors:
            self.factors.remove(factor)

    def get_factors(self):
        """
        Returns all the factors present in factor set.
        """
        return self.factors

    def product(self, factorset, inplace=True):
        r"""
        Return the factor sets product with the given factor sets

        Suppose :math:`\vec\phi_1` and :math:`\vec\phi_2` are two factor sets then their product is a another factors
        set :math:`\vec\phi_3 = \vec\phi_1 \cup \vec\phi_2`.

        Parameters
        ----------
        factorsets: FactorSet1, FactorSet2, ..., FactorSetn
            FactorSets to be multiplied

        inplace: A boolean (Default value True)
            If inplace = True , then it will modify the FactorSet object, if False, it will
            return a new FactorSet object.

        Returns
        --------
        If inpalce = False, will return a new FactorSet object, which is product of two factors
        """
        factor_set = self if inplace else self.copy()
        factor_set1 = factorset.copy()

        factor_set.add_factors(*factor_set1.factors)

        if not inplace:
            return factor_set

    def divide(self, factorset, inplace=True):
        r"""
        Returns a new factor set instance after division by the factor set

        Division of two factor sets :math:`\frac{\vec\phi_1}{\vec\phi_2}` basically translates to union of all the
        factors present in :math:`\vec\phi_2` and :math:`\frac{1}{\phi_i}` of all the factors present in
        :math:`\vec\phi_2`.

        Parameters
        ----------
        factorset: FactorSet
            The divisor

        inplace: A boolean (Default value True)
            If inplace = True ,then it will modify the FactorSet object, if False then will
            return a new FactorSet object.

        Returns
        --------
        If inplace = False, will return a new FactorSet Object which is division of
        given factors.
        """
        factor_set = self if inplace else self.copy()
        factor_set1 = factorset.copy()

        factor_set.add_factors(
            *[phi.identity_factor() / phi for phi in factor_set1.factors]
        )

        if not inplace:
            return factor_set

    def marginalize(self, variables, inplace=True):
        """
        Marginalizes the factors present in the factor sets with respect to the given variables.

        Parameters
        ----------
        variables: list, array-like
            List of the variables to be marginalized.

        inplace: boolean (Default value True)
            If inplace=True it will modify the factor set itself, would create a new factor set

        Returns
        -------
        If inplace = False, will return a new marginalized FactorSet object.
        """
        if isinstance(variables, str):
            raise TypeError("Expected list or array-like type got type str")

        factor_set = self if inplace else self.copy()

        factors_to_be_marginalized = set(
            filter(lambda x: set(x.scope()).intersection(variables), factor_set.factors)
        )

        for factor in factors_to_be_marginalized:
            variables_to_be_marginalized = list(
                set(factor.scope()).intersection(variables)
            )
            if inplace:
                factor.marginalize(variables_to_be_marginalized, inplace=True)
            else:
                factor_set.remove_factors(factor)
                factor_set.add_factors(
                    factor.marginalize(variables_to_be_marginalized, inplace=False)
                )

        if not inplace:
            return factor_set

    def __mul__(self, other):
        return self.product(other)

    def __truediv__(self, other):
        return self.divide(other)

    def __str__(self):
        return self.factors.__str__()

    def copy(self):
        """
        Create a copy of factor set.
        """
        # No need to have copies of factors as argument because __init__ method creates copies.
        return FactorSet(*self.factors)


def factorset_product(*factorsets_list):
    r"""
    Base method used for product of factor sets.

    Suppose :math:`\vec\phi_1` and :math:`\vec\phi_2` are two factor sets then their product is a another factors set
    :math:`\vec\phi_3 = \vec\phi_1 \cup \vec\phi_2`.

    Parameters
    ----------
    factorsets_list: FactorSet1, FactorSet2, ..., FactorSetn
        All the factor sets to be multiplied

    Returns
    -------
    Product of factorset in factorsets_list
    """
    if not all(isinstance(factorset, FactorSet) for factorset in factorsets_list):
        raise TypeError("Input parameters must be FactorSet instances")
    return reduce(lambda x, y: x.product(y, inplace=False), factorsets_list)


def factorset_divide(factorset1, factorset2):
    r"""
    Base method for dividing two factor sets.

    Division of two factor sets :math:`\frac{\vec\phi_1}{\vec\phi_2}` basically translates to union of all the factors
    present in :math:`\vec\phi_2` and :math:`\frac{1}{\phi_i}` of all the factors present in :math:`\vec\phi_2`.

    Parameters
    ----------
    factorset1: FactorSet
        The dividend

    factorset2: FactorSet
        The divisor

    Returns
    -------
    The division of factorset1 and factorset2
    """
    if not isinstance(factorset1, FactorSet) or not isinstance(factorset2, FactorSet):
        raise TypeError("factorset1 and factorset2 must be FactorSet instances")
    return factorset1.divide(factorset2, inplace=False)
