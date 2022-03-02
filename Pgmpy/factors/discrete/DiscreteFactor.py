from __future__ import division

from itertools import product
from collections import namedtuple
from warnings import warn
#import numba
#from numba import jit    #If your model is really large, use numba, otherwise don't use it
import numpy as np

from Pgmpy.factors.base import BaseFactor
from Pgmpy.utils import StateNameMixin
from Pgmpy.extern import tabulate

State = namedtuple("State", ["var", "state"])

class DiscreteFactor(BaseFactor, StateNameMixin):
    """
    Base class for DiscreteFactor.
    """

    def __init__(self, variables, cardinality, values, state_names={}):
        """
        Initialize a factor class.

        Defined above, we have the following mapping from variable
        assignments to the index of the row vector in the value field:
        +-----+-----+-----+-------------------+
        |  x1 |  x2 |  x3 |    phi(x1, x2, x3)|
        +-----+-----+-----+-------------------+
        | x1_0| x2_0| x3_0|     phi.value(0)  |
        +-----+-----+-----+-------------------+
        | x1_0| x2_0| x3_1|     phi.value(1)  |
        +-----+-----+-----+-------------------+
        | x1_0| x2_1| x3_0|     phi.value(2)  |
        +-----+-----+-----+-------------------+
        | x1_0| x2_1| x3_1|     phi.value(3)  |
        +-----+-----+-----+-------------------+
        | x1_1| x2_0| x3_0|     phi.value(4)  |
        +-----+-----+-----+-------------------+
        | x1_1| x2_0| x3_1|     phi.value(5)  |
        +-----+-----+-----+-------------------+
        | x1_1| x2_1| x3_0|     phi.value(6)  |
        +-----+-----+-----+-------------------+
        | x1_1| x2_1| x3_1|     phi.value(7)  |
        +-----+-----+-----+-------------------+

        Parameters
        ----------
        variables: list, array-like
            List of variables in the scope of the factor.

        cardinality: list, array_like
            List of cardinalities of each variable. `cardinality` array must have a value
            corresponding to each variable in `variables`.

        values: list, array_like
            List of values of factor.
            A DiscreteFactor's values are stored in a row vector in the value
            using an ordering such that the left-most variables as defined in
            `variables` cycle through their values the fastest.

        """
        if isinstance(variables, str):
            raise TypeError("Variables: Expected type list or array like, got string")

        values = np.array(values, dtype=float)

        if len(cardinality) != len(variables):
            raise ValueError(
                "Number of elements in cardinality must be equal to number of variables"
            )

        if values.size != np.product(cardinality):
            raise ValueError(
                "Values array must be of size: {size}".format(
                    size=np.product(cardinality)
                )
            )

        if len(set(variables)) != len(variables):
            raise ValueError("Variable names cannot be same")

        self.variables = list(variables)
        self.cardinality = np.array(cardinality, dtype=int)
        self.values = values.reshape(self.cardinality)
        
        # Set the state names
        super(DiscreteFactor, self).store_state_names(
            variables, cardinality, state_names
        )

    def scope(self):
        """
        Returns the scope of the factor.

        Returns
        -------
        list: List of variable names in the scope of the factor.
        """
        return self.variables

    #@jit
    def get_cardinality(self, variables):
        """
        Returns cardinality of a given variable

        Parameters
        ----------
        variables: list, array-like
                A list of variable names.

        Returns
        -------
        dict: Dictionary of the form {variable: variable_cardinality}
        {'x1': 2, 'x2': 3}
        """
        if isinstance(variables, str):
            raise TypeError("variables: Expected type list or array-like, got type str")

        if not all([var in self.variables for var in variables]):
            raise ValueError("Variable not in scope")

        return {var: self.cardinality[self.variables.index(var)] for var in variables}

    #@jit
    def assignment(self, index):
        """
        Returns a list of assignments for the corresponding index.

        Parameters
        ----------
        index: list, array-like
            List of indices whose assignment is to be computed

        Returns
        -------
        list: Returns a list of full assignments of all the variables of the factor.
        """
        index = np.array(index)

        max_possible_index = np.prod(self.cardinality) - 1
        if not all(i <= max_possible_index for i in index):
            raise IndexError("Index greater than max possible index")

        assignments = np.zeros((len(index), len(self.scope())), dtype=np.int)
        rev_card = self.cardinality[::-1]
        for i, card in enumerate(rev_card):
            assignments[:, i] = index % card
            index = index // card

        assignments = assignments[:, ::-1]

        return [
            [
                (key, self.get_state_names(key, val))
                for key, val in zip(self.variables, values)
            ]
            for values in assignments
        ]

    def identity_factor(self):
        """
        Returns the identity factor.

        Def: The identity factor of a factor has the same scope and cardinality as the original factor,
             but the values for all the assignments is 1. When the identity factor is multiplied with
             the factor it returns the factor itself.

        Returns
        -------
        DiscreteFactor: The identity factor
        """
        return DiscreteFactor(
            variables=self.variables,
            cardinality=self.cardinality,
            values=np.ones(self.values.size),
            state_names=self.state_names,
        )

    def marginalize(self, variables, inplace=True):
        """
        Modifies the factor with marginalized values.

        Parameters
        ----------
        variables: list, array-like
            List of variables over which to marginalize.

        inplace: boolean
            If inplace=True it will modify the factor itself, else would return
            a new factor.

        Returns
        -------
        DiscreteFactor or None: if inplace=True (default) returns None
                        if inplace=False returns a new `DiscreteFactor` instance.
        """

        if isinstance(variables, str):
            raise TypeError("variables: Expected type list or array-like, got type str")

        phi = self if inplace else self.copy()

        for var in variables:
            if var not in phi.variables:
                raise ValueError("{var} not in scope.".format(var=var))

        var_indexes = [phi.variables.index(var) for var in variables]

        index_to_keep = sorted(set(range(len(self.variables))) - set(var_indexes))
        phi.variables = [phi.variables[index] for index in index_to_keep]
        phi.cardinality = phi.cardinality[index_to_keep]
        phi.del_state_names(variables)

        phi.values = np.sum(phi.values, axis=tuple(var_indexes))

        if not inplace:
            return phi

    def maximize(self, variables, inplace=True):
        """
        Maximizes the factor with respect to `variables`.

        Parameters
        ----------
        variables: list, array-like
            List of variables with respect to which factor is to be maximized

        inplace: boolean
            If inplace=True it will modify the factor itself, else would return
            a new factor.

        Returns
        -------
        DiscreteFactor or None: if inplace=True (default) returns None
                        if inplace=False returns a new `DiscreteFactor` instance.
        """
        if isinstance(variables, str):
            raise TypeError("variables: Expected type list or array-like, got type str")

        phi = self if inplace else self.copy()

        for var in variables:
            if var not in phi.variables:
                raise ValueError("{var} not in scope.".format(var=var))

        var_indexes = [phi.variables.index(var) for var in variables]

        index_to_keep = sorted(set(range(len(self.variables))) - set(var_indexes))
        phi.variables = [phi.variables[index] for index in index_to_keep]
        phi.cardinality = phi.cardinality[index_to_keep]
        phi.del_state_names(variables)

        phi.values = np.max(phi.values, axis=tuple(var_indexes))

        if not inplace:
            return phi
    
    #@jit
    def normalize(self, inplace=True):
        """
        Normalizes the values of factor so that they sum to 1.

        Parameters
        ----------
        inplace: boolean
            If inplace=True it will modify the factor itself, else would return
            a new factor

        Returns
        -------
        DiscreteFactor or None: if inplace=True (default) returns None
                        if inplace=False returns a new `DiscreteFactor` instance.
        """
        phi = self if inplace else self.copy()

        phi.values = phi.values / phi.values.sum()

        if not inplace:
            return phi

    #@jit
    def reduce(self, values, inplace=True):
        """
        Reduces the factor to the context of given variable values.

        Parameters
        ----------
        values: list, array-like
            A list of tuples of the form (variable_name, variable_state).

        inplace: boolean
            If inplace=True it will modify the factor itself, else would return
            a new factor.

        Returns
        -------
        DiscreteFactor or None: if inplace=True (default) returns None
                        if inplace=False returns a new `DiscreteFactor` instance.
        """
        # Check if values is an array
        if isinstance(values, str):
            raise TypeError("values: Expected type list or array-like, got type str")

        if not all([isinstance(state_tuple, tuple) for state_tuple in values]):
            raise TypeError(
                "values: Expected type list of tuples, get type {type}", type(values[0])
            )

        # Check if all variables in values are in the factor
        for var, _ in values:
            if var not in self.variables:
                raise ValueError(f"The variable: {var} is not in the factor")

        phi = self if inplace else self.copy()
        # Convert the state names to state number. If state name not found treat them as
        # state numbers.
        new_state_names = dict()
        for i, (var, state_name) in enumerate(values):
            if type(state_name) == list:
                new_state_names[var] = state_name
                values[i] = (var, [self.get_state_no(var, no) for no in state_name])
            else:
                values[i] = (var, self.get_state_no(var, state_name))
        var_index_to_del = []
        slice_ = [slice(None)] * len(self.variables)
        point_query = True
        cardinality = phi.cardinality
        state_names = phi.state_names
        del_state_names = []
        for var, state in values:
            var_index = phi.variables.index(var)
            slice_[var_index] = state
            if type(state) == list:
                point_query = False
                cardinality[var_index] = len(state)
                state_names[var] = new_state_names[var]
            else:
                var_index_to_del.append(var_index)
                del_state_names.append(var)
        var_index_to_keep = sorted(
            set(range(len(phi.variables))) - set(var_index_to_del)
        )
        # set difference is not gaurenteed to maintain ordering
        phi.variables = [phi.variables[index] for index in var_index_to_keep]
        phi.cardinality = cardinality[var_index_to_keep]
        phi.del_state_names(del_state_names)
        phi.values = phi.values[tuple(slice_)]
        if not point_query:
            super(DiscreteFactor, phi).store_state_names(
                phi.variables, cardinality, state_names
            )
        
        if not inplace:
            return phi

    #@jit
    def sum(self, phi1, inplace=True):
        """
        DiscreteFactor sum with `phi1`.

        Parameters
        ----------
        phi1: `DiscreteFactor` instance.
            DiscreteFactor to be added.

        inplace: boolean
            If inplace=True it will modify the factor itself, else would return
            a new factor.

        Returns
        -------
        DiscreteFactor or None: if inplace=True (default) returns None
                        if inplace=False returns a new `DiscreteFactor` instance.
        """
        phi = self if inplace else self.copy()
        if isinstance(phi1, (int, float)):
            phi.values += phi1
        else:
            phi1 = phi1.copy()

            # modifying phi to add new variables
            extra_vars = set(phi1.variables) - set(phi.variables)
            if extra_vars:
                slice_ = [slice(None)] * len(phi.variables)
                slice_.extend([np.newaxis] * len(extra_vars))
                phi.values = phi.values[tuple(slice_)]

                phi.variables.extend(extra_vars)

                new_var_card = phi1.get_cardinality(extra_vars)
                phi.cardinality = np.append(
                    phi.cardinality, [new_var_card[var] for var in extra_vars]
                )

            # modifying phi1 to add new variables
            extra_vars = set(phi.variables) - set(phi1.variables)
            if extra_vars:
                slice_ = [slice(None)] * len(phi1.variables)
                slice_.extend([np.newaxis] * len(extra_vars))
                phi1.values = phi1.values[tuple(slice_)]

                phi1.variables.extend(extra_vars)
                # No need to modify cardinality as we don't need it.

            # rearranging the axes of phi1 to match phi
            for axis in range(phi.values.ndim):
                exchange_index = phi1.variables.index(phi.variables[axis])
                phi1.variables[axis], phi1.variables[exchange_index] = (
                    phi1.variables[exchange_index],
                    phi1.variables[axis],
                )
                phi1.values = phi1.values.swapaxes(axis, exchange_index)

            phi.values = phi.values + phi1.values

        if not inplace:
            return phi

    #@jit
    def product(self, phi1, inplace=True):
        """
        DiscreteFactor product with `phi1`.

        Parameters
        ----------
        phi1: `DiscreteFactor` instance
            DiscreteFactor to be multiplied.

        inplace: boolean
            If inplace=True it will modify the factor itself, else would return
            a new factor.

        Returns
        -------
        DiscreteFactor or None: if inplace=True (default) returns None
                        if inplace=False returns a new `DiscreteFactor` instance.
        """
        phi = self if inplace else self.copy()
        if isinstance(phi1, (int, float)):
            phi.values *= phi1
        else:
            phi1 = phi1.copy()

            # modifying phi to add new variables
            extra_vars = set(phi1.variables) - set(phi.variables)
            if extra_vars:
                slice_ = [slice(None)] * len(phi.variables)
                slice_.extend([np.newaxis] * len(extra_vars))
                phi.values = phi.values[tuple(slice_)]

                phi.variables.extend(extra_vars)

                new_var_card = phi1.get_cardinality(extra_vars)
                phi.cardinality = np.append(
                    phi.cardinality, [new_var_card[var] for var in extra_vars]
                )

            # modifying phi1 to add new variables
            extra_vars = set(phi.variables) - set(phi1.variables)
            if extra_vars:
                slice_ = [slice(None)] * len(phi1.variables)
                slice_.extend([np.newaxis] * len(extra_vars))
                phi1.values = phi1.values[tuple(slice_)]

                phi1.variables.extend(extra_vars)
                # No need to modify cardinality as we don't need it.

            # rearranging the axes of phi1 to match phi
            for axis in range(phi.values.ndim):
                exchange_index = phi1.variables.index(phi.variables[axis])
                phi1.variables[axis], phi1.variables[exchange_index] = (
                    phi1.variables[exchange_index],
                    phi1.variables[axis],
                )
                phi1.values = phi1.values.swapaxes(axis, exchange_index)

            phi.values = phi.values * phi1.values
            phi.add_state_names(phi1)

        if not inplace:
            return phi

    #@jit
    def divide(self, phi1, inplace=True):
        """
        DiscreteFactor division by `phi1`.

        Parameters
        ----------
        phi1 : `DiscreteFactor` instance
            The denominator for division.

        inplace: boolean
            If inplace=True it will modify the factor itself, else would return
            a new factor.

        Returns
        -------
        DiscreteFactor or None: if inplace=True (default) returns None
                        if inplace=False returns a new `DiscreteFactor` instance.
        """
        phi = self if inplace else self.copy()
        phi1 = phi1.copy()

        if set(phi1.variables) - set(phi.variables):
            raise ValueError("Scope of divisor should be a subset of dividend")

        # Adding extra variables in phi1.
        extra_vars = set(phi.variables) - set(phi1.variables)
        if extra_vars:
            slice_ = [slice(None)] * len(phi1.variables)
            slice_.extend([np.newaxis] * len(extra_vars))
            phi1.values = phi1.values[tuple(slice_)]

            phi1.variables.extend(extra_vars)

        # Rearranging the axes of phi1 to match phi
        for axis in range(phi.values.ndim):
            exchange_index = phi1.variables.index(phi.variables[axis])
            phi1.variables[axis], phi1.variables[exchange_index] = (
                phi1.variables[exchange_index],
                phi1.variables[axis],
            )
            phi1.values = phi1.values.swapaxes(axis, exchange_index)

        phi.values = phi.values / phi1.values

        # If factor division 0/0 = 0 but is undefined for x/0. In pgmpy we are using
        # np.inf to represent x/0 cases.
        phi.values[np.isnan(phi.values)] = 0

        if not inplace:
            return phi

    def copy(self):
        """
        Returns a copy of the factor.

        Returns
        -------
        DiscreteFactor: copy of the factor
        """
        # not creating a new copy of self.values and self.cardinality
        # because __init__ methods does that.
        return DiscreteFactor(
            self.scope(),
            self.cardinality,
            self.values,
            state_names=self.state_names.copy(),
        )

    def is_valid_cpd(self):
        return np.allclose(
            self.to_factor()
            .marginalize(self.scope()[:1], inplace=False)
            .values.flatten("C"),
            np.ones(np.product(self.cardinality[:0:-1])),
            atol=0.01,
        )

    def __str__(self):
        return self._str(phi_or_p="phi", tablefmt="grid")

    def _str(self, phi_or_p="phi", tablefmt="grid", print_state_names=True):
        """
        Generate the string from `__str__` method.

        Parameters
        ----------
        phi_or_p: 'phi' | 'p'
                'phi': When used for Factors.
                  'p': When used for CPDs.
        print_state_names: boolean
                If True, the user defined state names are displayed.
        """
        string_header = list(map(str, self.scope()))
        string_header.append(
            "{phi_or_p}({variables})".format(
                phi_or_p=phi_or_p, variables=",".join(string_header)
            )
        )

        value_index = 0
        factor_table = []
        for prob in product(*[range(card) for card in self.cardinality]):
            if self.state_names and print_state_names:
                prob_list = [
                    "{var}({state})".format(
                        var=list(self.variables)[i],
                        state=self.state_names[list(self.variables)[i]][prob[i]],
                    )
                    for i in range(len(self.variables))
                ]
            else:
                prob_list = [
                    "{s}_{d}".format(s=list(self.variables)[i], d=prob[i])
                    for i in range(len(self.variables))
                ]

            prob_list.append(self.values.ravel()[value_index])
            factor_table.append(prob_list)
            value_index += 1

        return tabulate(
            factor_table, headers=string_header, tablefmt=tablefmt, floatfmt=".4f"
        )

    def __repr__(self):
        var_card = ", ".join(
            [
                "{var}:{card}".format(var=var, card=card)
                for var, card in zip(self.variables, self.cardinality)
            ]
        )
        return "<DiscreteFactor representing phi({var_card}) at {address}>".format(
            address=hex(id(self)), var_card=var_card
        )

    def __mul__(self, other):
        return self.product(other, inplace=False)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __add__(self, other):
        return self.sum(other, inplace=False)

    def __radd__(self, other):
        return self.__add__(other)

    def __truediv__(self, other):
        return self.divide(other, inplace=False)

    __div__ = __truediv__

    def __eq__(self, other):
        if not (isinstance(self, DiscreteFactor) and isinstance(other, DiscreteFactor)):
            return False

        elif set(self.scope()) != set(other.scope()):
            return False

        else:
            phi = other.copy()
            for axis in range(self.values.ndim):
                exchange_index = phi.variables.index(self.variables[axis])
                phi.variables[axis], phi.variables[exchange_index] = (
                    phi.variables[exchange_index],
                    phi.variables[axis],
                )
                phi.cardinality[axis], phi.cardinality[exchange_index] = (
                    phi.cardinality[exchange_index],
                    phi.cardinality[axis],
                )
                phi.values = phi.values.swapaxes(axis, exchange_index)

            if phi.values.shape != self.values.shape:
                return False
            elif not np.allclose(phi.values, self.values):
                return False
            elif not all(self.cardinality == phi.cardinality):
                return False
            elif not (self.state_names == phi.state_names):
                return False
            else:
                return True

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        variable_hashes = [hash(variable) for variable in self.variables]
        sorted_var_hashes = sorted(variable_hashes)
        state_names_hash = hash(frozenset(self.state_names))
        phi = self.copy()
        for axis in range(phi.values.ndim):
            exchange_index = variable_hashes.index(sorted_var_hashes[axis])
            variable_hashes[axis], variable_hashes[exchange_index] = (
                variable_hashes[exchange_index],
                variable_hashes[axis],
            )
            phi.cardinality[axis], phi.cardinality[exchange_index] = (
                phi.cardinality[exchange_index],
                phi.cardinality[axis],
            )
            phi.values = phi.values.swapaxes(axis, exchange_index)
        return hash(
            str(sorted_var_hashes)
            + str(phi.values)
            + str(phi.cardinality)
            + str(state_names_hash)
        )
