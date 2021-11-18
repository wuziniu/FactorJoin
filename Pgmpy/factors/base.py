from abc import abstractmethod
from functools import reduce

class BaseFactor(object):
    """
    Base class for Factors. Any Factor implementation should inherit this class.
    """

    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def is_valid_cpd(self):
        pass


def factor_product(*args):
    """
    Returns factor product over `args`.

    Parameters
    ----------
    args: `BaseFactor` instances.
        factors to be multiplied

    Returns
    -------
    BaseFactor: `BaseFactor` representing factor product over all the `BaseFactor` instances in args.
    """
    if not all(isinstance(phi, BaseFactor) for phi in args):
        raise TypeError("Arguments must be factors")
    # Check if all of the arguments are of the same type
    elif len(set(map(type, args))) != 1:
        raise NotImplementedError(
            "All the args are expected to be instances of the same factor class."
        )
    return reduce(lambda phi1, phi2: phi1 * phi2, args)


def factor_divide(phi1, phi2):
    """
    Returns `DiscreteFactor` representing `phi1 / phi2`.

    Parameters
    ----------
    phi1: Factor
        The Dividend.

    phi2: Factor
        The Divisor.

    Returns
    -------
    DiscreteFactor: `DiscreteFactor` representing factor division `phi1 / phi2`.

    """
    if not isinstance(phi1, BaseFactor) or not isinstance(phi2, BaseFactor):
        raise TypeError("phi1 and phi2 should be factors instances")

    # Check if all of the arguments are of the same type
    elif type(phi1) != type(phi2):
        raise NotImplementedError(
            "All the args are expected to be instances of the same factor class."
        )

    return phi1.divide(phi2, inplace=False)
