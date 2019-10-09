import numpy as np
from util import substitute_value
from core import vector_jacobian_product_factory


def grad(func, arg_index=0):
    """An API for computing gradient function of a function `func`.
    TODO: implement narray version by a decorator

    Parameters
    ----------
    func : function
        A function to be computed gradient.
    arg_index : int
        The index of the argument in args.

    Returns
    -------
    gradient_func : function
        A gradient function that computes the gradient of `func`.

    """
    def gradient_func(*args, **kwargs):
        univariate_func = lambda x: func(
            substitute_value(args, arg_index, x), **kwargs)
        vector_jacobian_product, value = vector_jacobian_product_factory(
            univariate_func, args[arg_index])
        return vector_jacobian_product(np.ones_like(value))
    return gradient_func


def jacobian(func, x):
    """Short summary.

    Parameters
    ----------
    func : type
        Description of parameter `func`.
    x : type
        Description of parameter `x`.

    Returns
    -------
    type
        Description of returned object.

    """
    pass


def hessian(func, x):
    """Short summary.

    Parameters
    ----------
    func : type
        Description of parameter `func`.
    x : type
        Description of parameter `x`.

    Returns
    -------
    type
        Description of returned object.

    """
    pass
