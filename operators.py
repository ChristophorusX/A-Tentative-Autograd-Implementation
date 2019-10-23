import numpy as np
from util import substitute_value
from core import vector_jacobian_product_factory
from util import substitute_value, substitute_values


def unary_to_nary(unary_operator):
    """A decorator that converts unary operators to nary operators.

    Parameters
    ----------
    unary_operator : function
        A unary operator that returns a function taking one argument.

    Returns
    -------
    function
        Returns a nary operator that takes more than one argument.

    """
    def nary_operator(func, arg_index=0, *nary_args, **nary_kwargs):
        def nary_func(*args, **kwargs):
            if isinstance(arg_index, int):
                x = args[arg_index]
            else:
                x = tuple(args[index] for index in arg_index)

            def unary_func(x):
                if isinstance(arg_index, int):
                    substituted_args = substitute_value(args, arg_index, x)
                else:
                    substituted_args = substitute_values(
                        args, zip(arg_index, x))
                return func(*substituted_args, **kwargs)
            return unary_operator(unary_func, x, *nary_args, **nary_kwargs)
        return nary_func
    return nary_operator


@unary_to_nary
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


@unary_to_nary
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


@unary_to_nary
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
