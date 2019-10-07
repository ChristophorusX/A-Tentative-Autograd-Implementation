from util import substitute_values

class Node(object):
    """
    This is a dummy class to be overridden.
    """

    def __init__(self, parents):
        self.parents = parents
        self.func = None
        self.value = None
        self.args = []
        self.kwargs = {}
        self.parent_argnums = []

    def initialize_root(self, func):
        self.parents = []
        self.func = func

    # A class method as alternative constructor
    @classmethod
    def new_root(cls, *args, **kwargs):
        root = cls.__new__(cls)
        root.initialize_root(*args, **kwargs)
        return root


class Wrapper(object):
    """
    Equivalent to the Box class in the Autograd implementation.
    """
    def __init__(self, value, node):
        self._value = value
        self._node = node

    def __bool__(self):
        return self._value

def new_wrapper(value, propogation_level, node):
    """A function that returns a wrapper of certain data type. (TODO)

    Parameters
    ----------
    value : float
        Function value evaluated at the node.
    propogation_level : int
        The level of propagation from starting node.
    node : Node
        The node to be wrapped.

    Returns
    -------
    Wrapper
        A wrapper containing everything about the node in perticular run.

    """
    return Wrapper(value, propogation_level, node)


def primative():
    """
    Wrapper for function
    """
    def function_wrapped(*args, **kwargs):
        wrapped_args, trace, node_constructor = backtrace_top_wrapped_args(args)
        if wrapped_args:
            num_value_pair = [(argnum, wrapper._value) for argnum, wrapper in wrapped_args]
            argvals = substitute_values(args, num_value_pair)
            if function_wrapped in notrace_primitives[node_constructor]:
                return function_wrapped(*argvals, **kwargs)
            parents = tuple(wrapper._node for _, wrapper in wrapped_args)
            argnums = tuple(argnum for argnum, _ in wrapped_args)
            res = function_wrapped(*argvals, **kwargs)
            node = node_constructor(ans, function_wrapped, argvals, kwargs, argnums, parents)
            return new_wrapper(ans, trace, node)
        else:
            return function_raw(*args, **kwargs)
    function_wrapped.fun = function_raw
    function_wrapped.is_autograd_primitive = True
    return function_wrapped
