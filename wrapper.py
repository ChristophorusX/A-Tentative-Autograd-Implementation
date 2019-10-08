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

    def __init__(self, value, node, propogation_level):
        self._value = value
        self._node = node
        self._propogation_level = propogation_level

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


def primative(function_raw):
    """A wrapper function for primitive function `function_raw`, which builds
    the graph at the same time.

    Parameters
    ----------
    function_raw : function
        The raw function to be wrapped.

    Returns
    -------
    function_wrapped : function
        A wrapper function that builds the graph when envoked.

    """
    def function_wrapped(*args, **kwargs):
        wrapped_args, trace, node_constructor = backtrace_top_wrapped_args(
            args)
        if wrapped_args:
            num_value_pair = [(argnum, wrapper._value)
                              for argnum, wrapper in wrapped_args]
            argvals = substitute_values(args, num_value_pair)
            if function_wrapped in notrace_primitives[node_constructor]:
                return function_wrapped(*argvals, **kwargs)
            parents = tuple(wrapper._node for _, wrapper in wrapped_args)
            argnums = tuple(argnum for argnum, _ in wrapped_args)
            result = function_wrapped(*argvals, **kwargs)
            node = Node(result, function_wrapped,
                        argvals, kwargs, argnums, parents)
            return new_wrapper(result, trace, node)
        else:
            return function_raw(*args, **kwargs)
    function_wrapped.fun = function_raw
    function_wrapped.is_autograd_primitive = True
    return function_wrapped


def notrace_primitive(function_raw):
    """A wrapper function that evaluates with wrapped arguments.

    Parameters
    ----------
    function_raw : function
        A raw function to be wrapped.

    Returns
    -------
    function_wrapped : function
        A function only evaluated without doing any tracing.

    """
    def function_wrapped(*args, **kwargs):
        def get_value(x):
            return get_value(x._value) if isinstance(x, Wrapper) else x
        argvals = map(get_value, args)
        return function_raw(*argvals, **kwargs)
    return function_wrapped


def backtrace_top_wrapped_args(args):
    """Backtrace all the wrapped arguments with the largest propagation level.

    Parameters
    ----------
    args :
        All the arguments passed to the function.

    Returns
    -------
    top_wrappers, top_propogation_level : List[Tuple[int,Wrapper]], int
        All the wrapped arguments with their index in args along with their
        propagation level.

    """
    top_propogation_level = -1
    top_wrappers = []
    for arg_index, arg in enumerate(args):
        if isinstance(arg, Wrapper):
            if arg._propogation_level > top_propogation_level:
                top_wrppers = [(arg_index, arg)]
                top_propogation_level = arg._propogation_level
            elif arg._propogation_level == top_propogation_level:
                top_wrappers.append((arg_index, arg))
    return top_wrappers, top_propogation_level
