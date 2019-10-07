

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
    pass
