

class Node(object):
    """
    This is a dummy class to be overridden.
    """
    def __init__(self, parents):
        self.parents = parents


class Wrapper(object):
    """
    Equivalent to the Box class in the Autograd implementation.
    """
    def __init__(self, value, node):
        self._value = value
        self._node = node


def primative():
    pass
