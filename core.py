from collections import defaultdict
from wrapper import Node, Wrapper
import numpy as np
from util import topological_sort

def backpropagation(gradient, propagation_end_node):
    """A backpropagation function that computes the gradients up to the
    propagation_end_node.

    Parameters
    ----------
    gradient : narray
        gradient at the propagation_end_node.
    propagation_end_node : Node
        An end node during the propagation.

    Returns
    -------
    narray
        Returns the gradient to the original variables.

    """
    gradient_dict = {}
    # Starting with an end node on propagation map and traversing the
    # backpropagation graph
    gradient_dict[propagation_end_node] = gradient
    for node in topological_sort(propagation_end_node):
        gradient = gradient_dict.pop(node)
        # Compute the gradient of every parent
        for parent in node.parents:
            vector_jacobian_product = primative_vec_jac_prods[node.func]
            gradient_parent = vector_jacobian_product(gradient, node.value)
            previous_gradient = gradient_dict.get(parent)
            if previous_gradient:
                gradient_dict[parent] = gradient_parent + previous_gradient
            else:
                gradient_dict[parent] = gradient_parent
    return gradient


def forward_propagation(propagation_start_node, func, args):
    """A forward propagation starting at the `propagation_start_node` and
    wrapping the all the composition operations along the way.

    Parameters
    ----------
    propagation_start_node : type
        Description of parameter `propagation_start_node`.
    func : type
        Description of parameter `func`.
    args : type
        Description of parameter `args`.

    Returns
    -------
    type
        Description of returned object.

    """
    args = list(args)


def vector_jacobian_product_factory(func, x):
    """A factory that produces a vector jacobian product of a given node with
    a function `func` at variables `x`.

    Parameters
    ----------
    func : function
        A function specified at the given node.
    x : narray
        A set of variables at which the function `func` is evaluated.

    Returns
    -------
    function
        A vector jacobian product function with parameter `gradient`.

    """
    current_node = Node.new_root()
    # Propogating through the graph and finding the end node
    propagation_end_node, propogation_end_value = forward_propagation(current_node, func, x)
    # If the end node has no relationship to the current node, gradient is 0
    # Otherwise, the vector_jacobian_product is produced by backpropagation
    if propagation_end_node is None:
        return lambda gradient: np.zeros_like(x)
    else:
        return lambda gradient: backpropagation(gradient, propagation_end_node)


class VectorJacobianProductNode(Node):
    pass

class JacobianVectorProductNode(Node):
    pass

# Defaultdict is not going to throw an error when accessing a key that is not
# in the dictionary, rather it will just create an item with the default value,
# which is convenient.
primative_vec_jac_prods = defaultdict(dict)
