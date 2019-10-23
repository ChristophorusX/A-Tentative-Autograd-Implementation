from collections import defaultdict
from wrapper import Node, Wrapper, new_wrapper
import numpy as np
from util import topological_sort
from wrapper import MarkerStack

marker_stack = MarkerStack()


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
    current_gradient = None
    for node in topological_sort(propagation_end_node):
        current_gradient = gradient_dict.pop(node)
        # Compute the gradient of every parent
        for parent in node.parents:
            vector_jacobian_product = primative_vec_jac_prods[node.func]
            parent_gradient = vector_jacobian_product(
                current_gradient, node.value)
            # Handle the fan-out nodes
            previous_parent_gradient = gradient_dict.get(parent)
            if previous_parent_gradient:
                gradient_dict[parent] = parent_gradient + \
                    previous_parent_gradient
            else:
                gradient_dict[parent] = parent_gradient
    return current_gradient


def forward_propagation(propagation_start_node, func, x):
    """A forward propagation starting at the `propagation_start_node` and
    wrapping the all the composition operations along the way.

    Parameters
    ----------
    propagation_start_node : Node
        The node where the gradient function (or anything similar) is requested.
    func : function
        The function to apply at the node (most likely be a composition of functions).
    x : narray
        A set of parameters for the function.

    Returns
    -------
    Wrapper
        The ending wrapper wrapping the propagation end node.

    """
    trace_marker = marker_stack.get_marker()
    propagation_start_wrapper = new_wrapper(
        x, trace_marker, propagation_start_node)
    propagation_end_wrapper = func(propagation_start_wrapper)
    marker_stack.release_marker(trace_marker)
    if isinstance(propagation_end_wrapper, Wrapper) and propagation_end_wrapper._trace_marker == propagation_start_wrapper.trace_marker:
        return propagation_end_wrapper._value, propagation_end_wrapper._node
    else:
        return propagation_end_wrapper, None


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
    propagation_end_node, propogation_end_value = forward_propagation(
        current_node, func, x)
    # If the end node has no relationship to the current node, gradient is 0
    # Otherwise, the vector_jacobian_product is produced by backpropagation
    if propagation_end_node is None:
        return lambda gradient: np.zeros_like(x)
    else:
        return lambda gradient: backpropagation(gradient, propagation_end_node)

def define_vector_jacobian_product(func, *vector_jacobian_product_maker, **kwargs):
    """Defines the vectorjacobian product for a given primitive function.

    Parameters
    ----------
    func : type
        Description of parameter `func`.
    *vector_jacobian_product_maker : type
        Description of parameter `*vector_jacobian_product_maker`.
    **kwargs : type
        Description of parameter `**kwargs`.

    Returns
    -------
    type
        Description of returned object.

    """
    # TODO
    pass

class VectorJacobianProductNode(Node):
    pass


class JacobianVectorProductNode(Node):
    pass

# Defaultdict is not going to throw an error when accessing a key that is not
# in the dictionary, rather it will just create an item with the default value,
# which is convenient.
primative_vec_jac_prods = defaultdict()
