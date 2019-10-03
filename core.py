from collections import defaultdict
from wrapper import Node, Wrapper
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
    gradient = None
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



class VectorJacobianProductNode(Node):
    pass

class JacobianVectorProductNode(Node):
    pass

# Defaultdict is not going to throw an error when accessing a key that is not
# in the dictionary, rather it will just create an item with the default value,
# which is convenient.
primative_vec_jac_prods = defaultdict(dict)
