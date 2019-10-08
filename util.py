from wrapper import Node


def topological_sort(propagation_end_node):
    """A utility function to perform a topological sort on a DAG starting from
    the node to perform backpropagation.

    Parameters
    ----------
    propagation_end_node : Node
        The node to perform backpropagation.

    Returns
    -------
    List[Node]
        A list of nodes that have been topologically sorted, starting with
        propagation_end_node.

    """
    counting_dict = {}
    stack = []
    childness_nodes = []
    # Starting with the node where the propagation ends
    stack.append(propagation_end_node)
    childness_nodes.append(propagation_end_node)
    # Counting the number of children of each node where the direction of the
    # edges is the same as the direction of the propagation
    while stack:
        node = stack.pop()
        if node in counting_dict:
            counting_dict[node] += 1
        else:
            counting_dict[node] = 1
            stack.extend(node.parents)
    # Returns the nodes without children
    while childness_nodes:
        good_node = childness_nodes.pop()
        yield good_node
        for parent in good_node.parents:
            if counting_dict[parent] == 1:
                childness_nodes.append(parent)
            else:
                counting_dict[parent] -= 1


def substitute_values(x, index_value_pairs):
    x_list = list(x)
    for i, v in index_value_pairs:
        x_list[i] = v
    return tuple(x_list)


def substitute_value(x, index, value):
    x_list = list(x)
    x_list[index] = value
    return tuple(x_list)
