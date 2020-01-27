import networkx as nx


def generate_supersource(orig_graph, node_label):
    """Generates a supersource node for the graph.

    A supersource node is defined as a node that is connected to all others nodes. Referenced as a
    "super node" in https://arxiv.org/pdf/1511.05493.pdf. Node that is connected to all other
    nodes in the graph. Implemented by changing the networkx graph to a digraph (see below for
    why). Changing to digraph when the base representation is undirected can cause
    unexpected behavior for networkx calls to edges() and degree(), so this should be handled with
    care.

    For feature aggregation layers, the supersource node should accept messages, but should not
    pass messages. If not done this way, then every node becomes at least a second
    order neighbor of all others nodes. This is important for random walk (GraphSAGE) and GCN
    operators.

    The supersource has attribute label as 'supersource' and can be identified as 'supsersource'
    as its networkx hash. If it is a labeled graph, then the 'supersource' is assigned the type
    'supersource_type' to be a unique type.

    Args:
        orig_graph: A networkx UndirectedGraph object that needs a supersource added to it.
        node_label: The networkx key for the true node label (e.g. 'type' for AIDS), or None if
                    the graph is unlabelled.

    Returns:
        supersource_graph: A networkx DiGraph object with the same overall graph structure as
                           the original, but with a supersource node added.
        supersource_type: The type assigned to the supersource node (default 'supersource_type')

    """
    assert type(orig_graph) == nx.Graph, 'Currently not handling input graphs other than the ' \
                                         'default undirected graph.'

    supersource_graph = orig_graph.to_directed()

    # Define the attributes for the supersource node.
    supersource_id = 'supersource'
    supersource_type = 'supersource_type'
    attributes = {'label': supersource_id}
    if node_label:
        attributes[node_label] = supersource_type

    # Add the node to the graph and go through all the edges and add a directed edge to it.
    supersource_graph.add_node(supersource_id, attributes)
    for node in supersource_graph.nodes():
        if node == supersource_id:
            continue
        supersource_graph.add_edge(node, supersource_id)

    return supersource_graph, supersource_id, supersource_type
