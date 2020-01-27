import random


# Code modified from https://github.com/williamleif/GraphSAGE/blob/master/graphsage/utils.py

def generate_random_walks(graph, num_walks, walk_length):
    """Generates random walks for the graph.

    Random walk generation is as used in the original GraphSAGE implementation:
    (https://github.com/williamleif/GraphSAGE).

    Performs random walks for every single in the graph. This means that the result will have
    a list of neighbors for every node in the graph.

    Random walks are used to define the neighborhood of a node. They are defined by walk_length,
    which is how many nodes are visisted during a single walk. Since walks are used to define a
    neighborhood and are randomly sampled during training, a single random walk generates pairs of
    [original_node, neighbor_node] as data, where each neighbhor_node is a node visited along
    the walk.

    This implementation uses the original random walk implementation in GraphSAGE, which doesn't
    count self visits during the walk (can't have [orig_node, orig_node] as a pair), and actually
    generates at most (num_walks * (walk_length - 1) * num_nodes) pairs. In short,
    this is because they log pairs before walking, which seems to be an off-by-one error.

    Args:
        graph: A networkx graph object to perform the walk.
        num_walks: The number of walks performed per node in the graph.
        walk_length: The number of nodes visited in each walk.

    Returns:
        A list of dicts of walks of the form:
        [
            { <start_node_id>: [<neighbor>, ..., <neighbor>] },
            { <start_node_id>: [<neighbor>, ..., <neighbor>] },
            ...,
            { <start_node_id>: [<neighbor>, ..., <neighbor>] }
        ]
    """
    walks = {}
    nodes = graph.nodes()
    for node in nodes:
        assert node not in walks, 'node {} already in dict, is there a duplicate node id?'

        # Generate walks starting from the current node and create a big list of walk neighbors.
        for _ in range(num_walks):
            if node not in walks:
                walks[node] = []
            walk = _generate_single_walk(graph, node, walk_length)
            walks[node].extend(walk)

    return walks


def _generate_single_walk(graph, start_node, walk_length):
    """A single walk on a starting node.

    Args:
        graph: A Networkx graph.
        start_node: Id for the start_node as obtained from the graph.
        walk_length: The length of the walk.
    """
    # No walks to generate if no neighbors (a terminal node in a digraph).
    if not graph.neighbors(start_node):
        return []

    walk = []
    curr_node = start_node
    for i in range(walk_length):
        neighbors = graph.neighbors(curr_node)
        if not neighbors:
            # This is a terminal node, so just add it to the walk and get out.
            walk.append(curr_node)
            break

        next_node = random.choice(neighbors)

        # self co-occurrences are useless
        if curr_node != start_node:
            walk.append(curr_node)
        curr_node = next_node

    return walk
