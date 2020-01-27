import networkx as nx
from collections import deque


def reorder_nodes(nxgraph, method, exsiting_mapping, node_feat_name):
    """
    Copy and return a new Networkx graph with nodes reordered.
        The correct way to use the reordered graph is to call sorted(g.nodes()).
    :param nxgraph:
    :param method: 'bfs', 'degree', 'exsiting', None.
    :param exsiting_mapping: Only used when method == 'exsiting'.
           Supposed to be a dictionary returned by node_ordering().
    :param node_feat_name:
    :return: Another nxgraph, but with the desired ordering.
    """
    # TODO: this is going to break for supersource type because supersource uses a digraph. the
    # node hashes and label attributes are relabeled, so this is going to cause a real node to be
    # in a supersource node. get rid of the config import also!
    from config import FLAGS
    assert not FLAGS.supersource, 'Supersource and reorder_nodes not supported yet'
    ### end remove todo
    if not method:
        return nxgraph

    if method == 'existing':
        mapping = exsiting_mapping
    else:
        _, mapping = node_ordering(nxgraph, method, node_feat_name)

    # There are two steps to node relabeling, one by attributes and one by the
    # node relabeling function itself.
    relabeled = nx.relabel_nodes(nxgraph, mapping, copy=True)
    relabeled_attributes = nx.get_node_attributes(relabeled, 'label')
    if not relabeled_attributes:
        raise RuntimeError('Wrong graph: No node attribute "label"!')
    relabeled_mapping = {k: k for k, _ in relabeled_attributes.items()}
    nx.set_node_attributes(relabeled, 'label', relabeled_mapping)

    return relabeled


def node_ordering(graph, method, node_feat_name, last_order=[]):
    """
    Return the ordering of the input graph without modifying it.
    Some graphs may not have contiunous node ids.
        For example, a graph may have node ids:
        {'2', '9', '10'} (unordered set).
        graph.nodes() could return different ordering
        if g is loaded again from the disk,
        but once it is loaded, graph.nodes() would be consistent.
        The returned ordering in this case could be
        [2, 0, 1],
        denoting '10' --> '2' --> '9'
        if graph.nodes() == ['2', '9', '10'],
        and the returned mapping could be
        {'10': '2', '2': '9', '9': '10'}.
        To sum up, ordering is about integer indexing into
        g.nodes(), while mapping is about relabeling nodes
        regardless of the randomness of g.nodes().
    :param graph:
    :param method: 'bfs', 'degree', None.
    :param node_feat_name:
    :param last_order: List of node hashes in desired ending order or empty list. Ordering
                       ignores any matches in graph.nodes() that match last_order, then puts any
                       nodes that match last_order at the end of the returned ordering. Use to
                       force any nodes to be at the end of the ordering (specifically
                       supersource node).
    :return: ordering is a list of integers that can be used to
             reorder the N by D node input matrix or the N by N adj matrix
             by self.dense_node_inputs[self.order, :] or
             self.adj[np.ix_(self.order, self.order)].
             mapping is a dict mapping original node id --> new node id.
    """
    # Holds the nodes to append at the end, so make sure that they're valid nodes.
    sort_ignore_nodes = [node for node in last_order if node in graph.nodes()]
    assert len(sort_ignore_nodes) == len(last_order), 'Tried to ignore nodes that dont actually ' \
                                                      'exist in the graph. Double check ' \
                                                      'last_order call input.'
    # Get the ordered sequence with highest degree first. We presort the nodes so that the input
    # is stable before doing other sorts because
    nodes = _sorted_nodes_based_on_node_deg_and_types(sorted(graph.nodes()), graph, node_feat_name)
    if method == 'bfs':
        seq = [node for node in _bfs_seq(graph, nodes[0], node_feat_name, sort_ignore_nodes)]
    elif method == 'degree':
        seq = [node for node in nodes]
    else:
        raise RuntimeError('Unknown ordering method {}'.format(method))
    origin_seq = [node for node in graph.nodes()]
    ordering = []
    for e in seq:
        ordering.append(origin_seq.index(e))
    orig_nodes = graph.nodes()  # has randomness :(
    mapping = {orig_nodes[orig]: sorted(orig_nodes)[final]
               for final, orig in enumerate(ordering)}
    return ordering, mapping


def _bfs_seq(graph, start_id, node_feat_name, sort_ignore_nodes=[]):
    # sort_ignore_nodes is a list of nodes that should be ignored during the bfs search. We use
    # this as a special check for nodes to ignore that will be appended at the end of the ordering.
    dictionary = dict(_stable_bfs_successors(graph, start_id, sort_ignore_nodes))
    start = [start_id]
    output = [start_id]
    while len(start) > 0:
        next = []
        while len(start) > 0:
            current = start.pop(0)
            neighbors = dictionary.get(current)
            if neighbors is not None:
                neighbors = _sorted_nodes_based_on_node_types(neighbors, graph,
                                                              node_feat_name)
                next += neighbors
        output += next
        start = next
    # Put the ignored nodes in order at the end.
    output.extend(sort_ignore_nodes)
    assert len(output) == len(graph.nodes()), 'Mismatch graph nodes and output, something is ' \
                                              'probably wrong with the BFS successors with an ' \
                                              'ignored node.'
    return output


def _stable_bfs_successors(graph, start_id, sort_ignore_nodes=[]):
    # Replacement for nx.bfs_successors(). Provides stable bfs ordering by checking all
    # neighbors first, then adding relevant ones to the queue, but adding them sorted by id.
    queue = deque()
    seen_nodes = set()
    bfs_successors = {}

    # Init with the start node.
    seen_nodes.add(start_id)
    queue.append(start_id)

    while queue:
        current = queue.popleft()
        bfs_successors[current] = []
        neighbors = graph.neighbors(current)
        # Sort the neighbors first, then add them if they haven't been seen yet.
        for neighbor in sorted(neighbors):
            if neighbor in sort_ignore_nodes:
                seen_nodes.add(neighbor)
            if neighbor not in seen_nodes:
                bfs_successors[current].append(neighbor)
                seen_nodes.add(neighbor)
                queue.append(neighbor)
    # Make sure we've seen all the nodes. This is an assert check for disconnected graphs since
    # they are not currently handled.
    assert(len(seen_nodes) == len(graph.nodes()))
    return bfs_successors

def _sorted_nodes_based_on_node_deg_and_types(nodes, graph, node_feat_name):
    # For DiGraphs (supersource or similar), we want focus on out_degree, so a node with no
    # out_degree is at the end.
    if type(graph) == nx.Graph:
        degree_fn = graph.degree
    elif type(graph) == nx.DiGraph:
        degree_fn = graph.out_degree
    else:
        raise RuntimeError('Unidentified input graph type: {}'.format(type(graph)))
    # Sorting is based on 1. highest degree, 2. type (optional) 3. node.
    if node_feat_name:
        types = nx.get_node_attributes(graph, node_feat_name)
        decorated = [(degree_fn()[node], types[node], node)
                     for node in nodes]
        decorated.sort(key=lambda k: (-k[0], k[1], k[2]))
        return [node for deg, type, node in decorated]
    else:
        decorated = [(degree_fn()[node], node) for node in nodes]
        decorated.sort(key=lambda k: (-k[0], k[1]))
        return [node for deg, node in decorated]


def _sorted_nodes_based_on_node_types(nodes, graph, node_feat_name):
    if node_feat_name:
        types = nx.get_node_attributes(graph, node_feat_name)
        decorated = [(types[node], node) for node in nodes]
        decorated.sort()
        return [node for type, node in decorated]
    else:
        return nodes


def check_node_id_continuity(g):
    ids = sorted([int(idx) for idx in g.nodes()])
    should = 0
    rtn = True
    for id in ids:
        if id != should:
            rtn = False
        should += 1
    if not rtn:
        print('Wrong graph {} with node ids {}'.format(
            g.graph['gid'], ids))
    return rtn


if __name__ == '__main__':
    from utils import load_data

    gs = load_data('aids700nef', train=True).graphs
    # destructively_reorder_nodes(gs[236], 'bfs', None, 'type')
    # for i, g in enumerate(gs):
    #     # print(i)
    #     check_node_id_continuity(g)
