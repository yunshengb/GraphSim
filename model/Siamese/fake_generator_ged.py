from config import FLAGS
from dist_sim import normalized_dist_sim
import networkx as nx
from random import sample, choice


def graph_generator(nxgraph):
    new_g, ged = _perturb_graph_once(nxgraph)
    if FLAGS.ds_norm:
        print('@@@', nxgraph.number_of_nodes(), new_g.number_of_nodes(), ged)
        ged = normalized_dist_sim(ged, nxgraph, new_g)
        print('###', ged)
        print()
    return new_g, ged


def _perturb_graph_once(graph):
    new_g = graph.copy()
    ged = None
    _sanity_check(graph)

    op = choice(['add_node', 'add_edge', 'del_node', 'del_edge', 'change_node',
                 'isomorphic'])
    if op == 'add_node':
        print('add_node')
        new_g, ged, op = _add_node(new_g, ged, op)
    if op == 'add_edge':
        print('add_edge')
        new_g, ged, op = _add_edge(new_g, ged, op)
    if op == 'del_node':
        print('del_node')
        new_g, ged, op = _del_node(new_g, ged, op)
    if op == 'del_edge':
        print('del_edge')
        new_g, ged, op = _del_edge(new_g, ged, op)
    print('done 4 ops')
    if op == 'change_node':
        print('change_node')
        new_g, ged, op = _change_node(new_g, ged, op)
    print('done 5th op')
    if op == 'isomorphic':
        print('isomorphic')
        new_g = graph.copy()  # new_g may have been messed up, so re-copy
        ged = 0

    new_g.graph['gid'] = 'small perturbation from {}'.format(graph.graph['gid'])
    _sanity_check(graph)
    _sanity_check(new_g)
    assert (ged >= 0)
    return new_g, ged


def _add_node(new_g, ged, op):
    nodes = new_g.nodes()
    if len(nodes) < FLAGS.max_nodes:
        to_connect = choice(nodes)
        new_g, ged = _add_node_with_an_edge(new_g, nodes, to_connect)
    else:
        op = 'change_node'
    return new_g, ged, op


def _add_edge(new_g, ged, op):
    nodes = new_g.nodes()
    edges = new_g.edges()
    if len(nodes) >= 2:
        i, j = sample(nodes, 2)
        assert (i != j)
        if not new_g.has_edge(i, j):
            new_g.add_edge(i, j)
            assert (len(new_g.edges()) == len(edges) + 1)
            ged = 1
        else:
            op = 'change_node'
    else:
        op = 'change_node'
    return new_g, ged, op


def _del_node(new_g, ged, op):
    nodes = new_g.nodes()
    if len(nodes) >= 2:
        to_del = choice(nodes)
        degree = new_g.degree(to_del)
        new_g.remove_node(to_del)
        if not nx.is_connected(new_g):
            op = 'isomorphic'  # shouldn't have removed the edge
        else:  # fine
            ged = degree + 1
    else:
        op = 'change_node'
    return new_g, ged, op


def _del_edge(new_g, ged, op):
    nodes = new_g.nodes()
    edges = new_g.edges()
    if len(nodes) >= 2:
        i, j = sample(nodes, 2)
        if new_g.has_edge(i, j):
            new_g.remove_edge(i, j)
            assert (len(new_g.edges()) == len(edges) - 1)
            if not nx.is_connected(new_g):
                op = 'isomorphic'  # shouldn't have removed the edge
            else:  # fine
                ged = 1
        else:
            op = 'change_node'
    else:
        op = 'change_node'
    return new_g, ged, op


def _change_node(new_g, ged, op):
    nodes = new_g.nodes()
    if FLAGS.node_feat_name:
        to_change = choice(nodes)
        old_type = nx.get_node_attributes(new_g, FLAGS.node_feat_name)[to_change]
        new_type = _set_node_to_random_type(new_g, to_change)
        if old_type != new_type:
            ged = 1
        else:
            op = 'isomorphic'
    else:
        op = 'isomorphic'
    return new_g, ged, op


def _add_node_with_an_edge(graph, nodes, existing_node):
    nodes = [int(n) for n in nodes]
    new_node = str(max(nodes) + 1)
    graph.add_edge(existing_node, new_node)
    if FLAGS.node_feat_name:
        _set_node_to_random_type(graph, new_node)
    assert (len(graph.nodes()) == len(nodes) + 1)
    return graph, 2


def _set_node_to_random_type(g, node):
    exsiting_types = list(nx.get_node_attributes(g, FLAGS.node_feat_name).values())
    type = choice(exsiting_types)
    nx.set_node_attributes(g, FLAGS.node_feat_name, {node: type})
    return type


def _sanity_check(g):
    num_nodes = g.number_of_nodes()
    if num_nodes < 1:
        raise RuntimeError('Wrong graph {} with {} nodes < 1'.format(
            g.graph['gid'], num_nodes))
    if num_nodes > FLAGS.max_nodes:
        raise RuntimeError('Wrong graph {} with {} nodes > max nodes'.format(
            g.graph['gid'], num_nodes))
    if num_nodes >= 2 and len(nx.isolates(g)) != 0:
        raise RuntimeError('Wrong graph {} with {} isolated nodes'.format(
            g.graph['gid'], len(nx.isolates(g))))
