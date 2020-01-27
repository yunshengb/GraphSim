import struct
from utils import load_data

def convert_to_mivia(graph, labeled, label_key='label', label_dict={}):
    """Returns a binary representation of the graph that is consistent with:
    http://mivia.unisa.it/datasets/graph-database/arg-database/documentation/.

    BIG NOTE:
    McCreesh 2016 paper code only accepts labeled graphs. It can use a
    unlabeled setting, but the input format must be in mivia labeled. So to
    ignore the graph's labels, just use labeled=False.

    The edge numbering format is based double sorted. First by source node id,
    then for each node, by target node id. Be careful changing this because
    the answers from the cpp solver uses absolute edge numbering, so we have
    to keep the ordering consistent.

    Can handle labeled or unlabeled graphs.
    Does not handle directed graphs.
    Does not handle edge labels.

    Assumes that the gexf graph file starts with node id 0. This is because
    the mivia format uses 0 based indexing for everything without
    allowing graph ids.

    Args:
        graph: networkx graph, undirected, with all node ids starting from 0.
        labeled: Boolean whether or not the graph is labeled
                 (edge labels not supported yet).
        label_key: Default 'label'. Only for labeled graphs. The key that
                   should be used to access a networkx node to get its label.
        label_dict: A map from a node label to its unique id.
    Returns:
        List of results of struct.pack, to be iterated through and written.
        idx_to_node: A map of mivia index to the corresponding node from graph.nodes()
    """
    # 1. Number of nodes.
    data_bytes = []
    data_bytes.append(len(graph.nodes()))

    # 2. Node label for each node.
    idx_to_node = {}
    node_to_idx = {}
    node_iter = sorted(graph.nodes(data=True), key=lambda x: int(x[0]))
    for idx, (node, attr) in enumerate(node_iter):
        idx_to_node[idx] = int(node)
        node_to_idx[int(node)] = idx
        if labeled:
            current_label = attr[label_key]
            data_bytes.append(label_dict[current_label])
        else:
            # Just use any default label for all nodes.
            data_bytes.append(0)

    # 3. Adj list for each node, sorted by the source node id.
    adj_iter = sorted(graph.adj.items(), key=lambda x: int(x[0]))
    for source_id, adj_list in adj_iter:
        # 4. Add the length of the current node's adj list.
        data_bytes.append(len(adj_list))

        # 5. Add the indices of the connected nodes for each edge.
        for target_id, attr in sorted(adj_list.items(), key=lambda x: int(x[0])):
            # Edge labels are unsupported.
            edge_label = 0
            data_bytes.append(node_to_idx[int(target_id)])
            data_bytes.append(edge_label)

    return int_list_to_bytes_list(data_bytes), idx_to_node


def _convert_pure_unlabeled(graph):
    # Pure here means mivia unlabeled format. We only use labeled format for now,
    # and selectively ignore the labels.
    raise RuntimeError('Pure mivia unlabeled unsupported because mccreesh 2016 code '
                       'only reads mivia labeled graphs, even if setting is set '
                       'to unlabeled.')
    # 1. Number of nodes.
    data_bytes = []
    data_bytes.append(len(graph.nodes()))

    # 2. Adj list for each node, sorted by the source node id.
    adj_iter = sorted(graph.adj.items(), key=lambda x: int(x[0]))
    for idx, (source_id, adj_list) in enumerate(adj_iter):
        if idx != int(source_id):
            raise RuntimeError('Input graph node id error. The ' + str(idx) +
                               ' node does not have id = ' + str(idx) +
                               '. Probably the input graph does not start '
                               'with id=0 or ids are not ints or id skips '
                               'some values. Make sure all ids are 0 based '
                               'ints in +1 increments.')
        # 3. Add the length of the current node's adj list.
        data_bytes.append(len(adj_list))

        # 4. Add the indices of the connected nodes for each edge.
        for target_id, attr in adj_list.items():
            data_bytes.append(int(target_id))

    return int_list_to_bytes_list(data_bytes)


def int_list_to_bytes_list(int_list):
    format_string = '<H'  # 16 bit little endian.
    return [struct.pack(format_string, i) for i in int_list]


if __name__ == '__main__':
    test_data = load_data('aids700nef', train=False)
    train_data = load_data('aids700nef', train=True)

    for g in train_data.graphs[236:237]:
        bytes, idx_to_node = convert_to_mivia(g, labeled=True, label_key='type')
        filename = '{}.mivia'.format(g.graph['gid'])
        with open(filename, 'wb') as writefile:
            for byte in bytes:
                writefile.write(byte)
