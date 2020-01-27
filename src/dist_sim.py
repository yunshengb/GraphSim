from utils import get_root_path, exec_cmd, get_ts, create_dir_if_not_exists
from nx_to_gxl import nx_to_gxl
from os.path import isfile
from os import getpid
from time import time
from nx_to_mivia import convert_to_mivia
import fileinput
import json
import networkx as nx
import sys

# For HED.
sys.path.insert(0, '{}/model/aproximated_ged/aproximated_ged'.format(get_root_path()))


def ged(g1, g2, algo, debug=False, timeit=False, timeout=None):
    if algo in ['astar', 'hungarian', 'vj'] or 'beam' in algo:
        return handle_ged_gmt(g1, g2, algo, debug, timeit)
    elif algo in ['f2', 'f2lp', 'f24threads']:
        return handle_ged_fs(g1, g2, algo, debug, timeit)
    elif algo == 'hed':
        return handle_ged_hed(g1, g2, algo, debug, timeit)
    else:
        raise RuntimeError('Unknown ged algo {}'.format(algo))


def normalized_dist_sim(d, g1, g2, dec_gsize=False):
    g1_size = g1.number_of_nodes()
    g2_size = g2.number_of_nodes()
    if dec_gsize:
        g1_size -= 1
        g2_size -= 1
    return 2 * d / (g1_size + g2_size)


def unnormalized_dist_sim(d, g1, g2, dec_gsize=False):
    g1_size = g1.number_of_nodes()
    g2_size = g2.number_of_nodes()
    if dec_gsize:
        g1_size -= 1
        g2_size -= 1
    return d * (g1_size + g2_size) / 2


def handle_ged_gmt(g1, g2, algo, debug, timeit):
    # https://github.com/dan-zam/graph-matching-toolkit
    gp = get_gmt_path()
    append_str = get_append_str(g1, g2)
    src, t_datapath = setup_temp_data_folder(gp, append_str)
    meta1 = write_to_temp(g1, t_datapath, algo, 'g1')
    meta2 = write_to_temp(g2, t_datapath, algo, 'g2')
    if meta1 != meta2:
        if not ((meta1 in meta2) or (meta2 in meta1)):
            raise RuntimeError(
                'Different meta data {} vs {}'.format(meta1, meta2))
        else:
            if meta1 in meta2:
                meta1 = meta2
    prop_file = setup_property_file(src, gp, meta1, append_str)
    rtn = []
    lcnt, result_file, t = None, None, None
    if not exec_cmd(
            'cd {} && java {}'
            ' -classpath {}/src/graph-matching-toolkit/bin algorithms.GraphMatching '
            './properties/properties_temp_{}.prop'.format(
                gp, '-XX:-UseGCOverheadLimit -XX:+UseConcMarkSweepGC -Xmx100g'
                if algo == 'astar' else '', get_root_path(), append_str)):
        rtn.append(-1)
    else:
        d, t, lcnt, g1size, g2size, result_file = get_gmt_result(gp, algo, append_str)
        rtn.append(d)
        if g1size != g1.number_of_nodes():
            print('g1size {} g1.number_of_nodes() {}'.format(g1size, g1.number_of_nodes()))
        assert (g1size == g1.number_of_nodes())
        assert (g2size == g2.number_of_nodes())
    if debug:
        rtn += [lcnt, g1, g2]
    if timeit:
        rtn.append(t)
    clean_up([t_datapath, prop_file, result_file])
    if len(rtn) == 1:
        return rtn[0]
    return tuple(rtn)


def setup_temp_data_folder(gp, append_str):
    dir = gp + '/data'
    create_dir_if_not_exists(dir)
    tp = dir + '/temp_{}'.format(append_str)
    exec_cmd('rm -rf {} && mkdir {}'.format(tp, tp))
    src = get_root_path() + '/src/gmt_files'
    exec_cmd('cp {}/temp.xml {}/temp_{}.xml'.format(src, tp, append_str))
    return src, tp


def write_to_temp(g, tp, algo, g_name):
    node_attres, edge_attrs = nx_to_gxl(
        g, g_name, '{}/{}.gxl'.format(tp, g_name),
        ignore_node_attrs=['label'],
        ignore_edge_attrs=['id'])
    return algo + '_' + '_'.join(sorted(list(node_attres.keys())) + \
                                 sorted(list(edge_attrs.keys())))


def setup_property_file(src, gp, meta, append_str):
    destfile = '{}/properties/properties_temp_{}.prop'.format(
        gp, append_str)
    srcfile = '{}/{}.prop'.format(src, meta)
    if not isfile(srcfile):
        if 'beam' in meta:  # for beam
            metasp = meta.split('_')
            s = int(metasp[0][4:])
            if s <= 0:
                raise RuntimeError('Invalid s for beam search: {}'.format(s))
            newmeta = '_'.join(['beam'] + metasp[1:])
            srcfile = '{}/{}.prop'.format(src, newmeta)
        else:
            raise RuntimeError('File {} does not exist'.format(srcfile))
    exec_cmd('cp {} {}'.format(srcfile, destfile))
    for line in fileinput.input(destfile, inplace=True):
        line = line.rstrip()
        if line == 's=':  # for beam
            print('s={}'.format(s))
        else:
            print(line.replace('temp', 'temp_{}'.format(append_str)))
    return destfile


def get_gmt_result(gp, algo, append_str):
    result_file = '{}/result/temp_{}'.format(gp, append_str)
    with open(result_file) as f:
        lines = f.readlines()
        ln = 16 if 'beam' in algo else 15
        t = int(lines[ln].split(': ')[1])  # msec
        ln = 23 if 'beam' in algo else 22
        d = float(lines[ln]) * 2
        if (d - int(d) != 0) and algo != 'hungarian' and algo != 'vj':
            raise RuntimeError('{} != {}'.format(d, int(d)))
        d = int(d)
        if d < 0:
            d = -1  # in case rtn == -2
        ln = 26 if 'beam' in algo else 25
        g1size = int(lines[ln])
        ln = 27 if 'beam' in algo else 26
        g2size = int(lines[ln])
        ln = 28 if 'beam' in algo else 27
        lcnt = int(float(lines[ln]))
        return d, t, lcnt, g1size, g2size, result_file


def get_gmt_path():
    return get_root_path() + '/src/graph-matching-toolkit'


def get_append_str(g1, g2):
    return '{}_{}_{}_{}'.format(
        get_ts(), getpid(), g1.graph['gid'], g2.graph['gid'])


def clean_up(path_list):
    for path in path_list:
        exec_cmd('rm -rf {}'.format(path))


def handle_ged_fs(g1, g2, algo, debug, timeit):
    # https://drive.google.com/file/d/12MBjXcNko83mAUGKe9nVJqEKjLTjDJNd/view?usp=sharing
    gp = get_fs_path(algo)
    append_str = get_append_str(g1, g2)
    src, t_datapath = setup_temp_data_folder(gp, append_str)
    meta1 = write_to_temp(g1, t_datapath, algo, 'g1')
    meta2 = write_to_temp(g2, t_datapath, algo, 'g2')
    if meta1 != meta2:
        if not ((meta1 in meta2) or (meta2 in meta1)):
            raise RuntimeError(
                'Different meta data {} vs {}'.format(meta1, meta2))
    rtn = []
    result_file = t_datapath + '/reault.txt'
    t = None
    if not exec_cmd(
            'cd {}/symbolic && '
            'DISPLAY=:0 wine {}_symbolic_distance.exe '
            '1.0 1.0 0.0 1.0 {}/g1.gxl {}/g2.gxl | tee {}'.format(
                gp, algo.upper(), t_datapath, t_datapath, result_file)):
        rtn.append(-1)
    else:
        with open(result_file) as f:
            lines = f.readlines()
            assert (len(lines) == 1)
            line = lines[0]
            ls = line.rstrip().split(';')
            assert (len(ls) == 4)
            t = float(ls[1]) * 1000  # sec to msec
            d = int(float(ls[3]))
        rtn.append(d)
    if debug:
        rtn += [-1, g1, g2]
    if timeit:
        rtn.append(t)
    clean_up([t_datapath, result_file])
    if len(rtn) == 1:
        return rtn[0]
    return tuple(rtn)


def get_fs_path(algo):
    return get_root_path() + '/model/' + algo.upper()


def handle_ged_hed(g1, g2, algo, debug, timeit):
    # https://github.com/priba/aproximated_ged
    from VanillaHED import VanillaHED
    rtn = []
    assert (algo == 'hed')
    hed = VanillaHED(del_node=1.0, ins_node=1.0, del_edge=1.0, ins_edge=1.0,
                     node_metric='matching', edge_metric='matching')
    g1 = map_node_type_to_float(g1)
    g2 = map_node_type_to_float(g2)
    t = time()
    d, _ = hed.ged(g1, g2)
    rtn.append(d)
    if debug:
        rtn += [-1, g1, g2]
    if timeit:
        rtn.append((time() - t) * 1000)
    if len(rtn) == 1:
        return rtn[0]
    return tuple(rtn)


def map_node_type_to_float(g):
    for n, attr in g.nodes(data=True):
        if 'type' in attr:
            s = ''
            for c in attr['type']:
                num = ord(c)
                s = str(num)
            num = int(s)
            attr['hed_mapped'] = num
        else:
            attr['hed_mapped'] = 0
    for n1, n2, attr in g.edges(data=True):
        attr['hed_mapped'] = 0
    return g


""" MCS. """


def mcs(g1, g2, algo, labeled=False, label_key='', debug=False, timeit=False, timeout=None):
    """
    :param g1:
    :param g2:
    :param algo:
    :param debug:
    :param timeit:
    :param timeout: Timeout in seconds that the MCS run is allowed to take.
    :return: The actual size of MCS (# of edges) (integer),
             the node mapping (e.g. [{('1', '2'): ('A', 'B'), ('7', '9'): ('C', 'D')}, ...]),
             the edge_id mapping (e.g. [{'1': '2', '4': '3'}, ...]
             wall time in msec (if timeit==True).
    """
    if algo == 'mccreesh2016' or algo == 'mccreesh2017':
        return mcs_cpp_helper(g1, g2, algo, labeled, label_key, debug, timeit, timeout)
    else:
        return mcs_java_helper(g1, g2, algo, debug, timeit, timeout)


def mcs_java_helper(g1, g2, algo, debug=False, timeit=False, timeout=None):
    """See mcs function. Must match return format."""
    # Input format is ./model/mcs/data/temp_<ts>_<pid>_<gid1>_<gid2>/<gid1>_<gid2>_<algo>.json
    # Prepare the json file for java to read.
    gp = get_mcs_path()
    append_str = get_append_str(g1, g2)
    src, t_datapath = setup_temp_data_folder(gp, append_str)
    filepath = '{base}/{g1}_{g2}_{algo}.json'.format(
        base=t_datapath,
        g1=g1.graph['gid'],
        g2=g2.graph['gid'],
        algo=algo)
    write_java_input_file(g1, g2, algo, filepath)

    # Run the java program.
    java_src = get_mcs_path() + '/Java_MCS_algorithms/bin/'
    classpath = '{}:{}'.format(get_mcs_path() + '/Java_MCS_algorithms/bin/',
                               get_mcs_path() + '/Java_MCS_algorithms/lib/*')
    main_class = 'org.cisrg.mcsrun.RunMcs'
    exec_result = exec_cmd('java -Xmx20g -classpath "{classpath}" {main_class} {data_file}'.format(
        root=java_src, classpath=classpath, main_class=main_class, data_file=filepath), timeout=timeout)

    # Get out immediately with a -1 so the csv file logs failed test.
    # mcs_size = -1 means failed in the time limit.
    # mcs_size = -2 means failed by memory limit or other error.
    if not exec_result:
        return -1, -1, -1, timeout * 1000

    # Check if the output file exists, otherwise java threw an exception and didn't output anything.
    output_filepath = filepath + '.out'
    if not isfile(output_filepath):
        return -2, -1, -1, 0

    # Process the output data from the java program, original filename + .out (*.json.out).
    with open(output_filepath, 'r') as jsonfile:
        output_data = json.load(jsonfile)

    # Get the relevant data.
    edge_mappings = output_data['mcsEdgeMapIds']
    elapsed_time = output_data['elapsedTime']
    if len(edge_mappings) == 0:
        mcs_size = 0
    else:
        mcs_size = len(edge_mappings[0])

    mcs_node_id_maps, mcs_node_label_maps = get_mcs_info(g1, g2, edge_mappings)

    clean_up([t_datapath])

    if timeit:
        return mcs_size, mcs_node_label_maps, edge_mappings, elapsed_time
    else:
        return mcs_size, mcs_node_label_maps, edge_mappings


def mcs_cpp_helper(g1, g2, algo, labeled, label_key, debug=False, timeit=False, timeout=None):
    """See mcs function. Must match return format."""
    # Input format is ./model/mcs/data/temp_<ts>_<pid>_<gid1>_<gid2>/<gid1>.<extension>
    # Prepare both graphs to be read by the program.
    commands = []
    if algo == 'mccreesh2016':
        binary_name = 'solve_max_common_subgraph'
        extension = 'mivia'
        write_fn = write_mivia_input_file
        commands.append('' if labeled else '--unlabelled')
        commands.append('--connected')
        commands.append('--undirected')
    elif algo == 'mccreesh2017':
        binary_name = 'mcsp'  # 'mcsp_scai1'
        extension = 'mivia'
        write_fn = write_mivia_input_file
        commands.append('--labelled' if labeled else '')
        commands.append('--connected')
        commands.append('--quiet')
        if timeout:
            commands.append('--timeout={}'.format(timeout))
        commands.append('min_product')
    else:
        raise RuntimeError('{} not yet implemented in mcs_cpp_helper'.format(algo))

    gp = get_mcs_path()
    append_str = get_append_str(g1, g2)
    src, t_datapath = setup_temp_data_folder(gp, append_str)
    filepath_g1 = '{base}/{g1}.{extension}'.format(
        base=t_datapath,
        g1=g1.graph['gid'],
        extension=extension)
    filepath_g2 = '{base}/{g2}.{extension}'.format(
        base=t_datapath,
        g2=g2.graph['gid'],
        extension=extension)

    if labeled:
        label_map = _get_label_map(g1, g2, label_key)
    else:
        label_map = {}
    idx_to_node_1 = write_fn(g1, filepath_g1, labeled, label_key, label_map)
    idx_to_node_2 = write_fn(g2, filepath_g2, labeled, label_key, label_map)

    cpp_binary = '{mcs_path}/{algo}/{binary}'.format(mcs_path=get_mcs_path(), algo=algo,
                                                     binary=binary_name)

    # Run the solver.
    t = time()
    exec_result = exec_cmd('{bin} {commands} {g1} {g2}'.format(
        bin=cpp_binary, commands=' '.join(commands),
        g1=filepath_g1, g2=filepath_g2))
    elapsed_time = time() - t
    elapsed_time *= 1000  # sec to msec

    # Get out immediately with a -1 so the csv file logs failed test.
    # mcs_size = -1 means failed in the time limit.
    # mcs_size = -2 means failed by memory limit or other error.
    if not exec_result:
        return -1, -1, -1, timeout * 1000

    # Check if the output file exists, otherwise something failed with no output.
    output_filepath = t_datapath + '/output.csv'
    if not isfile(output_filepath):
        return -2, -1, -1, 0

    # Process the output data.
    with open(output_filepath, 'r') as readfile:
        num_nodes_mcis = int(readfile.readline().strip())
        idx_mapping = eval(readfile.readline().strip())
        mcs_node_id_mapping = {idx_to_node_1[idx1]: idx_to_node_2[idx2] for idx1, idx2 in idx_mapping.items()}
        # elapsed_time = int(readfile.readline().strip())

    edge_mapping = mcis_edge_map_from_nodes(g1, g2, mcs_node_id_mapping)
    mcs_size = num_nodes_mcis
    mcs_node_id_maps, mcs_node_label_maps = get_mcs_info(g1, g2, [edge_mapping])

    clean_up([t_datapath])

    if not debug:
        return mcs_size

    if timeit:
        return mcs_size, mcs_node_label_maps, [edge_mapping], elapsed_time
    else:
        return mcs_size, mcs_node_label_maps, [edge_mapping]


def _get_label_map(g1, g2, label_key):
    # Need this function because the two graphs needs consistent labelings in the mivia format. If they are called
    # separately, then they will likely have wrong labelings.
    label_dict = {}
    label_counter = 0
    # We make the labels into ints so that they can fit in the 16 bytes needed
    # for the labels in the mivia format. Each unique label encountered just gets a
    # unique label from 0 to edge_num - 1
    for g in [g1, g2]:
        for node, attr in g.nodes(data=True):
            current_label = attr[label_key]
            if current_label not in label_dict:
                label_dict[current_label] = label_counter
                label_counter += 1
    return label_dict


def mcis_edge_map_from_nodes(g1, g2, node_mapping):
    edge_map = {}
    induced_g1 = g1.subgraph([str(key) for key in node_mapping.keys()])
    induced_g2 = g2.subgraph([str(key) for key in node_mapping.values()])

    used_edge_ids_g2 = set()
    for u1, v1, edge1_attr in induced_g1.edges(data=True):
        u2 = str(node_mapping[int(u1)])
        v2 = str(node_mapping[int(v1)])
        edge1_id = edge1_attr['id']
        for temp1, temp2, edge2_attr in induced_g2.edges_iter(nbunch=[u2, v2], data=True):
            if (u2 == temp1 and v2 == temp2) or (u2 == temp2 and v2 == temp1):
                edge2_id = edge2_attr['id']
                if edge2_id in used_edge_ids_g2:
                    continue
                used_edge_ids_g2.add(edge2_id)
                edge_map[edge1_id] = edge2_id

    return edge_map


def write_mivia_input_file(graph, filepath, labeled, label_key, label_map):
    bytes, idx_to_node = convert_to_mivia(graph, labeled, label_key, label_map)
    with open(filepath, 'wb') as writefile:
        for byte in bytes:
            writefile.write(byte)
    return idx_to_node


def get_mcs_info_cpp(g1, g2, mivia_edge_mapping):
    g1_edge_map = get_mivia_edge_map(g1)
    g2_edge_map = get_mivia_edge_map(g2)

    # Translate the mivia edge map to nx edge id map.
    edge_map = {}
    for mivia_edge_1, mivia_edge_2 in mivia_edge_mapping.items():
        edge_1 = g1_edge_map[mivia_edge_1]
        edge_2 = g2_edge_map[mivia_edge_2]
        edge_map[edge_1] = edge_2

    mcs_node_id_maps, mcs_node_label_maps = get_mcs_info(g1, g2, [edge_map])

    return mcs_node_id_maps, mcs_node_label_maps, [edge_map]


def get_mivia_edge_map(graph):
    edge_map = {}

    # Go through same order as how we create the mivia graph file.
    adj_iter = sorted(graph.adj.items(), key=lambda x: int(x[0]))
    edge_num = 0
    for source_id, adj_list in adj_iter:
        for target_id, attr in sorted(adj_list.items(), key=lambda x: int(x[0])):
            edge_id = attr['id']
            edge_map[edge_num] = edge_id
            edge_num += 1

    return edge_map


def get_mcs_info(g1, g2, edge_mappings):
    id_edge_map1 = get_id_edge_map(g1)
    id_edge_map2 = get_id_edge_map(g2)

    mcs_node_id_maps = []
    mcs_node_label_maps = []
    for edge_mapping in edge_mappings:
        node_id_map = get_node_id_map_from_edge_map(id_edge_map1, id_edge_map2, edge_mapping)
        node_label_map = node_id_map_to_label_map(g1, g2, node_id_map)
        mcs_node_id_maps.append(node_id_map)
        mcs_node_label_maps.append(node_label_map)
    return mcs_node_id_maps, mcs_node_label_maps


def node_id_map_to_label_map(g1, g2, node_id_map):
    node_label_map = {}
    for (source1, target1), (source2, target2) in node_id_map.items():
        g1_edge = (g1.node[source1]['label'], g1.node[target1]['label'])
        g2_edge = (g2.node[source2]['label'], g2.node[target2]['label'])
        node_label_map[g1_edge] = g2_edge
    return node_label_map


def get_node_id_map_from_edge_map(id_edge_map1, id_edge_map2, edge_mapping):
    node_map = {}
    for edge1, edge2 in edge_mapping.items():
        nodes_edge1 = id_edge_map1[edge1]
        nodes_edge2 = id_edge_map2[edge2]
        nodes1 = (nodes_edge1[0], nodes_edge1[1])
        nodes2 = (nodes_edge2[0], nodes_edge2[1])
        node_map[nodes1] = nodes2
    return node_map


def get_id_edge_map(graph):
    id_edge_map = {}
    for u, v, edge_data in graph.edges(data=True):
        edge_id = edge_data['id']
        assert edge_id not in id_edge_map
        id_edge_map[edge_id] = (u, v)
    return id_edge_map


def get_mcs_path():
    return get_root_path() + '/model/mcs'


def write_java_input_file(g1, g2, algo, filepath):
    """Prepares and writes a file in JSON format for MCS calculation."""
    write_data = {}
    write_data['graph1'] = graph_as_dict(g1)
    write_data['graph2'] = graph_as_dict(g2)
    write_data['algorithm'] = algo
    # Assume there's at least one node and get its attributes
    test_node_attr = g1.nodes_iter(data=True).__next__()[1]
    # This is the actual key we want the MCS algorithm to use to compare node labels. The
    # Java MCS code has a default "unlabeled" key, so for unlabeled graphs, can just use that.
    write_data['nodeLabelKey'] = 'type' if 'type' in test_node_attr else 'unlabeled'

    with open(filepath, 'w') as jsonfile:
        json.dump(write_data, jsonfile)


def graph_as_dict(graph):
    dict = {}
    dict['directed'] = nx.is_directed(graph)
    dict['gid'] = graph.graph['gid']
    dict['nodes'] = []
    dict['edges'] = []
    for node, attr in graph.nodes(data=True):
        node_data = {}
        node_data['id'] = node
        node_data['label'] = attr['label']
        if 'type' in attr:
            node_data['type'] = attr['type']
        dict['nodes'].append(node_data)
    for source, target, attr in graph.edges(data=True):
        dict['edges'].append({'id': attr['id'], 'source': source, 'target': target})
    return dict


if __name__ == '__main__':
    from utils import load_data

    test_data = load_data('aids700nef', train=False)
    train_data = load_data('aids700nef', train=True)
    g1 = test_data.graphs[2]
    g2 = train_data.graphs[169]
    mcs_size, mcs_node_mapping, mcs_edge_mapping, elapsed_time = mcs(
        g1, g2, 'mccreesh2017', labeled=False, label_key='', timeit=True,
        debug=True)
    print('mcs_size', mcs_size)
    print('mcs_node_mapping', mcs_node_mapping)
    print('mcs_edge_mapping', mcs_edge_mapping)
    print('elapsed_time', elapsed_time)
    # d = ged(g1, g2, 'astar')
    # d = ged(g1, g2, 'beam80')
    # print(d)
    # nged = normalized_dist_sim(d, g1, g2)
    # print(nged)
    # original_ged = unnormalized_dist_sim(nged, g1, g2)
    # print(original_ged)
    # g1 = test_data.graphs[15]
    # g2 = train_data.graphs[761]
    #
    # # nx.write_gexf(g1, get_root_path() + '/temp/g1.gexf')
    # # nx.write_gexf(g2, get_root_path() + '/temp/g2.gexf')
    # g1 = nx.read_gexf(get_root_path() + '/temp/g1_small.gexf')
    # g2 = nx.read_gexf(get_root_path() + '/temp/g2_small.gexf')
    # print(astar_ged(g1, g2))
    # print(beam_ged(g1, g2, 2))
    # import networkx as nx
    #
    # g1 = nx.Graph(gid=1)
    # g1.add_node(1, label=1, type='red')
    # g1.add_node(2, label=2, type='red')
    # g1.add_node(3, label=3, type='blue')
    # g1.add_node(4, label=4, type='blue')
    # g1.add_node(5, label=5, type='blue')
    # g1.add_node(6, label=6, type='blue')
    # g1.add_node(7, label=7, type='blue')
    # g1.add_node(8, label=8, type='blue')
    # g1.add_edge(1, 2, id=1)
    # g1.add_edge(1, 3, id=2)
    # g1.add_edge(1, 4, id=3)
    # g1.add_edge(1, 8, id=4)
    # g1.add_edge(2, 5, id=5)
    # g1.add_edge(2, 6, id=6)
    # g1.add_edge(2, 7, id=7)
    #
    # g2 = nx.Graph(gid=2)
    # g2.add_node(1, label=1, type='red')
    # g2.add_node(2, label=2, type='red')
    # g2.add_node(3, label=3, type='deep_blue')
    # g2.add_node(4, label=4, type='red')
    # g2.add_node(5, label=5, type='yellow')
    # g2.add_node(6, label=6, type='green')
    # g2.add_edge(1, 2, id=1)
    # g2.add_edge(2, 3, id=2)
    # g2.add_edge(3, 4, id=3)
    # g2.add_edge(4, 5, id=4)
    # g2.add_edge(5, 6, id=5)
    # g2.add_edge(5, 1, id=6)
    # print(ged(g1, g2, 'astar'))
