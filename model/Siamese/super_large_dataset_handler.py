from fake_generator_ged import graph_generator
from fake_generator_mcs import graph_generator_mcs
from dist_sim import normalized_dist_sim
from samplers import SelfShuffleList
from config import FLAGS
import numpy as np
import networkx as nx
import random


def gen_data(from_info, gen_info, model_data):
    row_gs = _gen_row_gs(from_info, model_data)
    col_gs_list, true_ds_mat, sim_or_dist = _gen_col_gs(gen_info, row_gs)
    return row_gs, col_gs_list, true_ds_mat, sim_or_dist


def _gen_row_gs(from_info, model_data):
    sp, num_gs, what_gs = _parse_basic_info(from_info, 'from_info', 2)
    if what_gs == 'testgs':
        return _sample_from_test_gs(from_info, sp, num_gs, model_data)
    elif what_gs == 'fastgnp':
        return _gen_fastgnp_gs(from_info, sp, num_gs, model_data)
    else:
        raise RuntimeError('Unknown graph types {} in {}'.format(
            what_gs, from_info))


def _gen_col_gs(gen_info, row_gs):
    sp, num_gs, what_gs = _parse_basic_info(gen_info, 'gen_info', 2)
    if what_gs == 'isotop':
        return _gen_isotop_gs(gen_info, sp, row_gs, num_gs)
    elif what_gs == 'sp':
        return _gen_small_pertub_gs(row_gs, num_gs)
    elif what_gs == 'lp':
        return _gen_lp_gs(gen_info, sp, row_gs, num_gs)
    else:
        raise RuntimeError('Unknown graph types {} in {}'.format(
            what_gs, gen_info))


##############################  row_gs/from_gs ##############################


def _sample_from_test_gs(from_info, sp, num_gs, model_data):
    from_gs = [g.nxgraph for g in model_data.test_gs]
    if num_gs > len(from_gs):
        raise RuntimeError('Need {} gs but only {} gs {}'.format(
            num_gs, len(from_gs), from_info))
    if len(sp) > 2:
        return _sample_largest_n_gs(from_info, sp, from_gs, num_gs)
    else:
        random.Random(123).shuffle(from_gs)
        return from_gs[0:num_gs]


def _sample_largest_n_gs(from_info, sp, from_gs, num_gs):
    if sp[2] == 'largest':
        if len(sp) != 3:
            raise RuntimeError('Need largest how many graphs '
                               '{}'.format(from_info))
        gsizes = [g.number_of_nodes() for g in from_gs]
        indices = np.argsort(gsizes)[-num_gs:][::-1]
        rtn = []
        for id in indices:
            rtn.append(from_gs[id])
        return rtn
    else:
        raise RuntimeError('Unknown from_info[2] {}'.format(from_info))


def _gen_fastgnp_gs(from_info, sp, num_gs, model_data):
    if len(sp) != 4:
        raise RuntimeError('fastgnp need n and p to be specified '
                           '{}'.format(from_info))
    num_node = int(sp[2])
    prob = float(sp[3])
    if num_node <= 0 or prob < 0 or prob > 1:
        raise RuntimeError('Invalid num_node {} or prob {} in '
                           '{}'.format(num_node, prob, from_info))
    rtn = []
    for i in range(num_gs):
        while True:
            g = nx.fast_gnp_random_graph(num_node, prob, seed=123 + i)
            if g.number_of_nodes() > 1 and g.number_of_edges() >= 1:
                break
        g = _assign_glabel_ntype(g, model_data)
        g.graph['gid'] = 'Erdős-Rényi({}, {}) {}'.format(num_node, prob, i)
        rtn.append(g)
    return rtn


def _assign_glabel_ntype(g, model_data):
    g.graph['glabel'] = 0  # does not matter, just a fake graph label
    if FLAGS.node_feat_name:
        assert (FLAGS.node_feat_encoder == 'onehot')
        type = list(model_data.node_feat_encoder.feat_idx_dic.keys())[0]
        for node in g.nodes():
            nx.set_node_attributes(g, FLAGS.node_feat_name, {node: type})
    return g


############################## end of row_gs/from_gs ##############################

############################## col_gs/gen_gs ##############################


def _gen_isotop_gs(gen_info, sp, row_gs, n):
    remove_percent = _parse_percent_as_float(sp[2])
    if remove_percent <= 0 or remove_percent >= 1:
        raise RuntimeError('remove_percent {} must be in (0, 1) {}'.format(
            remove_percent, gen_info))
    return _gen_decoy_gs(row_gs, n, remove_percent, 0.0, 1)


def _gen_lp_gs(gen_info, sp, row_gs, n):
    if len(sp) != 5 or sp[2] != 'rmedge' or sp[3] != 'delta':
        raise RuntimeError('Large pertubation wrong spec {}'.format(gen_info))
    if FLAGS.ds_metric != 'ged':
        raise RuntimeError('Currently only support large perturbation for GED')
    delta_rmedge_percent = _parse_percent_as_float(sp[4])
    return _gen_decoy_gs(row_gs, n, 0.0, delta_rmedge_percent, 0)


def _gen_small_pertub_gs(row_gs, n):
    gen_gs = []
    ds_mat = np.zeros((len(row_gs), n))
    sim_or_dist = None
    for i, row_g in enumerate(row_gs):
        col_gs_for_row_g_i = []
        for j in range(n):
            if FLAGS.ds_metric == 'ged':
                new_g, d = graph_generator(row_g)
                col_gs_for_row_g_i.append(new_g)
                ds_mat[i][j] = d
                sim_or_dist = 'dist'
            else:
                r = graph_generator_mcs(row_g, '', 1, 1)[0]
                new_g, s = r.fake_graph, r.mcs
                col_gs_for_row_g_i.append(new_g)
                ds_mat[i][j] = s
                sim_or_dist = 'sim'
        gen_gs.append(col_gs_for_row_g_i)
    return gen_gs, ds_mat, sim_or_dist  # [[col_gs for row_gs[0], [col_gs for row_gs[1]], ...]


def _gen_decoy_gs(row_gs, n, start_rm_percent, delta_rm_percent, num_iso):
    if num_iso > n:
        raise RuntimeError('Cannot have {} iso graphs more than n={}'.format(
            num_iso, n))
    gen_gs = []
    ds_mat = np.zeros((len(row_gs), n))
    sim_or_dist = None
    for i, row_g in enumerate(row_gs):
        decoys, rm_edges_nums = _decoy_by_random_remove_edge(
            row_g, n, start_rm_percent, delta_rm_percent, num_iso)
        assert (len(decoys) == len(rm_edges_nums) == n)
        assert (rm_edges_nums[0] == 0)  # the first graph is always isomorphic
        gen_gs.append(decoys)
        for j in range(n):
            if FLAGS.ds_metric == 'ged':
                ged = rm_edges_nums[j]
                if FLAGS.ds_norm:
                    ged = normalized_dist_sim(ged, row_g, decoys[j])
                ds_mat[i][j] = ged
                sim_or_dist = 'dist'
            else:
                assert (FLAGS.ds_metric == 'mcs')
                if rm_edges_nums[j] == 0:
                    mcs = 1
                else:
                    # we don't know the exact MCS, 
                    # but know it must be smaller than 1, so just set to 0.
                    mcs = 0
                ds_mat[i][j] = mcs
                sim_or_dist = 'sim'
    return gen_gs, ds_mat, sim_or_dist  # [[col_gs for row_gs[0], [col_gs for row_gs[1]], ...]


def _decoy_by_random_remove_edge(nxgraph, num_gs,
                                 start_rm_percent, delta_rm_percent, num_iso):
    assert (num_iso <= num_gs)
    decoys, removed = [], []
    edges = SelfShuffleList(nxgraph.edges())
    num_edges = len(edges.li)
    num_edges_rm = int(num_edges * start_rm_percent)
    for i in range(num_gs):
        decoy = nxgraph.copy()
        if i < num_iso:
            num_edges_rm_i = 0
        else:
            if num_edges_rm > num_edges:
                raise RuntimeError('Cannot remove {] edges more than '
                                   '{}'.format(num_edges_rm, num_edges))
            ebunch = edges.get_next_n(num_edges_rm)
            assert (len(ebunch) == num_edges_rm)
            decoy.remove_edges_from(ebunch)
            # print('decoy', i, decoy.number_of_edges(), len(ebunch), nxgraph.number_of_edges())
            assert (decoy.number_of_edges() + len(ebunch) == nxgraph.number_of_edges())
            num_edges_rm_i = num_edges_rm
            if delta_rm_percent != 0:
                num_edges_rm += int(num_edges * delta_rm_percent)
        decoys.append(decoy)
        removed.append(num_edges_rm_i)
        decoy.graph['gid'] = 'rm {} edges from {}'.format(num_edges_rm, nxgraph.graph['gid'])
    return decoys, removed


############################## end of col_gs/gen_gs ##############################


def _parse_basic_info(info, what_info, at_least):
    sp = info.split('_')
    if len(sp) < at_least:
        raise RuntimeError('{} {} must have at least {} entries'.format(
            what_info, info, at_least))
    num_gs = int(sp[0])
    if num_gs <= 0:
        raise RuntimeError('{}[0] {} must indicate '
                           'how many graphs to sample'.format(what_info, info))
    return sp, num_gs, sp[1]


def _parse_percent_as_float(s):
    return float(s.strip('%')) / 100
