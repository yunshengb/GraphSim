from utils import get_save_path, get_result_path, \
    save, load, load_data, get_norm_str, create_dir_if_not_exists
from dist_sim import ged, normalized_dist_sim, mcs
from collections import OrderedDict
from pandas import read_csv
import numpy as np


class DistSimCalculator(object):
    def __init__(self, dataset, ds_metric, algo):
        if ds_metric == 'ged':
            self.dist_sim_func = ged
            ds = 'dist'
        elif ds_metric == 'glet': # graphlet similarity metric
            self.dist_sim_func = None # should be pre-computed and loaded
            ds = 'sim'
        elif ds_metric == 'mcs':
            self.dist_sim_func = mcs
            ds = 'dist'
        else:
            raise RuntimeError('Unknwon distance/similarity metric {}'.format(ds_metric))
        self.sfn = '{}/{}_{}_{}{}_gidpair_{}_map'.format(
            get_save_path(), dataset, ds_metric, algo,
            '' if algo == 'astar' or algo == 'graphlet' or algo == 'mccreesh2017' else '_revtakemin', ds)
        self.algo = algo
        self.gidpair_ds_map = load(self.sfn)
        if not self.gidpair_ds_map:
            self.gidpair_ds_map = OrderedDict()
            save(self.sfn, self.gidpair_ds_map)
            print('Saved dist/sim map to {} with {} entries'.format(
                self.sfn, len(self.gidpair_ds_map)))
        else:
            print('Loaded dist/sim map from {} with {} entries'.format(
                self.sfn, len(self.gidpair_ds_map)))

    def calculate_dist_sim(self, g1, g2, dec_gsize=False, return_neg1=False):
        gid1 = g1.graph['gid']
        gid2 = g2.graph['gid']
        pair = (gid1, gid2)
        d = self.gidpair_ds_map.get(pair)
        if d is None:
            if return_neg1:
                return -1, -1
            rev_pair = (gid2, gid1)
            rev_d = self.gidpair_ds_map.get(rev_pair)
            if rev_d:
                d = rev_d
            else:
                d = self.dist_sim_func(g1, g2, self.algo)
                if self.algo != 'astar':
                    d = min(d, self.dist_sim_func(g2, g1, self.algo))
            self.gidpair_ds_map[pair] = d
            print('{}Adding entry ({}, {}) to dist map'.format(
                ' ' * 80, pair, d))
            save(self.sfn, self.gidpair_ds_map)
        return d, normalized_dist_sim(d, g1, g2, dec_gsize=dec_gsize)

    def load(self, row_gs, col_gs, mats=None, csv_filenames=None, ds_metric=None,
             check_symmetry=False):
        """
        Load the internal distance map from external distance matrices,
            each of which is assumed to be m by n, or external csv files.
        Use this function if the pairwise distances have been calculate
            elsewhere, e.g. by the multiprocessing version of
            running the baselines as in one of the functions in exp.py.
        The distance map stored in this distance calculator will be
            enriched/expanded by the results.
        Be careful of the inputs! Check row_gs, col_gs, and the actual matrices
            or csv files match!
        :param row_gs: the corresponding row graphs
        :param col_gs: the corresponding column graphs
        :param mats:
        :param csv_filename:
        :param ds_metric: currently only support ged
        :param check_symmetry: whether to check if mat if symmetric or not
        :return:
        """
        exsiting_entries_list = None
        if mats:
            m, n = mats[0].shape
            assert (m == len(row_gs) and n == len(col_gs))
        else:
            assert (csv_filenames)
            print('Loading', csv_filenames)
            exsiting_entries_list = [load_from_exsiting_csv(csv_filename, dist_metric)
                                     for csv_filename in csv_filenames]
            print('Done loading with {} extries'.format([len(x) for x in exsiting_entries_list]))
        m, n = len(row_gs), len(col_gs)
        valid_count = 0
        for i in range(m):
            for j in range(n):
                i_gid = row_gs[i].graph['gid']
                j_gid = col_gs[j].graph['gid']
                d = self._min_vote_from(i, j, i_gid, j_gid, mats, exsiting_entries_list, ds_metric)
                if check_symmetry:
                    d_t = self._min_vote_from(j, i, j_gid, i_gid, mats, exsiting_entries_list, ds_metric)
                    if d != d_t:
                        raise RuntimeError(
                            'Asymmetric distance {} {}: {} and {}'.format(
                                i, j, d, d_t))
                gid1 = row_gs[i].graph['gid']
                gid2 = col_gs[j].graph['gid']
                pair = (gid1, gid2)
                d_m = self.gidpair_ds_map.get(pair)
                if d_m:
                    if d != d_m:
                        raise RuntimeError(
                            'Inconsistent distance {} {}: {} and {}'.format(
                                i, j, d, d_m))
                else:
                    if d != -1:
                        valid_count += 1
                    self.gidpair_ds_map[pair] = d
        save(self.sfn, self.gidpair_ds_map)
        print('{} valid entries loaded'.format(valid_count))

    def _min_vote_from(self, i, j, i_gid, j_gid, mats=None, exsiting_entries_list=None, ds_metric=None):
        """
        Assume the min needs to be taken. For MCS, may need the max (to be implemented).
        :param i:
        :param j:
        :param mats:
        :param exsiting_entries_list:
        :param ds_metric:
        :return:
        """
        if mats:
            return np.min([mat[i][j] for mat in mats])
        else:
            assert (exsiting_entries_list)
            cands = []
            for exsiting_entries in exsiting_entries_list:
                tmp = exsiting_entries.get((i_gid, j_gid))
                if tmp:
                    if ds_metric == 'ged':
                        _, _, _, _, d, _, _ = tmp
                    else:
                        raise NotImplementedError()  # TODO: for MCS, change min to max voting
                    cands.append(d)
            if len(cands) != len(exsiting_entries_list):  # not all finish computing for this pair
                return -1  # -1 indicates invalid ged
            else:
                return np.min(cands)


def get_train_train_dist_mat(dataset, dist_metric, dist_algo, norm):
    train_data = load_data(dataset, train=True)
    gs = train_data.graphs
    dist_sim_calculator = DistSimCalculator(dataset, dist_metric, dist_algo)
    return get_gs_ds_mat(gs, gs, dist_sim_calculator, 'train', 'train', dataset,
                           dist_metric, dist_algo, norm)


def get_gs_ds_mat(gs1, gs2, dist_sim_calculator, tvt1, tvt2,
                    dataset, dist_metric, dist_algo, norm, dec_gsize, return_neg1=False):
    mat_str = '{}({})_{}({})'.format(tvt1, len(gs1), tvt2, len(gs2))
    dir = '{}/ds_mat'.format(get_save_path())
    create_dir_if_not_exists(dir)
    sfn = '{}/{}_{}_ds_mat_{}{}_{}'.format(
        dir, dataset, mat_str, dist_metric,
        get_norm_str(norm), dist_algo)
    l = load(sfn)
    if l is not None:
        print('Loaded from {}'.format(sfn))
        return l
    m = len(gs1)
    n = len(gs2)
    dist_mat = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            g1 = gs1[i]
            g2 = gs2[j]
            d, normed_d = dist_sim_calculator.calculate_dist_sim(
                g1, g2, dec_gsize=dec_gsize, return_neg1=return_neg1)
            if norm:
                dist_mat[i][j] = normed_d
            else:
                dist_mat[i][j] = d
    save(sfn, dist_mat)
    print('Saved to {}'.format(sfn))
    return dist_mat


def load_from_exsiting_csv(csv_fn, ds_metric, skip_eval=True):
    rtn = {}
    if csv_fn:
        data = read_csv(csv_fn)
        for _, row in data.iterrows():
            i = int(row['i'])
            j = int(row['j'])
            # if j > 2:
            #     return rtn
            i_gid = int(row['i_gid'])
            j_gid = int(row['j_gid'])
            i_node = int(row['i_node'])
            j_node = int(row['j_node'])
            t = float(row['time(msec)'])
            if ds_metric == 'ged':
                lcnt = int(row['lcnt'])
                d = int(row['ged'])
                rtn[(i_gid, j_gid)] = (i_gid, j_gid, i_node, j_node, d, lcnt, t)
            elif ds_metric == 'mcs':
                d = int(row['mcs'])
                # Check the case where there was an error and we need to rerun it.
                if d < 0:
                    continue
                if skip_eval:
                    rtn[(i_gid, j_gid)] = (i_gid, j_gid, i_node, j_node, d, None, None, t)
                else:
                    # Check node mappings are right data format.
                    node_mappings = eval(row['node_mapping'])
                    assert isinstance(node_mappings, list), 'node_mapping must be a list'
                    if len(node_mappings) > 0:
                        assert isinstance(node_mappings[0], dict), 'node_mapping items must be dicts'
                        # assert isinstance(list(node_mappings[0].keys())[0], tuple), 'node_mapping keys must be tuples'
                        # assert isinstance(list(node_mappings[0].values())[0],
                        #                   tuple), 'node_mapping values must be tuples'
                    # Check edge mappings are right data format.
                    edge_mappings = eval(row['edge_mapping'])
                    assert isinstance(edge_mappings, list), 'edge_mapping must be a list'
                    if len(edge_mappings) > 0:
                        assert isinstance(edge_mappings[0], dict), 'edge_mapping items must be dicts'
                        # assert isinstance(list(edge_mappings[0].keys())[0], str), 'edge_mapping keys must be str'
                        # assert isinstance(list(edge_mappings[0].keys())[0], str), 'edge_mapping values must be str'
                    rtn[(i_gid, j_gid)] = (i_gid, j_gid, i_node, j_node, d, node_mappings, edge_mappings, t)
            else:
                raise RuntimeError('Did not handle ds_metric parsing for {}'.format(ds_metric))
    print('Loaded {} entries from {}'.format(len(rtn), csv_fn))
    return rtn


def find_graph_with_gid(gs, gid):
    for g in gs:
        if g.graph['gid'] == gid:
            return g
    return None


if __name__ == '__main__':
    dataset = 'mutag'
    dist_metric = 'ged'
    dist_algo = 'astar'
    dist_sim_calculator = DistSimCalculator(dataset, dist_metric, dist_algo)
    # The server qilin calculated all the pairwise distances between
    # the training graphs.
    # Thus, enrich the distance map (i.e. calculator) using the qilin results.
    csv3 = ('{}/{}/csv/{}.csv'.format(
        get_result_path(), dataset,
        'ged_mutag_beam80_2019-01-22T13:55:19.744928_qilin_all_20cpus'))
    csv1 = ('{}/{}/csv/{}.csv'.format(
        get_result_path(), dataset,
        'ged_mutag_hungarian_2019-01-22T14:09:43.111557_feilong_all_15cpus'))
    csv2 = ('{}/{}/csv/{}.csv'.format(
        get_result_path(), dataset,
        'ged_mutag_vj_2019-01-22T16:34:47.260820_qilin_all_20cpus'))
    row_gs = load_data(dataset, train=True).graphs
    col_gs = load_data(dataset, train=True).graphs
    dist_sim_calculator.load(row_gs, col_gs, csv_filenames=[csv1, csv2, csv3],
                         ds_metric='ged', check_symmetry=False)
    # dataset = 'webeasy'
    # ds_metric = 'glet'
    # dist_algo = 'graphlet'
    # dist_sim_calculator = DistSimCalculator(dataset, ds_metric, dist_algo)
    # mat = np.load('{}/{}/{}/{}.npy'.format(
    #     get_result_path(), dataset, ds_metric,
    #     'glet_glet_mat_webeasy_graphlet_all'))
    # row_gs = load_data(dataset, train=True).graphs
    # col_gs = load_data(dataset, train=True).graphs
    # dist_sim_calculator.load(row_gs, col_gs, [mat], check_symmetry=True)
    # dataset = 'aids80nef'
    # ds_metric = 'ged'
    # dist_algo = 'astar'
    # dist_sim_calculator = DistSimCalculator(dataset, ds_metric, dist_algo)
    # gs = load_data(dataset, True).graphs + load_data(dataset, False).graphs
    # g1 = find_graph_with_gid(gs, 31109)
    # g2 = find_graph_with_gid(gs, 34223)
    # print(dist_sim_calculator.calculate_dist_sim(g1, g2))
