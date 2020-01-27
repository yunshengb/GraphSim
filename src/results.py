from utils import get_result_path
from dist_sim import normalized_dist_sim, unnormalized_dist_sim
from glob import glob
import numpy as np


class Result(object):
    """
    The result object loads and stores the ranking result of a model
        for evaluation.
        Terminology:
            rtn: return value of a function.
            m: # of queries.
            n: # of database graphs.
    """

    def __init__(self, dataset, model, row_graphs, col_graphs, col_graphs_list,
                 ds_mat, ds_mat_normed, sim_or_dist, ds_metric, time_mat):
        if sim_or_dist == 'sim':
            # Either MCS or predicted GED.
            assert (ds_metric == 'mcs' or (ds_metric == 'ged' and ds_mat is not None))
        else:
            assert (sim_or_dist == 'dist')
            assert (ds_metric == 'ged')
        self.dataset = dataset
        self.model = model
        self.row_gs = row_graphs
        if col_graphs_list is None:
            # All the row graphs in row_gs share the same list of col graphs.
            assert (col_graphs is not None)
            self.col_gs = col_graphs
        else:
            # For each row graph in row_gs, there is a (unique) list of col graphs.
            # col_graphs_list is for super large dataset test.
            assert (col_graphs is None)
            self.col_gs_list = col_graphs_list
        self.ds_mat_normed = ds_mat_normed
        self.sim_or_dist = sim_or_dist
        self.ds_metric = ds_metric
        self.time_mat = time_mat
        if ds_mat is not None:
            if ds_mat_normed:
                self.ds_mat_normed = ds_mat
                self.ds_mat = self._copy_normalize_unnormalize_dist_mat(
                    ds_mat, row_graphs, col_graphs, col_graphs_list, 'unnorm')
            else:
                self.ds_mat = ds_mat
                self.ds_mat_normed = self._copy_normalize_unnormalize_dist_mat(
                    ds_mat, row_graphs, col_graphs, col_graphs_list, 'norm')
        else:
            # Need to load from disk.
            # A*, Beam, Mccreesh2017, etc.
            # Assume these models return unnormalized GEDs.
            assert ('siamese' not in model)
            assert (not ds_mat_normed)
            assert (col_graphs is not None and col_graphs_list is None)
            self.ds_mat = self._load_result_mat(
                self.ds_metric, self.model, len(row_graphs), len(col_graphs))
            self.time_mat = None
            self.ds_mat_normed = self._copy_normalize_unnormalize_dist_mat(
                self.ds_mat, row_graphs, col_graphs, col_graphs_list, 'norm')
        self.sort_id_mat = np.argsort(self.ds_mat, kind='mergesort')
        self.sort_id_mat_normed = np.argsort(self.ds_mat_normed, kind='mergesort')
        if sim_or_dist == 'sim':
            self.sort_id_mat = self.sort_id_mat[:, ::-1]
            self.sort_id_mat_normed = self.sort_id_mat_normed[:, ::-1]

    def get_model(self):
        """
        :return: The model name.
        """
        return self.model

    def m_n(self):
        return self.dist_sim_mat(norm=False).shape

    def has_single_col_gs(self):
        return hasattr(self, 'col_gs')

    def get_single_col_gs(self):
        return self.col_gs

    def get_row_gs(self):
        return self.row_gs

    def get_col_gs(self, qid):
        if hasattr(self, 'col_gs'):
            assert (not hasattr(self, 'col_gs_list'))
            return self.col_gs
        else:
            assert (hasattr(self, 'col_gs_list'))
            return self.col_gs_list[qid]

    def get_all_gs(self, col_first=True):
        if self.has_single_col_gs():
            col_gs = self.col_gs
        else:
            col_gs = [j for i in self.col_gs_list for j in i]
        if col_first:
            return col_gs + self.row_gs
        else:
            return self.row_gs + col_gs

    def dist_or_sim(self):
        return self.sim_or_dist

    def dist_sim_mat(self, norm):
        """
        Each result object stores either a distance matrix
            or a similarity matrix. It cannot store both.
        :param norm:
        :return: either the distance matrix or the similairty matrix.
        """
        if norm:
            return self.ds_mat_normed
        else:
            return self.ds_mat

    def dist_sim(self, qid, gid, norm):
        """
        :param qid: query id (0-indexed).
        :param gid: database graph id (0-indexed) (NOT g.graph['gid']).
        :param norm:
        :return: (metric, dist or sim between qid and gid)
        """
        return self.ds_metric, self.dist_sim_mat(norm)[qid][gid]

    def top_k_ids(self, qid, k, norm, inclusive, rm):
        """
        :param qid: query id (0-indexed).
        :param k: 
        :param norm: 
        :param inclusive: whether to be tie inclusive or not.
            For example, the ranking may look like this:
            7 (sim_score=0.99), 5 (sim_score=0.99), 10 (sim_score=0.98), ...
            If tie inclusive, the top 1 results are [7, 9].
            Therefore, the number of returned results may be larger than k.
            In summary,
                len(rtn) == k if not tie inclusive;
                len(rtn) >= k if tie inclusive.
        :return: for a query, the ids of the top k database graph
        ranked by this model.
        """
        sort_id_mat = self.get_sort_id_mat(norm)
        _, n = sort_id_mat.shape
        if k < 0 or k >= n:
            raise RuntimeError('Invalid k {}'.format(k))
        if not inclusive:
            return sort_id_mat[qid][:k]
        # Tie inclusive.
        dist_sim_mat = self.dist_sim_mat(norm)
        while k < n:
            cid = sort_id_mat[qid][k - 1]
            nid = sort_id_mat[qid][k]
            if abs(dist_sim_mat[qid][cid] - dist_sim_mat[qid][nid]) <= rm:
                k += 1
            else:
                break
        return sort_id_mat[qid][:k]

    def ranking(self, qid, gid, norm, one_based=True):
        """
        :param qid: query id (0-indexed).
        :param gid: database graph id (0-indexed) (NOT g.graph['gid']).
        :param norm:
        :param one_based: whether to return the 1-based or 0-based rank.
            True by default.
        :return: for a query, the rank of a database graph by this model.
        """
        # Assume self is ground truth.
        sort_id_mat = self.get_sort_id_mat(norm)
        finds = np.where(sort_id_mat[qid] == gid)
        assert (len(finds) == 1 and len(finds[0]) == 1)
        fid = finds[0][0]
        # Tie inclusive (always when find ranking).
        dist_sim_mat = self.dist_sim_mat(norm)
        while fid > 0:
            cid = sort_id_mat[qid][fid]
            pid = sort_id_mat[qid][fid - 1]
            if dist_sim_mat[qid][pid] == dist_sim_mat[qid][cid]:
                fid -= 1
            else:
                break
        if one_based:
            fid += 1
        return fid

    def time(self, qid, gid):
        return self.time_mat[qid][gid]

    def get_time_mat(self):
        return self.time_mat

    def get_ds_metric(self):
        return self.ds_metric

    def get_sort_id_mat(self, norm):
        """
        :param norm:
        :return: a m by n matrix representing the ranking result.
            rtn[i][j]: For query i, the id of the j-th most similar
                       graph ranked by this model.
        """
        if norm:
            return self.sort_id_mat_normed
        else:
            return self.sort_id_mat

    def pred_ds(self, qid, gid, norm):
        if norm:
            return self.ds_mat_normed[qid][gid]
        else:
            return self.ds_mat[qid][gid]

    def ranking_mat(self, norm, one_based=True):
        """
        :param norm:
        :param one_based:
        :return: a m by n matrix representing the ranking result.
                 Note it is different from sort_id_mat.
            rtn[i][j]: For query i, the ranking of the graph j.
        """
        m, n = self.m_n()
        rtn = np.zeros((m, n))
        for i in range(m):
            for j in range(n):
                rtn[i][j] = self.ranking(i, j, norm, one_based=one_based)
        return rtn

    def _copy_normalize_unnormalize_dist_mat(self, ds_mat,
                                             row_graphs, col_graphs,
                                             col_graphs_list, norm_unnorm):
        rtn = np.copy(ds_mat)
        m, n = ds_mat.shape
        assert (m == len(row_graphs))
        if col_graphs is not None:
            assert (col_graphs_list is None)
            assert (n == len(col_graphs))
        else:
            assert (col_graphs is None)
            assert (m == len(col_graphs_list))
        for i in range(m):
            for j in range(n):
                if col_graphs is None:
                    col_graphs = col_graphs_list[i]
                    assert (n == len(col_graphs))
                if norm_unnorm == 'norm':
                    rtn[i][j] = normalized_dist_sim(
                        rtn[i][j], row_graphs[i], col_graphs[j])
                else:
                    assert (norm_unnorm == 'unnorm')
                    rtn[i][j] = unnormalized_dist_sim(
                        rtn[i][j], row_graphs[i], col_graphs[j])
        return rtn

    def _load_result_mat(self, metric, model, m, n):
        file_p = get_result_path() + '/{}/{}/{}_{}_mat_{}_{}_*.npy'.format(
            self.dataset, metric, self.ds_metric, metric, self.dataset,
            model)
        li = glob(file_p)
        if not li:
            if 'astar' in model:
                if self.dataset not in ['imdbmulti', 'webeasy', 'linux_imdb', 'nci109', 'ptc', 'mutag']:
                    raise RuntimeError('Not imdbmulti/webeasy/linux_imdb/... and no astar results in {}!'.format(file_p))
                return self._load_merged_astar_from_other_three(metric, m, n)
            else:
                raise RuntimeError('No results found {}'.format(file_p))
        file = self._choose_result_file(li, m, n)
        return np.load(file)

    def _load_merged_astar_from_other_three(self, metric, m, n):
        self.model_ = 'merged_astar'
        if self.dataset in ['imdbmulti', 'linux_imdb', 'ptc', 'mutag']:
            merge_models = ['beam', 'hungarian', 'vj']
        else:
            assert (self.dataset in ['webeasy', 'nci109'])
            merge_models = ['beam', 'hungarian', 'vj']
        dms = [self._load_result_mat(self.ds_metric, model, m, n)
               for model in merge_models]
        tms = [None for model in merge_models]
        count = [0] * len(merge_models)
        for i in range(len(merge_models)):
            assert (dms[i].shape == (m, n))
        rtn = np.zeros((m, n))
        for i in range(m):
            for j in range(n):
                d_li = [dm[i][j] for dm in dms]
                d_argmin = int(np.argmin(d_li))
                count[d_argmin] += 1
                if metric == self.ds_metric:
                    rtn[i][j] = np.min(d_li)
                else:
                    assert (metric == 'time')
        print('Merge breakdown: {} {}'.format(merge_models, count))
        return rtn

    def _choose_result_file(self, files, m, n):
        cands = []
        shapes = []
        for file in files:
            temp = np.load(file)
            shapes.append(temp.shape)
            if temp.shape == (m, n):
                cands.append(file)
                if 'qilin' in file:
                    # print(file)
                    return file
        if cands:
            return cands[0]
        raise RuntimeError('No result files in {}; requires {} by {} but shapes: {}'.format(
            files, m, n, shapes))  # TODO: smart choice and cross-checking


def load_results_as_dict(dataset, models, row_graphs, col_graphs, col_graphs_list,
                         ds_mat, ds_mat_normed, sim_or_dist, ds_metric,
                         time_mat):
    rtn = {}
    for model in models:
        rtn[model] = load_result(dataset, model, row_graphs, col_graphs, col_graphs_list,
                                 ds_mat, ds_mat_normed, sim_or_dist, ds_metric,
                                 time_mat)
    return rtn


def load_result(dataset, model, row_graphs, col_graphs, col_graphs_list,
                ds_mat, ds_mat_normed, sim_or_dist, ds_metric, time_mat):
    return Result(dataset, model, row_graphs, col_graphs, col_graphs_list,
                  ds_mat, ds_mat_normed, sim_or_dist, ds_metric, time_mat)

# if __name__ == '__main__':
#     Graph2VecResult('aids10k', 'graph2vec', sim='dot')
