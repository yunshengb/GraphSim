import numpy as np
from scipy.stats import hmean, kendalltau, spearmanr


class Metric(object):
    def __init__(self, name, ylabel):
        self.name = name
        self.ylabel = ylabel

    def __str__(self):
        return self.name


def prec_at_ks(true_r, pred_r, norm, ks, rm, print_ids=[]):
    """
    Ranking-based. prec@ks.
    :param true_r: result object indicating the ground truth.
    :param pred_r: result object indicating the prediction.
    :param norm: whether to normalize the results or not.
    :param ks:
    :param print_ids: (optional) The ids of the query to print for debugging.
    :return: list of floats indicating the average precision at different ks.
    """
    m, n = true_r.m_n()
    assert (true_r.m_n() == pred_r.m_n())
    ps = np.zeros((m, len(ks)))
    for i in range(m):
        for k_idx, k in enumerate(ks):
            assert (type(k) is int and k > 0 and k < n)
            true_ids = true_r.top_k_ids(i, k, norm, inclusive=True, rm=rm)
            pred_ids = pred_r.top_k_ids(i, k, norm, inclusive=True, rm=rm)
            ps[i][k_idx] = \
                min(len(set(true_ids).intersection(set(pred_ids))), k) / k
        if i in print_ids:
            print('query {}\nks:    {}\nprecs: {}'.format(i, ks, ps[i]))
    return np.mean(ps, axis=0)


def mean_reciprocal_rank(true_r, pred_r, norm, print_ids=[]):
    """
    Ranking based. MRR.
    :param true_r: result object indicating the ground truth.
    :param pred_r: result object indicating the prediction.
    :param norm: whether to normalize the results or not.
    :param print_ids: (optional) the ids of the query to print for debugging.
    :return: float indicating the mean reciprocal rank.
    """
    m, n = true_r.m_n()
    assert (true_r.m_n() == pred_r.m_n())
    topanswer_ranks = np.zeros(m)
    for i in range(m):
        # There may be multiple graphs with the same dist/sim scores
        # as the top answer by the true_r model.
        # Select one with the lowest (minimum) rank
        # predicted by the pred_r model for mrr calculation.
        true_ids = true_r.top_k_ids(i, 1, norm, inclusive=True, rm=0)
        assert (len(true_ids) >= 1)
        min_rank = float('inf')
        for true_id in true_ids:
            pred_rank = pred_r.ranking(i, true_id, norm, one_based=True)
            min_rank = min(min_rank, pred_rank)
        topanswer_ranks[i] = min_rank
        if i in print_ids:
            print('query {}\nrank: {}'.format(i, min_rank))
    return 1.0 / hmean(topanswer_ranks)


def mean_squared_error(true_r, pred_r, ds_kernel, norm):
    """
    Regression-based. L2 difference between the ground-truth sim/dist
        and the predicted sim/dist.
    :return:
    """
    assert (true_r.m_n() == pred_r.m_n())
    A = true_r.dist_sim_mat(norm)
    A = _proc_true_ds_mat(A, true_r, pred_r, ds_kernel)
    B = pred_r.dist_sim_mat(norm)
    return ((A - B) ** 2).mean() / 2


def mean_deviation(true_r, pred_r, ds_kernel, norm):
    """
    Regression-based. L1 difference between the ground-truth sim/dist
        and the predicted sim/dist.
    :return:
    """
    assert (true_r.m_n() == pred_r.m_n())
    A = true_r.dist_sim_mat(norm)
    A = _proc_true_ds_mat(A, true_r, pred_r, ds_kernel)
    B = pred_r.dist_sim_mat(norm)
    return np.abs(A - B).mean()


def average_time(r):
    return np.mean(r.get_time_mat())


def accuracy(true_r, pred_r, thresh_pos, thresh_neg,
             thresh_pos_sim, thresh_neg_sim, norm):
    if not (thresh_pos >= 0 and thresh_neg >= 0 and thresh_pos <= thresh_neg):
        raise RuntimeError('Invalid thresholds {} and {}'.format(
            thresh_pos, thresh_neg))
    m, n = true_r.m_n()
    assert (true_r.m_n() == pred_r.m_n())
    if true_r.dist_or_sim() != 'dist':
        raise RuntimeError('The true result must be distance based')
    true_label_mat, true_num_poses, true_num_negs = \
        true_r.classification_mat(thresh_pos, thresh_neg,
                                  thresh_pos_sim, thresh_neg_sim, norm)
    pred_label_mat, pred_num_poses, pred_num_negs = \
        pred_r.classification_mat(thresh_pos, thresh_neg,
                                  thresh_pos_sim, thresh_neg_sim, norm)
    tps = 0
    tns = 0
    for i in range(m):
        for j in range(n):
            true_label = true_label_mat[i][j]
            pred_label = pred_label_mat[i][j]
            if true_label == 1:
                if pred_label == 1:
                    tps += 1
                else:
                    pass  # TODO: fair for pairwise baselines because can be 0?
            elif true_label == 0:  # do not care about the pair
                pass
            elif true_label == -1:
                if pred_label == -1:
                    tns += 1
            else:
                assert (False)
    pos_acc = tps / true_num_poses
    neg_acc = tns / true_num_negs
    acc = (tps + tns) / (true_num_poses + true_num_negs)
    return pos_acc, neg_acc, acc


def kendalls_tau(true_r, pred_r, norm):
    return _ranking_metric(kendalltau, true_r, pred_r, norm)


def spearmans_rho(true_r, pred_r, norm):
    return _ranking_metric(spearmanr, true_r, pred_r, norm)


def _ranking_metric(ranking_metric_func, true_r, pred_r, norm):
    y_true = true_r.ranking_mat(norm=norm)
    y_scores = pred_r.ranking_mat(norm=norm)
    scores = []
    m, n = true_r.m_n()
    assert (true_r.m_n() == pred_r.m_n())
    for i in range(m):
        scores.append(ranking_metric_func(y_true[i], y_scores[i])[0])
    return np.mean(scores)


def _proc_true_ds_mat(A, true_r, pred_r, ds_kernel):
    if true_r.dist_or_sim() != pred_r.dist_or_sim():
        if pred_r.dist_or_sim() == 'sim':
            assert (true_r.get_ds_metric() == 'ged')
            A = ds_kernel.dist_to_sim_np(A)
        else:
            assert (true_r.get_ds_metric() == 'mcs')
            A = ds_kernel.sim_to_dist_np(A)
    return A

if __name__ == '__main__':
    y_true = np.array([1, 2, 3, 4, 5])
    y_scores = np.array([1, 1, 1, 1, 1])
    kendalltau = kendalltau(y_true, y_scores)
    print(kendalltau)
