from config import FLAGS
from utils_siamese import get_flags
from utils import load_data
from exp import plot_preck, plot_single_number_metric, draw_emb_hist_heat, draw_ranking
from dist_sim_calculator import get_gs_ds_mat
from results import load_results_as_dict, load_result
from dist_sim_kernel import create_ds_kernel
from dist_sim import unnormalized_dist_sim
import numpy as np
from collections import OrderedDict
from os.path import join


class Eval(object):
    def __init__(self, data, dist_sim_calculator, need_val=True):
        self.test_row_gs = load_data(FLAGS.dataset_val_test, train=False).graphs
        self.test_col_gs = load_data(FLAGS.dataset_val_test, train=True).graphs
        self.baseline_models = []
        self.baseline_results_dict = load_results_as_dict(
            FLAGS.dataset_val_test, self.baseline_models,
            self.test_row_gs, self.test_col_gs, col_graphs_list=None,
            ds_mat=None, ds_mat_normed=False,  # load its own thing from disk
            sim_or_dist=self._ds_metric_is_dist_or_sim(),
            ds_metric=FLAGS.ds_metric, time_mat=None)
        if need_val:
            val_gs1, val_gs2 = self.get_val_gs_as_tuple(data)
            self.val_row_gs = self._to_nxgraph_list(val_gs1)
            self.val_col_gs = self._to_nxgraph_list(val_gs2)
            true_val_ds_mat = self._get_true_dist_sim_mat_for_val(
                data, dist_sim_calculator)
            self.true_val_result = load_result(
                FLAGS.dataset_val_test, FLAGS.ds_algo,
                self.val_row_gs, self.val_col_gs, col_graphs_list=None,
                ds_mat=true_val_ds_mat, ds_mat_normed=False,  # provide true ds mat
                sim_or_dist=self._ds_metric_is_dist_or_sim(),
                ds_metric=FLAGS.ds_metric, time_mat=None)
        self.true_test_result = load_result(
            FLAGS.dataset_val_test, FLAGS.ds_algo,
            self.test_row_gs, self.test_col_gs, col_graphs_list=None,
            ds_mat=None, ds_mat_normed=False,  # load its own thing from disk
            sim_or_dist=self._ds_metric_is_dist_or_sim(),
            ds_metric=FLAGS.ds_metric, time_mat=None)
        self.norms = [FLAGS.ds_norm]

    def get_val_gs_as_tuple(self, data):
        return data.val_gs, data.train_gs

    def get_test_gs_as_tuple(self, data):
        return data.test_gs, data.train_gs + data.val_gs

    def get_true_dist_sim(self, i, j, val_or_test, model):
        if val_or_test == 'val':
            r = self.true_val_result
        else:
            assert (val_or_test == 'test')
            r = self.true_test_result
        return model.get_true_dist_sim(i, j, r)

    def eval_for_val(self, ds_mat, loss_list, time_list, metrics):
        assert (ds_mat is not None)
        models = [FLAGS.model]
        pred_r = load_result(
            FLAGS.dataset_val_test, FLAGS.model,
            self.val_row_gs, self.val_col_gs, col_graphs_list=None,
            ds_mat=ds_mat, ds_mat_normed=FLAGS.ds_norm,  # provide pred ds mat
            sim_or_dist=FLAGS.pred_sim_dist,
            ds_metric=FLAGS.ds_metric, time_mat=time_list)
        rs = {FLAGS.model: pred_r, FLAGS.ds_algo: self.true_val_result}
        results = self._eval(models, rs, self.true_val_result,
                             metrics, False, loss_list=loss_list)
        rtn = OrderedDict()
        li = []
        for metric, num in results.items():
            if not 'loss' in metric:
                num = num[FLAGS.model]
                results[metric] = num
            metric = 'val_' + self._remove_norm_from_str(metric)
            rtn[metric] = num
            s = '{}={:.5f}'.format(metric, num)
            li.append(s)
        return rtn, ' '.join(li)

    def eval_for_test(self, ds_mat, metrics, saver, loss_list=None, time_list=None,
                      node_embs_dict=None, graph_embs_mat=None, attentions=None,
                      model=None, data=None, slt_collec=None):
        assert (ds_mat is not None)
        models = []
        extra_dir = saver.get_log_dir()
        assert (slt_collec is None)
        row_gs, col_gs, col_graphs_list, models, rs, true_r = \
            self._prepare_regular_results()
        models += [FLAGS.model]
        pred_r = load_result(
            FLAGS.dataset_val_test, FLAGS.model,
            row_gs, col_gs, col_graphs_list=col_graphs_list,
            ds_mat=ds_mat, ds_mat_normed=FLAGS.ds_norm,  # provide pred ds mat
            sim_or_dist=FLAGS.pred_sim_dist,
            ds_metric=FLAGS.ds_metric, time_mat=time_list)
        rs.update({FLAGS.model: pred_r})
        return self._eval(models, rs, true_r,
                          metrics, FLAGS.plot_results, loss_list,
                          node_embs_dict, graph_embs_mat, attentions,
                          extra_dir, model, data)

    def _prepare_slt_results(self, slt_collec, time_list, extra_dir):
        assert (slt_collec is not None)
        row_gs = self._to_nxgraph_list(slt_collec.row_gs)
        col_gs = None
        col_graphs_list = [self._to_nxgraph_list(li) for li in slt_collec.col_gs_list]
        extra_dir = join(extra_dir, slt_collec.short_name)
        true_r = load_result(
            FLAGS.dataset_val_test, 'decoy_true_result',
            row_gs, col_gs, col_graphs_list=col_graphs_list,
            ds_mat=slt_collec.true_ds_mat, ds_mat_normed=FLAGS.ds_norm,
            sim_or_dist=FLAGS.pred_sim_dist,
            ds_metric=FLAGS.ds_metric, time_mat=time_list)
        rs = {FLAGS.ds_algo: true_r}
        return row_gs, col_gs, col_graphs_list, extra_dir, rs, true_r

    def _prepare_regular_results(self):
        row_gs = self.test_row_gs
        col_gs = self.test_col_gs
        col_graphs_list = None
        models = ([FLAGS.ds_algo] + self.baseline_models)
        true_r = self.true_test_result
        rs = {FLAGS.ds_algo: true_r}
        if FLAGS.plot_results:
            rs.update(self.baseline_results_dict)
        return row_gs, col_gs, col_graphs_list, models, rs, true_r

    def _eval(self, models, rs, true_r, metrics, plot, loss_list=None,
              node_embs_dict=None, graph_embs_mat=None, attentions=None,
              extra_dir=None, model=None, data=None):
        rtn = OrderedDict()
        for metric in metrics:
            if metric == 'mrr' or metric == 'mse' or metric == 'dev' \
                    or metric == 'time' or 'acc' in metric \
                    or metric == 'kendalls_tau' or metric == 'spearmans_rho':
                d = plot_single_number_metric(
                    FLAGS.dataset_val_test, FLAGS.ds_metric, models, rs, true_r, metric,
                    self.norms,
                    ds_kernel=create_ds_kernel(get_flags('ds_kernel'),
                                               yeta=get_flags('yeta'),
                                               scale=get_flags('scale'))
                    if get_flags('ds_kernel') else None,
                    thresh_poss=[get_flags('thresh_val_test_pos')],
                    thresh_negs=[get_flags('thresh_val_test_neg')],
                    thresh_poss_sim=[0.5],
                    thresh_negs_sim=[0.5],
                    plot_results=plot, extra_dir=extra_dir)
                rtn.update(d)
            elif metric == 'draw_heat_hist':
                if node_embs_dict is not None:
                    draw_emb_hist_heat(
                        FLAGS.dataset_val_test,
                        node_embs_dict,
                        true_result=true_r,
                        ds_norm=FLAGS.ds_norm,
                        extra_dir=extra_dir + '/mne',
                        plot_max_num=FLAGS.plot_max_num)
            elif metric == 'ranking':
                em = self._get_node_mappings(data)
                draw_ranking(
                    FLAGS.dataset_val_test, FLAGS.ds_metric, true_r, rs[FLAGS.model],
                    node_feat_name=FLAGS.node_feat_name,
                    model_name=FLAGS.model_name,
                    plot_node_ids=FLAGS.dataset_val_test != 'webeasy' and em and not FLAGS.supersource,
                    plot_gids=False,
                    ds_norm=FLAGS.ds_norm,
                    existing_mappings=em,
                    extra_dir=extra_dir + '/ranking',
                    plot_max_num=FLAGS.plot_max_num)
            elif metric == 'attention':
                if attentions is not None:
                    draw_attention(
                        FLAGS.dataset_val_test, true_r, attentions,
                        extra_dir=extra_dir + '/attention',
                        plot_max_num=FLAGS.plot_max_num)
            elif 'prec@k' in metric:
                d = plot_preck(
                    FLAGS.dataset_val_test, FLAGS.ds_metric, models, rs, true_r, metric,
                    self.norms, plot, extra_dir=extra_dir)
                rtn.update(d)
            elif metric == 'loss':
                rtn.update({metric: np.mean(loss_list)})
            elif metric == 'train_pair_ged_vis':
                if FLAGS.ds_metric == 'ged':
                    plot_dist_hist(FLAGS.ds_metric, [FLAGS.dataset_val_test],
                                   [self._transform_train_pairs_to_dist_list(model)],
                                   extra_dir)
            elif metric == 'graph_classification':
                if graph_embs_mat is not None:
                    results_dict = graph_classification(
                        FLAGS.dataset_val_test, graph_embs_mat,
                        FLAGS.ds_metric,
                        extra_dir=extra_dir)
                    rtn.update(results_dict)
            else:
                raise RuntimeError('Unknown metric {}'.format(metric))
        return rtn

    def _get_true_dist_sim_mat_for_val(self, data, dist_sim_calculator):
        gs1, gs2 = self.get_val_gs_as_tuple(data)
        gs1 = self._to_nxgraph_list(gs1)
        gs2 = self._to_nxgraph_list(gs2)
        return get_gs_ds_mat(gs1, gs2, dist_sim_calculator, 'val', 'train',
                             FLAGS.dataset_val_test, FLAGS.ds_metric,
                             FLAGS.ds_algo, norm=False,  # load the raw dist/sim
                             dec_gsize=FLAGS.supersource)

    def _to_nxgraph_list(self, gs):
        return [g.nxgraph for g in gs]

    def _remove_norm_from_str(self, s):
        return s.replace('_norm', '').replace('_nonorm', '')

    def _transform_train_pairs_to_dist_list(self, model):
        mn = model.__class__.__name__
        assert (mn == 'SiameseRegressionModel')
        rtn = []
        for g1, g2, s_or_d in model.train_triples.li:
            if FLAGS.supply_sim_dist == 'sim':
                d = model.ds_kernel.sim_to_dist_np(s_or_d)
            else:
                d = s_or_d
            if FLAGS.ds_norm:
                orig_d = d
                g1nx, g2nx = g1.nxgraph, g2.nxgraph
                g1size, g2size = g1nx.number_of_nodes(), g2nx.number_of_nodes()
                if FLAGS.supersource:  # supersource changes #nodes, NOT ged/mcs
                    g1size -= 1
                    g2size -= 1
                d = unnormalized_dist_sim(
                    d, g1nx, g2nx, dec_gsize=FLAGS.supersource)
                if not abs(d - round(d)) < 1e-6:
                    raise RuntimeError(
                        'Wrong train pair: g1 gid {}, g2 gid {}, '
                        'g1 size {}, g2 size {}, '
                        'actual g1 size {}, actual g2 size {}, '
                        'd {}, round(d) {}, '
                        'orig_d {} '.format(
                            g1nx.graph['gid'], g2nx.graph['gid'],
                            g1nx.number_of_nodes(),
                            g2nx.number_of_nodes(),
                            g1size,
                            g2size,
                            d, round(d), orig_d))
                d = round(d)
            if d == np.inf:
                d = 999  # TODO: hacky; need to deal with ds_norm=False --> exp(big)=0
            else:
                d = int(d)
            rtn.append(d)
        return rtn

    def _get_node_mappings(self, data):
        # [train + val ... test]
        if data and hasattr(data.train_gs[0], 'mapping'):
            return [g.mapping for g in data.train_gs + data.val_gs + data.test_gs]
        else:
            return None

    def _ds_metric_is_dist_or_sim(self):
        if FLAGS.ds_metric == 'ged':
            return 'dist'
        else:
            assert FLAGS.ds_metric == 'mcs'
            return 'sim'
