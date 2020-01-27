#!/usr/bin/env python3
from utils import get_result_path, create_dir_if_not_exists, \
    load_data, get_ts, exec_turnoff_print, prompt, prompt_get_computer_name, \
    check_nx_version, prompt_get_cpu, format_float, get_norm_str, load_as_dict, \
    node_has_type_attrib
from metrics import Metric, prec_at_ks, mean_reciprocal_rank, \
    mean_squared_error, mean_deviation, average_time, accuracy, \
    kendalls_tau, spearmans_rho
from dist_sim import ged, mcs
from results import load_results_as_dict, load_result
from dist_sim_calculator import load_from_exsiting_csv
from classification import get_classification_labels_from_dist_mat
from node_ordering import reorder_nodes
import networkx as nx
import traceback
from os.path import dirname

check_nx_version()
import multiprocessing as mp
import numpy as np
import matplotlib
import matplotlib.colors as mcolors

# Fix font type for ACM paper submission.
matplotlib.use('Agg')
matplotlib.rc('font', **{'family': 'serif', 'size': 22})
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
from vis import vis, vis_small
import seaborn as sns
from collections import defaultdict
from warnings import warn
import random

BASELINE_MODELS = ['beam', 'hungarian', 'vj']
TRUE_MODEL = 'astar'

""" Plotting. """
args1 = {'astar': {'color': 'grey'},
         'beam': {'color': 'yellowgreen'},
         'hungarian': {'color': 'deepskyblue'},
         'vj': {'color': 'darkcyan'},
         'hed': {'color': 'lightskyblue'},
         'siamese': {'color': 'red'},
         'mccreesh2017': {'color': 'lightcoral'}}
args2 = {'astar': {'marker': '*', 'facecolors': 'none', 'edgecolors': 'grey'},
         'beam': {'marker': 'H', 'facecolors': 'none', 'edgecolors': 'yellowgreen'},
         'hungarian': {'marker': 'X', 'facecolors': 'none',
                       'edgecolors': 'deepskyblue'},
         'vj': {'marker': 'h', 'facecolors': 'none',
                'edgecolors': 'darkcyan'},
         'hed': {'marker': 'p', 'facecolors': 'none',
                 'edgecolors': 'lightskyblue'},
         'siamese': {'marker': 'P',
                     'facecolors': 'none', 'edgecolors': 'red'},
         'mccreesh2017': {'marker': 'd',
                          'facecolors': 'none', 'edgecolors': 'lightcoral'},
         }
TYPE_COLOR_MAP = {
    'C': '#ff6666',
    'O': 'lightskyblue',
    'N': 'yellowgreen',
    'movie': '#ff6666',
    'tvSeries': '#ff6666',
    'actor': 'lightskyblue',
    'actress': '#ffb3e6',
    'director': 'yellowgreen',
    'composer': '#c2c2f0',
    'producer': '#ffcc99',
    'cinematographer': 'gold'}
FAVORITE_COLORS = ['#ff6666', 'lightskyblue', 'yellowgreen', '#c2c2f0', 'gold',
                   '#ffb3e6', '#ffcc99', '#E0FFFF', '#7FFFD4', '#20B2AA',
                   '#FF8C00', '#ff1493',
                   '#FFE4B5', '#e6e6fa', '#7CFC00']
DATASET_ARGS = {
    'aids80nef': {
        'color': 'red',
        'marker': 'o',
        'label': 'AIDS'
    },
    'aids700nef': {
        'color': 'red',
        'marker': 'o',
        'label': 'AIDS'
    },
    'linux': {
        'color': 'blue',
        'marker': 'x',
        'label': 'LINUX'
    },
    'imdbmulti': {
        'color': 'green',
        'marker': '*',
        'label': 'IMDB'
    },
    'ptc': {
        'color': 'lightskyblue',
        'marker': 'o',
        'label': 'PTC'
    }
}


def exp1():
    """ Run baselines on real datasets. Take a while. """
    dataset = prompt('Which dataset?')
    row_train = prompt('Train or test for row graphs? (1/0)') == '1'
    row_graphs = load_data(dataset, train=row_train).graphs
    col_train = prompt('Train or test for col graphs? (1/0)') == '1'
    col_graphs = load_data(dataset, train=col_train).graphs
    ds_metric = prompt('Which metric (ged|mcs)?', options=['ged', 'mcs'])
    algo = prompt('Which algorthm?')
    timeout_temp = prompt('Time limit in min? Empty for no limit')
    timeout = float(timeout_temp) * 60 if timeout_temp else None
    exec_turnoff_print()
    num_cpu = prompt_get_cpu()
    computer_name = prompt_get_computer_name()
    try:
        real_dataset_run_helper(computer_name, dataset, ds_metric, algo, row_graphs, col_graphs,
                                num_cpu, timeout)
    except Exception as e:
        traceback.print_exc()


def real_dataset_run_helper(computer_name, dataset, ds_metric, algo, row_graphs, col_graphs,
                            num_cpu, timeout):
    if ds_metric == 'ged':
        func = ged
    elif ds_metric == 'mcs':
        func = mcs
        # For MCS, since the solver can handle labeled and unlabeled graphs, but the compressed
        # encoding must be labeled (need to tell it to ignore labels or not).
        # TODO: this should go in some kind of config file specific for mcs
        if node_has_type_attrib(row_graphs[0]):
            labeled = True
            label_key = 'type'
            print('Has node type')
        else:
            labeled = False
            label_key = ''
            print('Does not have node type')
    else:
        raise RuntimeError('Unknown distance similarity metric {}'.format(ds_metric))
    m = len(row_graphs)
    n = len(col_graphs)
    ds_mat = np.zeros((m, n))
    time_mat = np.zeros((m, n))
    outdir = '{}/{}'.format(get_result_path(), dataset)
    create_dir_if_not_exists(outdir + '/csv')
    create_dir_if_not_exists(outdir + '/{}'.format(ds_metric))
    create_dir_if_not_exists(outdir + '/time')
    exsiting_csv = prompt('File path to exsiting csv files?')
    exsiting_entries = load_from_exsiting_csv(exsiting_csv, ds_metric, skip_eval=False)
    is_symmetric = prompt('Is the ds matrix symmetric? (1/0)', options=['0', '1']) == '1'
    if is_symmetric:
        assert (m == n)
    smart_needed = prompt('Is smart pair sorting needed? (1/0)', options=['0', '1']) == '1'
    csv_fn = '{}/csv/{}_{}_{}_{}_{}_{}cpus.csv'.format(
        outdir, ds_metric, dataset, algo, get_ts(), computer_name, num_cpu)
    file = open(csv_fn, 'w')
    print('Saving to {}'.format(csv_fn))
    if ds_metric == 'ged':
        print_and_log('i,j,i_gid,j_gid,i_node,j_node,i_edge,j_edge,ged,lcnt,time(msec)',
                      file)
    else:
        print_and_log(
            'i,j,i_gid,j_gid,i_node,j_node,i_edge,j_edge,mcs,node_mapping,edge_mapping,time(msec)',
            file)
    # Multiprocessing.
    pool = mp.Pool(processes=num_cpu)
    # Submit to pool workers.
    results = {}
    pairs_to_run = get_all_pairs_to_run(row_graphs, col_graphs, smart_needed)
    for k, (i, j) in enumerate(pairs_to_run):
        g1, g2 = row_graphs[i], col_graphs[j]
        i_gid, j_gid = g1.graph['gid'], g2.graph['gid']
        if (i_gid, j_gid) in exsiting_entries:
            continue
        if is_symmetric and (j_gid, i_gid) in exsiting_entries:
            continue
        if ds_metric == 'mcs':
            results[(i, j)] = pool.apply_async(
                func, args=(g1, g2, algo, labeled, label_key, True, True, timeout,))
        else:
            results[(i, j)] = pool.apply_async(
                func, args=(g1, g2, algo, True, True, timeout,))
        print_progress(k, m, n, 'submit: {} {} {} {} cpus;'.
                       format(algo, dataset, computer_name, num_cpu))
    # Retrieve results from pool workers or a loaded csv file (previous run).
    for k, (i, j) in enumerate(pairs_to_run):
        print_progress(k, m, n, 'work: {} {} {} {} {} cpus;'.
                       format(ds_metric, algo, dataset, computer_name, num_cpu))
        g1, g2 = row_graphs[i], col_graphs[j]
        i_gid, j_gid = g1.graph['gid'], g2.graph['gid']
        if (i, j) not in results:
            lcnt, mcs_node_mapping, mcs_edge_mapping = None, None, None
            tmp = exsiting_entries.get((i_gid, j_gid))
            if tmp:
                if ds_metric == 'ged':
                    i_gid, j_gid, i_node, j_node, ds, lcnt, t = tmp
                else:
                    i_gid, j_gid, i_node, j_node, ds, mcs_node_mapping, mcs_edge_mapping, t = tmp
            else:
                assert (is_symmetric)
                get_from = exsiting_entries[(j_gid, i_gid)]
                if ds_metric == 'ged':
                    j_gid, i_gid, j_node, i_node, ds, lcnt, t = \
                        get_from
                else:
                    j_gid, i_gid, j_node, i_node, ds, mcs_node_mapping, mcs_edge_mapping, t = \
                        get_from
            if ds_metric == 'ged':
                assert (lcnt is not None)
                assert (g1.graph['gid'] == i_gid)
                assert (g2.graph['gid'] == j_gid)
                assert (g1.number_of_nodes() == i_node)
                assert (g2.number_of_nodes() == j_node)
                s = form_ged_print_string(i, j, g1, g2, ds, lcnt, t)
            else:
                assert (mcs_node_mapping is not None and
                        mcs_edge_mapping is not None)
                s = form_mcs_print_string(
                    i, j, g1, g2, ds, mcs_node_mapping, mcs_edge_mapping, t)
        else:
            if ds_metric == 'ged':
                ds, lcnt, g1_a, g2_a, t = results[(i, j)].get()
                i_gid, j_gid, i_node, j_node = \
                    g1.graph['gid'], g2.graph['gid'], \
                    g1.number_of_nodes(), g2.number_of_nodes()
                assert (g1.number_of_nodes() == g1_a.number_of_nodes())
                assert (g2.number_of_nodes() == g2_a.number_of_nodes())
                exsiting_entries[(i_gid, j_gid)] = \
                    (i_gid, j_gid, i_node, j_node, ds, lcnt, t)
                s = form_ged_print_string(i, j, g1, g2, ds, lcnt, t)
            else:  # MCS
                ds, mcs_node_mapping, mcs_edge_mapping, t = \
                    results[(i, j)].get()
                exsiting_entries[(i_gid, j_gid)] = \
                    (ds, mcs_node_mapping, mcs_edge_mapping, t)
                s = form_mcs_print_string(
                    i, j, g1, g2, ds, mcs_node_mapping, mcs_edge_mapping, t)
        print_and_log(s, file)
        if ds_metric == 'mcs' and (i_gid, j_gid) in exsiting_entries:
            # Save memory, clear the mappings since they're saved to file.
            exsiting_entries[(i_gid, j_gid)] = list(exsiting_entries[(i_gid, j_gid)])
            exsiting_entries[(i_gid, j_gid)][1] = {}
            exsiting_entries[(i_gid, j_gid)][2] = {}
        ds_mat[i][j] = ds
        time_mat[i][j] = t
    file.close()
    save_as_np(outdir, ds_metric, ds_mat, time_mat, get_ts(),
               dataset, row_graphs, col_graphs, algo, computer_name, num_cpu)


def get_all_pairs_to_run(row_gs, col_gs, smart_needed):
    m, n = len(row_gs), len(col_gs)
    rtn = []
    for i in range(m):
        for j in range(n):
            g1 = row_gs[i]
            g2 = col_gs[j]
            rtn.append((g1.number_of_nodes() + g2.number_of_nodes(), i, j))
    if smart_needed:
        rtn = sorted(rtn)
    for k in range(len(rtn)):
        _, i, j = rtn[k]
        # print(_, i, j)
        rtn[k] = (i, j)
    return rtn


def form_ged_print_string(i, j, g1, g2, ds, lcnt, t):
    return '{},{},{},{},{},{},{},{},{},{},{:.2f}'.format(
        i, j, g1.graph['gid'], g2.graph['gid'],
        g1.number_of_nodes(), g2.number_of_nodes(),
        g1.number_of_edges(), g2.number_of_edges(),
        ds, lcnt, t)


def form_mcs_print_string(i, j, g1, g2, ds, mcs_node_mapping, mcs_edge_mapping, t):
    return '"{}","{}","{}","{}","{}","{}","{}","{}","{}","{}","{}","{:.2f}"'.format(
        i, j, g1.graph['gid'], g2.graph['gid'],
        g1.number_of_nodes(), g2.number_of_nodes(),
        g1.number_of_edges(), g2.number_of_edges(),
        ds, mcs_node_mapping, mcs_edge_mapping, t)


def post_real_dataset_run_convert_csv_to_np():
    """ Use in case only csv is generated,
        and numpy matrices need to be saved. """
    dataset = 'imdbmulti'
    model = 'CDKMCS'
    ds_metric = 'mcs'
    row_graphs = load_data(dataset, False).graphs
    col_graphs = load_data(dataset, True).graphs
    num_cpu = 40
    computer_name = 'scai1_all'
    ts = '2018-10-09T13:41:13.942414'
    outdir = '{}/{}'.format(get_result_path(), dataset)
    csv_fn = '{}/csv/{}_{}_{}_{}_{}_{}cpus.csv'.format(
        outdir, ds_metric, dataset, model, ts, computer_name, num_cpu)
    data = load_from_exsiting_csv(csv_fn, ds_metric)
    m = len(row_graphs)
    n = len(col_graphs)
    # -3 is identifier that the csv the data came from didn't include the data point.
    ds_mat = np.full((m, n), -3)
    time_mat = np.full((m, n), -3)
    cnt = 0
    print('m: {}, n: {}, m*n: {}'.format(m, n, m * n))
    for (i, j), row_data in data.items():
        if cnt % 1000 == 0:
            print(cnt)
        ds_mat[i][j] = row_data[4]
        time_mat[i][j] = row_data[6] if ds_metric == 'ged' else row_data[7]
        cnt += 1
    print(cnt)
    assert (cnt == m * n)
    save_as_np(outdir, ds_metric, ds_mat, time_mat, ts,
               dataset, row_graphs, col_graphs, model, computer_name, num_cpu)


def save_as_np(outdir, ds_metric, ds_mat, time_mat, ts,
               dataset, row_graphs, col_graphs, model, computer_name, num_cpu):
    s = '{}_{}_{}_{}_{}cpus'.format(
        dataset,
        model, ts, computer_name, num_cpu)
    np.save('{}/{}/{}_{}_mat_{}'.format(
        outdir, ds_metric, ds_metric, ds_metric, s), ds_mat)
    np.save('{}/time/{}_time_mat_{}'.format(outdir, ds_metric, s), time_mat)


def print_progress(cur, m, n, label):
    tot = m * n
    print('----- {} progress: {}/{}={:.1%}'.format(label, cur, tot, cur / tot))


def print_and_log(s, file):
    print(s)
    file.write(s + '\n')
    # file.flush() # less disk I/O (hopefully)


def exp2():
    """ Plot ged and time. """
    dataset = 'aids50'
    models = BASELINE_MODELS
    rs = load_results_as_dict(
        dataset, models,
        row_graphs=load_data(dataset, train=False).graphs,
        col_graphs=load_data(dataset, train=True).graphs)
    metrics = [Metric('ged', 'ged'), Metric('time', 'time (msec)')]
    for metric in metrics:
        plot_ged_time_helper(dataset, models, metric, rs)


def plot_ged_time_helper(dataset, models, metric, rs):
    font = {'family': 'serif',
            'size': 22}
    matplotlib.rc('font', **font)

    plt.figure(0)
    plt.figure(figsize=(16, 10))

    xs = get_test_graph_sizes(dataset)
    so = np.argsort(xs)
    xs.sort()
    for model in models:
        mat = rs[model].mat(metric.name, norm=True)
        print('plotting for {}'.format(model))
        ys = np.mean(mat, 1)[so]
        plt.plot(xs, ys, **get_plotting_arg(args1, model))
        plt.scatter(xs, ys, s=200, label=model, **get_plotting_arg(args2, model))
    plt.xlabel('query graph size')
    ax = plt.gca()
    ax.set_xticks(xs)
    plt.ylabel('average {}'.format(metric.ylabel))
    plt.legend(loc='best', ncol=2)
    plt.grid(linestyle='dashed')
    plt.tight_layout()
    # plt.show()
    sp = get_result_path() + '/{}/{}/ged_{}_mat_{}_{}.png'.format( \
        dataset, metric, metric, dataset, '_'.join(models))
    plt.savefig(sp)
    print('Saved to {}'.format(sp))


def get_plotting_arg(args, model):
    for k, v in args.items():
        if k == model:
            return v
    for k, v in args.items():
        if k in model:
            return v
    raise RuntimeError('Unknown model {} in args {}'.format(model, args))


def get_test_graph_sizes(dataset):
    test_data = load_data(dataset, train=False)
    return [g.number_of_nodes() for g in test_data.graphs]


def exp3():
    dataset = 'imdbmulti'
    additional_models = ['graphlet']
    if dataset == 'imdbmulti':
        baseline_models = ['hungarian', 'vj', 'beam', 'hed']
    else:
        baseline_models = BASELINE_MODELS
    models = additional_models + baseline_models + [TRUE_MODEL]
    metric = 'prec@k'
    dsmetric = 'ged'
    norms = [True, False]
    row_graphs = load_data(dataset, train=False).graphs
    col_graphs = load_data(dataset, train=True).graphs
    rs = load_results_as_dict(
        dataset, models, row_graphs=row_graphs, col_graphs=col_graphs)
    # from utils import load_as_dict
    # d = load_as_dict(
    # pred_r = load_result(dataset, 'siamese_test', sim_mat=d['sim_mat'], time_mat=[],
    #                      row_graphs=row_graphs, col_graphs=col_graphs)
    # models += ['siamese_test']
    # rs['siamese_test'] = pred_r
    true_result = rs[TRUE_MODEL]
    plot_preck(dataset, dsmetric, models, rs, true_result, metric, norms)


def plot_preck(dataset, dsmetric, models, rs, true_result, metric, norms,
               plot_results=True, extra_dir=None):
    """ Plot prec@k. """
    create_dir_if_not_exists('{}/{}/{}'.format(
        get_result_path(), dataset, metric))
    rtn = {}
    for norm in norms:
        _, n = true_result.m_n()
        ks = range(1, n)
        d = plot_preck_helper(
            dataset, dsmetric, models, rs, true_result, metric, norm, ks,
            False, plot_results, extra_dir)
        rtn.update(d)
    return rtn


def plot_preck_helper(dataset, dsmetric, models, rs, true_result, metric, norm, ks,
                      logscale, plot_results, extra_dir):
    print_ids = []
    numbers = {}
    assert (metric[0:6] == 'prec@k')
    if len(metric) > 6:
        rm = float(metric.split('_')[1])
    else:
        rm = 0
    for model in models:
        precs = prec_at_ks(true_result, rs[model], norm, ks, rm, print_ids)
        numbers[model] = {'ks': ks, 'precs': precs}
    rtn = {'preck{}_{}'.format(get_norm_str(norm), rm): numbers}
    if not plot_results:
        return rtn
    plt.figure(figsize=(16, 10))
    for model in models:
        ks = numbers[model]['ks']
        inters = numbers[model]['precs']
        if logscale:
            pltfunc = plt.semilogx
        else:
            pltfunc = plt.plot
        pltfunc(ks, inters, **get_plotting_arg(args1, model))
        plt.scatter(ks, inters, s=200, label=shorten_name(model),
                    **get_plotting_arg(args2, model))
    plt.xlabel('k')
    # ax = plt.gca()
    # ax.set_xticks(ks)
    plt.ylabel(metric)
    plt.ylim([-0.06, 1.06])
    plt.legend(loc='best', ncol=2)
    plt.grid(linestyle='dashed')
    plt.tight_layout()
    # plt.show()
    kss = 'k_{}_{}'.format(min(ks), max(ks))
    bfn = '{}_{}_{}_{}_{}{}_{}'.format(
        dsmetric, metric, dataset, '_'.join(models), kss, get_norm_str(norm), rm)
    dir = '{}/{}/{}'.format(get_result_path(), dataset, metric)
    save_fig(plt, dir, bfn)
    if extra_dir:
        save_fig(plt, extra_dir, bfn)
    print(metric, 'plotted')
    return rtn


def create_siamese_result_from_test_info_pickle(fp, dataset, row_gs, col_gs):
    name = 'siamese_test'
    d = load_as_dict(fp)
    return name, load_result(dataset, name, sim_mat=d['sim_mat'],
                             row_graphs=row_gs, col_graphs=col_gs,
                             time_mat=[])


def plot_single_number_metric(dataset, dsmetric, models, rs, true_result, metric, norms,
                              ds_kernel=None,
                              thresh_poss=None, thresh_negs=None,
                              thresh_poss_sim=None, thresh_negs_sim=None,
                              plot_results=True,
                              extra_dir=None):
    """ Plot mrr or mse. """
    create_dir_if_not_exists('{}/{}/{}'.format(
        get_result_path(), dataset, metric))
    rtn = {}
    if norms and thresh_poss and thresh_negs:
        assert (len(norms) == len(thresh_poss) == len(thresh_negs))
    for i, norm in enumerate(norms):
        thresh_pos = thresh_poss[i] if thresh_poss else None
        thresh_neg = thresh_negs[i] if thresh_negs else None
        thresh_pos_sim = thresh_poss_sim[i] if thresh_poss_sim else None
        thresh_neg_sim = thresh_negs_sim[i] if thresh_negs_sim else None
        d = plot_single_number_metric_helper(
            dataset, dsmetric, models, rs, true_result, metric, norm, ds_kernel,
            thresh_pos, thresh_neg, thresh_pos_sim, thresh_neg_sim,
            plot_results, extra_dir)
        rtn.update(d)
    return rtn


def plot_single_number_metric_helper(dataset, dsmetric, models, rs, true_result,
                                     metric, norm,
                                     ds_kernel, thresh_pos, thresh_neg,
                                     thresh_pos_sim, thresh_neg_sim,
                                     plot_results, extra_dir):
    # dsmetric: distance/similarity metric, e.g. ged, mcs, etc.
    # metric: eval metric.
    print_ids = []
    rtn = {}
    val_list = []
    for model in models:
        if metric == 'mrr':
            val = mean_reciprocal_rank(
                true_result, rs[model], norm, print_ids)
        elif metric == 'mse':
            val = mean_squared_error(
                true_result, rs[model], ds_kernel, norm)
        elif metric == 'dev':
            val = mean_deviation(
                true_result, rs[model], ds_kernel, norm)
        elif metric == 'time':
            val = average_time(rs[model])
        elif 'acc' in metric:
            val = accuracy(
                true_result, rs[model], thresh_pos, thresh_neg,
                thresh_pos_sim, thresh_neg_sim, norm)
            pos_acc, neg_acc, acc = val
            if metric == 'pos_acc':
                val = pos_acc
            elif metric == 'neg_acc':
                val = neg_acc
            elif metric == 'acc':
                val = acc  # only the overall acc
            else:
                assert (metric == 'accall')
        elif metric == 'kendalls_tau':
            val = kendalls_tau(true_result, rs[model], norm)
        elif metric == 'spearmans_rho':
            val = spearmans_rho(true_result, rs[model], norm)
        else:
            raise RuntimeError('Unknown {}'.format(metric))
        # print('{} {}: {}'.format(metric, model, mrr_mse_time))
        rtn[model] = val
        val_list.append(val)
    rtn = {'{}{}'.format(metric, get_norm_str(norm)): rtn}
    if not plot_results:
        return rtn
    plt = plot_multiple_bars(val_list, models, metric)
    if metric == 'time':
        ylabel = 'time (msec)'
        norm = None
    elif metric == 'pos_acc':
        ylabel = 'pos_recall'
    elif metric == 'neg_acc':
        ylabel = 'neg_recall'
    elif metric == 'kendalls_tau':
        ylabel = 'Kendall\'s $\\tau$'
    elif metric == 'spearmans_rho':
        ylabel = 'Spearman\'s $\\rho$'
    else:
        ylabel = metric
    plt.ylabel(ylabel)
    if metric == 'time':
        plt.yscale('log')
    metric_addi_info = ''
    bfn = '{}_{}{}_{}_{}{}'.format(
        dsmetric, metric, metric_addi_info,
        dataset, '_'.join(models),
        get_norm_str(norm))
    sp = get_result_path() + '/{}/{}/'.format(dataset, metric)
    save_fig(plt, sp, bfn)
    if extra_dir:
        save_fig(plt, extra_dir, bfn)
    print(metric, 'plotted')
    return rtn


def plot_multiple_bars(val_list, xlabels, metric):
    plt.figure(figsize=(16, 10))
    ind = np.arange(len(val_list))  # the x locations for the groups
    val_lists = proc_val_list(val_list)
    width = 0.35  # the width of the bars
    if len(val_lists) > 1:
        width = width - 0.02 * len(val_lists)
    for i, val_list in enumerate(val_lists):
        bars = plt.bar(ind + i * width, val_list, width)
        for i, bar in enumerate(bars):
            bar.set_color(get_plotting_arg(args1, xlabels[i])['color'])
        autolabel(bars, metric)
    plt.xlabel('model')
    plt.xticks(ind + (width / 2) * (len(val_lists) - 1), shorten_names(xlabels))
    plt.grid(linestyle='dashed')
    plt.tight_layout()
    return plt


def autolabel(rects, metric):
    special_float_format = 3 if metric == 'mse' else None
    for rect in rects:
        height = rect.get_height()
        plt.text(
            rect.get_x() + rect.get_width() / 2., 1.005 * height,
            format_float(height, special_float_format),
            ha='center', va='bottom')


def shorten_names(models):
    return [shorten_name(model) for model in models]


def shorten_name(model):
    return '\n'.join(model.split('_'))


def plot_heatmap(gs1_str, gs2_str, dist_mat, thresh_pos, thresh_neg,
                 dataset, dist_metric, norm):
    m, n = dist_mat.shape
    label_mat, num_poses, num_negs, _, _ = \
        get_classification_labels_from_dist_mat(
            dist_mat, thresh_pos, thresh_neg)
    title = '{} pos pairs ({:.2%})\n{} neg pairs ({:.2%})'.format(
        num_poses, num_poses / (m * n), num_negs, num_negs / (m * n))
    sorted_label_mat = np.sort(label_mat, axis=1)[:, ::-1]
    mat_str = '{}({})_{}({})_{}_{}'.format(
        gs1_str, m, gs2_str, n, thresh_pos, thresh_neg)
    fn = '{}_acc_{}_labels_heatmap_{}{}'.format(dist_metric, mat_str,
                                                dataset, get_norm_str(norm))
    dir = '{}/{}/classif_labels'.format(get_result_path(), dataset)
    create_dir_if_not_exists(dir)
    plot_heatmap_helper(sorted_label_mat, title, dir, fn,
                        cmap='bwr')
    sorted_dist_mat = np.sort(dist_mat, axis=1)
    mat_str = '{}({})_{}({})'.format(
        gs1_str, m, gs2_str, n)
    fn = '{}_acc_{}_dist_heatmap_{}{}'.format(dist_metric, mat_str,
                                              dataset, get_norm_str(norm))
    plot_heatmap_helper(sorted_dist_mat, '', dir, fn,
                        cmap='tab20')


def plot_heatmap_helper(mat, title, dir, fn, cmap):
    plt.figure()
    fig, ax = plt.subplots()
    im = ax.imshow(mat, aspect='auto', cmap=plt.get_cmap(cmap))
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=15)
    ax.tick_params(labelsize=15)
    ax.set_title(title, fontsize=17)
    save_fig(plt, dir, fn, print_path=False)


def proc_val_list(val_list):
    assert (val_list)
    if type(val_list[0]) is tuple:
        rtn = [[] for _ in range(len(val_list[0]))]
        for val in val_list:
            for i, v in enumerate(val):
                rtn[i].append(v)
        return rtn
    else:
        return [val_list]


def exp5():
    """ Query visualization. """
    dataset = 'imdbmulti'
    model = 'astar'
    concise = True
    norms = [True, False]
    dir = get_result_path() + '/{}/query_vis/{}'.format(dataset, model)
    create_dir_if_not_exists(dir)
    info_dict = {
        # draw node config
        'draw_node_size': 150 if dataset != 'linux' else 10,
        'draw_node_label_enable': True,
        'node_label_name': None if dataset == 'linux' else 'type',
        'draw_node_label_font_size': 6,
        'draw_node_color_map': TYPE_COLOR_MAP,
        # draw edge config
        'draw_edge_label_enable': False,
        'edge_label_name': 'valence',
        'draw_edge_label_font_size': 6,
        # graph text info config
        'each_graph_text_list': [],
        'each_graph_text_font_size': 8,
        'each_graph_text_pos': [0.5, 1.05],
        # graph padding: value range: [0, 1]
        'top_space': 0.20 if concise else 0.26,  # out of whole graph
        'bottom_space': 0.05,
        'hbetween_space': 0.6 if concise else 1,  # out of the subgraph
        'wbetween_space': 0,
        # plot config
        'plot_dpi': 200,
        'plot_save_path_eps': '',
        'plot_save_path_png': ''
    }
    train_data = load_data(dataset, train=True)
    test_data = load_data(dataset, train=False)
    row_graphs = test_data.graphs
    col_graphs = train_data.graphs
    r = load_result(dataset, model, row_graphs=row_graphs, col_graphs=col_graphs)
    tr = load_result(dataset, TRUE_MODEL, row_graphs=row_graphs, col_graphs=col_graphs)
    for norm in norms:
        ids = r.get_sort_id_mat(norm)
        m, n = r.m_n()
        num_vis = 10
        for i in range(num_vis):
            q = test_data.graphs[i]
            gids = np.concatenate([ids[i][:3], [ids[i][int(n / 2)]], ids[i][-3:]])
            gs = [train_data.graphs[j] for j in gids]
            info_dict['each_graph_text_list'] = \
                [get_text_label(dataset, r, tr, i, i, q, model, norm, True, concise)] + \
                [get_text_label(dataset, r, tr, i, j,
                                train_data.graphs[j], model, norm, False, concise) \
                 for j in gids]
            # print(info_dict['each_graph_text_list'])
            info_dict['plot_save_path_png'] = '{}/query_vis_{}_{}_{}{}.{}'.format(
                dir, dataset, model, i, get_norm_str(norm), 'png')
            info_dict['plot_save_path_eps'] = '{}/query_vis_{}_{}_{}{}.{}'.format(
                dir, dataset, model, i, get_norm_str(norm), 'eps')
            vis(q, gs, info_dict)


def get_text_label(dataset, r, tr, qid, gid, g, model, norm, is_query, concise):
    rtn = ''
    if is_query:
        # rtn += '\n\n'
        pass
    else:
        trank = tr.ranking(qid, gid, norm)
        if r.get_model() == tr.get_model():
            rtn += 'rank: {}\n'.format(trank)
        else:
            pass
            # ged_str = get_ged_select_norm_str(tr, qid, gid, norm)
            # rtn = 'true ged: {}\ntrue rank: {}\n'.format(
            #     ged_str, trank)
    # rtn += 'gid: {}{}'.format(g.graph['gid'], get_graph_stats_text(g, concise))
    if is_query:
        # rtn += '\nquery {}\nmodel: {}'.format(dataset, model)
        rtn += 'query: {}\n'.format(dataset)
    else:
        pass
        # ged_sim_str, ged_sim = r.dist_sim(qid, gid, norm)
        # if ged_sim_str == 'ged':
        #     ged_str = get_ged_select_norm_str(r, qid, gid, norm)
        #     rtn += '\n {}: {}\n'.format(ged_sim_str, ged_str)
        # else:
        #     rtn += '\n {}: {:.2f}\n'.format(ged_sim_str, ged_sim)
        # t = r.time(qid, gid)
        # if t:
        #     rtn += 'time: {:.2f} msec'.format(t)
        # else:
        #     rtn += 'time: -'
    return rtn


def get_ged_select_norm_str(r, qid, gid, norm):
    ged = r.dist_sim(qid, gid, norm=False)[1]
    norm_ged = r.dist_sim(qid, gid, norm=True)[1]
    if norm:
        return '{:.2f}({})'.format(norm_ged, int(ged))
    else:
        return '{}({:.2f})'.format(int(ged), norm_ged)
    # if norm:
    #     return '{:.2f} ({})'.format(norm_ged, ged)
    # else:
    #     return '{} ({:.2f})'.format(ged, norm_ged)


def get_graph_stats_text(g, concise):
    return '' if concise else '\n#nodes: {}\n#edges: {}\ndensity: {:.2f}'.format(
        g.number_of_nodes(), g.number_of_edges(), nx.density(g))


def exp11():
    dataset = 'linux'
    l = load_as_dict(
        '/home/<>/Documents/GraphEmbedding/model/Siamese/logs/siamese_regression_linux_2018-11-04T22:07:15.428277(sepa, fix=10; check multi-scale)/test_info.klepto')
    weight = l['atts']
    node_embs_dict = l['node_embs_dict']
    draw_emb_hist_heat(dataset, node_embs_dict, True)  # TODO: fix


def draw_emb_hist_heat(dataset, node_embs_dict, ds_norm, true_result,
                       extra_dir=None, plot_max_num=np.inf):  # heatmap
    # node_embs_dict: a default dict {gcn_id: nel}, where
    # nel: a list of node embeddings (matrices of N by D): train_gs + test_gs.
    # Load the graphs from the train and test datasets.
    train_data = load_data(dataset, train=True)
    test_data = load_data(dataset, train=False)
    row_graphs = test_data.graphs
    col_graphs = train_data.graphs
    # ids is in np.argsort arrangement, indices of orig mat that would sort.
    ids = true_result.get_sort_id_mat(ds_norm)
    colors = ['Reds', 'Greens', 'Blues']
    for gcn_id in sorted(node_embs_dict.keys()):
        nel = node_embs_dict[gcn_id]
        cmap_color = colors[gcn_id % len(colors)]
        draw_emb_hist_heat_helper(gcn_id, nel, cmap_color, dataset,
                                  row_graphs, col_graphs, ids, true_result,
                                  ds_norm,
                                  plot_max_num, extra_dir)


def draw_emb_hist_heat_helper(gcn_id, nel, cmap_color, dataset,
                              row_graphs, col_graphs, ids, true_r, ds_norm,
                              plot_max_num, extra_dir):
    plt_cnt = 0
    for i in range(len(row_graphs)):
        # gids = column ids of [worst match, best match]
        gids = np.concatenate([ids[i][:1], ids[i][-1:]])
        for j in gids:
            _, d = true_r.dist_sim(i, j, ds_norm)
            # nel is [train + val ... test]
            query_nel_idx = len(col_graphs) + i
            match_nel_idx = j
            # result is dot product between the query (test) and match (train/val)
            result = np.dot(nel[query_nel_idx], nel[match_nel_idx].T)
            plt.figure()
            sns_plot = sns.heatmap(result, fmt='d', cmap=cmap_color)
            fig = sns_plot.get_figure()
            dir = '{}/{}/{}'.format(get_result_path(), dataset, 'heatmap')
            fn = '{}_{}_{}_gcn{}'.format(i, j, d, gcn_id)
            plt_cnt += save_fig(fig, dir, fn, print_path=False)
            if extra_dir:
                plt_cnt += save_fig(fig, extra_dir + '/heatmap', fn, print_path=False)
            plt.close()
            result_array = []
            for m in range(len(result)):
                for n in range(len(result[m])):
                    result_array.append(result[m][n])
            plt.figure()
            plt.xlim(-1, 1)
            plt.ylim(0, 100)
            sns_plot = sns.distplot(result_array, bins=16, color='r',
                                    kde=False, rug=False, hist=True)
            fig = sns_plot.get_figure()
            dir = '{}/{}/{}'.format(get_result_path(), dataset, 'histogram')
            fn = '{}_{}_{}_gcn{}'.format(i, j, d, gcn_id)
            plt_cnt += save_fig(fig, dir, fn, print_path=False)
            if extra_dir:
                plt_cnt += save_fig(fig, extra_dir + '/histogram', fn, print_path=False)
            plt.close()
        if plt_cnt > plot_max_num:
            print('Saved {} node embeddings mne plots for gcn{}'.format(plt_cnt, gcn_id))
            return
    print('Saved {} node embeddings mne plots for gcn{}'.format(plt_cnt, gcn_id))


def exp12():
    dataset = 'ptc'
    ds_algo = 'astar'
    ds_metric = 'ged'
    sim_or_dist = 'dist'
    dir = '/media/...)'
    row_graphs = load_data(dataset, False).graphs
    col_graphs = load_data(dataset, True).graphs
    tr_l = load_as_dict(dir + '/train_val_info.klepto')
    print(tr_l.keys())
    te_l = load_as_dict(dir + '/test_info.klepto')
    print(te_l.keys())
    true_r = load_result(dataset, ds_algo, row_graphs, col_graphs, None,
                         None, False, sim_or_dist, ds_metric, None)
    pred_r = load_result(dataset, 'siamese', row_graphs, col_graphs, None,
                         te_l['sim_mat'], True, sim_or_dist, ds_metric, None)
    draw_ranking(dataset, ds_metric, true_r, pred_r, 'Our Model',
                 tr_l['flags']['node_feat_name'],
                 plot_node_ids=False, plot_gids=False, ds_norm=True,
                 existing_mappings=None)


def draw_ranking(dataset, ds_metric, true_r, pred_r, model_name, node_feat_name,
                 plot_node_ids=False, plot_gids=False, ds_norm=True,
                 existing_mappings=None,
                 extra_dir=None, plot_max_num=np.inf):
    plot_what = 'query_demo'
    concise = True
    dir = get_result_path() + '/{}/{}/{}'.format(dataset, plot_what,
                                                 true_r.get_model())
    info_dict = {
        # draw node config
        'draw_node_size': 20,
        'draw_node_label_enable': True,
        'show_labels': plot_node_ids,
        'node_label_type': 'label' if plot_node_ids else 'type',
        'node_label_name': 'type',
        'draw_node_label_font_size': 6,
        'draw_node_color_map': get_color_map(true_r.get_all_gs()),
        # draw edge config
        'draw_edge_label_enable': False,
        'draw_edge_label_font_size': 6,
        # graph text info config
        'each_graph_text_list': [],
        'each_graph_text_font_size': 10,
        'each_graph_text_pos': [0.5, 1.05],
        # graph padding: value range: [0, 1]
        'top_space': 0.20 if concise else 0.26,  # out of whole graph
        'bottom_space': 0.05,
        'hbetween_space': 0.6 if concise else 1,  # out of the subgraph
        'wbetween_space': 0,
        # plot config
        'plot_dpi': 200,
        'plot_save_path_eps': '',
        'plot_save_path_png': ''
    }
    test_gs = true_r.get_row_gs()
    train_gs = None
    if true_r.has_single_col_gs():
        train_gs = true_r.get_single_col_gs()
        if plot_node_ids and existing_mappings:
            # existing_orderings: [train + val ... test]
            test_gs = reorder_gs_based_on_exsiting_mappings(
                test_gs, existing_mappings[len(train_gs):], node_feat_name)
            train_gs = reorder_gs_based_on_exsiting_mappings(
                train_gs, existing_mappings[0:len(train_gs)], node_feat_name)
    plt_cnt = 0
    ids_groundtruth = true_r.get_sort_id_mat(ds_norm)
    ids_rank = pred_r.get_sort_id_mat(ds_norm)
    for i in range(len(test_gs)):
        q = test_gs[i]
        if not true_r.has_single_col_gs():
            train_gs = true_r.get_col_gs(i)
        middle_idx = len(train_gs) // 2
        # Choose the top 6 matches, the overall middle match, and the worst match.
        selected_ids = list(range(6))
        selected_ids.extend([middle_idx, -1])
        # Get the selected graphs from the groundtruth and the model.
        gids_groundtruth = np.array(ids_groundtruth[i][selected_ids])
        gids_rank = np.array(ids_rank[i][selected_ids])
        # Top row graphs are only the groundtruth outputs.
        gs_groundtruth = [train_gs[j] for j in gids_groundtruth]
        # Bottom row graphs are the query graph + model ranking.
        gs_rank = [test_gs[i]]
        gs_rank = gs_rank + [train_gs[j] for j in gids_rank]
        gs = gs_groundtruth + gs_rank

        # Create the plot labels.
        text = []
        # First label is the name of the groundtruth algorithm, rest are scores for the graphs.
        text += [get_text_label_for_ranking(
            ds_metric, true_r, i, i, ds_norm, True, dataset, gids_groundtruth, plot_gids)]
        text += [get_text_label_for_ranking(
            ds_metric, true_r, i, j, ds_norm, False, dataset, gids_groundtruth, plot_gids)
            for j in gids_groundtruth]
        # Start bottom row labels, just ranked from 1 to N with some fancy formatting.
        text.append("Rank by\n{}".format(model_name))
        for j in range(len(gids_rank)):
            ds = format_ds(pred_r.pred_ds(i, gids_rank[j], ds_norm))
            if j == len(gids_rank) - 2:
                rtn = '\n ...   {}   ...\n{}'.format(int(len(train_gs) / 2), ds)
            elif j == len(gids_rank) - 1:
                rtn = '\n {}\n{}'.format(int(len(train_gs)), ds)
            else:
                rtn = '\n {}\n{}'.format(str(j + 1), ds)
            # rtn = '\n {}: {:.2f}'.format('sim', pred_r.sim_mat_[i][j])
            text.append(rtn)

        # Perform the visualization.
        info_dict['each_graph_text_list'] = text
        fn = '{}_{}_{}_{}{}'.format(
            plot_what, dataset, true_r.get_model(), i, get_norm_str(ds_norm))
        info_dict, plt_cnt = set_save_paths_for_vis(
            info_dict, dir, extra_dir, fn, plt_cnt)
        vis_small(q, gs, info_dict)
        if plt_cnt > plot_max_num:
            print('Saved {} query demo plots'.format(plt_cnt))
            return
    print('Saved {} query demo plots'.format(plt_cnt))


def set_save_paths_for_vis(info_dict, dir, extra_dir, fn, plt_cnt):
    info_dict['plot_save_path_png'] = '{}/{}.{}'.format(dir, fn, 'png')
    info_dict['plot_save_path_eps'] = '{}/{}.{}'.format(dir, fn, 'eps')
    if extra_dir:
        info_dict['plot_save_path_png'] = '{}/{}.{}'.format(
            extra_dir, fn, 'png')
        info_dict['plot_save_path_eps'] = '{}/{}.{}'.format(
            extra_dir, fn, 'eps')
    plt_cnt += 2
    return info_dict, plt_cnt


def format_ds(ds):
    return '{:.2f}'.format(ds)


def get_color_map(gs):
    fl = len(FAVORITE_COLORS)
    rtn = {}
    ntypes = defaultdict(int)
    for g in gs:
        for nid, node in g.nodes(data=True):
            ntypes[node.get('type')] += 1
    secondary = {}
    for i, (ntype, cnt) in enumerate(sorted(ntypes.items(), key=lambda x: x[1],
                                            reverse=True)):
        if ntype is None:
            color = None
            rtn[ntype] = color
        elif i >= fl:
            cmaps = plt.cm.get_cmap('hsv')
            color = cmaps((i - fl) / (len(ntypes) - fl))
            secondary[ntype] = color
        else:
            color = mcolors.to_rgba(FAVORITE_COLORS[i])[:3]
            rtn[ntype] = color
    if secondary:
        rtn.update((secondary))
    return rtn


def random_remap_dict(d):
    rtn = {}
    vs = list(d.values())
    random.Random(123).shuffle(vs)
    for i, k in enumerate(d.keys()):
        rtn[k] = vs[i]
    return rtn


def get_text_label_for_ranking(ds_metric, r, qid, gid, norm, is_query, dataset,
                               gids_groundtruth, plot_gids):
    rtn = ''
    # if is_query:
    #     rtn += '\n\n'
    # TODO: fix for graphlet
    if ds_metric == 'ged':
        if norm:
            ds_label = 'nGED'
        else:
            ds_label = 'GED'
    elif ds_metric == 'glet':
        ds_label = 'glet'
    elif ds_metric == 'mcs':
        if norm:
            ds_label = 'nMCS'
        else:
            ds_label = 'MCS'
    else:
        raise NotImplementedError()
    if is_query:
        if ds_metric == 'mcs':
            rtn += '{} by\nMcCreesh2017'.format(ds_label)
        else:
            if 'aids' in dataset or dataset == 'linux':
                rtn += '{} by\nA*'.format(ds_label)
            else:
                rtn += '{} by \nBeam-Hung.-VJ'.format(ds_label)
    else:
        ds_str, ged_sim = r.dist_sim(qid, gid, norm)
        if ds_str == 'ged':
            ged_str = get_ged_select_norm_str(r, qid, gid, norm)
            if gid != gids_groundtruth[6]:
                rtn += '\n {}'.format(ged_str.split('(')[0])
            else:
                rtn += '\n ...   {}   ...'.format(ged_str.split('(')[0])
        else:
            rtn += '\n {:.2f}'.format(ged_sim)  # in case it is a numpy.float64, use float()
    if plot_gids:
        rtn += '\nid: {}'.format(gid)
    return rtn


def reorder_gs_based_on_exsiting_mappings(gs, exsiting_mappings,
                                          node_feat_name):
    # [train + val ... test]
    for i, g in enumerate(gs):
        gs[i] = reorder_nodes(
            g, 'existing', exsiting_mappings[i], node_feat_name)
    return gs


def save_fig(plt, dir, fn, print_path=False):
    plt_cnt = 0
    if dir is None or fn is None:
        return plt_cnt
    final_path_without_ext = '{}/{}'.format(dir, fn)
    for ext in ['png', 'eps']:
        full_path = final_path_without_ext + '.' + ext
        create_dir_if_not_exists(dirname(full_path))
        try:
            plt.savefig(full_path, bbox_inches='tight')
        except:
            warn('savefig')
        if print_path:
            print('Saved to {}'.format(full_path))
        plt_cnt += 1
    return plt_cnt


if __name__ == '__main__':
    exp1()
