from utils import create_dir_if_not_exists
import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
from collections import OrderedDict
from os.path import dirname
import math
from colour import Color
import numpy as np


def vis(q=None, gs=None, info_dict=None):
    plt.figure()
    info_dict_preprocess(info_dict)

    # get num
    graph_num = 1 + len(gs)
    plot_m, plot_n = calc_subplot_size(graph_num)

    # draw query graph
    ax = plt.subplot(plot_m, plot_n, 1)
    draw_graph(q, info_dict)
    draw_extra(0, ax, info_dict,
               list_safe_get(info_dict['each_graph_text_list'], 0, ""))

    # draw graph candidates
    for i in range(len(gs)):
        ax = plt.subplot(plot_m, plot_n, i + 2)
        draw_graph(gs[i], info_dict)
        draw_extra(i, ax, info_dict,
                   list_safe_get(info_dict['each_graph_text_list'], i + 1, ""))

    # plot setting
    # plt.tight_layout()
    left = 0.01  # the left side of the subplots of the figure
    right = 0.99  # the right side of the subplots of the figure
    top = 1 - info_dict['top_space']  # the top of the subplots of the figure
    bottom = \
        info_dict['bottom_space']  # the bottom of the subplots of the figure
    wspace = \
        info_dict['wbetween_space']  # the amount of width reserved for blank space between subplots
    hspace = \
        info_dict['hbetween_space']  # the amount of height reserved for white space between subplots

    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top,
                        wspace=wspace, hspace=hspace)

    # save / display
    save_figs(info_dict)


def draw_graph(g, info_dict):
    if g is None:
        return
    pos = sorted_dict(graphviz_layout(g))
    if info_dict['node_label_name'] is not None:
        node_labels = sorted_dict(nx.get_node_attributes(g, info_dict['node_label_name']))
        color_values = [info_dict['draw_node_color_map'].get(node_label, 'yellow')
                        for node_label in node_labels.values()]
    else:
        node_labels = {}
        for i in range(len(pos.keys())):
            node_labels[str(i)] = ''
        node_labels = sorted_dict(node_labels)
        color_values = ['#ff6666'] * len(pos.keys())
    for key, value in node_labels.items():
        if len(value) > 6:
            value = value[0:6]  # shorten the label
        node_labels[key] = value
    if info_dict['node_label_name'] is not None:
        nx.draw_networkx(g, pos, nodelist=pos.keys(),
                         node_color=color_values, with_labels=True,
                         node_size=info_dict['draw_node_size'], labels=node_labels,
                         font_size=info_dict['draw_node_label_font_size'])
    else:
        nx.draw_networkx(g, pos, nodelist=pos.keys(), node_color=color_values,
                         with_labels=True, node_size=info_dict['draw_node_size'],
                         labels=node_labels)

    if info_dict['draw_edge_label_enable'] == True:
        edge_labels = nx.get_edge_attributes(g, info_dict['edge_label_name'])
        nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels,
                                     font_size=info_dict[
                                         'draw_edge_label_font_size'])


def vis_attention(q=None, gs=None, info_dict=None, weight=None, weight_max=0, weight_min=0):
    plt.figure(figsize=(12, 8))
    info_dict_preprocess(info_dict)

    # get num
    graph_num = 1 + len(gs)
    plot_m, plot_n = calc_subplot_size(graph_num)

    # draw query graph
    ax = plt.subplot(plot_m, plot_n, 1)
    draw_graph_attention(q, info_dict, weight[0], weight_max, weight_min)
    draw_extra(0, ax, info_dict,
               list_safe_get(info_dict['each_graph_text_list'], 0, ""))

    # draw graph candidates
    for i in range(len(gs)):
        ax = plt.subplot(plot_m, plot_n, i + 2)
        draw_graph_attention(gs[i], info_dict, weight[i + 1], weight_max, weight_min)
        draw_extra(i, ax, info_dict,
                   list_safe_get(info_dict['each_graph_text_list'], i + 1, ""))

    # plot setting
    # plt.tight_layout()
    left = 0.01  # the left side of the subplots of the figure
    right = 0.99  # the right side of the subplots of the figure
    top = 1 - info_dict['top_space']  # the top of the subplots of the figure
    bottom = \
        info_dict['bottom_space']  # the bottom of the subplots of the figure
    wspace = \
        info_dict['wbetween_space']  # the amount of width reserved for blank space between subplots
    hspace = \
        info_dict['hbetween_space']  # the amount of height reserved for white space between subplots

    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top,
                        wspace=wspace, hspace=hspace)

    # save / display
    save_figs(info_dict)


def vis_small(q=None, gs=None, info_dict=None):
    plt.figure(figsize=(8, 3))
    info_dict_preprocess(info_dict)

    # get num
    graph_num = 1 + len(gs)
    plot_m, plot_n = calc_subplot_size_small(graph_num)

    # draw query graph
    # info_dict['each_graph_text_font_size'] = 9
    ax = plt.subplot(plot_m, plot_n, 1)
    draw_graph_small(q, info_dict)
    draw_extra(0, ax, info_dict,
               list_safe_get(info_dict['each_graph_text_list'], 0, ""))

    # draw graph candidates
    # info_dict['each_graph_text_font_size'] = 12
    for i in range(len(gs)):
        ax = plt.subplot(plot_m, plot_n, i + 2)
        draw_graph_small(gs[i], info_dict)
        draw_extra(i, ax, info_dict,
                   list_safe_get(info_dict['each_graph_text_list'], i + 1, ""))

    # plot setting
    # plt.tight_layout()
    left = 0.01  # the left side of the subplots of the figure
    right = 0.99  # the right side of the subplots of the figure
    top = 1 - info_dict['top_space']  # the top of the subplots of the figure
    bottom = \
        info_dict['bottom_space']  # the bottom of the subplots of the figure
    wspace = \
        info_dict['wbetween_space']  # the amount of width reserved for blank space between subplots
    hspace = \
        info_dict['hbetween_space']  # the amount of height reserved for white space between subplots

    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top,
                        wspace=wspace, hspace=hspace)

    # save / display
    save_figs(info_dict)


def draw_graph_attention(g, info_dict, weight, weight_max, weight_min):
    if g is None:
        return
    pos = sorted_dict(graphviz_layout(g))
    weight_array = []
    for i in range(len(weight)):
        weight_array.append(weight[i][0])
    for i in range(len(weight_array)):
        if weight_array[i] > weight_max:
            weight_array[i] = weight_max
        if weight_array[i] < weight_min:
            weight_array[i] = weight_min
        weight_array[i] = (weight_array[i] - weight_min) / (weight_max - weight_min)

    if info_dict['node_label_name'] is not None:
        node_labels = sorted_dict(nx.get_node_attributes(g, info_dict['node_label_name']))
        # from light to dark
        # color_values = sorted(range(len(weight_array)), key=lambda k: weight_array[k])
        red = Color("red")
        white = Color("white")
        color_list = list(white.range_to(red, 10))
        color_values = []
        # print('weight_array:', weight_array)
        for i in range(len(pos.keys())):
            color_values.append(color_list[int(weight_array[i] * 10) - 1].hex_l)
        # print(color_values)
    else:
        node_labels = {}
        for i in range(len(pos.keys())):
            node_labels[str(i)] = ''
        node_labels = sorted_dict(node_labels)
        color_values = []
        for i in range(len(pos.keys())):
            if len(str(hex(int(255 - 255 * weight_array[i])))) <= 3:
                if len(str(hex(int(80 + 255 - 255 * weight_array[i])))) < 3:
                    color_values.append("#0" + str(hex(int(80 + 255 - 255 * weight_array[i])))[2:] + "0" +
                                        str(hex(int(255 - 255 * weight_array[i])))[2:] + "ff")
                else:
                    color_values.append("#" + str(hex(int(80 + 255 - 255 * weight_array[i])))[2:] + "0" +
                                        str(hex(int(255 - 255 * weight_array[i])))[2:] + "ff")
            else:
                if len(str(hex(int(80 + 255 - 255 * weight_array[i])))) < 3:
                    color_values.append("#0" + str(hex(int(80 + 255 - 255 * weight_array[i])))[2:] +
                                        str(hex(int(255 - 255 * weight_array[i])))[2:] + "ff")
                else:
                    color_values.append("#" + str(hex(int(255 - 255 * weight_array[i])))[2:] +
                                        str(hex(int(255 - 255 * weight_array[i])))[2:] + "ff")
            # print(color_values)
    for key, value in node_labels.items():
        if len(value) > 6:
            value = value[0:6]  # shorten the label
        node_labels[key] = value
    if info_dict['node_label_name'] is not None:
        # cmap=plt.cm.Blues or cmap=plt.cm.PuBu or cmap=plt.Reds it depends on you
        nx.draw_networkx(g, pos, nodelist=pos.keys(),
                         node_color=color_values, with_labels=True,
                         node_size=info_dict['draw_node_size'], labels=node_labels,
                         font_size=info_dict['draw_node_label_font_size'])
    else:
        nx.draw_networkx(g, pos, nodelist=pos.keys(), node_color=color_values,
                         with_labels=True, node_size=info_dict['draw_node_size'],
                         labels=node_labels)

    if info_dict['draw_edge_label_enable'] == True:
        edge_labels = nx.get_edge_attributes(g, info_dict['edge_label_name'])
        nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels,
                                     font_size=info_dict[
                                         'draw_edge_label_font_size'])


def draw_graph_small(g, info_dict):
    if g is None:
        return
    if g.number_of_nodes() > 1000:
        print('Graph to plot too large with {} nodes! skip...'.format(
            g.number_of_nodes()))
        return
    pos = sorted_dict(graphviz_layout(g))
    color_values = _get_node_colors(g, info_dict)
    node_labels = sorted_dict(nx.get_node_attributes(g, info_dict['node_label_type']))
    for key, value in node_labels.items():
        # Labels are not shown, but if the ids want to be plotted, then they are shown.
        if not info_dict['show_labels']:
            node_labels[key] = ''
    # print(pos)
    nx.draw_networkx(g, pos, nodelist=pos.keys(),
                     node_color=color_values, with_labels=True,
                     node_size=_get_node_size(g, info_dict),
                     labels=node_labels,
                     font_size=info_dict['draw_node_label_font_size'],
                     linewidths=_get_line_width(g), width=_get_edge_width(g, info_dict))

    if info_dict['draw_edge_label_enable'] == True:
        edge_labels = nx.get_edge_attributes(g, info_dict['edge_label_name'])
        nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels,
                                     font_size=info_dict[
                                         'draw_edge_label_font_size'])


def calc_subplot_size(area):
    h = int(math.sqrt(area))
    while area % h != 0:
        area += 1
    w = area / h
    return [h, w]


def calc_subplot_size_small(area):
    w = int(area)
    return [2, math.ceil(w / 2)]


def list_safe_get(l, index, default):
    try:
        return l[index]
    except IndexError:
        return default


def draw_extra(i, ax, info_dict, text):
    left = list_safe_get(info_dict['each_graph_text_pos'], 0, 0.5)
    bottom = list_safe_get(info_dict['each_graph_text_pos'], 1, 0.8)
    # print(left, bottom)
    ax.title.set_position([left, bottom])
    ax.set_title(text, fontsize=info_dict['each_graph_text_font_size'])
    plt.axis('off')


def info_dict_preprocess(info_dict):
    info_dict.setdefault('draw_node_size', 10)
    info_dict.setdefault('draw_node_label_enable', True)
    info_dict.setdefault('node_label_name', '')
    info_dict.setdefault('draw_node_label_font_size', 6)

    info_dict.setdefault('draw_edge_label_enable', False)
    info_dict.setdefault('edge_label_name', '')
    info_dict.setdefault('draw_edge_label_font_size', 6)

    info_dict.setdefault('each_graph_text_font_size', "")
    info_dict.setdefault('each_graph_text_pos', [0.5, 0.8])

    info_dict.setdefault('plot_dpi', 200)
    info_dict.setdefault('plot_save_path', "")

    info_dict.setdefault('top_space', 0.08)
    info_dict.setdefault('bottom_space', 0)
    info_dict.setdefault('hbetween_space', 0.5)
    info_dict.setdefault('wbetween_space', 0.01)


def sorted_dict(d):
    rtn = OrderedDict()
    for k in sorted(d.keys()):
        rtn[k] = d[k]
    return rtn


def save_figs(info_dict):
    save_path = info_dict['plot_save_path_png']
    print(save_path)
    if not save_path:
        # print('plt.show')
        plt.show()
    else:
        for full_path in [info_dict['plot_save_path_png'], info_dict['plot_save_path_eps']]:
            if not full_path:
                continue
            # print('Saving query vis plot to {}'.format(sp))
            create_dir_if_not_exists(dirname(full_path))
            plt.savefig(full_path, dpi=info_dict['plot_dpi'])
    plt.close()


def vis_multiple_mcs(q, gs, edge_mappings=[], node_mappings=[], info_dict=None):
    assert len(gs) == len(edge_mappings)

    plt.figure(figsize=(8, 3))
    info_dict_preprocess(info_dict)

    # get num
    graph_num = 2 * len(gs)
    plot_m, plot_n = calc_subplot_size_small(graph_num)

    # Populate the text for each subplot.
    info_dict['each_graph_text_list'] = []
    for i in range(len(gs)):
        info_dict['each_graph_text_list'].append('mcs size: {}'.format(len(node_mappings[i]['g1'])))

    for i in range(len(gs)):
        info_dict['each_graph_text_list'].append('gid: {}'.format(gs[i].graph['gid']))

    # draw query graphs (top row)
    # info_dict['each_graph_text_font_size'] = 9
    light_weight = info_dict['edge_weight_default']
    heavy_weight = info_dict['edge_weight_mcs']
    for i in range(len(gs)):
        # Query graph above.
        edge_selections = create_edge_selection(q, edge_mappings[i]['g1'])
        info_dict['edge_weights'] = [heavy_weight if x else light_weight for x in edge_selections]
        ax = plt.subplot(plot_m, plot_n, i + 1)
        draw_graph_small(q, info_dict)
        draw_extra(0, ax, info_dict,
                   list_safe_get(info_dict['each_graph_text_list'], i, ""))

        # Candidate graph below.
        edge_selections = create_edge_selection(gs[i], edge_mappings[i]['g2'])
        info_dict['edge_weights'] = [heavy_weight if x else light_weight for x in edge_selections]
        ax = plt.subplot(plot_m, plot_n, len(gs) + i + 1)
        draw_graph_small(gs[i], info_dict)
        draw_extra(0, ax, info_dict,
                   list_safe_get(info_dict['each_graph_text_list'], len(gs) + i, ""))

    # plot setting
    # plt.tight_layout()
    left = 0.01  # the left side of the subplots of the figure
    right = 0.99  # the right side of the subplots of the figure
    top = 1 - info_dict['top_space']  # the top of the subplots of the figure
    bottom = \
        info_dict['bottom_space']  # the bottom of the subplots of the figure
    wspace = \
        info_dict['wbetween_space']  # the amount of width reserved for blank space between subplots
    hspace = \
        info_dict['hbetween_space']  # the amount of height reserved for white space between subplots

    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top,
                        wspace=wspace, hspace=hspace)

    # save / display
    save_figs(info_dict)
    plt.close()


def create_edge_selection(g, mapping):
    selected = []
    for n1, n2, attr in g.edges(data=True):
        if attr['id'] in mapping:
            selected.append(True)
        else:
            selected.append(False)
    return selected


def _get_node_colors(g, info_dict):
    if info_dict['node_label_name'] is not None:
        color_values = []
        node_color_labels = sorted_dict(nx.get_node_attributes(g, info_dict['node_label_name']))
        for node_label in node_color_labels.values():
            color = info_dict['draw_node_color_map'].get(node_label, None)
            color_values.append(color)
    else:
        color_values = ['lightskyblue'] * g.number_of_nodes()
    # print(color_values)
    return color_values


def _get_node_size(g, info_dict):
    ns = info_dict['draw_node_size']
    return ns * np.exp(-0.02 * g.number_of_nodes())
    # return ns


def _get_line_width(g):
    lw = 5.0 * np.exp(-0.05 * g.number_of_edges())
    return lw


def _get_edge_width(g, info_dict):
    ew = info_dict.get('edge_weight_default', 1.0)
    ew = ew * np.exp(-0.0015 * g.number_of_edges())
    return info_dict.get('edge_weights', [ew] * len(g.edges()))
