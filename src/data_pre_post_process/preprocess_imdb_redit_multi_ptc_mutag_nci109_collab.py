import sys
from os.path import dirname, abspath

cur_folder = dirname(abspath(__file__))
sys.path.insert(0, '{}/..'.format(cur_folder))
from utils import get_data_path, get_file_base_id, create_dir_if_not_exists, sorted_nicely
from data import save_glabels_as_txt
import networkx as nx
from glob import glob
import random

random.seed(123)


class Conf(object):
    def __init__(self, infolder, outfolder, has_node_type, train_perc=0.8, need_sort=False):
        self.infolder = infolder
        self.outfolder = outfolder
        self.has_node_type = has_node_type
        self.train_perc_ = train_perc
        self.need_sort_ = need_sort


confs = {'imdbmulti': Conf('imdb_comedy_romance_scifi', 'IMDBMulti', False),
         'reddit5kmulti': Conf('reddit_multi_5K', 'RedditMulti5k', False),
         'ptc': Conf('ptc', 'PTC', True),
         'mutag': Conf('mutag', 'MUTAG', True),
         'nci109': Conf('nci109', 'NCI109', True, need_sort=True),
         'reddit_subreddit_10K': Conf('reddit_subreddit_10K', 'RedditMulti10k', False, 0.95),
         'collab': Conf('collab', 'COLLAB', False, need_sort=True),}

dataset = 'collab'

conf = confs[dataset]


def main():
    dirin = get_data_path() + '/{}/graph'.format(conf.infolder)
    k = float('inf')
    lesseqk = []
    glabel_map = read_graph_labels()
    info_map = {}
    disconnected = []
    files = glob(dirin + '/*.gexf')
    if conf.need_sort_:
        files = sorted_nicely(files)
    for i, file in enumerate(files):
        g = nx.read_gexf(file)
        gid = get_file_base_id(file)
        print(i, gid, g.number_of_nodes())
        if g.number_of_nodes() <= k:
            if not nx.is_connected(g):
                print(gid, 'is not connected')
                gsize = g.number_of_nodes()
                g = max(nx.connected_component_subgraphs(g), key=len)
                grmd = gsize - g.number_of_nodes()
                assert (grmd > 0)
                g_info = 'rm_{}_nodes'.format(grmd)
                disconnected.append(g)
            else:
                g_info = ''
                lesseqk.append(g)
            info_map[gid] = g_info
            g.graph['gid'] = gid
            g.graph['label'] = glabel_map[gid]
            for node, d in g.nodes(data=True):
                type = d['node_class']
                if conf.has_node_type:
                    d.pop('node_class')
                    d['type'] = type
            for edge in g.edges_iter(data=True):
                del edge[2]['weight']
    print(len(lesseqk))
    gen_dataset(lesseqk)
    gen_dataset(disconnected)
    save_glabels_as_txt(get_data_path() + '/{}/glabels'.format(conf.outfolder), glabel_map)
    save_glabels_as_txt(get_data_path() + '/{}/info'.format(conf.outfolder), info_map)


def gen_dataset(graphs):
    random.Random(123).shuffle(graphs)
    dirout_train = get_data_path() + '/{}/train'.format(conf.outfolder)
    dirout_test = get_data_path() + '/{}/test'.format(conf.outfolder)
    create_dir_if_not_exists(dirout_train)
    create_dir_if_not_exists(dirout_test)
    sp = int(len(graphs) * conf.train_perc_)
    for g in graphs[0:sp]:
        nx.write_gexf(g, dirout_train + '/{}.gexf'.format(g.graph['gid']))
    for g in graphs[sp:]:
        nx.write_gexf(g, dirout_test + '/{}.gexf'.format(g.graph['gid']))


def read_graph_labels():
    rtn = {}
    glabel_orig_new_map = {}
    with open(get_data_path() + '/{}/label/label.txt'.format(conf.infolder)) as f:
        for line in f:
            ls = line.rstrip().split()
            assert (len(ls) == 2)
            gid = get_file_base_id(ls[0])
            orig_glabel = int(float(ls[1]))
            glabel = glabel_orig_new_map.get(orig_glabel)
            if glabel is None:  # dangerous to test not glabel since glabel could be 0!
                glabel = len(glabel_orig_new_map)  # 0-based
                glabel_orig_new_map[orig_glabel] = glabel
            rtn[gid] = glabel
    return rtn


main()
