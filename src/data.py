from utils import get_train_str, get_data_path, get_save_path, sorted_nicely, \
    save, load, append_ext_to_filepath
import networkx as nx
import numpy as np
import random
from random import randint
from collections import OrderedDict
from glob import glob
from os.path import basename, join


class Data(object):
    def __init__(self, name_str):
        name = join(self.__class__.__name__, name_str + self.name_suffix())
        sfn = self.save_filename(name)
        temp = load(sfn, use_klepto=True)  # use klepto for faster saving and loading
        if temp:
            self.__dict__ = temp
            print('{} loaded from {}{}'.format(
                name, sfn,
                ' with {} graphs'.format(
                    len(self.graphs)) if
                hasattr(self, 'graphs') else ''))
        else:
            self.init()
            save(sfn, self.__dict__)
            print('{} saved to {}'.format(name, sfn))

    def init(self):
        raise NotImplementedError()

    def name_suffix(self):
        return ''

    def save_filename(self, name):
        return '{}/{}'.format(get_save_path(), name)

    def get_gids(self):
        return [g.graph['gid'] for g in self.graphs]


class SynData(Data):
    train_num_graphs = 20
    test_num_graphs = 10

    def __init__(self, train):
        if train:
            self.num_graphs = SynData.train_num_graphs
        else:
            self.num_graphs = SynData.test_num_graphs
        super(SynData, self).__init__(get_train_str(train))

    def init(self):
        self.graphs = []
        for i in range(self.num_graphs):
            n = randint(5, 20)
            m = randint(n - 1, n * (n - 1) / 2)
            g = nx.gnm_random_graph(n, m)
            g.graph['gid'] = i
            self.graphs.append(g)
        print('Randomly generated %s graphs' % self.num_graphs)

    def name_suffix(self):
        return '_{}_{}'.format(SynData.train_num_graphs,
                               SynData.test_num_graphs)


class AIDSData(Data):
    def __init__(self, train):
        self.train = train
        super(AIDSData, self).__init__(get_train_str(train))

    def init(self):
        self.graphs = []
        datadir = '{}/{}/{}'.format(
            get_data_path(), self.get_folder_name(), get_train_str(self.train))
        self.graphs = iterate_get_graphs(datadir)
        print('Loaded {} graphs from {}'.format(len(self.graphs), datadir))
        if 'nef' in self.get_folder_name():
            print('Removing edge features')
            for g in self.graphs:
                self._remove_valence(g)
        self.graphs, self.glabels = add_glabel_to_each_graph(self.graphs, '', True)
        assert (self.glabels is None)  # fake graph labels

    def get_folder_name(self):
        raise NotImplementedError()

    def _remove_valence(self, g):
        for n1, n2, d in g.edges(data=True):
            d.pop('valence', None)


class AIDS10kData(AIDSData):
    def get_folder_name(self):
        return 'AIDS10k'


class AIDS10knefData(AIDS10kData):
    def init(self):
        self.graphs = AIDS10kData(self.train).graphs
        for g in self.graphs:
            self._remove_valence(g)
        print('Processed {} graphs: valence removed'.format(len(self.graphs)))


class AIDS700nefData(AIDSData):
    def get_folder_name(self):
        return 'AIDS700nef'

    def _remove_valence(self, g):
        for n1, n2, d in g.edges(data=True):
            d.pop('valence', None)


class AIDS80nefData(AIDS700nefData):
    def init(self):
        self.graphs = AIDS700nefData(self.train).graphs
        random.Random(123).shuffle(self.graphs)
        if self.train:
            self.graphs = self.graphs[0:70]
        else:
            self.graphs = self.graphs[0:10]
        print('Loaded {} graphs: valence removed'.format(len(self.graphs)))
        self.graphs, self.glabels = add_glabel_to_each_graph(self.graphs, '', True)
        assert (self.glabels is None)  # fake graph labels


class LINUXData(Data):
    def __init__(self, train):
        self.train = train
        super(LINUXData, self).__init__(get_train_str(train))

    def init(self):
        self.graphs = []
        datadir = '{}/LINUX/{}'.format(
            get_data_path(), get_train_str(self.train))
        self.graphs = iterate_get_graphs(datadir)
        print('Loaded {} graphs from {}'.format(len(self.graphs), datadir))
        self.graphs, self.glabels = add_glabel_to_each_graph(self.graphs, '', True)
        assert (self.glabels is None)  # fake graph labels


class IMDB1kData(Data):
    def __init__(self, train):
        self.train = train
        super(IMDB1kData, self).__init__(get_train_str(train))

    def init(self):
        self.graphs = []
        datadir = '{}/IMDB1k{}/{}'.format(
            get_data_path(), self._identity(), get_train_str(self.train))
        self.graphs = iterate_get_graphs(datadir)
        print('Loaded {} graphs from {}'.format(len(self.graphs), datadir))

    def _identity(self):
        raise NotImplementedError()


class IMDB1kCoarseData(IMDB1kData):
    def _identity(self):
        return 'Coarse'


class IMDB1kFineData(IMDB1kData):
    def _identity(self):
        return 'Fine'


class IMDBMultiData(Data):
    def __init__(self, train):
        self.train = train
        super(IMDBMultiData, self).__init__(get_train_str(train))

    def init(self):
        dir = 'IMDBMulti'
        self.graphs = get_proc_graphs(dir, self.train)
        self.graphs, self.glabels = add_glabel_to_each_graph(self.graphs, dir)


class IMDBMulti800Data(Data):
    def __init__(self, train):
        self.train = train
        super(IMDBMulti800Data, self).__init__(get_train_str(train))

    def init(self):
        self.graphs = get_proc_graphs('IMDBMulti800', self.train)


class PTCData(Data):
    def __init__(self, train):
        self.train = train
        super(PTCData, self).__init__(get_train_str(train))

    def init(self):
        dir = 'PTC'
        self.graphs = get_proc_graphs(dir, self.train)
        self.graphs, self.glabels = add_glabel_to_each_graph(self.graphs, dir)
        assert (self.glabels is not None)  # real graph labels


def get_proc_graphs(datadir, train):
    datadir = '{}/{}/{}'.format(
        get_data_path(), datadir, get_train_str(train))
    graphs = iterate_get_graphs(datadir)
    print('Loaded {} graphs from {}'.format(len(graphs), datadir))
    return graphs


def iterate_get_graphs(dir):
    graphs = []
    for file in sorted_nicely(glob(dir + '/*.gexf')):
        gid = int(basename(file).split('.')[0])
        g = nx.read_gexf(file)
        g.graph['gid'] = gid
        graphs.append(g)
        if not nx.is_connected(g):
            print('{} not connected'.format(gid))
    return graphs


""" Graph labels. """


def add_glabel_to_each_graph(graphs, dir, use_fake_glabels=False):
    glabels = None
    if not use_fake_glabels:
        filepath = '{}/{}/glabels.txt'.format(get_data_path(), dir)
        glabels = load_glabels_from_txt(filepath)
    seen = set()  # check every graph id is seen only once
    for g in graphs:
        gid = g.graph['gid']
        assert (gid not in seen)
        seen.add(gid)
        if use_fake_glabels:
            glabel = randint(0, 9)  # randomly assign a graph label from {0, .., 9}
        else:
            glabel = glabels[gid]
        g.graph['glabel'] = glabel
    return graphs, glabels


def save_glabels_as_txt(filepath, glabels):
    filepath = append_ext_to_filepath('.txt', filepath)
    with open(filepath, 'w') as f:
        for id, glabel in OrderedDict(glabels).items():
            f.write('{}\t{}\n'.format(id, glabel))


def load_glabels_from_txt(filepath):
    filepath = append_ext_to_filepath('.txt', filepath)
    rtn = {}
    int_map = {}
    seen_glabels = set()
    with open(filepath) as f:
        for line in f:
            ls = line.rstrip().split()
            assert (len(ls) == 2)
            gid = int(ls[0])
            try:
                glabel = int(ls[1])
            except ValueError:
                label_string = ls[1]
                glabel = int_map.get(label_string)
                if glabel is None:
                    glabel = len(int_map)  # guarantee 0-based
                    int_map[label_string] = glabel  # increase the size of int_map by 1
            rtn[gid] = glabel
            seen_glabels.add(glabel)
    if 0 not in seen_glabels:  # check 0-based graph labels
        raise RuntimeError('{} has no glabel 0; {}'.format(filepath, seen_glabels))
    return rtn


if __name__ == '__main__':
    from utils import load_data
    from collections import defaultdict

    dataset = 'nci109'
    nn = []
    ntypes = defaultdict(int)
    train_gs = load_data(dataset, True).graphs
    test_gs = load_data(dataset, False).graphs
    gs = train_gs + test_gs
    glabels = set()
    disconnected = set()
    for g in gs:
        nn.append(g.number_of_nodes())
        print(g.graph['gid'], g.number_of_nodes(), g.graph.get('glabel'))
        glabels.add(g.graph.get('glabel'))
        for nid, node in g.nodes(data=True):
            ntypes[node.get('type')] += 1
        if not nx.is_connected(g):
            disconnected.add(g)
    print('train gs', len(train_gs))
    print('test_gs', len(test_gs))
    print('node types', ntypes)
    print('graph labels', glabels)
    print('{} disconnected graphs'.format(len(disconnected)))
    print('#node labels: {}\n#glabels: {}\n#graphs: {}\nAvg: {}\nStd: {}\nMin: {}\nMax: {}'.format(
        len(ntypes), len(glabels), len(gs), np.mean(nn), np.std(nn), np.min(nn), np.max(nn)))
