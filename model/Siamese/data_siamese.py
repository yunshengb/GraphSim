from data import Data
from config import FLAGS
from coarsening import coarsen, perm_data
from utils_siamese import get_coarsen_level, is_transductive
from utils import load_data, exec_turnoff_print
from node_ordering import node_ordering
from random_walk_generator import generate_random_walks
from supersource_generator import generate_supersource
from super_large_dataset_handler import gen_data
import scipy.sparse as sp
from sklearn.preprocessing import OneHotEncoder
import networkx as nx
import numpy as np

exec_turnoff_print()


class SiameseModelData(Data):
    """Represents a set of data for training a Siamese model.

    This object is saved to disk, so every time preprocessing is changed,
    the saved files should
    be archived and re-run so that the next training run uses the updated
    preprocessing. Each
    model data is uniquely identified by the dataset name and FLAGS settings.
    This object should
    contain everything required as data input that can be precomputed.
    The saved binary file's
    name has all parameters that uniquely identify it.
    The main objects representing the graphs
    are ModelGraph objects.
    """

    def __init__(self, dataset):
        # The attributes set below determine the save file name.
        self.dataset = dataset
        self.valid_percentage = FLAGS.valid_percentage
        self.node_feat_name = FLAGS.node_feat_name
        self.node_feat_encoder = FLAGS.node_feat_encoder
        self.ordering = FLAGS.ordering
        self.coarsening = FLAGS.coarsening
        self.supersource = FLAGS.supersource
        self.random_walk = FLAGS.random_walk
        if is_transductive():
            self.transductive_info = 'trans'
        super().__init__(self.get_name())
        print('{} train graphs; {} validation graphs; {} test graphs'.format(
            len(self.train_gs),
            len(self.val_gs),
            len(self.test_gs)))

    def init(self):
        """Creates the object from scratch,
        only if a saved version doesn't already exist."""
        orig_train_data = load_data(self.dataset, train=True)
        train_gs, val_gs = self._train_val_split(orig_train_data)
        test_gs = load_data(self.dataset, train=False).graphs
        # Note that <graph> and self.<graph> can have different graphs because
        # of the supersource
        # option. This turns the graph into a DiGraph and adds a node,
        # so the graph is
        # fundamentally changed with the supersource setting.
        # Use self.<graph> as truth.
        self.node_feat_encoder = self._create_node_feature_encoder(
            orig_train_data.graphs + test_gs)
        self.graph_label_encoder = GraphLabelOneHotEncoder(
            orig_train_data.graphs + test_gs)
        self._check_graphs_num(test_gs, 'test')
        self.train_gs = self.create_model_gs(train_gs, 'train')
        self.val_gs = self.create_model_gs(val_gs, 'val')
        self.test_gs = self.create_model_gs(test_gs, 'test')
        if is_transductive():
            self._assign_global_ids()
        assert (len(train_gs) + len(val_gs) == len(orig_train_data.graphs))

    def input_dim(self):
        return self.node_feat_encoder.input_dim()

    def num_graphs(self):
        return len(self.train_gs) + len(self.val_gs) + len(self.test_gs)

    def create_model_gs(self, gs, tvt):
        rtn = []
        for i, g in enumerate(gs):
            mg = ModelGraph(g, self.node_feat_encoder, self.graph_label_encoder)
            rtn.append(mg)
        return rtn

    def get_name(self):
        if hasattr(self, 'name'):
            return self.name
        else:
            li = []
            for k, v in sorted(self.__dict__.items(), key=lambda x: x[0]):
                li.append('{}'.format(v))
            self.name = '_'.join(li)
            return self.name

    def _train_val_split(self, orig_train_data):
        if self.valid_percentage < 0 or self.valid_percentage > 1:
            raise RuntimeError('valid_percentage {} must be in [0, 1]'.format(
                self.valid_percentage))
        gs = orig_train_data.graphs
        sp = int(len(gs) * (1 - self.valid_percentage))
        train_graphs = gs[0:sp]
        valid_graphs = gs[sp:]
        self._check_graphs_num(train_graphs, 'train')
        self._check_graphs_num(valid_graphs, 'validation')
        return train_graphs, valid_graphs

    def _check_graphs_num(self, graphs, label):
        if len(graphs) <= 2:
            raise RuntimeError('Insufficient {} graphs {}'.format(
                label, len(graphs)))

    def _create_node_feature_encoder(self, gs):
        if self.node_feat_encoder == 'onehot':
            return NodeFeatureOneHotEncoder(gs, self.node_feat_name)
        elif 'constant' in self.node_feat_encoder:
            return NodeFeatureConstantEncoder(gs, self.node_feat_name)
        else:
            raise RuntimeError('Unknown node_feat_encoder {}'.format(
                self.node_feat_encoder))

    def _assign_global_ids(self):
        # Global ids are at the level of the entire dataset.
        # It is 0-based,
        # different from g.nxgraph.graph['gid] which may have gap.
        # Only needed for transductive model which directly optimizes
        # graph-level embeddings.
        # Used to facilitate embedding lookup.
        cnt = 0
        for g in self.train_gs + self.val_gs + self.test_gs:
            g.global_id = cnt
            cnt += 1


class GeneratedGraphCollection(Data):
    def __init__(self, from_info, gen_info, model_data):
        self.from_info = from_info
        self.gen_info = gen_info
        self.model_data = model_data
        self.short_name = 'from_' + from_info + '_gen_' + gen_info
        self.full_name = self.short_name + \
                         '_using_SiameseModelData_' + model_data.get_name()
        super().__init__(self.full_name)

    def init(self):
        self.row_gs, self.col_gs_list, self.true_ds_mat, self.sim_or_dist = \
            gen_data(self.from_info, self.gen_info, self.model_data)
        assert (self.true_ds_mat.shape[0] ==
                len(self.row_gs) == len(self.col_gs_list))
        for col_gs_for_gi in self.col_gs_list:
            assert (self.true_ds_mat.shape[1] == len(col_gs_for_gi))
        # Check sim/dist consistency.
        if self.sim_or_dist == 'sim':
            assert (FLAGS.ds_metric == 'mcs')
        else:
            assert (self.sim_or_dist == 'dist')
            assert (FLAGS.ds_metric == 'ged')
        self.row_gs = self.model_data.create_model_gs(self.row_gs, 'row_gs')
        for i, col_gs_for_gi in enumerate(self.col_gs_list):
            self.col_gs_list[i] = \
                self.model_data.create_model_gs(
                    col_gs_for_gi, 'col_gs_for_g{}'.format(i))


class ModelGraph(object):
    """Defines all relevant graph properties required for training a Siamese model.

    This is a data model representation of a graph
    for use during the training stage.
    Each ModelGraph has precomputed parameters
    (Laplacian, inputs, adj matrix, etc) as needed by the
    network during training.
    """

    def __init__(self, nxgraph, node_feat_encoder, graph_label_encoder):
        self.glabel_position = graph_label_encoder.encode(nxgraph)
        # Check flag compatibility.
        self._error_if_incompatible_flags()

        self.nxgraph = nxgraph
        last_order = []  # Overriden by supersource flag if used.

        # Overrides nxgraph to DiGraph if supersource needed.
        # Also edits the node_feat_encoder
        # because we add a supersource type for labeled graphs,
        # which means that the encoder
        # needs to be aware of a new one-hot encoding.
        if FLAGS.supersource:
            nxgraph, supersource_id, supersource_label = generate_supersource(
                nxgraph, FLAGS.node_feat_name)
            # If we have a labeled graph,
            # we need to add the supersource's new label to the encoder
            # so it properly one-hot encodes it.
            if FLAGS.node_feat_name:
                node_feat_encoder.add_new_feature(supersource_label)
            self.nxgraph = nxgraph
            last_order = [supersource_id]

        # Generates random walks with parameters determined by the flags.
        # Walks are defined
        # by the ground truth node ids, so they do not depend on ordering,
        # but if a supersource
        # node is used it should be generated before the random walk.
        if FLAGS.random_walk:
            if FLAGS.supersource and type(nxgraph) != nx.DiGraph:
                raise RuntimeError(
                    'The input graph must be a DiGraph '
                    'in order to use random walks with '
                    'a supersource node so it is not used as a shortcut')
            params = FLAGS.random_walk.split('_')
            num_walks = int(params[0])
            walk_length = int(params[1])
            self.random_walk_data = generate_random_walks(nxgraph, num_walks,
                                                          walk_length)

        # Encode features.
        dense_node_inputs = node_feat_encoder.encode(nxgraph)

        # Determine ordering and reorder the dense inputs
        # based on the desired ordering.
        if FLAGS.ordering:
            if FLAGS.ordering == 'bfs':
                self.order, self.mapping = node_ordering(
                    nxgraph, 'bfs', FLAGS.node_feat_name, last_order)

            elif FLAGS.ordering == 'degree':
                self.order, self.mapping = node_ordering(
                    nxgraph, 'degree', FLAGS.node_feat_name, last_order)
            else:
                raise RuntimeError('Unknown ordering mode {}'.format(self.order))
            assert (len(self.order) == len(nxgraph.nodes()))
            # Apply the ordering.
            dense_node_inputs = dense_node_inputs[self.order, :]

        # Save matrix properties after reordering the nodes.
        self.sparse_node_inputs = self._preprocess_inputs(
            sp.csr_matrix(dense_node_inputs))
        # Only one laplacian.
        self.num_laplacians = 1
        adj = nx.to_numpy_matrix(nxgraph)
        # Fix ordering for adj.
        if FLAGS.ordering:
            # Reorders the adj matrix using the order provided earlier.
            adj = adj[np.ix_(self.order, self.order)]

        # Special handling for coarsening because it is
        # incompatible with other flags.
        if FLAGS.coarsening:
            self._coarsen(dense_node_inputs, adj)
        else:
            self.laplacians = [self._preprocess_adj(adj)]

    def _error_if_incompatible_flags(self):
        """Check flags and error for unhandled flag combinations.
            FLAGS.coarsening
            FLAGS.ordering
            FLAGS.supersource
            FLAGS.random_walk
        """
        if FLAGS.coarsening:
            if FLAGS.ordering or FLAGS.supersource or FLAGS.random_walk:
                raise RuntimeError(
                    'Cannot use coarsening with any of the following: ordering, '
                    'supersource, random_walk')
        else:
            if FLAGS.supersource and FLAGS.fake_generation:
                raise RuntimeError(
                    'Cannot use supersource with fake generation right now because'
                    ' fake_generation doesnt support digraphs and labeled graphs '
                    'will break since the supersource_type could be duplicated')

    def get_nxgraph(self):
        return self.nxgraph

    def get_node_inputs(self):
        if FLAGS.coarsening:
            return self.sparse_permuted_padded_dense_node_inputs
        else:
            return self.sparse_node_inputs

    def get_node_inputs_num_nonzero(self):
        return self.get_node_inputs()[1].shape

    def get_laplacians(self, gcn_id):
        if FLAGS.coarsening:
            return self.coarsened_laplacians[gcn_id]
        else:
            return self.laplacians

    def _coarsen(self, dense_node_inputs, adj):
        assert ('metis_' in FLAGS.coarsening)
        self.num_level = get_coarsen_level()
        assert (self.num_level >= 1)
        graphs, perm = coarsen(sp.csr_matrix(adj), levels=self.num_level,
                               self_connections=False)
        permuted_padded_dense_node_inputs = perm_data(
            dense_node_inputs.T, perm).T
        self.sparse_permuted_padded_dense_node_inputs = self._preprocess_inputs(
            sp.csr_matrix(permuted_padded_dense_node_inputs))
        self.coarsened_laplacians = []
        for g in graphs:
            self.coarsened_laplacians.append([self._preprocess_adj(g.todense())])
        assert (len(self.coarsened_laplacians) == self.num_laplacians * self.num_level + 1)

    def _preprocess_inputs(self, inputs):
        """Row-normalize feature matrix and convert to tuple representation"""
        rowsum = np.array(inputs.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        inputs = r_mat_inv.dot(inputs)
        return self._sparse_to_tuple(inputs)

    def _preprocess_adj(self, adj):
        """Preprocessing of adjacency matrix and conversion to tuple representation."""
        adj_normalized = self._normalize_adj(adj + sp.eye(adj.shape[0]))
        return self._sparse_to_tuple(adj_normalized)

    def _normalize_adj(self, adj):
        """Symmetrically normalize adjacency matrix."""
        adj = sp.coo_matrix(adj)
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

    def _sparse_to_tuple(self, sparse_mx):
        """Convert sparse matrix to tuple representation."""

        def to_tuple(mx):
            if not sp.isspmatrix_coo(mx):
                mx = mx.tocoo()
            coords = np.vstack((mx.row, mx.col)).transpose()
            values = mx.data
            shape = mx.shape
            return coords, values, shape

        if isinstance(sparse_mx, list):
            for i in range(len(sparse_mx)):
                sparse_mx[i] = to_tuple(sparse_mx[i])
        else:
            sparse_mx = to_tuple(sparse_mx)

        return sparse_mx


class NodeFeatureEncoder(object):
    def encode(self, g):
        raise NotImplementedError()

    def input_dim(self):
        raise NotImplementedError()


class NodeFeatureOneHotEncoder(NodeFeatureEncoder):
    def __init__(self, gs, node_feat_name):
        self.node_feat_name = node_feat_name
        # Go through all the graphs in the entire dataset
        # and create a set of all possible
        # labels so we can one-hot encode them.
        inputs_set = set()
        for g in gs:
            inputs_set = inputs_set | set(self._node_feat_dic(g).values())
        self.feat_idx_dic = {feat: idx for idx, feat in enumerate(inputs_set)}
        self._fit_onehotencoder()

    def _fit_onehotencoder(self):
        self.oe = OneHotEncoder().fit(
            np.array(list(self.feat_idx_dic.values())).reshape(-1, 1))

    def add_new_feature(self, feat_name):
        """Use this function if a new feature was added to the graph set."""
        # Add the new feature to the dictionary
        # as a unique feature and reinit the encoder.
        new_idx = len(self.feat_idx_dic)
        self.feat_idx_dic[feat_name] = new_idx
        self._fit_onehotencoder()

    def encode(self, g):
        node_feat_dic = self._node_feat_dic(g)
        temp = [self.feat_idx_dic[node_feat_dic[n]] for n in g.nodes()]
        return self.oe.transform(np.array(temp).reshape(-1, 1)).toarray()

    def input_dim(self):
        return self.oe.transform([[0]]).shape[1]

    def _node_feat_dic(self, g):
        return nx.get_node_attributes(g, self.node_feat_name)


class NodeFeatureConstantEncoder(NodeFeatureEncoder):
    def __init__(self, _, node_feat_name):
        self.input_dim_ = int(FLAGS.node_feat_encoder.split('_')[1])
        self.const = float(2.0)
        assert (node_feat_name is None)

    def encode(self, g):
        rtn = np.full((g.number_of_nodes(), self.input_dim_), self.const)
        return rtn

    def input_dim(self):
        return self.input_dim_


class GraphLabelOneHotEncoder(object):
    def __init__(self, gs):
        self.glabel_map = {}
        for g in gs:
            glabel = g.graph['glabel']
            if glabel not in self.glabel_map:
                self.glabel_map[glabel] = len(self.glabel_map)

    def encode(self, g):
        return self.glabel_map[g.graph['glabel']]
