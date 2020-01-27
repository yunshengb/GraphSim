import tensorflow as tf

# Hyper-parameters.
flags = tf.app.flags

# For data preprocessing.
""" dataset: aids80nef, aids700nef, linux, imdbmulti, ptc, reddit10k. """
dataset = 'aids80nef'
dataset_train = dataset
dataset_val_test = dataset
if 'aids' in dataset or dataset in ['alkane', 'webeasy', 'nci109', 'ptc', 'mutag']:
    node_feat_name = 'type'
    node_feat_encoder = 'onehot'
    max_nodes = 10
    num_glabels = 2
    if dataset == 'ptc':
        max_nodes = 109
elif dataset in ['linux'] or 'imdb' in dataset:
    node_feat_name = None
    node_feat_encoder = 'constant_1'
    if dataset == 'linux':
        max_nodes = 10
        num_glabels = 2
    else:
        assert ('imdb' in dataset)
        max_nodes = 90
        num_glabels = 3
else:
    assert (False)
if 'imdbmulti' in [dataset_train, dataset_val_test]:
    max_nodes = 90
flags.DEFINE_string('dataset_train', dataset_train, 'Dataset for training.')
flags.DEFINE_string('dataset_val_test', dataset_val_test, 'Dataset for testing.')
flags.DEFINE_integer('num_glabels', num_glabels, 'Number of graph labels in the dataset.')
flags.DEFINE_string('node_feat_name', node_feat_name, 'Name of the node feature.')
flags.DEFINE_string('node_feat_encoder', node_feat_encoder,
                    'How to encode the node feature.')
""" valid_percentage: (0, 1). """
flags.DEFINE_float('valid_percentage', 0.25,
                   '(# validation graphs) / (# validation + # training graphs.')
""" ds_metric: ged, glet, mcs. """
ds_metric = 'ged'
flags.DEFINE_string('ds_metric', ds_metric, 'Distance/Similarity metric to use.')
""" ds_algo: beam80, astar for ged, graphlet for glet, mccreesh2017 for mcs. """
ds_algo = 'astar' if ds_metric == 'ged' else 'mccreesh2017'
flags.DEFINE_string('ds_algo', ds_algo,
                    'Ground-truth distance algorithm to use.')
""" ordering: 'bfs', 'degree', None. """
flags.DEFINE_string('ordering', None, '')
""" coarsening: 'metis_<num_level>', None. """
flags.DEFINE_string('coarsening', None, 'Algorithm for graph coarsening.')

# For model.
""" model: 'siamese_regression'. """
model = 'siamese_regression'
flags.DEFINE_string('model', model, 'Model string.')
""" model_name: 'SimGNN', 'GSimCNN', None. """
flags.DEFINE_string('model_name', 'Our Model', 'Model name string.')
flags.DEFINE_integer('batch_size', 1, 'Number of graph pairs in a batch.')
flags.DEFINE_boolean('ds_norm', True,
                     'Whether to normalize the distance or not '
                     'when choosing the ground truth distance.')
flags.DEFINE_boolean('node_embs_norm', False,
                     'Whether to normalize the node embeddings or not.')
pred_sim_dist, supply_sim_dist = None, None
if model in ['siamese_regression']:
    """ ds_kernel: gaussian, exp, inverse, identity. """
    ds_kernel = 'exp'
    if ds_metric == 'glet':  # already a sim metric
        ds_kernel = 'identity'
    flags.DEFINE_string('ds_kernel', ds_kernel,
                        'Name of the similarity kernel.')
    if ds_kernel == 'gaussian':
        """ yeta:
         if ds_norm, try 0.6 for nef small, 0.3 for nef, 0.2 for regular;
         else, try 0.01 for nef, 0.001 for regular. """
        flags.DEFINE_float('yeta', 0.01, 'yeta for the gaussian kernel function.')
    elif ds_kernel == 'exp' or ds_kernel == 'inverse':
        flags.DEFINE_float('scale', 0.7, 'Scale for the exp/inverse kernel function.')
    pred_sim_dist = 'sim'
    if ds_metric == 'mcs':
        pred_sim_dist = 'sim'  # cannot support it for now
    supply_sim_dist = pred_sim_dist
    # Start of mse loss.
    lambda_msel = 1  # 1  # 1 #0.0001
    if lambda_msel > 0:
        flags.DEFINE_float('lambda_mse_loss', lambda_msel,
                           'Lambda for the mse loss.')
    # End of mse loss.
flags.DEFINE_string('pred_sim_dist', pred_sim_dist,
                    'dist/sim indicating whether the model is predicting dist or sim.')
flags.DEFINE_string('supply_sim_dist', supply_sim_dist,
                    'dist/sim indicating whether the model should supply dist or sim.')
layer = 0
layer += 1
if model in ['siamese_regression']:
    # '''
    # # --------------------------------- MNE+CNN ---------------------------------
    flags.DEFINE_string(
        'layer_{}'.format(layer),
        'GraphConvolution:output_dim=128,dropout=False,bias=True,'
        'act=relu,sparse_inputs=True', '')
    layer += 1
    # flags.DEFINE_string(
    #     'layer_{}'.format(layer),
    #     'GraphConvolution:input_dim=128,output_dim=128,dropout=False,bias=True,'
    #     'act=relu,sparse_inputs=False', '')
    # layer += 1
    flags.DEFINE_string(
        'layer_{}'.format(layer),
        'GraphConvolution:input_dim=128,output_dim=64,dropout=False,bias=True,'
        'act=relu,sparse_inputs=False', '')
    layer += 1
    flags.DEFINE_string(
        'layer_{}'.format(layer),
        'GraphConvolution:input_dim=64,output_dim=32,dropout=False,bias=True,'
        'act=identity,sparse_inputs=False', '')
    layer += 1
    # flags.DEFINE_string(
    #     'layer_{}'.format(layer),
    #     'GraphConvolution:input_dim=32,output_dim=32,dropout=False,bias=True,'
    #     'act=identity,sparse_inputs=False', '')
    # layer += 1
    # flags.DEFINE_string(
    #     'layer_{}'.format(layer),
    #     'GraphConvolution:input_dim=32,output_dim=32,dropout=False,bias=True,'
    #     'act=identity,sparse_inputs=False', '')
    # layer += 1
    # flags.DEFINE_string(
    #     'layer_6',
    #     'GraphConvolution:input_dim=32,output_dim=32,dropout=False,bias=True,'
    #     'act=relu,sparse_inputs=False', '')
    # flags.DEFINE_string(
    #     'layer_7',
    #     'GraphConvolution:input_dim=32,output_dim=32,dropout=False,bias=True,'
    #     'act=identity,sparse_inputs=False', '')
    # flags.DEFINE_string(
    #     'layer_4',
    #     'GraphConvolution:input_dim=32,output_dim=32,dropout=False,bias=True,'
    #     'act=identity,sparse_inputs=False', '')
    #  *************************** GraphConvolutionCollector Layer ***************************
    gcn_num = 3
    mode = 'separate'  # merge, separate (better)
    flags.DEFINE_string(
        'layer_{}'.format(layer),
        'GraphConvolutionCollector:gcn_num={},'
        'fix_size=54,mode=0,padding_value=0,align_corners=True'.format(gcn_num), '')
    layer += 1
    # --------------------------- MNEResize Layer ---------------------------
    # Fix_Size: 90 imdbmulti, 10 others
    # Mode: 0 Bilinear interpolation, 1 Nearest neighbor interpolation,
    #       2 Bicubic interpolation, 3 Area interpolation
    # flags.DEFINE_string(
    #     'layer_{}'.format(layer),
    #     'MNEResize:fix_size=10,mode=0,dropout=False,inneract=identity,'
    #     'padding_value=0,align_corners=True', '')
    # layer += 1
    # flags.DEFINE_string(
    #     'layer_{}'.format(layer),
    #     'PadandTruncate:padding_value=0', '')
    # layer += 1
    # flags.DEFINE_string(
    #     'layer_{}'.format(layer),
    #     'MNE:input_dim=32,dropout=False,inneract=identity', '')
    # layer += 1
    # flags.DEFINE_string(
    #     'layer_{}'.format(layer),
    #     'CNN:start_cnn=True,end_cnn=False,window_size=200,kernel_stride=1,'
    #     'in_channel=1,out_channel=16,'
    #     'padding=SAME,pool_size=3,dropout=False,act=relu,bias=True,'
    #     'mode={},gcn_num={}'.format(mode, gcn_num), '')
    # layer += 1
    flags.DEFINE_string(
        'layer_{}'.format(layer),
        'CNN:start_cnn=True,end_cnn=False,window_size=25,kernel_stride=1,'
        'in_channel=1,out_channel=16,'
        'padding=SAME,pool_size=3,dropout=False,act=relu,bias=True,'
        'mode={},gcn_num={}'.format(mode, gcn_num), '')
    layer += 1
    flags.DEFINE_string(
        'layer_{}'.format(layer),
        'CNN:start_cnn=False,end_cnn=False,window_size=10,kernel_stride=1,'
        'in_channel=16,out_channel=32,'
        'padding=SAME,pool_size=3,dropout=False,act=relu,bias=True,'
        'mode={},gcn_num={}'.format(mode, gcn_num), '')
    layer += 1
    flags.DEFINE_string(
        'layer_{}'.format(layer),
        'CNN:start_cnn=False,end_cnn=False,window_size=4,kernel_stride=1,'
        'in_channel=32,out_channel=64,'
        'padding=SAME,pool_size=3,dropout=False,act=relu,bias=True,'
        'mode={},gcn_num={}'.format(mode, gcn_num), '')
    layer += 1
    flags.DEFINE_string(
        'layer_{}'.format(layer),
        'CNN:start_cnn=False,end_cnn=True,window_size=2,kernel_stride=1,'
        'in_channel=64,out_channel=128,'
        'padding=SAME,pool_size=2,dropout=False,act=relu,bias=True,'
        'mode={},gcn_num={}'.format(mode, gcn_num), '')
    layer += 1
    # flags.DEFINE_string(
    #     'layer_{}'.format(layer),
    #     'CNN:start_cnn=False,end_cnn=False,window_size=5,kernel_stride=1,'
    #     'in_channel=128,out_channel=128,'
    #     'padding=SAME,pool_size=3,dropout=False,act=relu,bias=True,'
    #     'mode={},gcn_num={}'.format(mode, gcn_num), '')
    # layer += 1
    # #
    # flags.DEFINE_string(
    #     'layer_{}'.format(layer),
    #     'CNN:start_cnn=False,end_cnn=False,window_size=5,kernel_stride=1,'
    #     'in_channel=128,out_channel=128,'
    #     'padding=SAME,pool_size=3,dropout=False,act=relu,bias=True,'
    #     'mode={},gcn_num={}'.format(mode, gcn_num), '')
    # layer += 1
    #
    # flags.DEFINE_string(
    #     'layer_{}'.format(layer),
    #     'CNN:start_cnn=False,end_cnn=False,window_size=5,kernel_stride=1,'
    #     'in_channel=128,out_channel=128,'
    #     'padding=SAME,pool_size=3,dropout=False,act=relu,bias=True,'
    #     'mode={},gcn_num={}'.format(mode, gcn_num), '')
    # layer += 1
    # #
    # flags.DEFINE_string(
    #     'layer_{}'.format(layer),
    #     'CNN:start_cnn=False,end_cnn=True,window_size=5,kernel_stride=1,'
    #     'in_channel=128,out_channel=128,'
    #     'padding=SAME,pool_size=3,dropout=False,act=relu,bias=True,'
    #     'mode={},gcn_num={}'.format(mode, gcn_num), '')
    # layer += 1

    # flags.DEFINE_string(
    #     'layer_{}'.format(layer),
    #     'CNN:start_cnn=False,end_cnn=True,window_size=5,kernel_stride=1,'
    #     'in_channel=128,out_channel=128,'
    #     'padding=SAME,pool_size=3,dropout=False,act=relu,bias=True,'
    #     'mode={},gcn_num={}'.format(mode, gcn_num), '')
    # layer += 1
    #
    # flags.DEFINE_string(
    #     'layer_{}'.format(layer),
    #     'CNN:start_cnn=False,end_cnn=True,window_size=5,kernel_stride=1,in_channel=128,out_channel=128,'
    #     'padding=SAME,pool_size=3,dropout=False,act=relu,bias=True,mode=separate,gcn_num=3', '')
    # layer += 1

    # flags.DEFINE_string(
    #     'layer_{}'.format(layer),
    #     'Dense:input_dim=640,output_dim=512,dropout=False,'
    #     'act=relu,bias=True', '')
    # layer += 1

    flags.DEFINE_string(
        'layer_{}'.format(layer),
        'Dense:input_dim=384,output_dim=256,dropout=False,'
        'act=relu,bias=True', '')
    layer += 1

    flags.DEFINE_string(
        'layer_{}'.format(layer),
        'Dense:input_dim=256,output_dim=128,dropout=False,'
        'act=relu,bias=True', '')
    layer += 1
    flags.DEFINE_string(
        'layer_{}'.format(layer),
        'Dense:input_dim=128,output_dim=64,dropout=False,'
        'act=relu,bias=True', '')
    layer += 1
    flags.DEFINE_string(
        'layer_{}'.format(layer),
        'Dense:input_dim=64,output_dim=32,dropout=False,'
        'act=relu,bias=True', '')
    layer += 1
    flags.DEFINE_string(
        'layer_{}'.format(layer),
        'Dense:input_dim=32,output_dim=16,dropout=False,'
        'act=relu,bias=True', '')
    layer += 1
    flags.DEFINE_string(
        'layer_{}'.format(layer),
        'Dense:input_dim=16,output_dim=8,dropout=False,'
        'act=relu,bias=True', '')
    layer += 1
    flags.DEFINE_string(
        'layer_{}'.format(layer),
        'Dense:input_dim=8,output_dim=4,dropout=False,'
        'act=relu,bias=True', '')
    layer += 1
    flags.DEFINE_string(
        'layer_{}'.format(layer),
        'Dense:input_dim=4,output_dim=1,dropout=False,'
        'act=identity,bias=True', '')
    # '''
    flags.DEFINE_integer('layer_num', layer, 'Number of layers.')
# Start of graph loss.

# Generater and permutater.
fake = None
flags.DEFINE_string('fake_generation', fake,
                    'Whether to generate fake graphs for all graphs or not. '
                    '1st represents top num, and 2nd represents fake num.')
# Supersource node.
# Referenced as "super node" in https://arxiv.org/pdf/1511.05493.pdf. Node that is connected
# to all other nodes in the graph.
flags.DEFINE_boolean('supersource', False,
                     'Boolean. Whether or not to use a supersouce node in all of the graphs.')
# Random walk generation and usage.
# As used in the GraphSAGE model implementation: https://github.com/williamleif/GraphSAGE.
flags.DEFINE_string('random_walk', None,
                    'Random walk configuration. Set none to not use random walks. Format is: '
                    '<num_walks>_<walk_length>')

# Training (optimiztion) details.
flags.DEFINE_float('dropout', 0, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 0,
                   'Weight for L2 loss on embedding matrix.')
""" learning_rate: 0.01 recommended. """
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')

# For training and validating.
flags.DEFINE_integer('gpu', -1, 'Which gpu to use.')  # -1: cpu
flags.DEFINE_integer('iters', 10, 'Number of iterations to train.')
flags.DEFINE_integer('iters_val_start', 5,
                     'Number of iterations to start validation.')
flags.DEFINE_integer('iters_val_every', 2, 'Frequency of validation.')


# For testing.
flags.DEFINE_boolean('plot_results', True,
                     'Whether to plot the results '
                     '(involving all baselines) or not.')
flags.DEFINE_integer('plot_max_num', 10, 'Max number of plots per experiment.')
flags.DEFINE_integer('max_nodes', max_nodes, 'Maximum number of nodes in a graph.')

FLAGS = tf.app.flags.FLAGS
