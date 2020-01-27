from config import FLAGS
from layers import GraphConvolution, GraphConvolutionAttention, Coarsening, Average, \
    Attention, Dot, Dist, NTN, SLM, Dense, Padding, MNE, MNEMatch, CNN, ANPM, ANPMD, ANNH, \
    GraphConvolutionCollector, MNEResize, PadandTruncate, Supersource
import tensorflow as tf
import numpy as np
from math import exp


def create_layers(model, pattern, num_layers):
    layers = []
    for i in range(1, num_layers + 1):  # 1-indexed
        sp = FLAGS.flag_values_dict()['{}_{}'.format(pattern, i)].split(':')
        name = sp[0]
        layer_info = {}
        if len(sp) > 1:
            assert (len(sp) == 2)
            for spec in sp[1].split(','):
                ssp = spec.split('=')
                layer_info[ssp[0]] = ssp[1]
        if name == 'GraphConvolution':
            layers.append(create_GraphConvolution_layer(layer_info, model, i))
        elif name == 'GraphConvolutionAttention':
            layers.append(create_GraphConvolutionAttention_layer(layer_info, model, i))
        elif name == 'GraphConvolutionCollector':
            layers.append(create_GraphConvolutionCollector_layer(layer_info))
        elif name == 'Coarsening':
            layers.append(create_Coarsening_layer(layer_info))
        elif name == 'Average':
            layers.append(create_Average_layer(layer_info))
        elif name == 'Attention':
            layers.append(create_Attention_layer(layer_info))
        elif name == 'Supersource':
            layers.append(create_Supersource_layer(layer_info))
        elif name == 'Dot':
            layers.append(create_Dot_layer(layer_info))
        elif name == 'Dist':
            layers.append(create_Dist_layer(layer_info))
        elif name == 'SLM':
            layers.append(create_SLM_layer(layer_info))
        elif name == 'NTN':
            layers.append(create_NTN_layer(layer_info))
        elif name == 'ANPM':
            layers.append(create_ANPM_layer(layer_info))
        elif name == 'ANPMD':
            layers.append(create_ANPMD_layer(layer_info))
        elif name == 'ANNH':
            layers.append(create_ANNH_layer(layer_info))
        elif name == 'Dense':
            layers.append(create_Dense_layer(layer_info))
        elif name == 'Padding':
            layers.append(create_Padding_layer(layer_info))
        elif name == 'PadandTruncate':
            layers.append(create_PadandTruncate_layer(layer_info))
        elif name == 'MNE':
            layers.append(create_MNE_layer(layer_info))
        elif name == 'MNEMatch':
            layers.append(create_MNEMatch_layer(layer_info))
        elif name == 'MNEResize':
            layers.append(create_MNEResize_layer(layer_info))
        elif name == 'CNN':
            layers.append(create_CNN_layer(layer_info))
        else:
            raise RuntimeError('Unknown layer {}'.format(name))
    return layers


def create_GraphConvolution_layer(layer_info, model, layer_id):
    if not 5 <= len(layer_info) <= 6:
        raise RuntimeError('GraphConvolution layer must have 3-4 specs')
    input_dim = layer_info.get('input_dim')
    if not input_dim:
        if layer_id != 1:
            raise RuntimeError(
                'The input dim for layer must be specified'.format(layer_id))
        input_dim = model.input_dim
    else:
        input_dim = int(input_dim)
    return GraphConvolution(
        input_dim=input_dim,
        output_dim=int(layer_info['output_dim']),
        dropout=parse_as_bool(layer_info['dropout']),
        sparse_inputs=parse_as_bool(layer_info['sparse_inputs']),
        act=create_activation(layer_info['act']),
        bias=parse_as_bool(layer_info['bias']),
        featureless=False,
        num_supports=1)


def create_GraphConvolutionAttention_layer(layer_info, model, layer_id):
    if not 5 <= len(layer_info) <= 6:
        raise RuntimeError('GraphConvolution layer must have 3-4 specs')
    input_dim = layer_info.get('input_dim')
    if not input_dim:
        if layer_id != 1:
            raise RuntimeError(
                'The input dim for layer must be specified'.format(layer_id))
        input_dim = model.input_dim
    else:
        input_dim = int(input_dim)
    return GraphConvolutionAttention(
        input_dim=input_dim,
        output_dim=int(layer_info['output_dim']),
        dropout=parse_as_bool(layer_info['dropout']),
        sparse_inputs=parse_as_bool(layer_info['sparse_inputs']),
        act=create_activation(layer_info['act']),
        bias=parse_as_bool(layer_info['bias']),
        featureless=False,
        num_supports=1)


def create_GraphConvolutionCollector_layer(layer_info):
    if not len(layer_info) == 5:
        raise RuntimeError('GraphConvolutionCollector layer must have 5 spec')
    return GraphConvolutionCollector(gcn_num=int(layer_info['gcn_num']),
                                     fix_size=int(layer_info['fix_size']),
                                     mode=int(layer_info['mode']),
                                     padding_value=int(layer_info['padding_value']),
                                     align_corners=parse_as_bool(layer_info['align_corners']))


def create_Coarsening_layer(layer_info):
    if not len(layer_info) == 1:
        raise RuntimeError('Coarsening layer must have 1 spec')
    return Coarsening(pool_style=layer_info['pool_style'])


def create_Average_layer(layer_info):
    if not len(layer_info) == 0:
        raise RuntimeError('Average layer must have 0 specs')
    return Average()


def create_Attention_layer(layer_info):
    if not len(layer_info) == 5:
        raise RuntimeError('Attention layer must have 5 specs')
    return Attention(input_dim=int(layer_info['input_dim']),
                     att_times=int(layer_info['att_times']),
                     att_num=int(layer_info['att_num']),
                     att_style=layer_info['att_style'],
                     att_weight=parse_as_bool(layer_info['att_weight']))


def create_Supersource_layer(layer_info):
    if not len(layer_info) == 0:
        raise RuntimeError('Supersource layer must have 0 specs')
    return Supersource()


def create_Dot_layer(layer_info):
    if not len(layer_info) == 2:
        raise RuntimeError('Dot layer must have 2 specs')
    return Dot(output_dim=int(layer_info['output_dim']),
               act=create_activation(layer_info['act']))


def create_Dist_layer(layer_info):
    if not len(layer_info) == 1:
        raise RuntimeError('Dot layer must have 1 specs')
    return Dist(norm=layer_info['norm'])


def create_SLM_layer(layer_info):
    if not len(layer_info) == 5:
        raise RuntimeError('SLM layer must have 5 specs')
    return SLM(
        input_dim=int(layer_info['input_dim']),
        output_dim=int(layer_info['output_dim']),
        act=create_activation(layer_info['act']),
        dropout=parse_as_bool(layer_info['dropout']),
        bias=parse_as_bool(layer_info['bias']))


def create_NTN_layer(layer_info):
    if not len(layer_info) == 6:
        raise RuntimeError('NTN layer must have 6 specs')
    return NTN(
        input_dim=int(layer_info['input_dim']),
        feature_map_dim=int(layer_info['feature_map_dim']),
        dropout=parse_as_bool(layer_info['dropout']),
        inneract=create_activation(layer_info['inneract']),
        apply_u=parse_as_bool(layer_info['apply_u']),
        bias=parse_as_bool(layer_info['bias']))


def create_ANPM_layer(layer_info):
    if not len(layer_info) == 14:
        raise RuntimeError('ANPM layer must have 14 specs')
    return ANPM(
        input_dim=int(layer_info['input_dim']),
        att_times=int(layer_info['att_times']),
        att_num=int(layer_info['att_num']),
        att_style=layer_info['att_style'],
        att_weight=parse_as_bool(layer_info['att_weight']),
        feature_map_dim=int(layer_info['feature_map_dim']),
        dropout=parse_as_bool(layer_info['dropout']),
        bias=parse_as_bool(layer_info['bias']),
        ntn_inneract=create_activation(layer_info['ntn_inneract']),
        apply_u=parse_as_bool(layer_info['apply_u']),
        padding_value=int(layer_info['padding_value']),
        mne_inneract=create_activation(layer_info['mne_inneract']),
        # num_bins=int(layer_info['num_bins'])
        mne_method=layer_info['mne_method'],
        branch_style=layer_info['branch_style'])


def create_ANPMD_layer(layer_info):
    if not len(layer_info) == 22:
        raise RuntimeError('ANPMD layer must have 22 specs')
    return ANPMD(
        input_dim=int(layer_info['input_dim']),
        att_times=int(layer_info['att_times']),
        att_num=int(layer_info['att_num']),
        att_style=layer_info['att_style'],
        att_weight=parse_as_bool(layer_info['att_weight']),
        feature_map_dim=int(layer_info['feature_map_dim']),
        dropout=parse_as_bool(layer_info['dropout']),
        bias=parse_as_bool(layer_info['bias']),
        ntn_inneract=create_activation(layer_info['ntn_inneract']),
        apply_u=parse_as_bool(layer_info['apply_u']),
        padding_value=int(layer_info['padding_value']),
        mne_inneract=create_activation(layer_info['mne_inneract']),
        mne_method=layer_info['mne_method'],
        branch_style=layer_info['branch_style'],
        dense1_dropout=parse_as_bool(layer_info['dense1_dropout']),
        dense1_act=create_activation(layer_info['dense1_act']),
        dense1_bias=parse_as_bool(layer_info['dense1_bias']),
        dense1_output_dim=int(layer_info['dense1_output_dim']),
        dense2_dropout=parse_as_bool(layer_info['dense2_dropout']),
        dense2_act=create_activation(layer_info['dense2_act']),
        dense2_bias=parse_as_bool(layer_info['dense2_bias']),
        dense2_output_dim=int(layer_info['dense2_output_dim']))


def create_ANNH_layer(layer_info):
    if not len(layer_info) == 14:
        raise RuntimeError('ANNH layer must have 14 specs')
    return ANNH(
        input_dim=int(layer_info['input_dim']),
        att_times=int(layer_info['att_times']),
        att_num=int(layer_info['att_num']),
        att_style=layer_info['att_style'],
        att_weight=parse_as_bool(layer_info['att_weight']),
        feature_map_dim=int(layer_info['feature_map_dim']),
        dropout=parse_as_bool(layer_info['dropout']),
        bias=parse_as_bool(layer_info['bias']),
        ntn_inneract=create_activation(layer_info['ntn_inneract']),
        apply_u=parse_as_bool(layer_info['apply_u']),
        padding_value=int(layer_info['padding_value']),
        mne_inneract=create_activation(layer_info['mne_inneract']),
        # num_bins=int(layer_info['num_bins'])
        mne_method=layer_info['mne_method'],
        branch_style=layer_info['branch_style'])


def create_Dense_layer(layer_info):
    if not len(layer_info) == 5:
        raise RuntimeError('Dense layer must have 5 specs')
    return Dense(
        input_dim=int(layer_info['input_dim']),
        output_dim=int(layer_info['output_dim']),
        dropout=parse_as_bool(layer_info['dropout']),
        act=create_activation(layer_info['act']),
        bias=parse_as_bool(layer_info['bias']))


def create_Padding_layer(layer_info):
    if not len(layer_info) == 1:
        raise RuntimeError('Padding layer must have 1 specs')
    return Padding(
        padding_value=int(layer_info['padding_value']))


def create_PadandTruncate_layer(layer_info):
    if not len(layer_info) == 1:
        raise RuntimeError('PadandTruncate layer must have 1 specs')
    return PadandTruncate(
        padding_value=int(layer_info['padding_value']))


def create_MNE_layer(layer_info):
    if not len(layer_info) == 3:
        raise RuntimeError('MNE layer must have 3 specs')
    return MNE(
        input_dim=int(layer_info['input_dim']),
        dropout=parse_as_bool(layer_info['dropout']),
        inneract=create_activation(layer_info['inneract']))


def create_MNEMatch_layer(layer_info):
    if not len(layer_info) == 2:
        raise RuntimeError('MNEMatch layer must have 2 specs')
    return MNEMatch(
        input_dim=int(layer_info['input_dim']),
        inneract=create_activation(layer_info['inneract']))


def create_MNEResize_layer(layer_info):
    if not len(layer_info) == 6:
        raise RuntimeError('MNEResize layer must have 6 specs')
    return MNEResize(
        dropout=parse_as_bool(layer_info['dropout']),
        inneract=create_activation(layer_info['inneract']),
        fix_size=int(layer_info['fix_size']),
        mode=int(layer_info['mode']),
        padding_value=int(layer_info['padding_value']),
        align_corners=parse_as_bool(layer_info['align_corners']))


def create_CNN_layer(layer_info):
    if not 11 <= len(layer_info) <= 13:
        raise RuntimeError('CNN layer must have 11-13 specs')
    gcn_num = layer_info.get('gcn_num')
    mode = layer_info.get('mode')
    if not gcn_num:
        if layer_info['mode'] != 'merge':
            raise RuntimeError('The gcn_num for layer must be specified')
        gcn_num = None
    else:
        gcn_num = int(gcn_num)
    mode = 'merge' if not mode else mode

    return CNN(
        start_cnn=parse_as_bool(layer_info['start_cnn']),
        end_cnn=parse_as_bool(layer_info['end_cnn']),
        window_size=int(layer_info['window_size']),
        kernel_stride=int(layer_info['kernel_stride']),
        in_channel=int(layer_info['in_channel']),
        out_channel=int(layer_info['out_channel']),
        padding=layer_info['padding'],
        pool_size=int(layer_info['pool_size']),
        dropout=parse_as_bool(layer_info['dropout']),
        act=create_activation(layer_info['act']),
        bias=parse_as_bool(layer_info['bias']),
        mode=mode,
        gcn_num=gcn_num)


def create_activation(act, ds_kernel=None, use_tf=True):
    if act == 'relu':
        return tf.nn.relu if use_tf else relu_np
    elif act == 'identity':
        return tf.identity if use_tf else identity_np
    elif act == 'sigmoid':
        return tf.sigmoid if use_tf else sigmoid_np
    elif act == 'tanh':
        return tf.tanh if use_tf else np.tanh
    elif act == 'ds_kernel':
        return ds_kernel.dist_to_sim_tf if use_tf else \
            ds_kernel.dist_to_sim_np
    else:
        raise RuntimeError('Unknown activation function {}'.format(act))


def relu_np(x):
    return np.maximum(x, 0)


def identity_np(x):
    return x


def sigmoid_np(x):
    try:
        ans = exp(-x)
    except OverflowError:  # TODO: fix
        ans = float('inf')
    return 1 / (1 + ans)


def parse_as_bool(b):
    if b == 'True':
        return True
    elif b == 'False':
        return False
    else:
        raise RuntimeError('Unknown bool string {}'.format(b))
