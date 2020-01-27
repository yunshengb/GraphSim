from utils import format_float
import numpy as np


class SimilarityKernel(object):
    def name(self):
        return self.__class__.__name__.lower()

    def shortname(self):
        return self.name()

    def name_suffix(self):
        return ''

    def dist_to_sim_np(self, dist, max_dist):
        raise NotImplementedError()

    def dist_to_sim_tf(self, dist, max_dist):
        raise NotImplementedError()

    def sim_to_dist_np(self, sim):
        raise NotImplementedError()


class IdentityKernel:
    def dist_to_sim_np(self, dist, *unused):
        return self._d_to_s(dist)

    def dist_to_sim_tf(self, dist, *unused):
        return self._d_to_s(dist)

    def _d_to_s(self, dist):
        return dist


# class IdentityKernel:
#     def dist_to_sim_np(self, dist, max_dist):
#         return self._d_to_s(dist, max_dist)
#
#     def dist_to_sim_tf(self, dist, max_dist):
#         return self._d_to_s(dist, max_dist)
#
#     def _d_to_s(self, dist, max_dist):
#         return 1 - dist / max_dist


class GaussianKernel(SimilarityKernel):
    def __init__(self, yeta):
        self.yeta = yeta

    def name(self):
        return 'Gaussian_yeta={}'.format(format_float(self.yeta))

    def shortname(self):
        return 'g_{:.2e}'.format(self.yeta)

    def dist_to_sim_np(self, dist, *unused):
        return np.exp(-self.yeta * np.square(dist))

    def dist_to_sim_tf(self, dist, *unuse):
        import tensorflow as tf
        return tf.exp(-self.yeta * tf.square(dist))

    def sim_to_dist_np(self, sim):
        return np.sqrt(-np.log(sim) / self.yeta)


class ExpKernel(SimilarityKernel):
    def __init__(self, scale):
        self.scale = scale

    def name(self):
        return 'Exp_scale={}'.format(format_float(self.scale))

    def shortname(self):
        return 'e_{:.2e}'.format(self.scale)

    def dist_to_sim_np(self, dist, *unused):
        return np.exp(-self.scale * dist)

    def dist_to_sim_tf(self, dist, *unuse):
        import tensorflow as tf
        return tf.exp(-self.scale * dist)

    def sim_to_dist_np(self, sim):
        # if sim == 0.0:
        #     # e.g. if dist = 748, then sim = 0.0, and log(0.0) leads to a warning
        #     rtn = np.inf
        # else:
        rtn = -np.log(sim) / self.scale
        return rtn


class InverseKernel(SimilarityKernel):
    def __init__(self, scale):
        self.scale = scale

    def name(self):
        return 'Inverse_scale={}'.format(format_float(self.scale))

    def shortname(self):
        return 'i_{:.2e}'.format(self.scale)

    def dist_to_sim_np(self, dist, *unused):
        return 1 / (self.scale * dist + 1)

    def dist_to_sim_tf(self, dist, *unuse):
        return 1 / (self.scale * dist + 1)

    def sim_to_dist_np(self, sim):
        return (1 / sim - 1) / self.scale


# class LinearKernel(SimilarityKernel):
#     def __init__(self, max_thresh, min_thresh):
#         self.threshold = threshold
#
#     # TODO: simplify ranking into binary classification


def create_ds_kernel(kernel_name, yeta=None, scale=None):
    if kernel_name == 'identity':
        return IdentityKernel()
    elif kernel_name == 'gaussian':
        return GaussianKernel(yeta)
    elif kernel_name == 'exp':
        return ExpKernel(scale)
    elif kernel_name == 'inverse':
        return InverseKernel(scale)
    else:
        raise RuntimeError('Unknown sim kernel {}'.format(kernel_name))
