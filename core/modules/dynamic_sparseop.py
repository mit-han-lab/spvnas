import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsparse.nn.functional as spf
from torchsparse import SparseTensor

__all__ = ['make_divisible', 'SparseDynamicConv3d', 'SparseDynamicBatchNorm']


def make_divisible(x):
    return int((x // 4) * 4)


# TBD: kernel_size = 1 special case.
class SparseDynamicConv3d(nn.Module):

    def __init__(self,
                 inc,
                 outc,
                 kernel_size=3,
                 stride=1,
                 dilation=1,
                 transposed=False):
        super().__init__()
        self.inc = inc
        self.outc = outc
        self.ks = kernel_size
        self.k = self.ks ** 3
        self.s = stride
        self.d = dilation
        self.kernel = nn.Parameter(torch.zeros(
            self.k, inc, outc)) if self.k > 1 else nn.Parameter(
                torch.zeros(inc, outc))
        self.t = transposed
        self.init_weight()
        self.runtime_outc = None
        self.runtime_inc = None
        self.runtime_inc_constraint = None

        if kernel_size == 1:
            assert not transposed

    def extra_repr(self) -> str:
        return 'imax={}, omax={}, s={}, d={}, t={}'.format(
            self.inc, self.outc, self.s, self.d, self.t)

    def init_weight(self):
        std = 1. / math.sqrt(self.outc if self.t else self.inc * self.k)
        self.kernel.data.uniform_(-std, std)

    def set_in_channel(self, in_channel=None, constraint=None):
        if in_channel is not None:
            self.runtime_inc = in_channel
        elif constraint is not None:
            self.runtime_inc_constraint = torch.from_numpy(
                np.array(constraint)).long()
        else:
            raise NotImplementedError

    def set_output_channel(self, channel):
        self.runtime_outc = channel

    def forward(self, inputs):
        cur_kernel = self.kernel
        if self.runtime_inc_constraint is not None:
            if self.ks > 1:
                cur_kernel = cur_kernel[:, self.runtime_inc_constraint, :]
            else:
                cur_kernel = cur_kernel[self.runtime_inc_constraint]
        elif self.runtime_inc is not None:
            if self.ks > 1:
                cur_kernel = cur_kernel[:, torch.arange(self.runtime_inc), :]
            else:
                cur_kernel = cur_kernel[torch.arange(self.runtime_inc)]
        else:
            raise ValueError('Number of channels not specified!')
        cur_kernel = cur_kernel[..., torch.arange(self.runtime_outc)]

        return spf.conv3d(inputs,
                          cur_kernel,
                          self.ks,
                          stride=self.s,
                          dilation=self.d,
                          transposed=self.t)


class SparseDynamicBatchNorm(nn.Module):
    SET_RUNNING_STATISTICS = False

    def __init__(self, c, cr_bounds=(0.25, 1.0), eps=1e-5, momentum=0.1):
        super().__init__()
        self.c = c
        self.eps = eps
        self.momentum = momentum
        self.cr_bounds = cr_bounds
        self.bn = nn.BatchNorm1d(c, eps=eps, momentum=momentum)
        self.channels = []
        self.runtime_channel = None

    def __repr__(self):
        return 'SparseDynamicBatchNorm(cmax=%d)' % self.c

    def set_channel(self, channel):
        self.runtime_channel = channel

    def bn_foward(self, x, bn, feature_dim):
        if (bn.num_features == feature_dim
                or SparseDynamicBatchNorm.SET_RUNNING_STATISTICS):
            return bn(x)
        else:
            exponential_average_factor = 0.0

            if bn.training and bn.track_running_stats:
                if bn.num_batches_tracked is not None:
                    bn.num_batches_tracked += 1
                    if bn.momentum is None:  # use cumulative moving average
                        exponential_average_factor = 1.0 / float(
                            bn.num_batches_tracked)
                    else:  # use exponential moving average
                        exponential_average_factor = bn.momentum
            return F.batch_norm(
                x,
                bn.running_mean[:feature_dim],
                bn.running_var[:feature_dim],
                bn.weight[:feature_dim],
                bn.bias[:feature_dim],
                bn.training or not bn.track_running_stats,
                exponential_average_factor,
                bn.eps,
            )

    def forward(self, inputs):
        output_features = self.bn_foward(inputs.F, self.bn, inputs.F.shape[-1])
        output_tensor = SparseTensor(output_features, inputs.C, inputs.s)
        output_tensor.cmaps = inputs.cmaps
        output_tensor.kmaps = inputs.kmaps

        return output_tensor
