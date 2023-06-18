import numpy as np
import torch
import torchsparse
import torchsparse.nn.functional as spf
from torchsparse import SparseTensor
from torchsparse.nn.functional.devoxelize import calc_ti_weights
from torchsparse.nn.utils import *
from torchsparse.utils import *
from torchsparse.utils.tensor_cache import TensorCache
import torch_scatter
from typing import Union, Tuple

__all__ = ["initial_voxelize", "point_to_voxel", "voxel_to_point", "PointTensor"]


class PointTensor(SparseTensor):
    def __init__(
        self,
        feats: torch.Tensor,
        coords: torch.Tensor,
        stride: Union[int, Tuple[int, ...]] = 1,
    ):
        super().__init__(feats=feats, coords=coords, stride=stride)
        self._caches.idx_query = dict()
        self._caches.idx_query_devox = dict()
        self._caches.weights_devox = dict()


def sphashquery(query, target, kernel_size=1):
    hashmap_keys = torch.zeros(
        2 * target.shape[0], dtype=torch.int64, device=target.device
    )
    hashmap_vals = torch.zeros(
        2 * target.shape[0], dtype=torch.int32, device=target.device
    )
    hashmap = torchsparse.backend.GPUHashTable(hashmap_keys, hashmap_vals)
    hashmap.insert_coords(target[:, [1, 2, 3, 0]])
    kernel_size = make_ntuple(kernel_size, 3)
    kernel_volume = np.prod(kernel_size)
    kernel_size = make_tensor(kernel_size, device=target.device, dtype=torch.int32)
    stride = make_tensor((1, 1, 1), device=target.device, dtype=torch.int32)
    results = (
        hashmap.lookup_coords(
            query[:, [1, 2, 3, 0]], kernel_size, stride, kernel_volume
        )
        - 1
    )[: query.shape[0]]
    return results


# z: PointTensor
# return: SparseTensor
def initial_voxelize(z, init_res, after_res):
    new_float_coord = torch.cat(
        [z.C[:, 0].view(-1, 1), (z.C[:, 1:] * init_res) / after_res], 1
    )
    # optimization TBD: init_res = after_res
    new_int_coord = torch.floor(new_float_coord).int()
    sparse_coord = torch.unique(new_int_coord, dim=0)
    idx_query = sphashquery(new_int_coord, sparse_coord).reshape(-1)

    sparse_feat = torch_scatter.scatter_mean(z.F, idx_query.long(), dim=0)
    new_tensor = SparseTensor(sparse_feat, sparse_coord, 1)
    z._caches.idx_query[z.s] = idx_query
    z.C = new_float_coord
    return new_tensor


# x: SparseTensor, z: PointTensor
# return: SparseTensor
def point_to_voxel(x, z):
    if z._caches.idx_query.get(x.s) is None:
        # Note: x.C has a smaller range after downsampling.
        new_int_coord = torch.cat(
            [
                z.C[:, 0].int().view(-1, 1),
                torch.floor(z.C[:, 1:] / x.s[0]).int(),
            ],
            1,
        )
        idx_query = sphashquery(new_int_coord, x.C)
        z._caches.idx_query[x.s] = idx_query
    else:
        idx_query = z._caches.idx_query[x.s]
    # Haotian: This impl. is not elegant
    idx_query = idx_query.clamp_(0)
    sparse_feat = torch_scatter.scatter_mean(z.F, idx_query.long(), dim=0)
    new_tensor = SparseTensor(sparse_feat, x.C, x.s)
    new_tensor._caches = x._caches

    return new_tensor


# x: SparseTensor, z: PointTensor
# return: PointTensor
def voxel_to_point(x, z, nearest=False):
    if (
        z._caches.idx_query_devox.get(x.s) is None
        or z._caches.weights_devox.get(x.s) is None
    ):
        point_coords_float = torch.cat(
            [z.C[:, 0].int().view(-1, 1), z.C[:, 1:] / x.s[0]],
            1,
        )
        point_coords_int = torch.floor(point_coords_float).int()
        idx_query = sphashquery(point_coords_int, x.C, kernel_size=2)
        weights = calc_ti_weights(point_coords_float[:, 1:], idx_query, scale=1)

        if nearest:
            weights[:, 1:] = 0.0
            idx_query[:, 1:] = -1
        new_feat = spf.spdevoxelize(x.F, idx_query, weights)
        new_tensor = PointTensor(new_feat, z.C)
        new_tensor._caches = z._caches
        new_tensor._caches.idx_query_devox[x.s] = idx_query
        new_tensor._caches.weights_devox[x.s] = weights
        z._caches.idx_query_devox[x.s] = idx_query
        z._caches.weights_devox[x.s] = weights

    else:
        new_feat = spf.spdevoxelize(
            x.F, z._caches.idx_query_devox.get(x.s), z._caches.weights_devox.get(x.s)
        )
        new_tensor = PointTensor(new_feat, z.C)
        new_tensor._caches = z._caches

    return new_tensor
