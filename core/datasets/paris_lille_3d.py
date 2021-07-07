import os
import numpy as np
import glob
import torch

from torchsparse import SparseTensor
from torchsparse.utils import sparse_collate_fn, sparse_quantize

from plyfile import PlyData

__all__ = ['ParisLille3D']

label_name_mapping = {
    0: "unclassified",
    1: "ground",
    2: "building",
    3: "pole-road_sign-traffic_light",
    4: "bollard-small_pole",
    5: "trash_can",
    6: "barrier",
    7: "pedestrian",
    8: "car",
    9: "natural-vegetation"
}

kept_labels = [
    "ground", "building", "pole-road_sign-traffic_light", "bollard-small_pole",
    "trash_can", "barrier", "pedestrian", "car", "natural-vegetation"
]


class ParisLille3D(dict):
    def __init__(self, root, voxel_size, num_points, **kwargs):
        submit_to_server = kwargs.get('submit', False)
        if submit_to_server:
            super(ParisLille3D, self).__init__({
                'train':
                ParisLille3DInternal(root,
                                      voxel_size,
                                      num_points,
                                      split='train',
                                      submit=True),
                'test':
                ParisLille3DInternal(root,
                                      voxel_size,
                                      num_points,
                                      split='test')
            })
        else:
            super(ParisLille3D, self).__init__({
                'train':
                ParisLille3DInternal(root,
                                      voxel_size,
                                      num_points,
                                      split='train'),
                'test':
                ParisLille3DInternal(root,
                                      voxel_size,
                                      num_points,
                                      split='val')
            })


class ParisLille3DInternal:
    def __init__(self,
                 root,
                 voxel_size,
                 num_points,
                 split,
                 submit=False):

        self.root = root
        self.split = split
        self.voxel_size = voxel_size
        self.num_points = num_points
        self.files = []
        if split == 'train':
            train_path = os.path.join(self.root, "train", "*.ply")
            self.files = glob.glob(train_path)
            if submit:
                val_path = os.path.join(self.root, "val", "*.ply")
                files_temp = glob.glob(val_path)
                self.files.append(files_temp)
        elif self.split == 'val':
            val_path = os.path.join(self.root, "val", "*.ply")
            self.files = glob.glob(val_path)
        elif self.split == 'test':
            test_path = os.path.join(self.root, "test", "*.ply")
            self.files = glob.glob(test_path)

        self.label_map = np.zeros(10)
        cnt = 0
        for label_id in label_name_mapping:
            if label_name_mapping[label_id] in kept_labels:
                self.label_map[label_id] = cnt
                cnt += 1
            else:
                self.label_map[label_id] = 9

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        # 1. prepare point clouds
        data = PlyData.read(self.files[index])['vertex']

        block_ = np.zeros((data['x'].shape[-1], 4), dtype=np.float32) # NOTE Used -1 instead of 0
        block_[:, 0] = data['x']
        block_[:, 1] = data['y']
        block_[:, 2] = data['z']
        block_[:, 3] = data['reflectance']

        # Data augmentation
        block = np.zeros_like(block_)
        if 'train' in self.split:
            theta = np.random.uniform(0, 2 * np.pi)
            scale_factor = np.random.uniform(0.95, 1.05)
            # Read current angle and create the rotation matrix
            rot_mat = np.array([[np.cos(theta),
                                 np.sin(theta), 0],
                                [-np.sin(theta),
                                 np.cos(theta), 0], [0, 0, 1]])

            block[:, :3] = np.dot(block_[:, :3], rot_mat) * scale_factor
            #block[:, 3:] = block_[:, 3:] + np.random.randn(3) * 0.1
        else:
            theta = 0.0
            transform_mat = np.array([[np.cos(theta),
                                       np.sin(theta), 0],
                                      [-np.sin(theta),
                                       np.cos(theta), 0], [0, 0, 1]])
            block[...] = block_[...]
            block[:, :3] = np.dot(block[:, :3], transform_mat)

        block[:, 3] = block_[:, 3]
        pc_ = np.round(block[:, :3] / self.voxel_size)
        pc_ -= pc_.min(0, keepdims=1)

        feat_ = block

        # 2. prepare labels
        all_labels = np.array(data['class'], dtype=np.int32).reshape(-1)
        labels_ = self.label_map[all_labels].astype(np.int64)

        # 3. sparse quantize
        # The way to convert a point cloud to SparseTensor so that it can be consumed by
        # networks built with Sparse Convolution or Sparse Point-Voxel Convolution is to
        # use the function torchsparse.utils.sparse_quantize.
        # The inds denotes unique indices in the point cloud coordinates, and inverse_map
        # denotes the unique index each point is corresponding to. The inverse map is used
        # to restore full point cloud prediction from downsampled prediction.
        # In the SparseTensor representation, the coordinates are four dimensions as the
        # first three are x, y, and z, and the last one is the batch index.
        inds, labels, inverse_map = sparse_quantize(pc_,
                                                    feat_,
                                                    labels_,
                                                    return_index=True,
                                                    return_invs=True)

        # 4. sample voxels
        if len(inds) > self.num_points:
            # Not all GPU's can handle the original point cloud tile size.
            # Try to split a tile using 'pdal split --capacity 1000000'
            print("NOTE! GPU may not handle this large point cloud tile size." +
                "Please consider using a smaller point cloud tile size.")
            # If true, randomly select unique indices in the point cloud coordinates
            if 'train' in self.split:
                inds = np.random.choice(inds, self.num_points, replace=False)

        pc = pc_[inds]
        feat = feat_[inds]
        labels = labels_[inds]
        lidar = SparseTensor(feat, pc)
        labels = SparseTensor(labels, pc)
        labels_ = SparseTensor(labels_, pc_)
        inverse_map = SparseTensor(inverse_map, pc_)

        return {
            'lidar': lidar,
            'targets': labels,
            'targets_mapped': labels_,
            'inverse_map': inverse_map,
            'file_name': self.files[index]
        }
    
    @staticmethod
    def collate_fn(inputs):
        return sparse_collate_fn(inputs)
