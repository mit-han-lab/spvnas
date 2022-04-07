"""Visualization code for point clouds and 3D bounding boxes with mayavi.

Modified by Charles R. Qi
Date: September 2017
"""

import argparse
import os

import mayavi.mlab as mlab
import numpy as np
import open3d as o3d
import torch
from torchsparse import SparseTensor
from torchsparse.utils.quantize import sparse_quantize

from model_zoo import minkunet, spvcnn, spvnas_specialized


def process_point_cloud(input_point_cloud, input_labels=None, voxel_size=0.05):
    input_point_cloud[:, 3] = input_point_cloud[:, 3]
    pc_ = np.round(input_point_cloud[:, :3] / voxel_size)
    pc_ -= pc_.min(0, keepdims=1)

    label_map = create_label_map()
    if input_labels is not None:
        labels_ = label_map[input_labels & 0xFFFF].astype(
            np.int64)  # semantic labels
    else:
        labels_ = np.zeros(pc_.shape[0], dtype=np.int64)

    feat_ = input_point_cloud

    if input_labels is not None:
        out_pc = input_point_cloud[labels_ != labels_.max(), :3]
        pc_ = pc_[labels_ != labels_.max()]
        feat_ = feat_[labels_ != labels_.max()]
        labels_ = labels_[labels_ != labels_.max()]
    else:
        out_pc = input_point_cloud
        pc_ = pc_

    coords_, inds, inverse_map = sparse_quantize(pc_,
                                                 return_index=True,
                                                 return_inverse=True)

    pc = np.zeros((inds.shape[0], 4))
    pc[:, :3] = pc_[inds]

    feat = feat_[inds]
    labels = labels_[inds]
    lidar = SparseTensor(
        torch.from_numpy(feat).float(),
        torch.from_numpy(pc).int())
    return {
        'pc': out_pc,
        'lidar': lidar,
        'targets': labels,
        'targets_mapped': labels_,
        'inverse_map': inverse_map
    }


mlab.options.offscreen = True


def create_label_map(num_classes=19):
    name_label_mapping = {
        'unlabeled': 0,
        'outlier': 1,
        'car': 10,
        'bicycle': 11,
        'bus': 13,
        'motorcycle': 15,
        'on-rails': 16,
        'truck': 18,
        'other-vehicle': 20,
        'person': 30,
        'bicyclist': 31,
        'motorcyclist': 32,
        'road': 40,
        'parking': 44,
        'sidewalk': 48,
        'other-ground': 49,
        'building': 50,
        'fence': 51,
        'other-structure': 52,
        'lane-marking': 60,
        'vegetation': 70,
        'trunk': 71,
        'terrain': 72,
        'pole': 80,
        'traffic-sign': 81,
        'other-object': 99,
        'moving-car': 252,
        'moving-bicyclist': 253,
        'moving-person': 254,
        'moving-motorcyclist': 255,
        'moving-on-rails': 256,
        'moving-bus': 257,
        'moving-truck': 258,
        'moving-other-vehicle': 259
    }

    for k in name_label_mapping:
        name_label_mapping[k] = name_label_mapping[k.replace('moving-', '')]
    train_label_name_mapping = {
        0: 'car',
        1: 'bicycle',
        2: 'motorcycle',
        3: 'truck',
        4: 'other-vehicle',
        5: 'person',
        6: 'bicyclist',
        7: 'motorcyclist',
        8: 'road',
        9: 'parking',
        10: 'sidewalk',
        11: 'other-ground',
        12: 'building',
        13: 'fence',
        14: 'vegetation',
        15: 'trunk',
        16: 'terrain',
        17: 'pole',
        18: 'traffic-sign'
    }

    label_map = np.zeros(260) + num_classes
    for i in range(num_classes):
        cls_name = train_label_name_mapping[i]
        label_map[name_label_mapping[cls_name]] = min(num_classes, i)
    return label_map.astype(np.int64)


cmap = np.array([
    [245, 150, 100, 255],
    [245, 230, 100, 255],
    [150, 60, 30, 255],
    [180, 30, 80, 255],
    [255, 0, 0, 255],
    [30, 30, 255, 255],
    [200, 40, 255, 255],
    [90, 30, 150, 255],
    [255, 0, 255, 255],
    [255, 150, 255, 255],
    [75, 0, 75, 255],
    [75, 0, 175, 255],
    [0, 200, 255, 255],
    [50, 120, 255, 255],
    [0, 175, 0, 255],
    [0, 60, 135, 255],
    [80, 240, 150, 255],
    [150, 240, 255, 255],
    [0, 0, 255, 255],
])
cmap = cmap[:, [2, 1, 0, 3]]  # convert bgra to rgba


def draw_lidar(pc,
               color=None,
               fig=None,
               bgcolor=(1, 1, 1),
               pts_scale=0.06,
               pts_mode='2dcircle',
               pts_color=None):
    if fig is None:
        fig = mlab.figure(figure=None,
                          bgcolor=bgcolor,
                          fgcolor=None,
                          engine=None,
                          size=(800, 500))
    if color is None:
        color = pc[:, 2]
    pts = mlab.points3d(pc[:, 0],
                        pc[:, 1],
                        pc[:, 2],
                        color,
                        mode=pts_mode,
                        scale_factor=pts_scale,
                        figure=fig)
    pts.glyph.scale_mode = 'scale_by_vector'
    pts.glyph.color_mode = 'color_by_scalar'  # Color by scalar
    pts.module_manager.scalar_lut_manager.lut.table = cmap
    pts.module_manager.scalar_lut_manager.lut.number_of_colors = cmap.shape[0]

    mlab.view(azimuth=180,
              elevation=70,
              focalpoint=[12.0909996, -1.04700089, -2.03249991],
              distance=62,
              figure=fig)

    return fig


# visualize by open3d
label_name_mapping = {
    0: 'unlabeled',
    1: 'outlier',
    10: 'car',
    11: 'bicycle',
    13: 'bus',
    15: 'motorcycle',
    16: 'on-rails',
    18: 'truck',
    20: 'other-vehicle',
    30: 'person',
    31: 'bicyclist',
    32: 'motorcyclist',
    40: 'road',
    44: 'parking',
    48: 'sidewalk',
    49: 'other-ground',
    50: 'building',
    51: 'fence',
    52: 'other-structure',
    60: 'lane-marking',
    70: 'vegetation',
    71: 'trunk',
    72: 'terrain',
    80: 'pole',
    81: 'traffic-sign',
    99: 'other-object',
    252: 'moving-car',
    253: 'moving-bicyclist',
    254: 'moving-person',
    255: 'moving-motorcyclist',
    256: 'moving-on-rails',
    257: 'moving-bus',
    258: 'moving-truck',
    259: 'moving-other-vehicle'
}

kept_labels = [
    'road', 'sidewalk', 'parking', 'other-ground', 'building', 'car', 'truck',
    'bicycle', 'motorcycle', 'other-vehicle', 'vegetation', 'trunk', 'terrain',
    'person', 'bicyclist', 'motorcyclist', 'fence', 'pole', 'traffic-sign'
]


class BinVisualizer:

    def __init__(self):
        self.points = np.zeros((0, 3), dtype=np.float32)
        self.sem_label = np.zeros((0, 1), dtype=np.uint32)  # [m, 1]: label
        self.sem_label_color = np.zeros((0, 3),
                                        dtype=np.float32)  # [m ,3]: color

        # label map
        reverse_label_name_mapping = {}
        self.label_map = np.zeros(260)
        cnt = 0
        for label_id in label_name_mapping:
            if label_id > 250:
                if label_name_mapping[label_id].replace('moving-',
                                                        '') in kept_labels:
                    self.label_map[label_id] = reverse_label_name_mapping[
                        label_name_mapping[label_id].replace('moving-', '')]
                else:
                    self.label_map[label_id] = 255
            elif label_id == 0:
                self.label_map[label_id] = 255
            else:
                if label_name_mapping[label_id] in kept_labels:
                    self.label_map[label_id] = cnt
                    reverse_label_name_mapping[
                        label_name_mapping[label_id]] = cnt
                    cnt += 1
                else:
                    self.label_map[label_id] = 255
        self.reverse_label_name_mapping = reverse_label_name_mapping

    def read_pc_label(self, points, label):
        assert points.shape[0] == label.shape[0]
        label = label.reshape(-1)
        self.sem_label = label
        self.points = points[:, :3]

    def show_cloud(self, window_name='open3d'):
        # make color table
        color_dict = {}
        for i in range(19):
            color_dict[i] = cmap[i, :]
        color_dict[255] = [0, 0, 0, 255]

        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(self.points)
        cloud_color = [color_dict[i] for i in list(self.sem_label)]
        self.sem_label_color = np.array(cloud_color).reshape(
            (-1, 4))[:, :3] / 255
        pc.colors = o3d.utility.Vector3dVector(self.sem_label_color)

        o3d.visualization.draw_geometries([pc], window_name)

    def run_visualize(self, points, label, window_name):
        self.read_pc_label(points, label)
        self.show_cloud(window_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--velodyne-dir', type=str, default='sample_data')
    parser.add_argument('--model',
                        type=str,
                        default='SemanticKITTI_val_SPVNAS@65GMACs')
    parser.add_argument('--visualize_backend',
                        type=str,
                        default='open3d',
                        help='visualization beckend, default=open3d')
    args = parser.parse_args()
    output_dir = os.path.join(args.velodyne_dir, 'outputs')
    os.makedirs(output_dir, exist_ok=True)

    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'

    if 'MinkUNet' in args.model:
        model = minkunet(args.model, pretrained=True)
    elif 'SPVCNN' in args.model:
        model = spvcnn(args.model, pretrained=True)
    elif 'SPVNAS' in args.model:
        model = spvnas_specialized(args.model, pretrained=True)
    else:
        raise NotImplementedError

    model = model.to(device)

    input_point_clouds = sorted(os.listdir(args.velodyne_dir))
    for point_cloud_name in input_point_clouds:
        if not point_cloud_name.endswith('.bin'):
            continue
        label_file_name = point_cloud_name.replace('.bin', '.label')
        vis_file_name = point_cloud_name.replace('.bin', '.png')
        gt_file_name = point_cloud_name.replace('.bin', '_GT.png')

        pc = np.fromfile(f'{args.velodyne_dir}/{point_cloud_name}',
                         dtype=np.float32).reshape(-1, 4)
        if os.path.exists(f'{args.velodyne_dir}/{label_file_name}'):
            label = np.fromfile(f'{args.velodyne_dir}/{label_file_name}',
                                dtype=np.int32)
        else:
            label = None
        feed_dict = process_point_cloud(pc, label)
        inputs = feed_dict['lidar'].to(device)
        outputs = model(inputs)
        predictions = outputs.argmax(1).cpu().numpy()
        predictions = predictions[feed_dict['inverse_map']]

        if args.visualize_backend == 'mayavi':
            fig = draw_lidar(feed_dict['pc'], predictions.astype(np.int32))
            mlab.savefig(f'{output_dir}/{vis_file_name}')
            if label is not None:
                fig = draw_lidar(feed_dict['pc'], feed_dict['targets_mapped'])
                mlab.savefig(f'{output_dir}/{gt_file_name}')
        elif args.visualize_backend == 'open3d':
            # visualize prediction
            bin_vis = BinVisualizer()
            bin_vis.run_visualize(feed_dict['pc'], predictions.astype(np.int32),
                                  'Predictions')
            if label is not None:
                bin_vis = BinVisualizer()
                bin_vis.run_visualize(feed_dict['pc'],
                                      feed_dict['targets_mapped'],
                                      'Ground turth')
        else:
            raise NotImplementedError
