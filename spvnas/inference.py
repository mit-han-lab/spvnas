''' Inference code for point clouds semantic segmentation

Modified by Sandeep N Menon
Date: December 2020

Ref: https://github.com/mit-han-lab/e3d/blob/master/spvnas/visualize.py

'''

import os
import argparse
import numpy as np

import torch
from torchsparse.utils import sparse_quantize, sparse_collate
from torchsparse import SparseTensor
from model_zoo import spvcnn


def process_point_cloud(input_point_cloud, voxel_size=0.05):
    input_point_cloud[:, 3] = input_point_cloud[:, 3]
    pc_ = np.round(input_point_cloud[:, :3] / voxel_size)
    pc_ -= pc_.min(0, keepdims=1)

    labels_ = np.zeros(pc_.shape[0], dtype=np.int64)
    feat_ = input_point_cloud
    out_pc = input_point_cloud
    pc_ = pc_

    inds, labels, inverse_map = sparse_quantize(pc_,
                                                feat_,
                                                labels_,
                                                return_index=True,
                                                return_invs=True)
    pc = np.zeros((inds.shape[0], 4))
    pc[:, :3] = pc_[inds]

    feat = feat_[inds]
    labels = labels_[inds]
    lidar = SparseTensor(
        torch.from_numpy(feat).float(),
        torch.from_numpy(pc).int()
    )
    return {
        'pc': out_pc,
        'lidar': lidar,
        'targets': labels,
        'targets_mapped': labels_,
        'inverse_map': inverse_map
    }


def get_inference(model, point_cloud):
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'

    if 'MinkUNet' in model:
        model = minkunet(model, pretrained=True)
    elif 'SPVCNN' in model:
        model = spvcnn(model, pretrained=True)
    elif 'SPVNAS' in model:
        model = spvnas_specialized(model, pretrained=True)
    else:
        raise NotImplementedError

    model = model.to(device)
    with torch.no_grad():
        feed_dict = process_point_cloud(point_cloud)
        inputs = feed_dict['lidar'].to(device)

        outputs = model(inputs)
        predictions = outputs.argmax(1).cpu().numpy()
        predictions = predictions[feed_dict['inverse_map']]

    return predictions.astype(np.uint8)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='SemanticKITTI_val_SPVCNN@119GMACs')
    parser.add_argument('--file', type=str, default=None)
    args = parser.parse_args()

    point_cloud_name = args.file
    pc = np.fromfile(point_cloud_name, dtype=np.float32).reshape(-1, 4)
    get_inference(args.model, pc)
