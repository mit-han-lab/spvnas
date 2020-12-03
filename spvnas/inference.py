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

cmap = np.array([
    [245, 150, 100, 255],# car
    [245, 230, 100, 255],# pedestrian with object
    [245, 230, 100, 255], # pedestrian with object
    [180, 30, 80, 255], # Vehicle_Truck
    [255, 0, 0, 255], # -------
    [30, 30, 255, 255],# Pedestrian adult
    [245, 230, 100, 255], # pedestrian with object
    [245, 230, 100, 255], # pedestrian with object
    [255, 0, 255, 255], # drivable region
    [255, 0, 255, 255],# drivable region
    [75, 0, 75, 255], # ground
    [255, 0, 255, 255],# drivable region
    [0, 200, 255, 255],# static
    [0, 200, 255, 255],# static
    [0, 175, 0, 255],# hard vegetation
    [0, 175, 0, 255],# hard vegetation
    [75, 0, 75, 255],# ground
    [0, 200, 255, 255],# static
    [0, 200, 255, 255],# static
    #[255, 255, 255, 0]
])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='SemanticKITTI_val_SPVCNN@119GMACs')
    parser.add_argument('--file', type=str, default=None)
    args = parser.parse_args()
    
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
    
    point_cloud_name = args.file
    pc = np.fromfile(point_cloud_name, dtype=np.float32).reshape(-1, 4)
    feed_dict = process_point_cloud(pc)
    inputs = feed_dict['lidar'].to(device)

    print("Prediction started")
    outputs = model(inputs)
    predictions = outputs.argmax(1).cpu().numpy()
    predictions = predictions[feed_dict['inverse_map']]
    predictions.astype(np.int32).tofile(f"{point_cloud_name}_color.bin")
    print("Predictions saved")
