import numpy as np


def create_label_map(num_classes=19):
    name_label_mapping = {
        'unlabeled': 0, 'outlier': 1, 'car': 10, 'bicycle': 11,
        'bus': 13, 'motorcycle': 15, 'on-rails': 16, 'truck': 18,
        'other-vehicle': 20, 'person': 30, 'bicyclist': 31,
        'motorcyclist': 32, 'road': 40, 'parking': 44,
        'sidewalk': 48, 'other-ground': 49, 'building': 50,
        'fence': 51, 'other-structure': 52, 'lane-marking': 60,
        'vegetation': 70, 'trunk': 71, 'terrain': 72, 'pole': 80,
        'traffic-sign': 81, 'other-object': 99, 'moving-car': 252,
        'moving-bicyclist': 253, 'moving-person': 254, 'moving-motorcyclist': 255,
        'moving-on-rails': 256, 'moving-bus': 257, 'moving-truck': 258,
        'moving-other-vehicle': 259
    }
    
    for k in name_label_mapping:
        name_label_mapping[k] = name_label_mapping[k.replace('moving-', '')]
    train_label_name_mapping = {
        0: 'car', 1: 'bicycle', 2: 'motorcycle', 3: 'truck', 4:
        'other-vehicle', 5: 'person', 6: 'bicyclist', 7: 'motorcyclist',
        8: 'road', 9: 'parking', 10: 'sidewalk', 11: 'other-ground',
        12: 'building', 13: 'fence', 14: 'vegetation', 15: 'trunk',
        16: 'terrain', 17: 'pole', 18: 'traffic-sign'
    }

    label_map = np.zeros(260)+num_classes
    for i in range(num_classes):
        cls_name = train_label_name_mapping[i]
        label_map[name_label_mapping[cls_name]] = min(num_classes,i)
    return label_map.astype(np.int64)