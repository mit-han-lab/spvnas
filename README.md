# e3d: <u>E</u>fficient Methods for <u>3D</u> Deep Learning

We open source e3d: Efficient Methods for 3D Deep Learning, a repository containing our recent advances in efficient 3D point cloud understanding.

<img src="https://hanlab.mit.edu/projects/spvnas/figures/overview.png" width="1080">


## News

**[2020-08]** Please check out our ECCV 2020 tutorial on [AutoML for Efficient 3D Deep Learning](https://www.youtube.com/watch?v=zzJR07LMXxs), which summarizes the methods released in this codebase.

**[2020-07]** Our [paper](https://arxiv.org/abs/2007.16100) Searching Efficient 3D Architectures with Sparse Point-Voxel Convolution is accepted to ECCV 2020.

**[2020-03]** Our work [PVCNN](https://arxiv.org/abs/1907.03739) is deployed on MIT Driverless racing cars, please check of this [video](https://www.youtube.com/watch?v=WW9paispAW0).

**[2019-12]** We give the spotlight talk of [PVCNN](https://arxiv.org/abs/1907.03739) at NeurIPS 2019.



## Content

- [Installation](#Installation)
- [PVCNN (NeurIPS 2019, spotlight)](#PVCNN)
- [SPVNAS (ECCV 2020)](#SPVNAS)



## Installation

Please run:

```bash
git clone https://github.com/mit-han-lab/e3d --recurse-submodules
```

to clone this code base. If you forget to add the `—recursive-submodules` flag when cloning the codebase, please run:

```bash
git submodule update --init
```

after you run:

```bash
git clone https://github.com/mit-han-lab/e3d
```

To use all the codebases presented in this repository, please following the instructions in each folder. 



## PVCNN
```bibtex
@inproceedings{liu2019pvcnn,
  title={Point-Voxel CNN for Efficient 3D Deep Learning},
  author={Liu, Zhijian and Tang, Haotian and Lin, Yujun and Han, Song},
  booktitle={Advances in Neural Information Processing Systems},
  year={2019}
}
```

[[Paper]](https://arxiv.org/abs/1907.03739) [[NeurIPS 2019 spotlight talk]](https://youtu.be/h6zbQe5U1v4) [[Deploy on MIT Driverless]](https://youtu.be/WW9paispAW0) [[NVIDIA Jetson Community Project Spotlight]](https://news.developer.nvidia.com/point-voxel-cnn-3d/?ncid=so-twit-99540#cid=em02_so-twit_en-us) [[Playlist]](https://www.youtube.com/playlist?list=PL80kAHvQbh-oUjbb-kFfaSZyWqkCjrUnD) [[Website]](http://pvcnn.mit.edu/)

<img src="https://hanlab.mit.edu/projects/pvcnn/figures/pvcnn_results_on_edge.png" width="1080">

### Overview

In [PVCNN](https://arxiv.org/abs/1907.03739), we present a new efficient 3D deep learning module, Point-Voxel Convolution (PVConv) as is illustrated below.

<img src="https://hanlab.mit.edu/projects/pvcnn/figures/overview.png" width="1080">

PVConv takes advantage of the regularity of volumetric representation and small footprint of point cloud representation, achieving significantly faster inference speed and much lower memory footprint comparing with both point cloud-based and voxel-based 3D deep learning methods.



Here is a demo comparing PVCNN and PointNet in 3D shape part segmentation on NVIDIA Jetson Nano:

<img src="https://hanlab.mit.edu/projects/pvcnn/figures/gif/pvcnn_demo_shape_480p.gif" width="720">

### Evaluation

To test the PVCNN models, please run `cd pvcnn` first and download our pretrained models as is indicated in the README file. Then, please run this code template

```bash
python train.py [config-file] --devices [gpu-ids] --evaluate --configs.evaluate.best_checkpoint_path [path to the model checkpoint]
```

to do the evaluation. If you want to do inference on S3DIS with GPU 0,1, you can run:

```bash
python train.py configs/s3dis/pvcnn/area5.py --devices 0,1 --evaluate --configs.evaluate.best_checkpoint_path s3dis.pvcnn.area5.c1.pth.tar
```



## SPVNAS
```bibtex
@inproceedings{
  tang2020searching,
  title = {Searching Efficient 3D Architectures with Sparse Point-Voxel Convolution},
  author = {Tang, Haotian* and Liu, Zhijian* and Zhao, Shengyu and Lin, Yujun and Lin, Ji and Wang, Hanrui and Han, Song},
  booktitle = {European Conference on Computer Vision},
  year = {2020}
}
```

[[Paper]](https://arxiv.org/abs/2007.16100) [[ECCV 10-min Talk]](https://www.youtube.com/watch?v=zC_4k5Pnqss) [[MIT News]](http://news.mit.edu/2020/artificial-intelligence-ai-carbon-footprint-0423) [[State-of-the-Art on SemanticKITTI Leaderboard]](http://semantic-kitti.org/tasks.html#semseg)

<img src="https://hanlab.mit.edu/projects/spvnas/figures/spvnas_barchart.png" width="1080">

### Overview

We present [SPVNAS](https://arxiv.org/abs/2007.16100), the **first** AutoML method for efficient 3D scene understanding. In this work, we first adapt Point-Voxel Convolution to large-scale outdoor LiDAR scans by introducing Sparse Point-Voxel Convolution (SPVConv):

<img src="https://hanlab.mit.edu/projects/spvnas/figures/spvconv.png" width="1080">

We then apply 3D Neural Architecture Search (3D-NAS) to automatically search for the best architectures built from SPVConv under efficiency constraints.

<img src="https://hanlab.mit.edu/projects/spvnas/figures/3dnas.png" width="1080">

Here is a demo comparing SPVNAS and MinkowskiNet, SPVNAS reaches 9.1 FPS on NVIDIA GTX1080Ti, which is 3 times faster than MinkowskiNet.

<img src="https://hanlab.mit.edu/projects/spvnas/figures/spvnas_vs_mink.gif" width="640">

### Evaluation

Please run `cd spvnas` and run the following code template to evaluate pretrained SPVNAS/SPVCNN/MinkowskiNet models:

```bash
torchpack dist-run -np [num_of_gpus] python evaluate.py configs/semantic_kitti/default.yaml --name [num_of_net]
```

For example, if you want to run inference with the model [SemanticKITTI_val_SPVNAS@65GMACs](https://hanlab.mit.edu/files/SPVNAS/spvnas_specialized/SemanticKITTI_val_SPVNAS@65GMACs/), you may run:

```bash
torchpack dist-run -np 1 python evaluate.py configs/semantic_kitti/default.yaml --name SemanticKITTI_val_SPVNAS@65GMACs
```



