# e3d: <u>E</u>fficient Methods for <u>3D</u> Deep Learning

We open source e3d: Efficient Methods for 3D Deep Learning, a repository containing our recent advances in efficient 3D point cloud understanding.

<img src="https://hanlab.mit.edu/projects/spvnas/figures/overview.png" width="1080">


## News

**[2020-09]** **[NEW!!]** We release baseline training code for SPVCNNs and MinkowskiNets in [spvnas](https://github.com/mit-han-lab/e3d/tree/master/spvnas) repo, please have a look!

**[2020-08]** Please check out our ECCV 2020 tutorial on [AutoML for Efficient 3D Deep Learning](https://www.youtube.com/watch?v=zzJR07LMXxs), which summarizes the methods released in this codebase. We also made the hands-on tutorial available in colab:

<p class="aligncenter">

    <a href="https://colab.research.google.com/github/mit-han-lab/e3d/blob/master/tutorial/e3d.ipynb" target="_parent"><img src="https://camo.githubusercontent.com/52feade06f2fecbf006889a904d221e6a730c194/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667" alt="Open In Colab" data-canonical-src="https://colab.research.google.com/assets/colab-badge.svg"></a> 
</p>


**[2020-07]** Our [paper](https://arxiv.org/abs/2007.16100) Searching Efficient 3D Architectures with Sparse Point-Voxel Convolution is accepted to ECCV 2020.

**[2020-03]** Our work [PVCNN](https://arxiv.org/abs/1907.03739) is deployed on MIT Driverless racing cars, please check of this [video](https://www.youtube.com/watch?v=WW9paispAW0).

**[2019-12]** We give the spotlight talk of [PVCNN](https://arxiv.org/abs/1907.03739) at NeurIPS 2019.



## Content

- [Installation](#Installation)
- [SPVNAS (ECCV 2020)](#SPVNAS)
- [PVCNN (NeurIPS 2019, spotlight)](#PVCNN)



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



## SPVNAS: Searching Efficient 3D Architectures with Sparse Point-Voxel Convolution

[[Tutorial at ECCV NAS Workshop]](https://www.youtube.com/watch?v=zzJR07LMXxs) [[ECCV 10-min Talk]](https://www.youtube.com/watch?v=zC_4k5Pnqss) [[MIT News]](http://news.mit.edu/2020/artificial-intelligence-ai-carbon-footprint-0423) [[State-of-the-Art on SemanticKITTI Leaderboard]](http://semantic-kitti.org/tasks.html#semseg)

```bibtex
@inproceedings{
  tang2020searching,
  title = {Searching Efficient 3D Architectures with Sparse Point-Voxel Convolution},
  author = {Tang, Haotian* and Liu, Zhijian* and Zhao, Shengyu and Lin, Yujun and Lin, Ji and Wang, Hanrui and Han, Song},
  booktitle = {European Conference on Computer Vision},
  year = {2020}
}
```



### Overview

We release the PyTorch code of our paper SPVNAS: Searching Efficient 3D Architectures with Sparse Point Voxel Convolution ([arXiv version](https://arxiv.org/abs/2007.16100)). It achieves state-of-the-art performance on the SemanticKITTI [leaderboard](http://semantic-kitti.org/tasks.html#semseg), and outperforms [MinkowskiNet](https://arxiv.org/abs/1904.08755) with **3x speedup, 8x MACs reduction**.

#### SPVNAS Uniformly Outperforms MinkowskiNet

<img src="https://hanlab.mit.edu/projects/spvnas/figures/spvnas_tradeoff_curves.png" width="1080">

#### SPVNAS Achieves Lower Error on Safety Critical Small Objects

<img src="https://hanlab.mit.edu/projects/spvnas/figures/error_map.png" width="1080">

#### SPVNAS is Much Faster Than MinkowskiNet

<img src="https://hanlab.mit.edu/projects/spvnas/figures/spvnas_vs_mink.gif" width="1080">



### Content

- [Prerequisites](#prerequisites)
- [Data Preparation](#data-preparation)
  * SemanticKITTI
- [Code](#code)
- [Pretrained Models](#pretrained-models)
  * SemanticKITTI
- [Testing Pretrained Models](#testing-pretrained-models)
- [Visualizations](#visualizations)
- [Training](#training)
- [Searching](#searching)

### Prerequisites

The code is built with following libraries:
- Python >= 3.6
- [PyTorch](https://github.com/pytorch/pytorch) >= 1.6
- [tensorboardX](https://github.com/lanpa/tensorboardX) >= 1.2
- [tqdm](https://github.com/tqdm/tqdm)
- [torchpack](https://github.com/mit-han-lab/torchpack)
- [torchsparse](https://github.com/mit-han-lab/torchsparse)



### Data Preparation

#### SemanticKITTI

Please follow the instructions from [here](http://www.semantic-kitti.org) to download the SemanticKITTI dataset (both KITTI Odometry dataset and SemanticKITTI labels) and extract all the files in the `sequences` folder to `/dataset/semantic-kitti`. You shall see 22 folders 00, 01, …, 21; each with subfolders named `velodyne` and `labels`.



### Code

The code is based on [torchsparse](https://github.com/mit-han-lab/torchsparse/), a high-performance GPU computing library for 3D sparse convolution operations. It is significantly faster than existing implementation [MinkowskiEngine](https://github.com/StanfordVL/MinkowskiEngine) and supports more diverse operations, such as the new 3D module proposed in this paper, Sparse Point-Voxel Convolution, or in short SPVConv (see [core/models/semantic_kitti/spvcnn.py](core/models/semantic_kitti/spvcnn.py) for details):

```python
# x: sparse tensor, z: point_tensor
x_new = point_to_voxel(x, z)
x_new = sparse_conv_net(x_new)
z_ew = voxel_to_point(x_new, z) + point_transforms(z.F)
```

<img src="https://hanlab.mit.edu/projects/spvnas/figures/spvconv.png" width="1080">

We further propose 3D-NAS to automatically search for efficient 3D architectures built with SPVConv. The 3D-NAS super network implementation can be found in [core/models/semantic_kitti/spvnas.py](core/models/semantic_kitti/spvnas.py). 

<img src="https://hanlab.mit.edu/projects/spvnas/figures/3dnas.png" width="1080">



### Pretrained Models

#### SemanticKITTI

We share the pretrained models for MinkowskiNets, our manually designed SPVCNN models and also SPVNAS models found by our 3D-NAS pipeline. All the pretrained models are available in the [Model Zoo](model_zoo.py). Currently, we release the models trained on sequences 00-07 and 09-10 and evaluated on sequence 08.

|                            Models                            | \#Params (M) | MACs (G) | mIoU (paper) | mIoU (reprod.) |
| :----------------------------------------------------------: | :----------: | :------: | :----------: | :------------: |
| [SemanticKITTI_val_MinkUNet@29GMACs](https://hanlab.mit.edu/files/SPVNAS/minkunet/SemanticKITTI_val_MinkUNet@29GMACs/) |     5.5      |   28.5   |     58.9     |      59.3      |
| [SemanticKITTI_val_SPVCNN@30GMACs](https://hanlab.mit.edu/files/SPVNAS/spvcnn/SemanticKITTI_val_SPVCNN@30GMACs/) |     5.5      |   30.0   |     60.7     |   60.8 ± 0.5   |
| [SemanticKITTI_val_SPVNAS@20GMACs](https://hanlab.mit.edu/files/SPVNAS/spvnas_specialized/SemanticKITTI_val_SPVNAS@20GMACs/) |     3.3      |   20.0   |     61.5     |       -        |
| [SemanticKITTI_val_SPVNAS@25GMACs](https://hanlab.mit.edu/files/SPVNAS/spvnas/SemanticKITTI_val_SPVNAS@25GMACs/) |     4.5      |   24.6   |     62.9     |       -        |
| [SemanticKITTI_val_MinkUNet@46GMACs](https://hanlab.mit.edu/files/SPVNAS/minkunet/SemanticKITTI_val_MinkUNet@46GMACs/) |     8.8      |   45.9   |     60.3     |      60.0      |
| [SemanticKITTI_val_SPVCNN@47GMACs](https://hanlab.mit.edu/files/SPVNAS/spvcnn/SemanticKITTI_val_SPVCNN@47GMACs/) |     8.8      |   47.4   |     61.4     |   61.5 ± 0.2   |
| [SemanticKITTI_val_SPVNAS@35GMACs](https://hanlab.mit.edu/files/SPVNAS/spvnas_specialized/SemanticKITTI_val_SPVNAS@35GMACs/) |     7.0      |   34.7   |     63.5     |       -        |
| [SemanticKITTI_val_MinkUNet@114GMACs](https://hanlab.mit.edu/files/SPVNAS/minkunet/SemanticKITTI_val_MinkUNet@114GMACs/) |     21.7     |  113.9   |     61.1     |      61.9      |
| [SemanticKITTI_val_SPVCNN@119GMACs](https://hanlab.mit.edu/files/SPVNAS/spvcnn/SemanticKITTI_val_SPVCNN@119GMACs/) |     21.8     |  118.6   |     63.8     |   63.7 ± 0.4   |
| [SemanticKITTI_val_SPVNAS@65GMACs](https://hanlab.mit.edu/files/SPVNAS/spvnas_specialized/SemanticKITTI_val_SPVNAS@65GMACs/) |     10.8     |   64.5   |     64.7     |       -        |

Here, the results are reproduced using 8 NVIDIA RTX 2080Ti GPUs. Result variation for each single model is due to the existence of floating point atomic addition operation in our [torchsparse](https://github.com/mit-han-lab/torchsparse) CUDA backend.



### Testing Pretrained Models

You can run the following command to test the performance of SPVNAS / SPVCNN / MinkUNet models.

```bash
torchpack dist-run -np [num_of_gpus] python evaluate.py configs/semantic_kitti/default.yaml --name [num_of_net]
```

For example, to test the model [SemanticKITTI_val_SPVNAS@65GMACs](https://hanlab.mit.edu/files/SPVNAS/spvnas_specialized/SemanticKITTI_val_SPVNAS@65GMACs/) on one GPU, you may run

```bash
torchpack dist-run -np 1 python evaluate.py configs/semantic_kitti/default.yaml --name SemanticKITTI_val_SPVNAS@65GMACs
```



### Visualizations

You can run the following command (on a headless server) to visualize the predictions of SPVNAS / SPVCNN / MinkUNet models:

```bash
xvfb-run --server-args="-screen 0 1024x768x24" python visualize.py
```

If you are running the code on a computer with monitor, you may also directly run

```bash
python visualize.py
```

The visualizations will be generated in `sample_data/outputs`.



### Training

#### SemanticKITTI

We currently release the training code for manually-designed baseline models (SPVCNN and MinkowskiNets). You may run the following code to train the model from scratch:

```bash
torchpack dist-run -np [num_of_gpus] python train.py configs/semantic_kitti/[model name]/[config name].yaml
```

For example, to train the model [SemanticKITTI_val_SPVCNN@30GMACs](https://hanlab.mit.edu/files/SPVNAS/spvcnn/SemanticKITTI_val_SPVCNN@30GMACs/), you may run

```bash
torchpack dist-run -np [num_of_gpus] python train.py configs/semantic_kitti/spvcnn/cr0p5.yaml
```



### Searching

The code related to architecture search will be coming soon!



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
