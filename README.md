# SPVNAS

### [video](https://youtu.be/zzJR07LMXxs) | [paper](https://arxiv.org/abs/2007.16100) | [website](http://spvnas.mit.edu/) [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mit-han-lab/spvnas/blob/master/tutorial.ipynb)

[Searching Efficient 3D Architectures with Sparse Point-Voxel Convolution](https://arxiv.org/abs/2007.16100)

[Haotian Tang\*](http://kentang.net/), [Zhijian Liu\*](http://zhijianliu.com/), Shengyu Zhao, Yujun Lin, [Ji Lin](http://linji.me/), [Hanrui Wang](http://hanruiwang.me/), [Song Han](https://songhan.mit.edu/)

ECCV 2020

<img src="https://hanlab.mit.edu/projects/spvnas/figures/spvnas_vs_mink.gif" width="1080">

SPVNAS achieves state-of-the-art performance on the SemanticKITTI [leaderboard](http://semantic-kitti.org/tasks.html#semseg) (as of July 2020) and outperforms [MinkowskiNet](https://arxiv.org/abs/1904.08755) with **3x speedup, 8x MACs reduction**.

## News

**\[2020-09\]** We release the baseline training code for SPVCNN and MinkowskiNet.

**\[2020-08\]** Please check out our ECCV 2020 tutorial on [AutoML for Efficient 3D Deep Learning](https://www.youtube.com/watch?v=zzJR07LMXxs), which summarizes the algorithm in this codebase.

**\[2020-07\]** Our paper is accepted to ECCV 2020.

## Usage

### Prerequisites

The code is built with following libraries:

- Python >= 3.6, \<3.8
- [PyTorch](https://github.com/pytorch/pytorch) >= 1.6
- [tqdm](https://github.com/tqdm/tqdm)
- [torchpack](https://github.com/mit-han-lab/torchpack)
- [torchsparse](https://github.com/mit-han-lab/torchsparse)
- [numba](http://numba.pydata.org/)
- [cv2](https://github.com/opencv/opencv)

#### Recommended Installation

For easy installation, use [conda](https://docs.conda.io/projects/conda/en/latest/):

```
conda create -n torch python=3.7
conda activate torch
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
conda install numba opencv
pip install torchpack
pip install --upgrade git+https://github.com/mit-han-lab/torchsparse.git
```

### Data Preparation

#### SemanticKITTI

Please follow the instructions from [here](http://www.semantic-kitti.org) to download the SemanticKITTI dataset (both KITTI Odometry dataset and SemanticKITTI labels) and extract all the files in the `sequences` folder to `/dataset/semantic-kitti`. You shall see 22 folders 00, 01, …, 21; each with subfolders named `velodyne` and `labels`.

### Model Zoo

#### SemanticKITTI

We share the pretrained models for MinkowskiNets, our manually designed SPVCNN models and also SPVNAS models found by our 3D-NAS pipeline. All the pretrained models are available in the [model zoo](model_zoo.py). Currently, we release the models trained on sequences 00-07 and 09-10 and evaluated on sequence 08.

|                                       | #Params (M) | MACs (G) | mIoU (paper) | mIoU (reprod.) |
| :-----------------------------------: | :---------: | :------: | :----------: | :------------: |
| `SemanticKITTI_val_MinkUNet@29GMACs`  |     5.5     |   28.5   |     58.9     |      59.3      |
|  `SemanticKITTI_val_SPVCNN@30GMACs`   |     5.5     |   30.0   |     60.7     |   60.8 ± 0.5   |
|  `SemanticKITTI_val_SPVNAS@20GMACs`   |     3.3     |   20.0   |     61.5     |       -        |
|  `SemanticKITTI_val_SPVNAS@25GMACs`   |     4.5     |   24.6   |     62.9     |       -        |
| `SemanticKITTI_val_MinkUNet@46GMACs`  |     8.8     |   45.9   |     60.3     |      60.0      |
|  `SemanticKITTI_val_SPVCNN@47GMACs`   |     8.8     |   47.4   |     61.4     |   61.5 ± 0.2   |
|  `SemanticKITTI_val_SPVNAS@35GMACs`   |     7.0     |   34.7   |     63.5     |       -        |
| `SemanticKITTI_val_MinkUNet@114GMACs` |    21.7     |  113.9   |     61.1     |      61.9      |
|  `SemanticKITTI_val_SPVCNN@119GMACs`  |    21.8     |  118.6   |     63.8     |   63.7 ± 0.4   |
|  `SemanticKITTI_val_SPVNAS@65GMACs`   |    10.8     |   64.5   |     64.7     |       -        |

Here, the results are reproduced using 8 NVIDIA RTX 2080Ti GPUs. Result variation for each single model is due to the existence of floating point atomic addition operation in our [TorchSparse](https://github.com/mit-han-lab/torchsparse) CUDA backend.

### Testing Pretrained Models

You can run the following command to test the performance of SPVNAS / SPVCNN / MinkUNet models.

```bash
torchpack dist-run -np [num_of_gpus] python evaluate.py configs/semantic_kitti/default.yaml --name [num_of_net]
```

For example, to test the model `SemanticKITTI_val_SPVNAS@65GMACs` on one GPU, you may run

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

The visualizations will be generated in `assets/`.

### Training

#### SemanticKITTI

We currently release the training code for manually-designed baseline models (SPVCNN and MinkowskiNets). You may run the following code to train the model from scratch:

```bash
torchpack dist-run -np [num_of_gpus] python train.py configs/semantic_kitti/[model name]/[config name].yaml
```

For example, to train the model `SemanticKITTI_val_SPVCNN@30GMACs`, you may run

```bash
torchpack dist-run -np [num_of_gpus] python train.py configs/semantic_kitti/spvcnn/cr0p5.yaml
```

To train the model in a non-distributed environment without MPI, i.e. on a single GPU, you may directly call `train.py` with the `--distributed False` argument:

```bash
python train.py configs/semantic_kitti/spvcnn/cr0p5.yaml --distributed False
```

### Searching

The code related to architecture search will be coming soon!

## Citation

If you use this code for your research, please cite our paper.

```@inproceedings{
@inproceedings{tang2020searching,
  title = {Searching Efficient 3D Architectures with Sparse Point-Voxel Convolution},
  author = {Tang, Haotian* and Liu, Zhijian* and Zhao, Shengyu and Lin, Yujun and Lin, Ji and Wang, Hanrui and Han, Song},
  booktitle = {European Conference on Computer Vision},
  year = {2020}
}
```
