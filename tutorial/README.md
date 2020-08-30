# Hands-on Tutorial of torchsparse and SPVNAS [[youtube]](https://www.youtube.com/watch?v=zzJR07LMXxs), [[bilibili]](https://www.bilibili.com/video/BV14i4y1g7eQ)

<p class="aligncenter">
    <a href="https://colab.research.google.com/github/mit-han-lab/e3d/blob/master/tutorial/e3d.ipynb" target="_parent"><img src="https://camo.githubusercontent.com/52feade06f2fecbf006889a904d221e6a730c194/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667" alt="Open In Colab" data-canonical-src="https://colab.research.google.com/assets/colab-badge.svg"></a> 
</p>


This is a hands-on tutorial for our efficient 3D sparse computation library, [**torchsparse**](https://github.com/mit-han-lab/torchsparse) and latest AutoML framework for efficient 3D scene understanding, [**SPVNAS**](https://arxiv.org/abs/2007.16100).

```bibtex
@inproceedings{
  tang2020searching,
  title = {Searching Efficient 3D Architectures with Sparse Point-Voxel Convolution},
  author = {Tang, Haotian* and Liu, Zhijian* and Zhao, Shengyu and Lin, Yujun and Lin, Ji and Wang, Hanrui and Han, Song},
  booktitle = {European Conference on Computer Vision},
  year = {2020}
}
```

In this notebook, we will demonstrate 
- how to build 3D sparse computation DNNs with **torchsparse** and perform training
- how to evaluate pretrained [SPVNAS](https://arxiv.org/abs/2007.16100) models and visualize the predictions

Required packages:
```bash
pip install --upgrade pip
pip install --upgrade jupyter notebook
```

Then, please clone this repository to your computer using:

```bash
git clone https://github.com/mit-han-lab/e3d.git
```

After cloning is finished, you may go to the directory of this tutorial and run

```bash
jupyter notebook --port 8888
```

to start a jupyter notebook and access it through the browser. Then, please click `e3d.ipynb` and start your exploration!