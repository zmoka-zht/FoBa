<div align="center">

<h3>FoBa: A Foreground-Background co-Guided Method and New Benchmark for Remote Sensing Semantic Change Detection</h3>

[Haotian Zhang](https://scholar.google.com/citations?user=c7uR6NUAAAAJ&hl=zh-CN)<sup>1</sup>, Han Guo<sup>1</sup>, Keyan Chen<sup>1</sup>, Hao Chen<sup>2</sup>, Zhengxia Zou<sup>1</sup>, Zhenwei Shi<sup>1, *</sup>

<sup>1</sup>  北京航空航天大学,  <sup>2</sup> 上海人工智能实验室.

<sup>*</sup> 通讯作者


[![TGRS paper](https://img.shields.io/badge/TGRS-paper-00629B.svg)](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=11268372)  [![arXiv paper](https://img.shields.io/badge/arXiv-paper-b31b1b.svg)](https://arxiv.org/abs/2509.15788)

[**简介**](#overview) | [**开始使用**](#%EF%B8%8Flets-get-started) | [**结果下载**](#%EF%B8%8Fresults-taken-away) | [**其他**](#q--a)

</div>

## 🛎️更新日志
* **` 通知`**: FoBa 已经被 [IEEE TGRS](https://ieeexplore.ieee.org/document/11268372)接收! 仓库的代码已更新完毕！如果对您的研究有所帮助，请考虑给该仓库一个 ⭐️**star**⭐️ !!

* **` 2025年11月18日`**: FoBa 已经被 [IEEE TGRS](https://ieeexplore.ieee.org/document/11268372)接收!!

## 🔭简介

* [**FoBa**](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=11268372) 作为语义变化检测任务的一个重要基准。 

* **LevirSCD数据集**
<p align="center">
  <img src="figures/levirscd.png" alt="accuracy" width="90%">
</p>

* **FoBa的模型结构**

<p align="center">
  <img src="figures/foba_overview.png" alt="accuracy" width="90%">
</p>


## 🗝️开始使用!
### `一、安装`

此仓库的代码是在 **Linux** 系统下运行的。我们尚未测试是否能在其他操作系统下运行。.

首先需要安装 [VMama 仓库](https://github.com/MzeroMiko/VMamba) 或 [ChangeMamba 仓库](https://github.com/ChenHongruixuan/ChangeMamba)。以下安装顺序取自VMama仓库。

**步骤1： 克隆仓库：**

克隆该版本库并导航至项目目录：
```bash
git clone https://github.com/zmoka-zht/FoBa.git
cd FoBa
```


**步骤2： 环境配置：**

建议设置 conda 环境并通过 pip 安装依赖项。使用以下命令设置环境：

***创建并激活新的环境***

```bash
conda create -n foba
conda activate foba
```

***安装依赖项***

```bash
pip install -r requirements.txt
cd kernels/selective_scan && pip install .
```


***检测和分割任务的依赖库（在 VMamba 中为可选项）***

```bash
pip install mmengine==0.10.1 mmcv==2.1.0 opencv-python-headless ftfy regex
pip install mmdet==3.3.0 mmsegmentation==1.2.2 mmpretrain==1.2.0
```
### `二、 下载预训练权重`
另外，请下载[VMamba-Tiny](https://drive.google.com/file/d/160PXughGMNZ1GyByspLFS68sfUdrQE2N/view?usp=drive_link), [VMamba-Small](https://drive.google.com/file/d/1dxHtFEgeJ9KL5WiLlvQOZK5jSEEd2Nmz/view?usp=drive_link), 和[VMamba-Base](https://drive.google.com/file/d/1kUHSBDoFvFG58EmwWurdSVZd8gyKWYfr/view?usp=drive_link) 在ImageNet上的预训练权重并把它们放在下述文件夹中
```bash
project_path/FoBa/pretrained_weight/
```

### `三、 数据准备`
***语义变化检测***

语义变化检测任务的数据集为 [SECOND dataset](https://captain-whu.github.io/SCD/)。 请下载该数据集，并使其具有以下文件夹/文件结构。 

或者，欢迎您直接下载并使用经过我们 **预处理后的 SECOND 、LevirSCD、JL1数据集**：
- **百度网盘:** [Download link](https://pan.baidu.com/s/11mcbilrctWH02sazZEt3Xg)  
  **提取码:** `foba`  

```
${DATASET_ROOT} # Dataset root directory, for example: /home/username/data/SECOND
├── train
│   ├── im1
│   │   ├──00001.png
│   │   ├──00002.png
│   │   ├──00003.png
│   │   ...
│   │
│   ├── im2
│   │   ├──00001.png
│   │   ... 
│   │
│   ├── label   # Binary change map
│   │   ├──00001.png 
│   │   ... 
│   │
│   ├── label1   # Land-cover map of T1
│   │   ├──00001.png 
│   │   ...  
│   │
│   └── label2   # Land-cover map of T2
│       ├──00001.png 
│       ...  
│   
├── test
│   ├── ...
│   ...
├── list 
│   ├──train.txt
└── ├──test.txt
```

### `四、 模型训练`
在训练模型之前，请进入 [`changedetection`] 文件夹，其中包含网络定义、训练和测试的所有代码。 

```bash
cd <project_path>/FoBa/changedetection
```
***语义变化检测***

运行以下命令在 SECOND 数据集上训练和评估 FoBa：
```bash
python script/train_foba.py  --dataset 'SECOND' \
                                 --batch_size 2 \
                                 --crop_size 512 \
                                 --max_iters 480000 \
                                 --model_type FoBa \
                                 --model_param_path '<project_path>/FoBa/changedetection/saved_models' \ 
                                 --train_dataset_path '<dataset_path>/SECOND/train' \
                                 --train_data_list_path '<dataset_path>/SECOND/list/train_list.txt' \
                                 --test_dataset_path '<dataset_path>/SECOND/test' \
                                 --test_data_list_path '<dataset_path>/SECOND/list/test_list.txt'
                                 --cfg '<project_path>/FoBa/changedetection/configs/vssm1/vssm_small_224.yaml' \
                                 --pretrained_weight_path '<project_path>/FoBa/pretrained_weight/vssm_small_0229_ckpt_epoch_222.pth'
```
### `五、 模型推理`

推理前，请先通过命令行进入 [`changedetection`] 文件夹。 
```bash
cd <project_path>/FoBa/changedetection
```

***语义变化检测***

以下命令展示了如何在 SECOND 数据集上使用训练完成的 FoBa:
```bash
python script/infer_foba_second.py  --dataset 'SECOND'  \
                                 --model_type 'FoBaMambaBased' \
                                 --test_dataset_path '<dataset_path>/SECOND/test' \
                                 --test_data_list_path '<dataset_path>/SECOND/list\test_list.txt' \
                                 --cfg '<project_path>/FoBa/changedetection/configs/vssm1/vssm_base_224.yaml' \
                                 --pretrained_weight_path '<project_path>/FoBa/pretrained_weight/vssm_base_0229_ckpt_epoch_237.pth'
                                 --resume '<saved_model_path>/[your_trained_model].pth'
```


## ⚗️结果下载


* *我们上传到Github的代码是经过重新组织整理的。下面提供的模型权重也是采用重新组织整理后的代码训练得到的。因此精度可能会和原始论文不完全一致。*

### `一、 VMamba (编码器)的预训练权重`

|      方法      | ImageNet (ckpt) | 
|:------------:| :---: |
| VMamba-Tiny  | [[GDrive](https://drive.google.com/file/d/160PXughGMNZ1GyByspLFS68sfUdrQE2N/view?usp=drive_link)]    
| VMamba-Small | [[GDrive](https://drive.google.com/file/d/1dxHtFEgeJ9KL5WiLlvQOZK5jSEEd2Nmz/view?usp=drive_link)] 
| VMamba-Base  |  [[GDrive](https://drive.google.com/file/d/1kUHSBDoFvFG58EmwWurdSVZd8gyKWYfr/view?usp=drive_link)]

### `二、 语义变化检测`
|          方法          |                                                    权重                                                    |
|:--------------------:|:--------------------------------------------------------------------------------------------------------:|
|    FoBaMambaBased    | [SECOND + LevirSCD + JL1](https://pan.baidu.com/s/11mcbilrctWH02sazZEt3Xg) (**Extraction code:** `foba`) 
| FoBaTransformerBased | [SECOND + LevirSCD + JL1](https://pan.baidu.com/s/11mcbilrctWH02sazZEt3Xg) (**Extraction code:** `foba`) 

## 📜引用

如果我们的代码有助于您的研究，请考虑引用我们的论文，并给我们一个 ⭐️ :)
```
@ARTICLE{11268372,
  author={Zhang, Haotian and Guo, Han and Chen, Keyan and Chen, Hao and Zou, Zhengxia and Shi, Zhenwei},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={FoBa: A Foreground-Background co-Guided Method and New Benchmark for Remote Sensing Semantic Change Detection}, 
  year={2025},
  volume={},
  number={},
  pages={1-1},
  keywords={Semantics;Remote sensing;Transformers;Feature extraction;Annotations;Roads;Multitasking;Spatial resolution;Landsat;Land surface;Semantic change detection (SCD);foreground-background co-guided;bi-temporal interaction;mamba;new benchmark},
  doi={10.1109/TGRS.2025.3636947}}

@ARTICLE{10902569,
  author={Zhang, Haotian and Chen, Keyan and Liu, Chenyang and Chen, Hao and Zou, Zhengxia and Shi, Zhenwei},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={CDMamba: Incorporating Local Clues Into Mamba for Remote Sensing Image Binary Change Detection}, 
  year={2025},
  volume={63},
  number={},
  pages={1-16},
  keywords={Feature extraction;Transformers;Remote sensing;Convolutional neural networks;Visualization;Artificial intelligence;Spatiotemporal phenomena;Computational modeling;Attention mechanisms;Computer vision;Bi-temporal interaction;change detection (CD);high-resolution optical remote sensing image;Mamba;state-space model},
  doi={10.1109/TGRS.2025.3545012}}

@ARTICLE{10471555,
  author={Zhang, Haotian and Chen, Hao and Zhou, Chenyao and Chen, Keyan and Liu, Chenyang and Zou, Zhengxia and Shi, Zhenwei},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={BiFA: Remote Sensing Image Change Detection With Bitemporal Feature Alignment}, 
  year={2024},
  volume={62},
  number={},
  pages={1-17},
  keywords={Feature extraction;Task analysis;Remote sensing;Transformers;Interference;Decoding;Optical flow;Bitemporal interaction (BI);change detection (CD);feature alignment;flow field;high-resolution optical remote sensing image;implicit neural representation},
  doi={10.1109/TGRS.2024.3376673}}
```

## 🤝致谢
本项目采用和借鉴了 VMamba ([paper](https://arxiv.org/abs/2401.10166), [code](https://github.com/MzeroMiko/VMamba)), ChangeMamba ([paper](https://ieeexplore.ieee.org/document/10565926), [code](https://github.com/ChenHongruixuan/ChangeMamba))等优秀的工作!!

## 🙋联系我们 
***如有任何问题，请随时 [联系我们。](haotianzhang@buaa.edu.cn)***

