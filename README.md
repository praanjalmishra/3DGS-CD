[![arXiv](https://img.shields.io/badge/arXiv-2411.03706-b31b1b.svg)](https://arxiv.org/abs/2411.03706)
# 3DGS-CD
3D Gaussian Splatting-based Change Detection for Physical Object Rearrangement

![1](https://github.com/user-attachments/assets/659806cd-d127-48aa-addb-771db4458926)


TLDR: We estimate **3D object-level changes** from two sets of unaligned RGB images using **3D Gaussian Splatting** as the scene representation, enabling **accurate** recovery of shapes and pose changes of rearranged objects in **cluttered environments** within **tens of seconds** using **sparse (as few as one) new images**.

[![Watch the video](https://via.placeholder.com/100)](https://github.com/user-attachments/assets/ef073079-6bed-4a06-8f0e-4e765a5fd680)

## Data

The **3DGS-CD dataset** can be found [here](https://drive.google.com/drive/folders/1OPUu643bkbAoryASNMi8_iDJGnypotc0?usp=drive_link).
All the RGB images have been pre-processed (i.e. downscaled and undistorted).
Below is the structure of the data folder:
```
scene_name
  -rgb: Pre-change images
  -rgb_new: Post-change images
    - Images at indices 0, 2, 4, ... are used for change detection
    - Images at indices 1, 3, 5, ... are used for evaluation
  -masks_gt: Ground truth change masks for evaluation images
```

## Installation

### 1. Install nerfstudio dependencies

#### 1.0 Create conda environment

```bash
conda create --name gscd -y python=3.8
conda activate gscd
pip install --upgrade pip
```

#### 1.1 Install CUDA dependencies

Install PyTorch with CUDA and [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn).
`cuda-toolkit` is required for building `tiny-cuda-nn`.

For CUDA 11.8:

```bash
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

See [Dependencies](https://github.com/nerfstudio-project/nerfstudio/blob/main/docs/quickstart/installation.md#dependencies)
in the Installation documentation for more.

#### 1.2 Install nerfstudio dependencies

```
git clone https://github.com/520xyxyzq/3DGS-CD.git 3dgscd
cd 3dgscd
pip install --upgrade pip setuptools
pip install -e .
```

### 2. Install 3DGS-CD dependencies

#### 2.1 Install EfficientSAM

Follow EfficientSAM [instructions](https://github.com/yformer/EfficientSAM)

**OR** if you prefer pip install:

```bash
pip install git+https://github.com/yformer/EfficientSAM.git@c9408a74b1db85e7831977c66e9462c6f4891729
```
Download the EfficientSAM model weight from [here](https://github.com/yformer/EfficientSAM/blob/main/weights/efficient_sam_vits.pt.zip) and change line 21 of [this file](https://github.com/yformer/EfficientSAM/blob/main/efficient_sam/build_efficient_sam.py) in your python lib to point to the downloaded weight. 

#### 2.2 Install HLoc

```bash
pip install git+https://github.com/cvg/Hierarchical-Localization.git@73a3cb0f59659306eb6c15c7213137b2196c5ceb
```


## Instructions

### Run on our data

```
python nerfstudio/scripts/change_det.py \
  --config <data_folder>/<scene_name>/config.yml \
  --transform <data_folder>/<scene_name>/transforms.json \
  --output <data_folder>/<scene_name> \
  --ckpt <data_folder>/<scene_name>/nerfstudio_models/
```

**NOTE**: 
1. All output masks are saved under `<data_folder>/<scene_name>/masks_new/`. The `mask_*.png` files are the object move-out masks (previous location), and the `mask_new_*.png` files are the move-in masks (new location).
2. We have uploaded the pre-change 3DGS models with the data. This means you do not need to train the pre-change 3DGS models.
3. The post-change camera pose estimation is already handled for you, and the poses are stored in the transforms.json file.


### Run on custom data

Our code will be released soon!
