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

### Run on our data

```
python nerfstudio/scripts/change_det.py \
  -c <data_folder>/<scene_name>/config.yml \
  -t <data_folder>/<scene_name>/transforms.json \
  -o <data_folder>/<scene_name> \
  -ckpt <data_folder>/<scene_name>/nerfstudio_models/
```


### Run on custom data

Our code will be released soon!
