# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Input/output utils.
"""

import json
import numpy as np
import os
import torch
from pathlib import Path
from PIL import Image
from nerfstudio.cameras.cameras import Cameras


def load_from_json(filename: Path):
    """Load a dictionary from a JSON filename.

    Args:
        filename: The filename to load from.
    """
    assert filename.suffix == ".json"
    with open(filename, encoding="UTF-8") as file:
        return json.load(file)


def write_to_json(filename: Path, content: dict):
    """Write data to a JSON file.

    Args:
        filename: The filename to write to.
        content: The dictionary data to write.
    """
    assert filename.suffix == ".json"
    with open(filename, "w", encoding="UTF-8") as file:
        json.dump(content, file)


def read_imgs(paths, device='cpu'):
    """
    Read images from img paths

    Args:
        paths (N-str-list): Image paths
        device (str): Device to move the images to

    Returns:
        imgs (N, 3, H, W): RGB images
    """
    imgs = []
    for path in paths:
        assert os.path.isfile(path), f"{path} does not exist"
        img = Image.open(path)
        img = torch.from_numpy(np.array(img)) / 255
        if "mask" in os.path.basename(path):
            img= 1 - img
            img = img.bool().unsqueeze(-1)
        else:
            if img.shape[-1] == 4:
                img = img[..., :-1]
        imgs.append(img)
    imgs = torch.stack(imgs).to(device).permute(0, 3, 1, 2)
    return imgs


def read_masks(paths, device='cpu'):
    """
    Read masks from img paths

    Args:
        paths (N-str-list): Image paths
        device (str): Device to move the images to

    Returns:
        masks (N, 1, H, W): RGB images
    """
    masks = []
    for path in paths:
        assert os.path.isfile(path), f"{path} does not exist"
        mask = Image.open(path)
        mask = torch.from_numpy(np.array(mask)) / 255
        mask = mask.bool().unsqueeze(0)
        masks.append(mask)
    masks = torch.stack(masks).to(device)
    return masks


def params_to_cameras(poses, intrinsics, dist_coeffs, H, W):
    """
    Convert camera params to NeRFSutdio Cameras object
    
    Args:
        poses (N, 4, 4): Camera-to-world poses (OpenCV convention)
        intrinsics (N, 3, 3): Camera intrinsics
        dist_coeffs (N, 4): Camera distortion coefficients
        H, W (int): Image height and width
    
    Returns:
        cameras (Cameras): NeRFSutdio Cameras object
    """
    assert len(poses) == len(intrinsics) == len(dist_coeffs)
    # Distortion coefficients [k1, k2, p1, p2] to [k1, k2, k3, k4, p1, p2]
    dist_coeffs_6 = torch.zeros(len(poses), 6).to(intrinsics)
    dist_coeffs_6[:, [0, 1, 4, 5]] = dist_coeffs
    # Camera poses OpenGL to OpenCV
    poses_gl = poses.clone()
    poses_gl[:, 0:3, 1:3] = -poses_gl[:, 0:3, 1:3]
    cameras = Cameras(
        fx=intrinsics[:, 0, 0].cpu(), fy=intrinsics[:, 1, 1].cpu(),
        cx=intrinsics[:, 0, 2].cpu(), cy=intrinsics[:, 1, 2].cpu(),
        distortion_params=dist_coeffs_6.cpu(),
        height=H, width=W,
        camera_to_worlds=poses_gl[:, :3].cpu()
    )
    return cameras


def read_transforms(transforms_json, read_images=True, mode="train", device="cuda"):
    """
    Read camera parameters for training views from transforms.json

    Args:
        json (str): Path to transforms.json
        read_images (bool): If True, read images
        mode (str): Read info for train or val or other images

    Returns:
        images (N, 3, H, W or None): RGB training images
        img_fnames (N-Path-list): Image filenames
        poses (N, 4, 4): Camera-to-world poses (OpenCV convention)
        intrinsics (N, 3, 3): Camera intrinsics
        dist_coeffs (N, 4): Camera distortion coefficients
        cameras (Cameras): NeRFSutdio Cameras object
    """
    with open(transforms_json, 'r') as infile:
        data = json.load(infile)
    # read images
    data_path = Path(transforms_json).parent
    assert mode in ["train", "val", "other"], \
        "train_or_eval must be either train or val or other"
    if mode == "other":
        filenames_base = [
            fr["file_path"] for fr in data["frames"] 
            if fr["file_path"] not in 
            data["train_filenames"] + data["val_filenames"]
        ]
    else:
        filenames_base = data[f"{mode}_filenames"]
    filenames = [f"{data_path}/{fn}" for fn in filenames_base]
    if len(filenames) == 0:
        return None, [], None, None, None, None
    img_fnames = [Path(tf) for tf in filenames_base]
    H, W = read_imgs(filenames[:1]).shape[-2:]
    if read_images:
        images = read_imgs(filenames) # read in cpu to avoid OOM
    else:
        images = None
    frames = data['frames']
    # read poses
    frames_dict = {
        fr["file_path"]: torch.from_numpy(np.array(fr["transform_matrix"]))
        for fr in frames
    }
    poses = [frames_dict[fn] for fn in filenames_base] 
    poses = torch.stack(poses).to(device).float()
    poses[:, 0:3, 1:3] = -poses[:, 0:3, 1:3] # OpenGL to OpenCV
    # read camera params
    intrinsics = torch.eye(3)
    intrinsics[0, 0], intrinsics[1, 1] = data["fl_x"], data["fl_y"]
    intrinsics[0, -1], intrinsics[1, -1] = data["cx"], data["cy"]
    intrinsics = intrinsics.unsqueeze(0).repeat(len(poses), 1, 1)
    intrinsics = intrinsics.to(device)
    dist_coeffs = torch.zeros(4)
    dist_coeffs[0], dist_coeffs[1] = data["k1"], data["k2"]
    dist_coeffs[2], dist_coeffs[3] = data["p1"], data["p2"]
    dist_coeffs = dist_coeffs.unsqueeze(0).repeat(len(poses), 1)
    dist_coeffs = dist_coeffs.to(intrinsics)
    # Create NeRFStudio Cameras object
    cameras = params_to_cameras(
        poses, intrinsics, dist_coeffs, H, W
    )
    return images, img_fnames, poses, intrinsics, dist_coeffs, cameras


def read_dataset(train_dataset, read_images=True, device="cuda"):
    """
    Read images and cameras from NeRFstudio InputDataset

    Args:
        train_dataset (InputDataset): NeRFStudio InputDataset
        indices (int-list): Indices for the images and cameras to read
        read_images (bool): If True, read images, 
            otherwise only cam params (faster)
    
    Returns:
        images (N, 3, H, W or None): RGB training images
        poses (N, 4, 4): Camera-to-world poses (OpenCV convention)
        intrinsics (N, 3, 3): Camera intrinsics
        dist_coeffs (N, 4): Camera distortion coefficients
    """
    if read_images:
        images = [
            train_dataset.get_image(i) for i in range(len(train_dataset))
        ]
        images = torch.stack(images).to(device).permute(0, 3, 1, 2)
    else:
        images = None
    # Read camera poses
    cameras = train_dataset.cameras
    c2w = cameras.camera_to_worlds.to(device)
    c2w = torch.cat([c2w, torch.zeros(c2w.shape[0], 1, 4).to(c2w)], dim=1)
    c2w[:, 3, 3] = 1
    c2w[:, 0:3, 1:3] = -c2w[:, 0:3, 1:3] # OpenGL to OpenCV
    # Read intrinsics
    K = cameras.get_intrinsics_matrices().to(device)
    # Read distortion coefficients
    dist_params = cameras.distortion_params[:, [0, 1, 4, 5]].to(device)
    return images, c2w, K, dist_params