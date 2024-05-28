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
import shutil
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.utils.poses import to4x4


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
        json.dump(content, file, indent=4)


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


def save_imgs(imgs, paths):
    """
    Save images to img paths

    Args:
        imgs (N, 3, H, W): RGB images
        paths (N-str-list): Image paths
    """
    assert len(imgs.shape) == 4 and imgs.shape[1] == 3
    assert len(imgs) == len(paths)
    for img, path in tqdm(zip(imgs, paths), desc="Saving images"):
        img = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        img = Image.fromarray(img).save(path)


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


def save_masks(masks, paths):
    """
    Save masks to img paths

    Args:
        masks (N, 1, H, W): Masks
        paths (N-str-list): Image paths
    """
    assert len(masks.shape) == 4 and masks.shape[1] == 1
    assert len(masks) == len(paths)
    for mask, path in tqdm(zip(masks, paths), desc="Saving masks"):
        mask = mask.squeeze(0).cpu().numpy() * 255
        mask = Image.fromarray(mask.astype(np.uint8))
        mask.save(path)


def cameras_to_params(cameras, device="cuda"):
    """
    Convert a Cameras intrinsic and extrinsic and dimension parameters

    Args:
        cameras (Cameras): Cameras object

    Returns:
        poses (Bx4x4): Camera parameters
        Ks (Bx3x3): Camera intrinsics
        dist_params (Bx4): Camera distortion parameters
        H (int): Image height
        W (int): Image width
    """
    poses = to4x4(cameras.camera_to_worlds).to(device)
    poses[:, 0:3, 1:3] = -poses[:, 0:3, 1:3] # OpenGL to OpenCV
    Ks = cameras.get_intrinsics_matrices().to(device)
    dist_params6 = cameras.distortion_params.to(device)
    dist_params = dist_params6[:, [0, 1, 4, 5]]
    H = cameras.height.squeeze()[0].item()
    W = cameras.width.squeeze()[0].item()
    return poses, Ks, dist_params, H, W


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


def save_alpha_transparent_train_data(
    masks, masks_new, pose_change, data_dir, output_dir, gt_pose_change=None
):
    """
    Save alpha transparency training data
    NOTE: Data must have the NeRF Update format
    -rgb:
        -frame_{i:05g}.png
    -rgb_new
        -frame_{i:05g}.png
    -masks
        -mask_{i:05g}.png
    -masks_new
        -mask_{i:05g}.png
    -transforms.json

    Args:
        masks (N, 1, H, W): Masks
        masks_new (N, 1, H, W): New masks
        pose_change (4, 4): Pose change matrix
        data_dir (str): Data directory
        output_dir (str): Output directory
    """
    from nerfstudio.utils.img_utils import rgb2rgba
    assert os.path.isdir(data_dir), f"{data_dir} does not exist"
    assert pose_change.shape == (4, 4)
    if type(pose_change) == torch.Tensor:
        pose_change = pose_change.detach().cpu().numpy()
    assert len(masks.shape) == len(masks_new.shape) == 4
    assert masks.shape[1] == masks_new.shape[1] == 1
    if gt_pose_change is None:
        print("Warning: Not passing gt_pose_change lead to incorret eval")
    else:
        assert gt_pose_change.shape == (4, 4)
        if type(gt_pose_change) == torch.Tensor:
            gt_pose_change = gt_pose_change.detach().cpu().numpy()

    # Load transforms.json
    tjson = load_from_json(Path(f"{data_dir}/transforms.json"))
    frames = tjson["frames"]
    trainfiles = tjson["train_filenames"]
    trainfiles_old = [
        f"{data_dir}/{f}" for f in trainfiles if not "rgb_new" in f
    ]
    trainfiles_new = [
        f"{data_dir}/{f}" for f in trainfiles if "rgb_new" in f
    ]
    assert len(trainfiles_old) == len(masks)
    assert len(trainfiles_new) == len(masks_new)

    # Save masks to temp folders
    mask_files_old = [
        f"{output_dir}/obj_masks/{os.path.basename(f)}" for f in trainfiles_old
    ]
    mask_files_new = [
        f"{output_dir}/obj_masks_new/{os.path.basename(f)}"
        for f in trainfiles_new
    ]    
    if not os.path.exists(output_dir / "obj_masks"):
        os.makedirs(output_dir / "obj_masks")
        save_masks(masks, mask_files_old)
    if not os.path.exists(output_dir / "obj_masks_new"):
        os.makedirs(output_dir / "obj_masks_new")
        save_masks(masks_new, mask_files_new)

    # Convert rgb + masks to rgba
    if not os.path.exists(output_dir / "rgba"):
        os.makedirs(output_dir / "rgba")
    rgb2rgba(trainfiles_old, mask_files_old, f"{output_dir}/rgba")
    if not os.path.exists(output_dir / "rgba_new"):
        os.makedirs(output_dir / "rgba_new")
    rgb2rgba(trainfiles_new, mask_files_new, f"{output_dir}/rgba_new")

    if os.path.exists(output_dir / "obj_masks"):
        shutil.rmtree(output_dir / "obj_masks")
    if os.path.exists(output_dir / "obj_masks_new"):
        shutil.rmtree(output_dir / "obj_masks_new")

    # Update filenames in transforms.json
    for i in range(len(tjson["train_filenames"])):
        tjson["train_filenames"][i] = \
            tjson["train_filenames"][i].replace("rgb", "rgba")
    if gt_pose_change is not None:
        for i in range(len(tjson["test_filenames"])):
            tjson["test_filenames"][i] = \
                tjson["test_filenames"][i].replace("rgb", "rgba")
        for i in range(len(tjson["val_filenames"])):
            tjson["val_filenames"][i] = \
                tjson["val_filenames"][i].replace("rgb", "rgba")
    
    # Update camera poses and remove masks
    for frame in frames:
        # remove masks
        frame.pop("mask_path", None)
        # Update camera poses
        if "rgb_new" in frame["file_path"]:
            pose = np.array(frame["transform_matrix"])
            pose[0:3, 1:3] = -pose[0:3, 1:3] # OpenGL to OpenCV
            if frame["file_path"] in trainfiles_new or gt_pose_change is None:
                pose = np.linalg.inv(pose_change) @ pose
            else:
                pose = np.linalg.inv(gt_pose_change) @ pose
            pose[0:3, 1:3] = -pose[0:3, 1:3] # OpenCV to OpenGL
            frame["transform_matrix"] = pose.tolist()
        frame["file_path"] = frame["file_path"].replace("rgb", "rgba")
    
    write_to_json(Path(f"{output_dir}/transforms_obj.json"), tjson)

    print("Run the following command to reconstruct the full object model:")
    print(
        f"ns-train splatfacto --pipeline.model.background-color random" + 
        " --pipeline.model.random_scale 0.5"+
        f" nerfstudio-data --data {output_dir}/transforms_obj.json"
    )


