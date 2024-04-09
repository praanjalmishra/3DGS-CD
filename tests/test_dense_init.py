# Test using accurate dense initialization for 3DGS training
# MUST use kubric generated data: hdri{ID}_{obj}_remove
import argparse
import os
import cv2
import torch
from PIL import Image
import numpy as np
import open3d as o3d
from pathlib import Path
from matplotlib import pyplot as plt
from nerfstudio.utils.io import load_from_json, write_to_json
from nerfstudio.utils.proj_utils import depths_to_points


def compute_depth_map(camera_to_world_pose, intrinsics, height, width):
    """
    Compute depth map to the z=0 ground plane in the world coordinate

    Args:
    - camera_to_world_pose (4x4): camera pose in world coordinates
    - intrinsics (3x3): camera intrinsics matrix
    - height (int): image height
    - width (int): image width

    Returns:
    - depth_map (HxW):, depth map to the z=0 ground plane
    """
    # Generate a grid of pixel coordinates
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    # Create homogeneous coordinates for the pixel grid
    pixels_homog = np.stack(
        [x.flatten(), y.flatten(), np.ones_like(x.flatten())], axis=0
    )
    # Invert intrinsics for backprojection
    inv_intrinsics = np.linalg.inv(intrinsics)
    # Back-project pixels to rays in camera space
    rays_camera = np.dot(inv_intrinsics, pixels_homog)
    # Normalize rays to ensure they represent directions
    rays_camera /= rays_camera[2, :]
    # Transform rays to world space
    rays_world = np.dot(camera_to_world_pose[:3, :3], rays_camera)
    # Compute scale factor 't' for intersection with ground plane
    camera_position_world = camera_to_world_pose[:3, 3]
    t = -camera_position_world[2] / rays_world[2, :]
    # Calculate depths: Since we're interested in z-coordinates, 
    # depth can be directly calculated
    depth_map = t.reshape(height, width)
    return depth_map

parser = argparse.ArgumentParser()
parser.add_argument(
    "--tjson", "-t", type=str, required=True,
    help="Path to transforms.json in hdri{ID}_{obj}_remove"
)
parser.add_argument(
    "--ind", "-i", type=int, nargs="+", required=True,
    help="Indices of the sparse view images"
)
parser.add_argument(
    "--out", "-o", type=str, required=True,
    help="Path to save the new transforms.json"
)
parser.add_argument(
    "--eval", "-e", action="store_true",
)
args = parser.parse_args()

assert os.path.isfile(args.tjson), f"{args.tjson} doesn't exist"
assert os.path.isdir(args.out), f"{args.out} doesn't exist"


# Modify the transforms.json file
if args.eval:
    train_indices = args.ind[0::2]
    eval_indices = args.ind[1::2]
else:
    train_indices = args.ind
    eval_indices = []
data = load_from_json(Path(args.tjson))
data["train_filenames"] = [f"rgb_new/frame_{i:05d}.png" for i in train_indices]
data["val_filenames"] = [f"rgb_new/frame_{i:05d}.png" for i in eval_indices]
data["test_filenames"] = data["val_filenames"]


# Read the camera intrinsics and extrinsics
fx, fy, cx, cy = data["fl_x"], data["fl_y"], data["cx"], data["cy"]
K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
H, W = data["h"], data["w"]
# dist_params = data["k1"], data["k2"], data["p1"], data["p2"]
train_poses, new_frames = [], []
for fr in data["frames"]:
    if fr["file_path"] in data["train_filenames"]:
        pose = fr["transform_matrix"]
        pose = np.array(pose)
        pose[0:3, 1:3] = -pose[0:3, 1:3] # OpenGL to OpenCV
        train_poses.append(pose)
        new_frames.append(fr)
    elif fr["file_path"] in data["val_filenames"]:
        new_frames.append(fr)
train_poses = np.stack(train_poses)
data["frames"] = new_frames
assert len(train_poses) == len(train_indices), \
    f"One or more input indices doesn't exist in {args.tjson} file"

# Read masks
masks = []
for i in train_indices:
    mask = Image.open(Path(args.tjson).parent / f"masks_new/mask_{i:05d}.png")
    mask = np.array(mask)
    masks.append(mask)

# Read images
colors = []
for i in train_indices:
    img = Image.open(Path(args.tjson).parent / f"rgb_new/frame_{i:05d}.png")
    img = np.array(img)[..., :3]
    img = img[mask>0, :]
    colors.append(img)
colors = np.concatenate(colors, axis=0)

# Compute the camera depth maps to the z=0 ground plane in the world coordinate
depths = []
for pose in train_poses:
    depth_map = compute_depth_map(pose, K, H, W)
    depths.append(depth_map)
depths = np.stack(depths)

# Backproject the depths to 3D points
points = []
for depth, pose, mask in zip(depths, train_poses, masks):
    xx, yy = np.meshgrid(np.arange(W), np.arange(H))
    pixels = np.stack([xx, yy], axis=-1)
    pixels = torch.from_numpy(pixels).float()
    depth = torch.from_numpy(depth).float().reshape(H, W, 1)
    point = depths_to_points(
        pixels, depth,
        torch.from_numpy(pose).float(), torch.from_numpy(K).float()
    )
    point = point[mask > 0]
    points.append(point.numpy())
points = np.concatenate(points, axis=0)

# Save the open3d color pcd
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points.reshape(-1, 3))
pcd.colors = o3d.utility.Vector3dVector(colors.reshape(-1, 3) / 255)
pcd = pcd.voxel_down_sample(0.001)
o3d.io.write_point_cloud(f"{args.out}/sparse_pc.ply", pcd)
data["ply_file_path"] = "sparse_pc.ply"
# Save the new transforms.json
write_to_json(Path(args.out) / "transforms.json", data)

# Move images and masks to new folders
import shutil
if not os.path.isdir(f"{args.out}/rgb_new"):
    os.makedirs(f"{args.out}/rgb_new")
for i in train_indices + eval_indices:
    shutil.copyfile(
        Path(args.tjson).parent / f"rgb_new/frame_{i:05d}.png",
        f"{args.out}/rgb_new/frame_{i:05d}.png"
    )

if not os.path.isdir(f"{args.out}/masks_new"):
    os.makedirs(f"{args.out}/masks_new")
for i in train_indices + eval_indices:
    shutil.copyfile(
        Path(args.tjson).parent / f"masks_new/mask_{i:05d}.png",
        f"{args.out}/masks_new/mask_{i:05d}.png"
    )
