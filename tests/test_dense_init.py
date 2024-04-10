# Test using accurate dense initialization for 3DGS training
# MUST use kubric generated data: hdri{ID}_{obj}_remove
import argparse
import os
import cv2
import shutil
import torch
from PIL import Image
import numpy as np
import open3d as o3d
from pathlib import Path
from matplotlib import pyplot as plt
from nerfstudio.utils.io import load_from_json, write_to_json
from nerfstudio.utils.proj_utils import depths_to_points
from nerfstudio.utils.debug_utils import debug_depths
from transformers import pipeline
from scipy.ndimage import distance_transform_cdt
from scipy.optimize import least_squares


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


def masked_mahalanobis_distance(mask):
    # Compute Manhattan distance of a masked image:
    # inside the mask, distances are 0s, outside they are positive
    mask = mask.astype(float) / 255
    distance_outside = distance_transform_cdt(1 - mask, metric='taxicab')
    # Set distances inside the mask to 0
    distance_outside[mask > 0] = 0
    return distance_outside


def inpaint_depth_map(gt_depth, mask, disp, method="wls"):
    """
    Use predicted disparity map to fill the mask in the GT depth map

    Args:
    - depth (HxW): GT depth map
    - mask (HxW bool): mask == True is the area to inpaint
    - disp (HxW): predicted disparity map
    - method (str): "Inpainting" method: 1/depth = A * disp + B
                    "gt": A = 1 / d_min - 1/d_max; B = 1/d_max
                    "ls": A, B = lsq(1/depth, disp)
                    "wls": A, B = wlsq(1/depth, disp)

    Returns:
    - depth (HxW): "Inpainted" depth map

    References:
    - GT: https://github.com/heyoeyo/muggled_dpt/blob/main/.readme_assets/results_explainer.md
    - Least-squares: https://github.com/isl-org/MiDaS/issues/171
    """
    assert gt_depth.shape == mask.shape == disp.shape
    # Normalize predicted disparity to [0, 1]
    disp = disp / disp.max()
    # Convert GT depth to GT disparity
    inv_depth = 1 / gt_depth
    inv_depth_min = inv_depth[mask <= 0].min()
    inv_depth_max = inv_depth[mask <= 0].max()
    if method == "gt":
        A = inv_depth_max - inv_depth_min
        B = inv_depth_min
    elif method in ["ls", "wls"]:
        disp_masked = disp[mask <= 0].flatten()
        inv_depth_masked = inv_depth[mask <= 0].flatten()
        disp_homo = np.vstack([disp_masked, np.ones_like(disp_masked)]).T
        if method == "ls":
            weights = np.ones_like(inv_depth_masked)
        else:
            weights = masked_mahalanobis_distance(mask)[mask <= 0].flatten()
            weights = 1 / weights**2
        def residuals(beta, X, y, weights):
            return np.sqrt(weights) * (y - X.dot(beta))
        beta_init = np.array([inv_depth_max - inv_depth_min, inv_depth_min])
        result = least_squares(
            residuals, beta_init, args=(disp_homo, inv_depth_masked, weights),
        )
        assert result.success, "Least squares fitting failed!"
        A, B = result.x[0], result.x[1]
    else:
        raise ValueError(f"Method: {method} not supported!")
    out_depth = np.zeros_like(gt_depth)
    out_depth[mask <= 0] = gt_depth[mask <= 0]
    # "Inpaint" the GT disparity using the predicted disparity
    out_depth[mask > 0] = 1 / (A * disp[mask > 0] + B)
    return out_depth


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
gt_depths = []
for pose in train_poses:
    depth_map = compute_depth_map(pose, K, H, W)
    gt_depths.append(depth_map)
gt_depths = np.stack(gt_depths, axis=0)

# Depth-anything to predict disparity maps
pipe = pipeline(
    task="depth-estimation", model="LiheYoung/depth-anything-small-hf"
)
depths = []
for ii, ind in enumerate(train_indices):
    img = Image.open(Path(args.tjson).parent / f"rgb_new/frame_{ind:05d}.png")
    disp = pipe(img)["predicted_depth"]
    disp = torch.nn.functional.interpolate(
        disp.unsqueeze(1), size=img.size[::-1], mode="bicubic",
        align_corners=False,
    )
    pred_depth = inpaint_depth_map(
        gt_depths[ii], masks[ii], disp.squeeze().numpy()
    )
    depths.append(pred_depth)


# Evaluate the depth accuracy with L1 error
l1_errors = []
for gt, pred, mask in zip(gt_depths, depths, masks):
    l1_error = np.abs(gt[mask > 0] - pred[mask > 0]).mean()
    l1_errors.append(l1_error)
print(f"Mean L1 error: {np.mean(l1_errors)}")


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


print("Train splatfacto with the following command:")
print(
    "ns-train splatfacto --vis viewer+tensorboard" +
    "--experiment-name <exp_name> --steps_per_eval_all_images 100 " +
    "--pipeline.model.warmup_length 0" + # No warmup
    "--pipeline.model.refine_every 50000 " + # No densification
    "--pipeline.model.use_scale_regularization True " + # Encourage isotropic
    "--pipeline.model.num_downscales 0 " + # Don't downscale
    "--pipeline.model.continue_cull_post_densification False " +
    "--pipeline.model.reset_alpha_every 30000 " + # No opacity reset
    "--pipeline.model.sh_degree_interval 10 " +
    "--max-num-iterations 1000" +
    f"nerfstudio-data --data {args.out}/transforms.json" +
    " --auto-scale-poses=False --center-method none --orientation-method none"
)