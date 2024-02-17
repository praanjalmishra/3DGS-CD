# Merge colmap generated transforms.json files

import argparse
import json
import math
import os
import re

import numpy as np
import torch
from pathlib import Path

from nerfstudio.cameras.camera_utils import auto_orient_and_center_poses


def merge_json(
        json_file_old, json_file_new, output_file, 
        new_view_indices = [0, 1, 2, 3], split_fraction: float = 0.9, 
        replace_old_views: bool = True, eval_on_new: bool = True
):
    """Merge two transforms.json files to generate data for object re-configs.

    Args:
        json_file_old: Path to the json file containing old view info.
        json_file_new: Path to the json file contianing new view info.
        new_view_indices: Indices of new images to use in training.
        output_file: Path to the output json file.
        split_fraction: number of training images / number of all images.
        replace_old_views: True to replace old views with new ones.
        eval_on_new: True to eval on new views, False to eval on old views.
        eval_on_all_pix: True to remove mask for eval images.
    """
    # load json files
    with open(json_file_old) as f1, open(json_file_new) as f2:
        data_old = json.load(f1)
        data_new = json.load(f2)

    # Train-Eval split logic for the old views (borrowed from nerfstudio_dataparser.py)
    num_images = len(data_old['frames'])
    num_train_images = math.ceil(num_images * split_fraction)
    i_all = np.arange(num_images)
    i_train = np.linspace(0, num_images - 1, num_train_images, dtype=int)
    i_eval = np.setdiff1d(i_all, i_train)

    # Specify training and evaluation file names
    filenames = [f['file_path'] for f in data_old['frames']]
    data_old['train_filenames'] = [filenames[i] for i in i_train]
    data_old['val_filenames'] = [filenames[i] for i in i_eval]
    data_old['test_filenames'] = data_old['val_filenames']

    # sort data_new frames by filename
    data_new_frames = sorted(data_new['frames'], key=lambda x: x['file_path'])
    if replace_old_views:
        for idx in new_view_indices:
            new_view_base_filename = data_new_frames[idx]['file_path'].split('/')[-1]
            # create old view filename using the id
            old_view_filename = 'rgb/' + new_view_base_filename
            if old_view_filename in data_old['train_filenames']:
                # replace the matching old view with the new view
                data_old['train_filenames'][data_old['train_filenames'].index(old_view_filename)] =\
                    data_new_frames[idx]['file_path']
            else:
                # no matching old view found, replace the next available old view instead
                # wrap around if reached the end of the list
                new_view_id = int(re.search('frame_(\d+).png', new_view_base_filename).group(1))
                while old_view_filename not in data_old['train_filenames']:
                    new_view_id = (new_view_id + 1) % len(data_old['train_filenames'])
                    # replace the next available old view with the new view
                    old_view_filename = f"rgb/frame_{new_view_id:05d}.png"
                data_old['train_filenames'][data_old['train_filenames'].index(old_view_filename)] =\
                    data_new_frames[idx]['file_path']
                print(
                    f"Warning: {new_view_base_filename} in old views is used for evaluation. "
                    f"Replacing {old_view_filename} with new view instead."
                )
    else:
        new_filenames = [
            data_new_frames[idx]['file_path'] for idx in new_view_indices
        ]
        data_old['train_filenames'] = new_filenames + data_old['train_filenames']

    # Concatenate old and new views
    data_old['frames'] += data_new_frames

    if eval_on_new:
        # Evaluate on all new views not used for training
        data_old['val_filenames'] = [
            frame["file_path"] for frame in data_new_frames
            if frame['file_path'] not in data_old['train_filenames']
        ]
        data_old['test_filenames'] = data_old['val_filenames']

    with open(output_file, 'w') as fout:
        json.dump(data_old, fout, indent=4)


def transform_colmap_pcd(
    ply_file_path, transform_matrix, scale_factor, ply_out_path
):
    """
    Transform the point cloud positions and colors from .ply
    (adpated from nerfstudio)

    Args:
        ply_file_path: Path to sparse_pc.ply file
        transform_matrix: Matrix to transform world coordinates
        scale_factor: How much to scale the camera origins by
        ply_out_path: Path to save the transformed .ply file
    """
    assert os.path.isfile(ply_file_path), f"{ply_file_path} does not exist."
    assert type(transform_matrix) is torch.Tensor and \
        transform_matrix.shape[0] in [3, 4] and \
        transform_matrix.shape[1]==4, "transform_mat must be 4x4 Tensor"
    assert scale_factor > 0, "scale_factor must be positive"
    if transform_matrix.shape[0] == 3:
        transform_matrix = torch.cat((
                transform_matrix,
                torch.tensor([[0.0, 0, 0, 1.0]]).to(transform_matrix)
        ))
    # Importing open3d is slow, so we only do it if we need it.
    import open3d as o3d
    # Read the COLMAP point cloud
    pcd = o3d.io.read_point_cloud(str(ply_file_path))
    assert len(pcd.points) > 0, "No points found in the point cloud."
    # Transform the point cloud
    points3D = torch.from_numpy(np.asarray(pcd.points, dtype=np.float32))
    points3D = torch.cat((points3D, torch.ones_like(points3D[..., :1])), -1) @\
        transform_matrix.T
    points3D *= scale_factor
    # Save the transformed point cloud
    pcd.points = o3d.utility.Vector3dVector(points3D[..., :3].numpy())
    o3d.io.write_point_cloud(str(ply_out_path), pcd)


def extract_last_number(s):
    # finds the last group of digits in the string
    match = re.search(r'(\d+)(?!.*\d)', s)  
    if match:
        return int(match.group())
    else:
        raise ValueError(f"No number found in string {s}")


def merge_colmap_json(
        json_old, json_new, out_json_pretrain, out_json_recfg,
        recenter=True, rescale=True, new_view_indices=None,
        split_fraction=0.9, replace_old_views=True, eval_on_new=True
    ):
    """
    Convert the colmap dataset to our dataset format from kubric

    Args:
        json_old (str): transforms.json for the pre-training dataset
        json_new (str): transforms.json for the new sparse view images
        json_pretrain (str): The output json path for pre-training
        json_recfg (str): The output json path for post-recfg dataset
        recenter (bool): True to recenter the poses
        rescale (bool): True to rescale the poses
        new_view_indices (list or None): Indices of new views to use in training
        split_fraction (float): Fraction of training images
    """
    assert os.path.exists(json_old), f"{json_old} does not exist."
    assert os.path.exists(json_new), f"{json_new} does not exist."

    with open(json_old, 'r') as infile:
        data_old = json.load(infile)
    with open(json_new, 'r') as infile:
        data_new = json.load(infile)

    # --- Re-center, reorient, and rescale the pre-training poses ---
    # (borrowed from nerfstudio)
    poses = [
        torch.from_numpy(np.array(fr["transform_matrix"]))
        for fr in data_old["frames"]
    ]
    poses = torch.stack(poses, dim=0).float()
    if recenter:
        poses, transform = auto_orient_and_center_poses(poses)
    if rescale: 
        scale_factor = float(torch.max(torch.abs(poses[:, :3, 3])))
        poses[:, :3, 3] /= scale_factor
    poses = poses.numpy().tolist()

    # --- Revise the pre-training frames names ---
    # Get the number from the first pose in the frames
    idx0 = extract_last_number(data_old["frames"][0]["file_path"])
    # Modify the frames in the original data
    new_frames = []
    for ii, frame in enumerate(data_old["frames"]):
        new_frame_number = extract_last_number(frame["file_path"]) - idx0
        new_file_path = f"rgb/frame_{new_frame_number:05d}.png"
        new_frame = {
            "file_path": new_file_path,
            "transform_matrix": poses[ii] + [[0.0, 0.0, 0.0, 1.0]]
        }
        new_frames.append(new_frame)
    data_old["frames"] = new_frames
    # Write to the output json file
    with open(out_json_pretrain, 'w') as outfile:
        json.dump(data_old, outfile, indent=4)

    # --- recenter, reorient, and rescale the new sparse view poses ---
    poses_new = [
        torch.from_numpy(np.array(fr["transform_matrix"]))
        for fr in data_new["frames"]
    ]
    poses_new = torch.stack(poses_new, dim=0).float()
    if recenter:
        poses_new = transform @ poses_new
    if rescale:
        poses_new[:, :3, 3] /= scale_factor
    poses_new = poses_new.numpy().tolist()
    # 
    data_new["frames"] = [
        {
            "file_path": frame["file_path"],
            "transform_matrix": poses_new[ii] + [[0.0, 0.0, 0.0, 1.0]]
        }
        for ii, frame in enumerate(data_new["frames"])
    ]
    with open(out_json_recfg, 'w') as outfile:
        json.dump(data_new, outfile, indent=4)

    # --- recenter, reorient, and rescale the COLMAP point cloud ---
    ply_file_path = f"{os.path.dirname(json_old)}/sparse_pc.ply"
    assert os.path.isfile(ply_file_path), f"{ply_file_path} does not exist."
    if recenter:
        transform_colmap_pcd(
            ply_file_path, transform, 1.0, ply_file_path
        )
    if rescale:
        transform_colmap_pcd(
            ply_file_path, torch.eye(4), 1.0 / scale_factor, ply_file_path
        )

    # --- Merge the pre-training and new sparse view transforms.json ---
    if new_view_indices is None:
        data_path = Path(json_old).parent / "rgb_new"
        png_files = data_path.glob("*.png")
        new_view_indices = [i for i, _ in enumerate(png_files)]
    merge_json(
        out_json_pretrain, out_json_recfg, out_json_recfg, 
        new_view_indices=new_view_indices, split_fraction=split_fraction, 
        replace_old_views=replace_old_views, eval_on_new=eval_on_new
    )
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-jo", "--json_old", type=str, required=True)
    parser.add_argument("-jn", "--json_new", type=str, required=True)
    parser.add_argument("-oo", "--out_json_pretrain", type=str, required=True)
    parser.add_argument("-on", "--out_json_recfg", type=str, required=True)
    parser.add_argument("-n", "--new_view_indices", nargs="+", type=int, default=None)
    parser.add_argument("-f", "--split_fraction", type=float, default=0.9)
    args = parser.parse_args()

    merge_colmap_json(
        args.json_old, args.json_new, args.out_json_pretrain, args.out_json_recfg,
        new_view_indices=args.new_view_indices, split_fraction=args.split_fraction,
    )