# 3DGS-based change detection
import argparse
from ast import Not
import json
import os
import re
from pathlib import Path
import datetime

import open3d as o3d
import torchvision.utils as vutils
import torchvision.transforms.functional as TF

import cv2
import numpy as np
import torch
from lightglue import LightGlue, SuperPoint, viz2d
from matplotlib import pyplot as plt
from PIL import Image
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from nerfstudio.cameras.camera_paths import get_path_from_json
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.models.splatfacto import SplatfactoModel
from nerfstudio.utils.debug_utils import (
    debug_image_pairs, debug_images, debug_masks, debug_matches,
    debug_point_cloud, debug_point_prompts, debug_depths
)
from nerfstudio.utils.effsam_utils import (
    effsam_predict, effsam_embedding, effsam_refine_masks,
    effsam_batch_predict, compute_2D_bbox, expand_2D_bbox,
    get_effsam_embedding_in_masks
)
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils.gauss_utils import transform_gaussians
from nerfstudio.utils.img_utils import (
    extract_depths_at_pixels, image_align, filter_features_with_mask,
    in_image, split_masks, dilate_masks
)
from nerfstudio.utils.io import (
    load_from_json, write_to_json, read_dataset, read_imgs, read_transforms,
    save_masks, params_to_cameras, cameras_to_params, save_imgs
)
from nerfstudio.utils.misc import extract_last_number
from nerfstudio.utils.obj_3d_seg import Object3DSeg, Obj3DFeats
from nerfstudio.utils.pcd_utils import (
    compute_3D_bbox, compute_point_cloud, expand_3D_bbox,
    point_cloud_filtering, nn_distance, pcd_size, bbox2voxel
)
from nerfstudio.utils.proj_utils import (
    depths_to_points, proj_check_3D_points, project_points
)
from nerfstudio.utils.poses import to4x4
from nerfstudio.utils.render_utils import render_cameras, render_3dgs_at_cam


def camera_clone(cameras):
    """
    Clone a Cameras object

    Args:
        cameras (Cameras): Cameras object to clone

    Returns:
        cameras_new (Cameras): Cloned Cameras object
    """
    cameras_new = Cameras(
        camera_to_worlds=cameras.camera_to_worlds.clone(),
        fx=cameras.fx.clone(), fy=cameras.fy.clone(),
        cx=cameras.cx.clone(), cy=cameras.cy.clone(),
        distortion_params=cameras.distortion_params.clone(),
        width=cameras.width, height=cameras.height
    )
    return cameras_new



class ChangeDet:
    """
    Export a 3D segmentation for a target object
    """
    #debug_dir = "/local/home/pmishra/cvg/3dgscd/debug/Mustard"

    """Directory to save debug output"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    """Device"""
    extractor = SuperPoint(max_num_keypoints=4096).eval().to(device)
    """SuperPoint extractor"""
    matcher = LightGlue(features='superpoint').eval().to(device)
    """LightGlue matcher"""

    def __init__(self, load_config: Path, output_dir: Path, debug=False):
        # Path to the config.yml file of the pretrained 3DGS
        self.load_config = load_config
        # Path to save the output 3D segmentation
        self.output_dir = output_dir

        # Path to save the debug output
        if debug:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.debug_dir = Path("debug") / f"{timestamp}"
            os.makedirs(self.debug_dir, exist_ok=True)
            print(f"[Debug] debug outputs will be saved to: {self.debug_dir}")
        else:
            self.debug_dir = None

    def image_diff(self, capture, render, threshold=1e-2, kernel_ratio=0.03):
        """
        Image differencing for change detection
        TODO: weird that capture and render cannot be swapped

        Args:
            capture (1x3xHxW): Captured image
            render (1x3xHxW): Rendered image
            threshold (float): Threshold for mask area below which it's ignored
            kernel_ratio (float): Gaussian blur kernel fractional size
            (no blur if < 0)

        Return:
            masks (Mx1xHxW): Masks for the changed regions
            masks_all (Mx1xHxW): masks + change regions occupying small areas
        """
        H, W = capture.shape[-2:]
        device = render.device
        # Align render to capture
        render, align_mask = image_align(capture, render)
        capture = capture[0].permute(1, 2, 0).cpu().numpy()
        render = render[0].permute(1, 2, 0).cpu().numpy()
        align_mask = align_mask.squeeze().cpu().numpy().astype(np.uint8)

        # Gaussian blur to filter high-freq signal that 3DGS fails to fit
        if kernel_ratio > 0:
            kernel_size = int(W * kernel_ratio)
            kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size
            capture = cv2.GaussianBlur(capture, (kernel_size, kernel_size), 0)
            render = cv2.GaussianBlur(render, (kernel_size, kernel_size), 0)

        # # Uncomment to debug
        # viz2d.plot_images([capture, render])
        # viz2d.save_plot(f"{self.debug_dir}/debug.png")
        # plt.close()

        # Get pixel-aligned image embeddings using sam_embedding
        emb1 = effsam_embedding(capture)
        emb2 = effsam_embedding(render)
        # Calculate cosine similarity between embeddings
        norm1 = torch.nn.functional.normalize(emb1, p=2, dim=1)
        norm2 = torch.nn.functional.normalize(emb2, p=2, dim=1)
        # # Uncomment to debug
        # save_imgs(norm1[:, :3], [f"{self.debug_dir}/feat1.png"])
        # save_imgs(norm2[:, :3], [f"{self.debug_dir}/feat2.png"])
        similarity_map = torch.nn.functional.cosine_similarity(
            norm1, norm2, dim=1
        )
        similarity_map = similarity_map.squeeze().cpu().numpy()
        similarity_map = (similarity_map * 255).astype(np.uint8)
        # Uncomment to debug
        ##cv2.imwrite(f"{self.debug_dir}/similarity_map.png", similarity_map)
        # Threshold the SAM cosine similarity map
        thresh = cv2.threshold(
            similarity_map, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
        )[1]
        # remove the influence of the black region in the aligned image
        thresh = thresh * align_mask
        # Uncomment to debug
        ##cv2.imwrite(f"{self.debug_dir}/thresh.png", thresh)
        # Find contours in the thresholded binary image
        contours, _ = cv2.findContours(
            thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        # Obtain masks for large enough contours
        masks, masks_all = [], []
        for contour in contours:
            mask = np.zeros((H, W))
            cv2.drawContours(
                mask, [contour], -1, (255, 255, 255), thickness=cv2.FILLED
            )
            mask = torch.from_numpy(mask).unsqueeze(0)
            masks_all.append(mask)
            if cv2.contourArea(contour) >= threshold * H * W:
                masks.append(mask)
        masks = torch.stack(masks, dim=0).to(device)
        masks_all = torch.stack(masks_all, dim=0).to(device)
        # Uncomment to debug
        # for i, mask in enumerate(masks):
        #     cv2.imwrite(
        #         f"{self.debug_dir}/mask_{i}.png",
        #         mask.squeeze().cpu().numpy()
        #     )
        return masks, masks_all


    def masks_to_3D(self, masks_list, depths, Ks, cam_poses):
        """
        Convert 2D masks to 3D point clouds.

        Args:
            masks_list (List[Tensor]): List of binary masks, each (1, H, W)
            depths (Tensor): (N, 1, H, W) Depth maps rendered from 3DGS
            Ks (Tensor): (N, 3, 3) Intrinsics per view
            cam_poses (Tensor): (N, 4, 4) Extrinsics (camera-to-world) per view

        Returns:
            pcds (List[np.ndarray]): List of 3D point clouds (K_i, 3) per mask
        """
        assert len(masks_list) == depths.shape[0], \
            f"Mismatch: {len(masks_list)} masks vs {depths.shape[0]} depths"

        device = depths.device
        pcds = []

        for i in range(len(masks_list)):
            mask_tensor = masks_list[i]  # shape: (1, H, W)
            print(f"mask_tensor: {mask_tensor.shape}")
            mask = mask_tensor.squeeze().bool()  # (H, W)
            depth = depths[i, 0]  # (H, W)
            K = Ks[i]  # (3, 3)
            cam_pose = cam_poses[i]  # (4, 4)

            # Get image pixel coordinates
            y, x = torch.where(mask)
            z = depth[y, x]
            valid = z > 0
            x, y, z = x[valid], y[valid], z[valid]

            if x.numel() == 0:
                pcds.append(torch.empty((0, 3)))
                continue

            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]

            X = (x - cx) * z / fx
            Y = (y - cy) * z / fy
            Z = z

            pts_cam = torch.stack([X, Y, Z, torch.ones_like(Z)], dim=1).T  # (4, N)
            pts_world = (cam_pose @ pts_cam).T[:, :3]  # (N, 3)

            pcds.append(pts_world.cpu().numpy()) 

        return pcds

    def save_pcds(self, pcds, output_dir):
        """
        Save 3D point clouds to PCD files.

        Args:
            pcds (List[np.ndarray]): List of 3D point clouds (K_i, 3) per mask
            output_dir (Path): Directory to save the PCD files
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcds)
        o3d.io.write_point_cloud(output_dir, pcd)
        print(f"[INFO] Saved point cloud to {output_dir}")

    def project_points_to_image(self, points_3d, K, cam_pose, image_shape):
        """
        Project 3D points into 2D image space.

        Args:
            points_3d (np.ndarray): (N, 3) points in world frame
            K (np.ndarray): (3, 3) camera intrinsics
            cam_pose (np.ndarray): (4, 4) camera-to-world matrix
            image_shape (tuple): (H, W) of the image

        Returns:
            projected_pts (np.ndarray): (M, 2) projected pixel coords
        """
        # Invert the camera pose to get world-to-camera
        w2c = np.linalg.inv(cam_pose)
        N = points_3d.shape[0]

        # Convert to homogeneous
        pts_h = np.concatenate([points_3d, np.ones((N, 1))], axis=1).T  # (4, N)
        pts_cam = (w2c @ pts_h).T[:, :3]  # (N, 3)

        # Remove points behind the camera
        valid = pts_cam[:, 2] > 0
        pts_cam = pts_cam[valid]

        # Project to image plane
        pts_2d = (K @ pts_cam.T).T  # (N, 3)
        pts_2d = pts_2d[:, :2] / pts_2d[:, 2:3]

        # Filter those falling inside the image
        H, W = image_shape
        x, y = pts_2d[:, 0], pts_2d[:, 1]
        in_bounds = (x >= 0) & (x < W) & (y >= 0) & (y < H)

        return pts_2d[in_bounds].astype(np.int32)

    def visualize_projection(self, image, points_2d, output_path):
        vis = image.copy()
        for x, y in points_2d:
            cv2.circle(vis, (x, y), 2, (0, 255, 0), -1)
        cv2.imwrite(output_path, vis)
        print(f"[DEBUG] Saved projection overlay: {output_path}")

    def get_features_in_masks(self, rgbs, masks, flip=False):
        """
        Extract SuperPoint descriptors in the masked regions

        Args:
            rgbs (Nx3xHxW): RGB images
            masks (N-list of Mx1xHxW): Image masks

        Returns:
            feats (N-list of M-list of TxC): SuperPoint descriptors
        """
        assert rgbs.shape[1] == 3 and len(rgbs) == len(masks)
        if flip:
            # Rotate images by 180 degrees if flip is True
            rgbs = torch.flip(rgbs, [2, 3])
            for m in masks:
                m = torch.flip(m, [2, 3])
        feats_all = []
        for i in range(len(rgbs)):
            feat_i = []
            for j in range(len(masks[i])):
                feat = self.extractor.extract(rgbs[i])
                # Ensure keypoints are within image
                feat["keypoints"].clamp_(min=0)
                # Filter keypoints using masks
                feat = filter_features_with_mask(feat, masks[i][j:j+1])
                if flip:
                    H, W = rgbs.shape[2], rgbs.shape[3]
                    feat['keypoints'][:, 0] = W - feat['keypoints'][:, 0] - 1
                    feat['keypoints'][:, 1] = H - feat['keypoints'][:, 1] - 1
                feat_i.append(feat)
            feats_all.append(feat_i)
        return feats_all

    def match_move_out(
        self, rgbs, depths, masks, poses, Ks, pcd_filter=0.9, embed_sim_thresh=0.4
    ):
        """
        Associate and fuse 2D move-out masks across post-change views
        to obtain obj templates
        NOTE: This is only for moved or removed objects

        Args:
            rgbs (Nx3xHxW): RGB images
            depths (Nx1xHxW): Depth images
            masks (N-list of Mx1xHxW): Sampling masks
            poses (Nx4x4): Camera poses wrt world
            Ks (Nx3x3): Camera intrinsics

        Returns:
            pcds (K-list of Lx3): Object point clouds
            pcd_feats (K-list of Obj3DFeats): Object SuperPoint descriptors
        """
        assert rgbs.shape[1] == 3
        assert depths.shape[1] == 1
        assert poses.shape[1:] == (4, 4)
        assert Ks.shape[1:] == (3, 3)
        assert len(rgbs) == len(depths) == len(masks) == len(poses) == len(Ks)
        device = rgbs.device
        N = len(rgbs)
        def compute_feats_3D(feats, depth, pose, K):
            # Extract 3D positions of keypoints
            depths_at_kps = extract_depths_at_pixels(
                feats["keypoints"].squeeze(), depth
            )
            pts_at_kps = depths_to_points(
                feats["keypoints"].squeeze(), depths_at_kps, pose, K
            )
            return pts_at_kps
        # Extract SuperPoint descriptors in multi-view masked RGB images
        feats = self.get_features_in_masks(rgbs, masks)
        # Extract EfficientSAM embeddings for objects across multi-view images
        embeds = get_effsam_embedding_in_masks(rgbs, masks)
        # Initialize object pcds
        pcds, pcd_sizes, pcd_counts, pcd_feats, pcd_embeds = [], [], [], [], []
        for j in range(len(masks[0])):
            pcd = compute_point_cloud(
                depths[0:1], poses[0:1], Ks[0:1], masks[0][j:j+1]
            )
            pcd = point_cloud_filtering(pcd, pcd_filter)
            pcds.append(pcd)
            pcd_sizes.append(pcd_size(pcd))
            pcd_counts.append(1)
            # Extract 3D positions of keypoints
            pts3D = compute_feats_3D(
                feats[0][j], depths[0:1], poses[0], Ks[0]
            )
            pcd_feats.append(Obj3DFeats([feats[0][j]], [pts3D]))
            pcd_embeds.append(embeds[0][j:j+1, :])
        # Associate move-out masks with the object point clouds w/ NN matching
        for i in range(1, N):
            dist_mat = torch.tensor(pcd_sizes).reshape(-1, 1).to(device)
            dist_mat = dist_mat.repeat(1, len(masks[i]) + len(pcds))
            new_pcds = []
            for j in range(len(masks[i])):
                pcd = compute_point_cloud(
                    depths[i:i+1], poses[i:i+1], Ks[i:i+1], masks[i][j:j+1]
                )
                pcd = point_cloud_filtering(pcd, pcd_filter)
                for k in range(len(pcds)):
                    dist_mat[k, j] = nn_distance(pcds[k], pcd)
                new_pcds.append(pcd)
            # print(f"dist_mat:\n {dist_mat.cpu().numpy()}")
            row_ind, col_ind = linear_sum_assignment(dist_mat.cpu().numpy())
            # print(f"row_ind: {row_ind}, col_ind: {col_ind}")
            # Update existing object point clouds
            for r, c in zip(row_ind, col_ind):
                # check feature sim btw matched object segments
                embed_sim = torch.cosine_similarity(
                    pcd_embeds[r], embeds[i][c], dim=-1
                )
                if c < len(masks[i]):
                    if embed_sim.max() > embed_sim_thresh:
                        pcds[r] = torch.cat((pcds[r], new_pcds[c]), dim=0)
                        pcd_sizes[r] = pcd_size(pcds[r])
                        pcd_counts[r] += 1
                        pts3D = compute_feats_3D(
                            feats[i][c], depths[i:i+1], poses[i], Ks[i]
                        )
                        pcd_feats[r].add_feats(feats[i][c], pts3D)
                        pcd_embeds[r] = torch.cat(
                            (pcd_embeds[r], embeds[i][c:c+1]), dim=0
                        )
                    else:
                        pcds.append(new_pcds[c])
                        pcd_sizes.append(pcd_size(new_pcds[c]))
                        pcd_counts.append(1)
                        pts3D = compute_feats_3D(
                            feats[i][c], depths[i:i+1], poses[i], Ks[i]
                        )
                        pcd_feats.append(Obj3DFeats([feats[i][c]], [pts3D]))
                        pcd_embeds.append(embeds[i][c:c+1, :])
            # Add new object point clouds
            for k in range(len(masks[i])):
                if k not in col_ind:
                    pcds.append(new_pcds[k])
                    pcd_sizes.append(pcd_size(new_pcds[k]))
                    pcd_counts.append(1)
                    pts3D = compute_feats_3D(
                        feats[i][k], depths[i:i+1], poses[i], Ks[i]
                    )
                    pcd_feats.append(Obj3DFeats([feats[i][k]], [pts3D]))
                    pcd_embeds.append(embeds[i][k:k+1, :])
        # Filter out object point clouds that appear in <25% of images
        pcds = [p for p, ct in zip(pcds, pcd_counts) if ct > N * 0.25]
        pcd_feats = [
            e for e, ct in zip(pcd_feats, pcd_counts) if ct > N * 0.25
        ]
        for idx, pcd in enumerate(pcds):
            self.save_pcds(
                pcd.cpu().numpy(),  
                self.debug_dir / f"move_out_obj_{idx}.ply" 
            )

        return pcds, pcd_feats

    def match_move_in(self, rgbs, masks):
        """
        Associate and fuse move-in masks across post-change views
        to obtain per-object move-in masks
        NOTE: This is only for inserted objects

        Args:
            rgbs (Nx3xHxW): RGB images
            masks (N-list of Mx1xHxW): Sampling masks

        Returns:
            masks_move_in (K-list of Lx1xHxW): Per-object move-in across views
            view_indices (K-list of L): view indices of the move-in masks
        """
        assert rgbs.shape[1] == 3
        assert len(rgbs) == len(masks)
        # Extract EffSAM embeddings for objects across multi-view images
        embeds = get_effsam_embedding_in_masks(rgbs, masks)
        # Initialize move-in masks
        obj_masks_move_in = [masks[0][i:i+1] for i in range(len(masks[0]))]
        view_indices = [[0] for _ in range(len(masks[0]))]
        for ii, (masks_i, embeds_i) in enumerate(zip(masks[1:], embeds[1:])):
            if embeds_i.size(0) == 0:
                continue
            sim_mat = torch.cosine_similarity(
                embeds[0][:,None,:], embeds_i[None,:,:], dim=-1
            )
            row_ind, col_ind = linear_sum_assignment(-sim_mat.cpu().numpy())
            # Update existing objects
            for r, c in zip(row_ind, col_ind):
                if c < len(masks_i):
                    obj_masks_move_in[r] = torch.cat(
                        (obj_masks_move_in[r], masks_i[c:c+1]), dim=0
                    )
                    view_indices[r].append(ii+1)
            # Add new objects
            for k in range(len(masks_i)):
                if k not in col_ind:
                    obj_masks_move_in.append(masks_i[k:k+1])
                    view_indices.append([ii+1])
        # Filter out move-in masks that appear in <25% of images
        obj_masks_move_in = [
            m for m in obj_masks_move_in if m.size(0) > len(rgbs) * 0.25
        ]
        view_indices = [i for i in view_indices if len(i) > len(rgbs) * 0.25]


        # Save move-in masks for debugging
        debug_dir = self.debug_dir / "move_in_masks"
        os.makedirs(debug_dir, exist_ok=True)

        for obj_id, (masks_per_obj, views) in enumerate(zip(obj_masks_move_in, view_indices)):
            for i, (mask, v_idx) in enumerate(zip(masks_per_obj, views)):
                # Convert mask to image (uint8)
                mask_img = (mask.squeeze(0).cpu().numpy() * 255).astype("uint8")

                # Save binary mask
                path = debug_dir / f"obj{obj_id}_view{v_idx}_mask.png"
                print(f"Saving mask to {path}")
                TF.to_pil_image(mask_img).save(path)

        return obj_masks_move_in, view_indices

    def masks_to_bbox3d(
        self, masks, poses, Ks, dist_params, gauss_filter_percent=0.5,
        obj_pts_filter_percent=0.8, num_sample=1000000, proj_check_cutoff=0.99
    ):
        """
        Obj masks on multi views to rough object bbox3D
        for finer obj segmentation of *inserted* objects

        Args:
            masks (Nx1xHxW): Object move-out masks on the sparse views
            poses (Nx4x4): Camera poses wrt world
            Ks (Nx3x3): Camera intrinsics
            dist_params (Nx4): Camera distortion parameters

        Returns:
            bbox3d (2-tuple of 3-tuple of floats): MinMax xyz of the obj bbox3D
        """
        # Get the 3D bbox of all Gaussians in the 3DGS model
        gauss_means = self.pipeline_pretrain.model.gauss_params.means
        gauss_means = point_cloud_filtering(gauss_means, gauss_filter_percent)
        device = gauss_means.device
        min_xyz = gauss_means.min(dim=0)[0]
        max_xyz = gauss_means.max(dim=0)[0]
        # Sample points in the 3D bbox
        pts_sampled = torch.rand(num_sample, 3, device=device) * \
            (max_xyz - min_xyz) + min_xyz
        occupied = proj_check_3D_points(
            pts_sampled, poses, Ks, dist_params, masks,
            cutoff=proj_check_cutoff
        )
        pts_occupy = pts_sampled[occupied]
        pts_occupy = point_cloud_filtering(
            pts_occupy, obj_pts_filter_percent
        )
        bbox3d = (
            pts_occupy.min(dim=0)[0].detach().cpu().numpy(),
            pts_occupy.max(dim=0)[0].detach().cpu().numpy()
        )
        return bbox3d


    def main(
        self, transforms_json=None, configs=None, checkpoint_dir=None,
        refine_pose=True, debug=False
    ):
        """
        Estimate moved objects' 3D masks and pose changes

        Args:
            transforms_json (Path or str):
                transforms.json for the post-reconfig training dataset
            configs (Path or str): hyperparameters
            refine_pose (bool): Refine object pose change and camera poses

        Returns:
            obj_3D_seg (list of Obj3DSeg): Object 3D segmentation
        """
        if configs is None:
            configs = {
                "sam_threshold": 0.95,
                "mask_refine_sparse_view": 0.0,
                "area_threshold": 0.01,
                "pcd_filtering": 0.98,
                "pre_train_pred_bbox_expand": 0.05,
                "voxel_dim": 300,
                "bbox3d_expand": 1.8,
                "mask3d_dilate_uniform": 1,
                "mask3d_dilate_top": 0,
                "pose_change_break": None,
                "pose_refine_lr": 1e-3,
                "pose_refine_epochs": 50,
                "pose_refine_patience": 20,
                "vis_check_threshold": 0.8,
                "proj_check_cutoff": 0.95,
                "val_move_in_dilate_3d": 0.05,
                "val_move_out_dilate_3d": 0.05,
            }
        else:
            json_path = Path(configs)
            assert json_path.exists(), f"{json_path} does not exist"
            with open(json_path, "r") as f:
                configs = json.load(f)

        assert self.output_dir.exists(), f"{self.output_dir} does not exist"
        assert transforms_json is not None, "Need transforms.json for CD!"

        # Load pre-trained 3DGS
        assert os.path.isfile(self.load_config)
        _, self.pipeline_pretrain, _, _ = eval_setup(
            self.load_config, test_mode="inference",
            checkpoint_dir=checkpoint_dir, data_path=Path(transforms_json).parent
        )

        device = self.device
        # ----------------------------Load data -------------------------------
        # Load pre-training images and camera info + new images and camera info
        color_images, depth_images, img_fnames, c2w, K, dist_params, cameras = \
            read_transforms(transforms_json)
        # Undistort images
        assert dist_params.sum() < 1e-6, \
            "All images must be undistorted before change detection"
        sparse_view_file_ids, train_file_ids = [], []
        sparse_view_indices, pretrain_indices = [], []
        for ii, path in enumerate(img_fnames):
            id_int = extract_last_number(path.name)
            if "rgb_new" in path.as_posix():
                sparse_view_file_ids.append(id_int)
                sparse_view_indices.append(ii)
            else:
                pretrain_indices.append(ii)
            train_file_ids.append(id_int)

        N, _, H, W = color_images.shape
        # Get sparse-view captured images
        rgbs_captured_sparse_view = \
            color_images[sparse_view_indices].to(device)

        # Get sparse-view captured depths
        depths_captured_sparse_view = depth_images.to(device)
        # Get sparse view camera parameters
        cameras_sparse_view = cameras[torch.tensor(sparse_view_indices, dtype=torch.long)]
        cam_poses_sparse_view = c2w[sparse_view_indices]
        Ks_sparse_view = K[sparse_view_indices]
        dist_params_sparse_view = dist_params[sparse_view_indices]
        # Get pre-training images and cameras
        color_images_pretrain_view = color_images[pretrain_indices]
        cam_poses_pretrain_view = c2w[pretrain_indices]
        Ks_pretrain_view = K[pretrain_indices]
        dist_params_pretrain_view = dist_params[pretrain_indices]
        # -------------------------------------------------------

        # Render images at the sparse viewpoints
        rgbs_render_sparse_view, depths_sparse_view = render_cameras(
            self.pipeline_pretrain, cameras_sparse_view, device=device
        )
        # if debug:
        #     debug_image_pairs(
        #         rgbs_render_sparse_view, rgbs_captured_sparse_view,
        #         self.debug_dir
        #     )

        # Sec.IV.C: 2D Change detection on post-change views
        masks_changed_sparse, masks_changed_sparse_all = [], []
        points_changed_sparse = []


        for ii in range(len(sparse_view_file_ids)): 
            masks_changed, masks_changed_all = self.image_diff(
                rgbs_render_sparse_view[ii:ii+1],
                rgbs_captured_sparse_view[ii:ii+1]
            )
            
            masks_changed_sparse.append(masks_changed)
            masks_changed_sparse_all.append(masks_changed_all)

            # depth
            depth = depths_captured_sparse_view[ii, 0].squeeze() # (H, W)
            #depth = depths_sparse_view[ii, 0] # (H, W)

            k = Ks_sparse_view[ii] # (3, 3)
            cam_pose = cam_poses_sparse_view[ii]

            for m in range(masks_changed.shape[0]):
                mask = masks_changed[m, 0].bool() # (H, W)

                # Resize mask to match depth resolution
                mask_resized = torch.nn.functional.interpolate(
                    mask.unsqueeze(0).unsqueeze(0).float(),  # Add batch and channel dims
                    size=(depth.shape[0], depth.shape[1]),  # Match depth resolution
                    mode='nearest'
                ).squeeze().bool()
                
                # Get coordinates from resized mask
                y, x = torch.where(mask_resized)
                
                z = depth[y, x] # depth at mask coordinates
                
                # Filter valid depths
                valid = z > 0
                x, y, z = x[valid], y[valid], z[valid]
                
                if x.numel() == 0:
                    print(f"No valid points found for mask {m} in view {ii}")
                    continue
                    
                # Calculate scaling to map back to original image coordinates if needed
                scale_x = mask.shape[1] / depth.shape[1]  # Width ratio
                scale_y = mask.shape[0] / depth.shape[0]  # Height ratio
                
                # Calculate camera coordinates
                # Map pixel coordinates back to original resolution
                x_orig = x.float() * scale_x
                y_orig = y.float() * scale_y
                
                # Extract camera intrinsics
                fx, fy = k[0, 0], k[1, 1]
                cx, cy = k[0, 2], k[1, 2]
                
                # Convert to camera coordinates
                X = (x_orig - cx) * z / fx
                Y = (y_orig - cy) * z / fy
                Z = z

                pts_cam = torch.stack([X, Y, Z, torch.ones_like(Z)], dim=1).T  # (4, N)
                pts_world = (cam_pose @ pts_cam).T[:, :3]
                points_changed_sparse.append(pts_world.cpu().numpy()) 

                # 
                image = rgbs_captured_sparse_view[ii].permute(1, 2, 0).cpu().numpy() * 255
                image = image.astype(np.uint8)

                # Load points for reprojection
                points_2d = self.project_points_to_image(
                    pts_world.cpu().numpy(), Ks_sparse_view[ii].cpu().numpy(),
                    cam_poses_sparse_view[ii].cpu().numpy(),
                    image.shape[:2]
                )
                
                # Visualize the projection
                self.visualize_projection(image, points_2d, self.debug_dir / f"projected_overlay_{ii}.png")
                # Save the point cloud
                self.save_pcds(pts_world.cpu().numpy(), self.debug_dir / f"changed_mask_{ii}_{m}.ply")


        # debug
        print(f"[INFO] Extracted {len(points_changed_sparse)} 3D point clusters from masks.")
        print(f"[INFO] Example shape: {points_changed_sparse[0].shape if points_changed_sparse else 'Empty'}")

        all_changed_points = np.concatenate(points_changed_sparse, axis=0)
        if debug:

            depths_np = depths_captured_sparse_view.squeeze(2).squeeze(1).cpu().numpy()  # Result shape: [4, 256, 192]
            for i, depth in enumerate(depths_np):
                # Ensure we have valid depth values to avoid division by zero
                depth_max = np.max(depth)
                if depth_max > 0:
                    normalized_depth = (depth / depth_max * 65535).astype(np.uint16)
                else:
                    normalized_depth = np.zeros_like(depth, dtype=np.uint16)
                
                # Save the depth map
                cv2.imwrite(str(self.debug_dir / f"depth_view_{i:03d}.png"), normalized_depth)
                
                # # Optional: Also save a color-mapped version for better visualization
                # depth_color = cv2.applyColorMap(
                #     (normalized_depth / 256).astype(np.uint8),  # Convert to 8-bit for color mapping
                #     cv2.COLORMAP_JET
                # )
                # cv2.imwrite(str(self.debug_dir / f"depth_view_color_{i:03d}.png"), depth_color)


        if debug:
            masks_changed_tensor = torch.cat(masks_changed_sparse, dim=0)
            save_masks(
                masks_changed_tensor / 255.0, [
                    f"{self.debug_dir}/masks_changed{i}.png"
                    for i in range(len(masks_changed_tensor))
                ]
            )

        #########################
        masks_move_out_sparse_view = []

        for ii, masks_changed in enumerate(masks_changed_sparse):
            masks_render, scores_render = effsam_refine_masks(
                rgbs_render_sparse_view[ii:ii+1], masks_changed,
                expand=-0.1
            )

            # Right after effsam_refine_masks call
            # if self.debug_dir:
            #     # Save scores to a text file
            #     with open(f"{self.debug_dir}/view{ii}_scores.txt", "w") as f:
            #         for i, score in enumerate(scores_render):
            #             f.write(f"Mask {i}: Score {score}\n")
                        
            masks_out = [
                masks_render[i:i+1] for i, s in enumerate(scores_render)
                if s > 0.9
            ]

            if self.debug_dir:
                save_masks(
                    masks_render,
                    [
                        f"{self.debug_dir}/masks_render_view{ii}_mask{j}.png"
                        for j in range(masks_render.shape[0])
                    ]
                )

            if len(masks_out) > 0:
                masks_out = torch.cat(masks_out, dim=0)
                masks_out = split_masks(masks_out, configs["area_threshold"])
            else:
                masks_out = torch.empty((0, 1, H, W)).to(device)
            
            masks_move_out_sparse_view.append(masks_out)
        
        num_move_out = max([m.size(0) for m in masks_move_out_sparse_view])
        print(f"[INFO] Found {num_move_out} masks for moved objects")

        # Filter out masks with too few points
        no_overlap_ind = []
        for i in range(len(masks_move_out_sparse_view)):
            if masks_move_out_sparse_view[i].size(0) >= num_move_out:
                no_overlap_ind.append(i)
                print(f"[INFO] View {i} has enough masks")
        if debug:
            masks_to_save = torch.cat(masks_move_out_sparse_view, dim=0)
            save_masks(masks_to_save, [
                f"{self.debug_dir}/masks_move_out{i}.png"
                for i in range(len(masks_to_save))
            ])


        ## Object Association across
        pcds, pcd_feats = self.match_move_out(
            rgbs_render_sparse_view[no_overlap_ind],
            depths_sparse_view[no_overlap_ind],
            [masks_move_out_sparse_view[i] for i in no_overlap_ind],
            cam_poses_sparse_view[no_overlap_ind],
            Ks_sparse_view[no_overlap_ind],
            pcd_filter=configs["pcd_filtering"]
        )

        ## Multi view move-in mask association across post-change views
        masks_move_in_sparse_view = []
        for ii, masks_changed in enumerate(masks_changed_sparse):
            masks_captured, scores_captured = effsam_refine_masks(
                rgbs_captured_sparse_view[ii:ii+1], masks_changed,
                expand=-0.1
            )
            # Move-in masks have SAM prediction score > 0.95 on captured image
            masks_in = [
                masks_captured[i:i+1] for i, s in enumerate(scores_captured)
                if s > 0.5
            ]
            if self.debug_dir:
                save_masks(
                    masks_captured,
                    [
                        f"{self.debug_dir}/masks_captured_view{ii}_mask{j}.png"
                        for j in range(masks_captured.shape[0])
                    ]
                )
            if len(masks_in) > 0:
                masks_in = torch.cat(masks_in, dim=0)
                masks_in = split_masks(masks_in, configs["area_threshold"])
            else:
                masks_in = torch.empty(0, 1, H, W, device=device)
            print(f"[View {ii}] Move-in candidates after SAM filtering: {len(masks_in)}")
            masks_move_in_sparse_view.append(masks_in)


        

        # Move-in masks w/ few inlier matches to obj templates are for inserted
        feats_move_in = self.get_features_in_masks(
            rgbs_captured_sparse_view, masks_move_in_sparse_view
        )
        masks_move_in_inserted = []
        for feats_i, masks_i in zip(feats_move_in, masks_move_in_sparse_view):
            masks_move_in_inserted_i = []
            for feat_i, mask_i in zip(feats_i, masks_i):
                num_inlier_max = 0
                for pcd_feat in pcd_feats:
                    _, num_inlier, _ = pcd_feat.PnP(
                        feat_i, Ks_sparse_view[0], H, W
                    )
                    num_inlier_max = max(num_inlier_max, num_inlier)
                print(f"[View {ii}] Inserted mask kept (inliers max: {num_inlier_max})")
                if num_inlier_max < 10:
                    masks_move_in_inserted_i.append(mask_i[None])
            if len(masks_move_in_inserted_i) > 0:
                masks_move_in_inserted_i = torch.cat(
                    masks_move_in_inserted_i, dim=0
                )
            else:
                masks_move_in_inserted_i = torch.empty(
                    0, 1, H, W, device=device
                )
            masks_move_in_inserted.append(masks_move_in_inserted_i)
        
        # debug
        total_masks = sum(m.size(0) for m in masks_move_in_inserted)
        print(f"[match_move_in] Total move-in mask candidates: {total_masks}")


        obj_masks_move_in, obj_move_in_view_indices = self.match_move_in(
            rgbs_captured_sparse_view, masks_move_in_inserted
        )
        print(f"[match_move_in] Found {len(obj_masks_move_in)} move-in objects")
        print(f"[match_move_in] Found {len(obj_move_in_view_indices)} view indices")

        print(f"obj_masks_move_in: {obj_masks_move_in}")
        print(f"obj_move_in_view_indices: {obj_move_in_view_indices}")










        # # Sec.IV.F: Object pose change estimate
        # # Extract features only within changed regions (dilated)
        # feat_masks = [
        #     dilate_masks(m.any(dim=0, keepdim=True), 10)
        #     for m in masks_changed_sparse_all
        # ]
        # feats = self.get_features_in_masks(
        #     rgbs_captured_sparse_view, feat_masks
        # )
        # debug_point_prompts(
        #     rgbs_captured_sparse_view[0:1], feats[0][0]["keypoints"],
        #     self.debug_dir
        # )
        # pose_changes = []
        # num_sparse_views = len(masks_move_out_sparse_view)

        # for ii, pcd_feat in enumerate(pcd_feats):
        #     num_inliers, num_matches = 0, 0
        #     pose_change = None            
        #     for idx in tqdm(range(num_sparse_views), desc="pose estimation"):
        #         pose_change_i, num_inlier_i, num_match_i = pcd_feat.PnP(
        #             feats[idx][0], Ks_sparse_view[idx], H, W, self.matcher
        #         )
        #         if debug:
        #             m2d, m3d = pcd_feat.match(feats[idx][0], self.matcher)
        #             m3d_proj, _ = project_points(
        #                 m3d, cam_poses_sparse_view[0:1], Ks_sparse_view[0:1],
        #                 dist_params_sparse_view[0:1], H, W
        #             )
        #             debug_matches(
        #                 rgbs_render_sparse_view[0:1], 
        #                 rgbs_captured_sparse_view[idx:idx+1],
        #                 m3d_proj[:, :, :], [m2d[:, :]],
        #                 torch.arange(m2d.shape[0])[None, :, None].repeat(1, 1, 2),
        #                 self.debug_dir
        #             )
        #         if pose_change_i is None:
        #             continue
        #         # Equation in the paper
        #         pose_change_i = \
        #             cam_poses_sparse_view[idx] @ pose_change_i.inverse()
        #         if configs["pose_change_break"] is not None and \
        #             configs["pose_change_break"][ii] is not None and \
        #             idx == configs["pose_change_break"][ii]:
        #             num_inliers = num_inlier_i
        #             num_matches = num_match_i
        #             pose_change = pose_change_i
        #             break
        #         if num_inlier_i > num_inliers:
        #             num_inliers = num_inlier_i
        #             num_matches = num_match_i
        #             pose_change = pose_change_i
        #     if pose_change is None:
        #         print(f"Object pose change est. for object {ii} failed!")
        #         print(f"Object {ii} is removed from the scene!")
        #     else:
        #         print(f"pose_change: \n {pose_change.cpu().numpy()}")
        #     print(f"inlier_ratio: {num_inliers} / {num_matches}")
        #     pose_changes.append(pose_change)
        # if debug:
        #     debug_point_cloud(pcds[-1], self.debug_dir)
        # num_moved = len([_ for pc in pose_changes if pc is not None])
        # print(f"# Moved objects: {num_moved}")
        # print(f"# Removed objects: {len(pose_changes) - num_moved}")
        # print(f"# Inserted objects: {len(obj_masks_move_in)}")



        






if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3DGS change detection")
    parser.add_argument(
        "--config", "-c", required=True, type=str,
        help="Path to the config.yml file of the pretrained 3DGS"
    )
    parser.add_argument(
        "--output", "-o", required=True, type=str,
        help="Path to save the output 3D segmentation"
    )
    parser.add_argument(
        "--transform", "-t", type=str,
        help="Path to transforms.json with info on both old and new images"
    )
    parser.add_argument(
        "--ckpt", "-ckpt", type=str, default=None,
        help="Path to the parent folder of 3DGS checkpoint"
    )
    parser.add_argument(
        "--debug", "-d", action="store_true",
        help="Debug mode"
    )
    args = parser.parse_args()

    # Load hyperparams
    hyperparams = f"{os.path.dirname(args.transform)}/configs.json"
    hyperparams = hyperparams if os.path.exists(hyperparams) else None
    # Detect changes
    change_det = ChangeDet(Path(args.config), Path(args.output), debug=args.debug)
    change_det.main(
        transforms_json=args.transform, configs=hyperparams,
        checkpoint_dir=Path(args.ckpt), debug=args.debug
    )