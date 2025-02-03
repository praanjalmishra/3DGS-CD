# 3DGS-based change detection
import argparse
import json
import os
import re
from pathlib import Path

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
    effsam_batch_predict, compute_2D_bbox, expand_2D_bbox
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
    debug_dir = "/home/ziqi/Desktop/test/"
    """Directory to save debug output"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    """Device"""
    extractor = SuperPoint(max_num_keypoints=4096).eval().to(device)
    """SuperPoint extractor"""
    matcher = LightGlue(features='superpoint').eval().to(device)
    """LightGlue matcher"""

    def __init__(self, load_config: Path, output_dir: Path):
        # Path to the config.yml file of the pretrained 3DGS
        self.load_config = load_config
        # Path to save the output 3D segmentation
        self.output_dir = output_dir

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
        # cv2.imwrite(f"{self.debug_dir}/similarity_map.png", similarity_map)
        # Threshold the SAM cosine similarity map
        thresh = cv2.threshold(
            similarity_map, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
        )[1]
        # remove the influence of the black region in the aligned image
        thresh = thresh * align_mask
        # Uncomment to debug
        # cv2.imwrite(f"{self.debug_dir}/thresh.png", thresh)
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

    def match_move_out(self, rgbs, depths, masks, poses, Ks, pcd_filter=0.9):
        """
        Match move-out masks across multiple images

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
        # Initialize object pcds
        pcds, pcd_sizes, pcd_counts, pcd_feats = [], [], [], []
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
                if c < len(masks[i]):
                    pcds[r] = torch.cat((pcds[r], new_pcds[c]), dim=0)
                    pcd_sizes[r] = pcd_size(pcds[r])
                    pcd_counts[r] += 1
                    pts3D = compute_feats_3D(
                        feats[i][c], depths[i:i+1], poses[i], Ks[i]
                    )
                    pcd_feats[r].add_feats(feats[i][c], pts3D)
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
        # Filter out object point clouds that appear in <25% of images
        pcds = [p for p, ct in zip(pcds, pcd_counts) if ct > N * 0.25]
        pcd_feats = [
            e for e, ct in zip(pcd_feats, pcd_counts) if ct > N * 0.25
        ]
        return pcds, pcd_feats

    def pretrain_iteration(self, rgbs, masks, cameras, gaussians):
        """
        Forward pass through the transformed pre-trained 3DGS

        Args:
            rgbs (Nx3xHxW): Captured sparse view RGB images
            masks (Nx1xHxW): Sampling masks on the sparse views
            cameras (Cameras): NeRFStudio Cameras object of size N
            gaussians (dict): Transformed Gaussian parameters

        Returns:
            rgb_loss: RGB loss btw the captured and rendered pixels
        """
        rgbs = rgbs.permute(0, 2, 3, 1)
        masks = masks.permute(0, 2, 3, 1)
        batches = [
            { "image": rgb, "mask": mask, "image_idx": ii} 
            for ii, (rgb, mask) in enumerate(zip(rgbs, masks))
        ]
        loss_accumulated = torch.tensor(0.0, device=rgbs.device)
        for ii, batch in enumerate(batches):
            camera = cameras[ii:ii+1]
            # outputs = self.pipeline_pretrain.model(camera)
            color, _ = render_3dgs_at_cam(camera, gaussians)
            outputs = {
                "rgb": color.squeeze().permute(1, 2, 0), "background": None
            }
            loss_dict = self.pipeline_pretrain.model.get_loss_dict(
                outputs, batch, None
            )
            loss = sum(loss_dict.values())
            loss_accumulated += loss
        # Uncomment to vis the pose regression process
        # blend = rgbs[-1] * 0.2 + outputs["rgb"] * masks[-1] * 0.8
        # blend = (blend.detach().cpu().numpy() * 255).astype(np.uint8)
        # Image.fromarray(blend).save(self.debug_dir + f"blend{debug_count}.png")
        return loss_accumulated

    def refine_obj_pose_change(
        self, rgbs, obj_segs, cameras, lr=1e-3, epochs=100, patience=20,
        optim="obj+cam"
    ):
        """
        Refine object pose change to make the object pose pixel-perfect

        Args:
            rgbs (Nx3xHxW): Captured sparse view RGB images
            obj_segs (M-list of Obj3DSeg): Object 3D segments
            cameras (Cameras): NeRFStudio Cameras object
            batch_size (int): Batch size for training
            epochs (int): Number of epochs
            patience (int): Number of epochs to wait for plateau
            optim (str): Variables to optim, choices: "obj+cam", "obj", "cam"
        
        Returns:
            poses_refined (M-list of 4x4): Refined object pose change
            cameras (Cameras): Cameras w/ refined camera poses
        """
        from nerfstudio.cameras.lie_groups import exp_map_SO3xR3
        assert hasattr(self, "pipeline_pretrain"), \
            "Pre-training pipeline not loaded yet"
        device = rgbs.device
        cameras.camera_to_worlds = cameras.camera_to_worlds.to(device)
        c2w, Ks, dist, H, W = cameras_to_params(cameras, device)
        # Project object 3D seg voxel grid points to have obj's masks        
        in_objs, poses_init, obj_masks = [], [], []
        for obj_seg in obj_segs:
            in_obj = obj_seg.query(self.pipeline_pretrain.model.means)
            pose_init = obj_seg.pose_change.clone().to(device)
            obj_mask = ~obj_seg.project(c2w, Ks, dist, H, W)
            in_objs.append(in_obj)
            poses_init.append(pose_init)
            obj_masks.append(obj_mask)
        obj_masks = torch.all(torch.stack(obj_masks, dim=0), dim=0)
        # Uncomment to debug
        # debug_masks(obj_masks, self.debug_dir)
        # Pre-trained Gaussians
        gauss0 = {
            name: self.pipeline_pretrain.model.gauss_params[name].data.clone()
            for name in [
                "means", "scales", "quats", "features_dc", "features_rest",
                "opacities"
            ]
        }
        cam0 = camera_clone(cameras)
        # Make a pose update parameter
        poses_update = torch.nn.Parameter(
            torch.zeros((len(poses_init), 6), device=device)
        )
        cam_pose_update = torch.nn.Parameter(
            torch.zeros((len(cameras), 6), device=device)
        )
        param = []
        if "obj" in optim:
            param.append(poses_update)
        if "cam" in optim:
            param.append(cam_pose_update)
        assert len(param) > 0, "No parameters to optimize"
        optimizer = torch.optim.Adam(param, lr=lr)
        # Training loop
        best_loss, initial_loss = float("inf"), None
        plateau_count = 0
        with tqdm(total=epochs, desc="pose change opt") as pbar:
            for idx in range(epochs):
                optimizer.zero_grad()
                poses_update4x4 = to4x4(exp_map_SO3xR3(poses_update))
                poses_update4x4 = poses_update4x4.reshape(-1, 4, 4)
                cam_pose_update4x4 = to4x4(exp_map_SO3xR3(cam_pose_update))
                cam_pose_update4x4 = cam_pose_update4x4.reshape(-1, 4, 4)
                # Transform object Gaussians
                assert self.pipeline_pretrain.model.means.shape[0] > 0
                means, quats = gauss0["means"], gauss0["quats"]
                for pose_init, in_obj, pose_update4x4 in zip(
                    poses_init, in_objs, poses_update4x4
                ):
                    means_trans, quats_trans = transform_gaussians(
                        pose_init @ pose_update4x4,
                        gauss0["means"], gauss0["quats"]
                    )
                    means = torch.where(
                        in_obj.unsqueeze(-1).repeat(1, 3), means_trans, means
                    )
                    quats = torch.where(
                        in_obj.unsqueeze(-1).repeat(1, 4), quats_trans, quats
                    )
                gauss = {
                    name : gauss0[name] for name in [
                        "scales", "features_dc", "features_rest", "opacities"
                    ]
                }
                gauss["means"], gauss["quats"] = means, quats
                # Update camera pose
                cameras.camera_to_worlds = \
                    cam0.camera_to_worlds @ cam_pose_update4x4
                # Forward pass
                rgb_loss = self.pretrain_iteration(
                    rgbs, obj_masks, cameras, gauss
                )
                # Backward pass
                rgb_loss.backward()
                optimizer.step()
                pbar.set_postfix(
                    {'Epoch': idx+1, 'RGB Loss': f'{rgb_loss.item():.4f}'}
                )
                pbar.update(1)
                if initial_loss is None:
                    initial_loss = rgb_loss.item()
                if rgb_loss.item() < best_loss:
                    best_loss = rgb_loss.item()
                    plateau_count = 0
                else:
                    plateau_count += 1
                    if plateau_count > patience:
                        print(f"Early stopping at epoch {idx+1} after plateau")
                        break
        if rgb_loss.item() > initial_loss:
            print("Warning: RGB loss increased after pose change refinement")
        poses_refined = [
            pose_init @ pose_update4x4.detach() for pose_init, pose_update4x4
            in zip(poses_init, poses_update4x4)
        ]
        return poses_refined, cameras

    def check_visibility(
        self, pcds, masks, poses, Ks, dist_params, H, W, threshold=0.95
    ):
        """
        Check visibility of object point clouds

        Args:
            pcds (M-list of Lx3): Object point clouds
            masks (MxNx1xHxW): Object move-out masks on the sparse views
            poses (Nx4x4): Camera poses wrt world
            Ks (Nx3x3): Camera intrinsics
            dist_params (Nx4): Camera distortion parameters
            H (int): Image height
            W (int): Image width
            
        Returns:
            vis (M-list of N-list of int): Views where obj pcd is fully visible
        """
        assert len(pcds) == len(masks)
        assert masks.shape[1] == len(poses)
        vis = []
        for ii in range(len(pcds)):
            pcd_proj, _ = project_points(
                pcds[ii], poses, Ks, dist_params, H, W
            )
            # We count how many object points can project in masks
            vis_ii = []
            for jj, proj in enumerate(pcd_proj):
                proj = proj.round().long().unique(dim=0)
                proj_in = in_image(proj, H, W)
                proj_in_ratio = proj_in.sum().item() / proj.size(0)
                proj = proj[proj_in]
                in_mask_count = masks[ii, jj, 0][proj[:, 1], proj[:, 0]].sum()
                in_mask_ratio = in_mask_count / proj.size(0)
                if in_mask_ratio > threshold and proj_in_ratio > threshold:
                    vis_ii.append(jj)
            vis.append(vis_ii)
        return vis
    
    def occl_aware_mask_proj(self, cams, obj_segs, dilate=0.15, new=False):
        """
        Project object 3D segmentation to target cameras w/ occlusion-awareness
        
        cams (Cameras): Target camera views
        obj_segs (M-list of Obj3DSeg): Object 3D segments
        dilate (float): Dilate the 3D segments to check if points in mask
        new (bool): Use object's new pose for mask projection

        Returns:
            masks (Nx1xHxW): 2D move-out or -in masks on the target views
        """
        # Render depths at target cameras
        poses, Ks, dist, H, W = cameras_to_params(cams)
        if not new:
            _, depths = render_cameras(
                self.pipeline_pretrain, cams, device=self.device
            )
        else:
            gauss0 = {
                name: self.pipeline_pretrain.model.gauss_params[name].data.clone()
                for name in [
                    "means", "scales", "quats", "features_dc", "features_rest",
                    "opacities"
                ]
            }
            means, quats = gauss0["means"], gauss0["quats"]
            for obj_seg in obj_segs:
                in_obj = obj_seg.query(self.pipeline_pretrain.model.means)
                means_trans, quats_trans = transform_gaussians(
                    obj_seg.pose_change, gauss0["means"], gauss0["quats"]
                )
                means = torch.where(
                    in_obj.unsqueeze(-1).repeat(1, 3), means_trans, means
                )
                quats = torch.where(
                    in_obj.unsqueeze(-1).repeat(1, 4), quats_trans, quats
                )
            gauss = {
                name : gauss0[name] for name in [
                    "scales", "features_dc", "features_rest", "opacities"
                ]
            }
            gauss["means"], gauss["quats"] = means, quats
            depths = []
            for ii in range(len(cams)):
                _, depth = render_3dgs_at_cam(cams[ii:ii+1], gauss)
                depths.append(depth)
            depths = torch.cat(depths, dim=0)
        # debug_depths(depths, self.debug_dir)
        # Project object 3D segmentation to target
        masks_no_occl_all_obj = []
        for obj_seg in obj_segs:
            if not new:
                masks = obj_seg.project(poses, Ks, dist, H, W)
            else:
                masks = obj_seg.project_new(poses, Ks, dist, H, W)
            # dilate the 3D segments due to noise
            voxel_dilated = obj_seg.dilate_uniform(
                int(obj_seg.voxel.size(0) * dilate)
            )
            masks_no_occl = []
            for dd, pp, kk, mm in zip(depths, poses, Ks, masks):
                pcd_in_mask = compute_point_cloud(
                    dd[None], pp[None], kk[None], mm[None]
                )
                if not new:
                    not_occluded = obj_seg.query(pcd_in_mask, voxel_dilated)
                else:
                    not_occluded = obj_seg.query_new(pcd_in_mask, voxel_dilated)
                change_inds = (mm==1).nonzero()
                change_inds_no_occl = change_inds[not_occluded]
                mm_no_occl = torch.zeros_like(mm)
                mm_no_occl[
                    0, change_inds_no_occl[:, 1], change_inds_no_occl[:, 2]
                ] = 1
                masks_no_occl.append(mm_no_occl)
            masks_no_occl = torch.stack(masks_no_occl, dim=0)
            masks_no_occl_all_obj.append(masks_no_occl)
        masks_no_occl_union = torch.any(
            torch.stack(masks_no_occl_all_obj, dim=0), dim=0
        )
        return masks_no_occl_union

    def main(
        self, transforms_json=None, configs=None, refine_pose=True
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
                "mask_refine_sparse_view": 0.0,
                "pcd_filtering": 0.9,
                "pre_train_pred_bbox_expand": 0.1,
                "pre_train_refine_bbox_expand": 0.1,
                "voxel_dim": 400,
                "bbox3d_expand": 1.8,
                "mask3d_dilate_uniform": 1,
                "mask3d_dilate_top": 1,
                "move_out_dilate": 31,
                "pose_change_break": None,
                "pose_refine_lr": 1e-3,
                "pose_refine_epochs": 100,
                "pose_refine_patience": 20,
                "vis_check_threshold": 0.8,
                "proj_check_cutoff": 0.95
            }
        else:
            json_path = Path(configs)
            assert json_path.exists(), f"{json_path} does not exist"
            with open(json_path, "r") as f:
                configs = json.load(f)

        assert self.output_dir.exists(), f"{self.output_dir} does not exist"

        # Load pre-trained 3DGS
        assert os.path.isfile(self.load_config)
        _, self.pipeline_pretrain, _, _ = eval_setup(
            self.load_config, test_mode="inference"
        )

        device = self.device
        # ----------------------------Load data -------------------------------
        # Load pre-training images and camera info + new images and camera info
        if transforms_json is not None:
            color_images, img_fnames, c2w, K, dist_params, cameras = \
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

        else: # hardcode sparse view indices
            # TODO: Remove the hard-coded GT input between the lines
            num_images, split_fraction = 200, 0.9
            num_train_images = int(np.ceil(num_images * split_fraction))
            train_file_ids = np.linspace(
                0, num_images-1, num_train_images, dtype=int
            ).tolist()
            sparse_view_file_ids = [80, 85, 90, 95]
            sparse_view_indices = [
                train_file_ids.index(ii) for ii in sparse_view_file_ids
            ]
            assert self.pipeline_pretrain.datamanager.train_dataset is not None
            pretrain_dataset = self.pipeline_pretrain.datamanager.train_dataset
            # Read pre-training images and cameras
            color_images, c2w, K, dist_params = read_dataset(
                pretrain_dataset, read_images=True
            )
            cameras = pretrain_dataset.cameras
            N, _, H, W = color_images.shape
            # Get pre-training view indices
            full_indices = torch.arange(N, device=device)
            is_pretrain = torch.ones(N, dtype=torch.bool, device=device)
            is_pretrain[sparse_view_indices] = False
            pretrain_indices = full_indices[is_pretrain]
            prefix="/home/ziqi/Packages/nerfstudio/data/nerfstudio/gt_masks/"
            rgbs_sparse_paths=[
                f"{prefix}/cube_rgb_new/frame_{idx:05g}.png"
                for idx in sparse_view_file_ids
            ]
            rgbs_captured_sparse_view = \
                read_imgs(rgbs_sparse_paths).to(device)
        # Get sparse view camera parameters
        cameras_sparse_view = cameras[torch.tensor(sparse_view_indices)]
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
        # # Uncomment to debug
        # debug_image_pairs(
        #     rgbs_render_sparse_view, rgbs_captured_sparse_view, self.debug_dir
        # )

        # Change detection on sparse views
        masks_changed_sparse, masks_changed_sparse_all = [], []
        for ii in range(len(sparse_view_file_ids)): 
            masks_changed, masks_changed_all = self.image_diff(
                rgbs_render_sparse_view[ii:ii+1],
                rgbs_captured_sparse_view[ii:ii+1]
            )
            masks_changed_sparse.append(masks_changed)
            masks_changed_sparse_all.append(masks_changed_all)
        # Uncomment to debug
        # masks_changed_tensor = torch.cat(masks_changed_sparse, dim=0)
        # save_masks(
        #     masks_changed_tensor / 255.0, [
        #         f"{self.debug_dir}/masks_changed{i}.png"
        #         for i in range(len(masks_changed_tensor))
        #     ]
        # )
        masks_move_out_sparse_view = []
        for ii, masks_changed in enumerate(masks_changed_sparse):
            masks_render, scores_render = effsam_refine_masks(
                rgbs_render_sparse_view[ii:ii+1], masks_changed,
                expand=configs["mask_refine_sparse_view"]
            )
            # Move-out masks have SAM prediction score > 0.95 on rendered image
            masks_out = [
                masks_render[i:i+1] for i, s in enumerate(scores_render)
                if s > configs["sam_threshold"]
            ]
            if len(masks_out) > 0:
                masks_out = torch.cat(masks_out, dim=0)
                masks_out = split_masks(masks_out, 0.005)
            else:
                masks_out = torch.empty(0, 1, H, W, device=device)
            masks_move_out_sparse_view.append(masks_out)
        # Ignore views with overlapped move-out regions
        num_move_out = max([m.size(0) for m in masks_move_out_sparse_view])

        no_overlap_ind = []
        for i in range(len(masks_move_out_sparse_view)):
            if masks_move_out_sparse_view[i].size(0) >= num_move_out:
                no_overlap_ind.append(i)
        # # Uncomment to debug
        # masks_to_save = torch.cat(masks_move_in_sparse_view, dim=0)
        # save_masks(masks_to_save, [
        #     f"{self.debug_dir}/masks_move_in{i}.png" 
        #     for i in range(len(masks_to_save))
        # ])
        # masks_to_save = torch.cat(masks_move_out_sparse_view, dim=0)
        # save_masks(masks_to_save, [
        #     f"{self.debug_dir}/masks_move_out{i}.png"
        #     for i in range(len(masks_to_save))
        # ])
        # Multi-view move-out mask association 
        pcds, pcd_feats = self.match_move_out(
            rgbs_render_sparse_view[no_overlap_ind],
            depths_sparse_view[no_overlap_ind],
            [masks_move_out_sparse_view[i] for i in no_overlap_ind],
            cam_poses_sparse_view[no_overlap_ind],
            Ks_sparse_view[no_overlap_ind],
            pcd_filter=configs["pcd_filtering"]
        )
        print(f"Number of moved objects: {len(pcds)}")
        # Object pose change estimate
        # Extract features only within changed regions (dilated)
        feat_masks = [
            dilate_masks(m.any(dim=0, keepdim=True), 10)
            for m in masks_changed_sparse_all
        ]
        feats = self.get_features_in_masks(
            rgbs_captured_sparse_view,
            # Try this if pose change est fail for moved objs
            # [torch.ones(1, 1, H, W).bool().to(device)] * len(rgbs_captured_sparse_view)
            feat_masks
        )
        # debug_point_prompts(
        #     rgbs_captured_sparse_view,
        #     torch.cat([f[0]["keypoints"] for f in feats], dim=0),
        #     self.debug_dir
        # )
        pose_changes = []
        num_sparse_views = len(masks_move_out_sparse_view)
        for ii, pcd_feat in enumerate(pcd_feats):
            num_inliers, num_matches = 0, 0
            pose_change = None            
            for idx in tqdm(range(num_sparse_views), desc="pose estimation"):
                pose_change_i, num_inlier_i, num_match_i = pcd_feat.PnP(
                    feats[idx][0], Ks_sparse_view[idx], H, W, self.matcher
                )
                # # Uncomment to debug
                # m2d, m3d = pcd_feat.match(feats[idx][0], self.matcher)
                # m3d_proj, _ = project_points(
                #     m3d, cam_poses_sparse_view[0:1], Ks_sparse_view[0:1],
                #     dist_params_sparse_view[0:1], H, W
                # )
                # debug_matches(
                #     rgbs_render_sparse_view[0:1], 
                #     rgbs_captured_sparse_view[idx:idx+1],
                #     m3d_proj[:, :, :], [m2d[:, :]],
                #     torch.arange(m2d.shape[0])[None, :, None].repeat(1, 1, 2),
                #     self.debug_dir
                # )
                if pose_change_i is None:
                    continue
                # Equation in the paper
                pose_change_i = \
                    cam_poses_sparse_view[idx] @ pose_change_i.inverse()
                if configs["pose_change_break"] is not None and \
                    configs["pose_change_break"][ii] is not None and \
                    idx == configs["pose_change_break"][ii]:
                    num_inliers = num_inlier_i
                    num_matches = num_match_i
                    pose_change = pose_change_i
                    break
                if num_inlier_i > num_inliers:
                    num_inliers = num_inlier_i
                    num_matches = num_match_i
                    pose_change = pose_change_i
            pose_changes.append(pose_change)
            assert pose_change is not None, "Object pose change est. failed!"
            print(f"pose_change: \n {pose_change.cpu().numpy()}")
            print(f"inlier_ratio: {num_inliers} / {num_matches}")
        # # Uncomment to debug
        # debug_point_cloud(pcds[0], self.debug_dir)


        # Project the object point cloud to dense old views to get 2D bboxes
        bboxes2d = []
        for pcd in pcds:
            pcd_proj, is_point_in_img = project_points(
                pcd, cam_poses_pretrain_view, Ks_pretrain_view,
                dist_params_pretrain_view, H, W
            )
            if not is_point_in_img.all():
                print("WARN: Some points are out of the pretraining images")
            # debug_point_prompts(
            #     color_images_pretrain_view, pcd_proj, self.debug_dir
            # )
            bbox2d = compute_2D_bbox(pcd_proj)
            # Slightly expand 2D bboxes to improve SAM predictions
            bbox2d = expand_2D_bbox(
                bbox2d, configs["pre_train_pred_bbox_expand"]
            )
            bboxes2d.append(bbox2d)
        bboxes2d = torch.stack(bboxes2d, dim=1) # NxMx4

        # SAM predict all move-out masks (batched for multi-object)
        masks_move_out_pretrain_view, scores = [], []
        for img, bbox2d in tqdm(
            zip(color_images_pretrain_view, bboxes2d), desc="SAM predict"
        ):
            mask, score = effsam_batch_predict(
                img[None].to(self.device), bbox2d
            )
            masks_move_out_pretrain_view.append(mask)
            scores.append(score)
        masks_move_out_pretrain_view = torch.stack(
            masks_move_out_pretrain_view, dim=1
        ) # MxNx1xHxW
        scores = [list(t) for t in zip(*scores)] # M-list of N-list
        # # Uncomment to debug
        # debug_masks(masks_move_out_pretrain_view[0, ...], self.debug_dir)

        # Get high score mask indices
        high_score_inds = []
        for ss in scores:
            high_score = [i for i, x in enumerate(ss) if x > 0.95]
            if len(high_score) > 0:
                print(f"High score masks: {len(high_score)} / {len(ss)}")
            else:
                print("All masks look great!!")
            high_score_inds.append(high_score)
        # Check visibility of object point clouds
        visible = self.check_visibility(
            pcds, masks_move_out_pretrain_view, cam_poses_pretrain_view,
            Ks_pretrain_view, dist_params_pretrain_view, H, W,
            threshold=configs["vis_check_threshold"]
        )
        for vv in visible:
            print(f"Visible views: {len(vv)} / {len(cam_poses_pretrain_view)}")
        # Views having high-score masks and from which object is fully visible
        high_score_inds = [
            list(set(hs) & set(vis))
            for hs, vis in zip(high_score_inds, visible)
        ]
        for inds in high_score_inds:
            print(f"#Views for 3D seg: {len(inds)} / {len(Ks_pretrain_view)}")
        
        # # Uncomment to train object-3DGS
        # obj_ind_to_save = 0
        # masks_to_save = masks_move_out_pretrain_view[obj_ind_to_save]\
        #     [high_score_inds[obj_ind_to_save]]
        # if not os.path.isdir(f"{self.output_dir}/rgb_obj"):
        #     os.makedirs(f"{self.output_dir}/rgb_obj", exist_ok=True)
        # good_inds = [
        #     train_file_ids[pretrain_indices[ii]]
        #     for ii in high_score_inds[obj_ind_to_save]
        # ]
        # rgb_img_list = [
        #     f"{self.output_dir}/rgb/frame_{ii:05g}.png" for ii in good_inds
        # ]
        # mask_img_list = [
        #     f"{self.debug_dir}/mask_pretrain{ii}.png" for ii in good_inds
        # ]
        # save_masks(masks_to_save, mask_img_list)
        # from nerfstudio.utils.img_utils import rgb2rgba
        # rgb2rgba(rgb_img_list, mask_img_list, f"{self.output_dir}/rgb_obj/")
        # pretrain_json = \
        #     Path(transforms_json).parent / "transforms_pretrain.json"
        # assert os.path.isfile(pretrain_json), "Pretrain data missing"
        # pretrain_data = load_from_json(pretrain_json)
        # obj_data_frames = []
        # for frame in pretrain_data["frames"]:
        #     frame_id = extract_last_number(frame["file_path"])
        #     if frame_id in good_inds:
        #         frame["file_path"] = f"rgb_obj/frame_{frame_id:05g}.png"
        #         obj_data_frames.append(frame)
        # pretrain_data["frames"] = obj_data_frames
        # write_to_json(
        #     Path(transforms_json).parent / "transforms_obj.json",
        #     pretrain_data
        # )
        # # train object 3DGS by running the following command
        # # ns-train splatfacto --pipeline.model.background-color random

        # from nerfstudio.utils.io import save_alpha_transparent_train_data
        # gt_pose_change=torch.tensor(
        #     [[1, 0,  0, 2.5 / 11.0], 
        #     [0,  0, -1, 3.0 / 11.0], 
        #     [0,  1,  0, 0.5 / 11.0], 
        #     [0,  0,  0, 1]]
        # ).to(pose_change)        
        # save_alpha_transparent_train_data(
        #     masks, masks_move_in_sparse_view, pose_change, 
        #     Path(transforms_json).parent, self.output_dir,
        #     gt_pose_change=gt_pose_change
        # )


        # Compute finer object 3D segmentation
        obj_segs = []
        for ii in range(len(pcds)):
            bbox3d = compute_3D_bbox(pcds[ii])
            print(f"bbox3d: {bbox3d}")
            expanded_bbox3d = expand_3D_bbox(bbox3d, configs["bbox3d_expand"])
            voxel = bbox2voxel(
                expanded_bbox3d, configs["voxel_dim"], self.device
            )
            occ_grid = proj_check_3D_points(
                voxel, cam_poses_pretrain_view[high_score_inds[ii]], 
                Ks_pretrain_view[high_score_inds[ii]],
                dist_params_pretrain_view[high_score_inds[ii]],
                masks_move_out_pretrain_view[ii][high_score_inds[ii]],
                cutoff=configs["proj_check_cutoff"]
            )
            obj3Dseg = Object3DSeg(
                *expanded_bbox3d, occ_grid, pose_changes[ii], bbox3d,
                configs["mask3d_dilate_uniform"],
                configs["mask3d_dilate_top"]
            )
            # # Uncomment to debug
            # obj3Dseg.visualize(self.debug_dir)
            obj_segs.append(obj3Dseg)
        # Global pose refinement
        if refine_pose:
            new_cameras = params_to_cameras(
                cam_poses_sparse_view, Ks_sparse_view, 
                dist_params_sparse_view, H, W
            )
            # for ii in range(len(obj_segs)):
            pose_changes, new_cameras = self.refine_obj_pose_change(
                rgbs_captured_sparse_view, obj_segs, new_cameras,
                lr=configs["pose_refine_lr"],
                epochs=configs["pose_refine_epochs"],
                patience=configs["pose_refine_patience"]
            )
            for ii, pose_change in enumerate(pose_changes):
                obj_segs[ii].pose_change = pose_change
                print(
                    f"refined: \n {obj_segs[ii].pose_change.cpu().numpy()}"
                )
        # Save object 3D segmentation w/ updated pose changes
        for ii, obj_seg in enumerate(obj_segs):
            obj_seg.save(self.output_dir / f"obj3Dseg{ii}.pt")
        
        # Optimize eval camera poses
        if refine_pose:
            rgbs_eval, eval_fnames, _, _, _, cams_eval = \
                read_transforms(transforms_json, mode="val")
            _, cams_eval = self.refine_obj_pose_change(
                rgbs_eval.to(device), obj_segs, cams_eval,
                lr=configs["pose_refine_lr"],
                epochs=configs["pose_refine_epochs"],
                patience=configs["pose_refine_patience"], optim="cam"
            )
            eval_file_ids = []
            for ii, path in enumerate(eval_fnames):
                id_int = extract_last_number(path.name)
                eval_file_ids.append(id_int)

        # Update sparse-view training camera poses
        finetune_tjson = \
            Path(transforms_json).parent / "transforms_finetune.json"
        assert os.path.isfile(finetune_tjson), "Finetuning data missing"
        finetune_data = load_from_json(finetune_tjson)
        for ii, frame in enumerate(finetune_data["frames"]):
            frame_id = extract_last_number(frame["file_path"])
            if frame_id in sparse_view_file_ids:
                frame_ind = sparse_view_file_ids.index(frame_id)
                cam_pose_updated = new_cameras[frame_ind].camera_to_worlds
                cam_pose_updated = to4x4(cam_pose_updated)
                finetune_data["frames"][ii]["transform_matrix"] = \
                    cam_pose_updated.detach().cpu().numpy().tolist()
            if frame_id in eval_file_ids:
                frame_ind = eval_file_ids.index(frame_id)
                cam_pose_updated = cams_eval[frame_ind].camera_to_worlds
                cam_pose_updated = to4x4(cam_pose_updated)
                finetune_data["frames"][ii]["transform_matrix"] = \
                    cam_pose_updated.detach().cpu().numpy().tolist()
        write_to_json(finetune_tjson, finetune_data)


        # Save all-white masks to mask_new folder
        if not os.path.exists(self.output_dir / "masks_new"):
            os.makedirs(self.output_dir / "masks_new")
        mask_files = [
            self.output_dir / "masks_new" / f"mask_{ii:05g}.png"
            for ii in sparse_view_file_ids
        ]
        save_masks(torch.ones(len(mask_files), 1, H, W), mask_files)
        # Save eval masks
        _, val_files, _, _, _, _ = read_transforms(
            transforms_json, read_images=False, mode="val"
        )
        val_masks_move_out_no_occl = self.occl_aware_mask_proj(
            cams_eval, obj_segs, new=False
        )
        val_masks_move_in_no_occl = self.occl_aware_mask_proj(
            cams_eval, obj_segs, new=True
        )
        val_file_ids = []
        for ii, path in enumerate(val_files):
            id_int = extract_last_number(path.name)
            val_file_ids.append(id_int)
        mask_files = [
            self.output_dir / "masks_new" / f"mask_{ii:05g}.png"
            for ii in val_file_ids
        ]
        save_masks(val_masks_move_out_no_occl, mask_files)


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
    args = parser.parse_args()

    # Load hyperparams
    hyperparams = f"{os.path.dirname(args.transform)}/configs.json"
    hyperparams = hyperparams if os.path.exists(hyperparams) else None
    # Detect changes
    change_det = ChangeDet(Path(args.config), Path(args.output))
    change_det.main(
        transforms_json=args.transform, configs=hyperparams
    )