# 3DGS-based change detection
import argparse
import json
import os
import re
import statistics
from pathlib import Path

import cv2
import numpy as np
import pycolmap
import torch
from lightglue import viz2d
from matplotlib import pyplot as plt
from pyquaternion import Quaternion
from tqdm import tqdm

from nerfstudio.cameras.camera_paths import get_path_from_json
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.models.splatfacto import SplatfactoModel
from nerfstudio.utils.debug_utils import (
    debug_image_pairs, debug_images, debug_masks, debug_matches
)
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils.img_utils import (
    extract_depths_at_pixels, image_align, image_matching, undistort_images
)
from nerfstudio.utils.io import (
    read_dataset, read_imgs, read_transforms, save_masks, params_to_cameras
)
from nerfstudio.utils.obj_3d_seg import Object3DSeg
from nerfstudio.utils.pcd_utils import (
    compute_3D_bbox, compute_point_cloud, expand_3D_bbox, point_cloud_filtering
)
from nerfstudio.utils.proj_utils import (
    depths_to_points, proj_check_3D_points, project_points, undistort_points
)
from nerfstudio.utils.render_utils import render_cameras
from nerfstudio.utils.sam_utils import (
    compute_2D_bbox, expand_2D_bbox, sam_embedding, sam_predict,
    sam_refine_masks
)


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
    )
    return cameras_new


def new_cams_to_pretrain(cameras, pose_change):
    """
    Map new cameras poses back to corresponding pre-training camera poses,
    so that rendering with the pre-trained 3DGS at the updated poses
    gives (almost) the same object appearance

    Args:
        cameras (Cameras): Camera poses wrt world
        pose_change (4x4 tensor): Pose change

    Returns:
        c2w_new (Cameras): Updated camera poses wrt world
    """
    assert pose_change.shape == (4, 4)
    # Read 4x4 camera poses
    cameras = camera_clone(cameras)
    c2w = cameras.camera_to_worlds.to(pose_change.device)
    c2w = torch.cat([c2w, torch.zeros(c2w.shape[0], 1, 4).to(c2w)], dim=1)
    c2w[:, 3, 3] = 1
    c2w[:, 0:3, 1:3] = -c2w[:, 0:3, 1:3] # OpenGL to OpenCV
    # Apply the pose change
    c2w_new = pose_change.inverse() @ c2w
    c2w_new[:, 0:3, 1:3] = -c2w_new[:, 0:3, 1:3] # OpenCV to OpenGL
    cameras.camera_to_worlds = c2w_new[:, 0:3, :]
    return cameras


class ChangeDet:
    """
    Export a 3D segmentation for a target object
    """
    debug_dir = "/home/ziqi/Desktop/test/"
    """Directory to save debug output"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    """Device"""

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
        emb1 = sam_embedding(capture)
        emb2 = sam_embedding(render)
        # Calculate cosine similarity between embeddings
        norm1 = torch.nn.functional.normalize(emb1, p=2, dim=1)
        norm2 = torch.nn.functional.normalize(emb2, p=2, dim=1)
        similarity_map = torch.nn.functional.cosine_similarity(
            norm1, norm2, dim=1
        )
        similarity_map = similarity_map.squeeze().cpu().numpy()
        similarity_map = (similarity_map * 255).astype(np.uint8)
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
        masks = []
        for contour in contours:
            if cv2.contourArea(contour) < threshold * H * W:
                continue
            mask = np.zeros((H, W))
            cv2.drawContours(
                mask, [contour], -1, (255, 255, 255), thickness=cv2.FILLED
            )
            mask = torch.from_numpy(mask).unsqueeze(0)
            masks.append(mask)
        masks = torch.stack(masks, dim=0).to(device)
        # Uncomment to debug
        # for i, mask in enumerate(masks):
        #     cv2.imwrite(
        #         f"{self.debug_dir}/mask_{i}.png",
        #         mask.squeeze().cpu().numpy()
        #     )
        return masks

    def pretrain_iteration(self, rgbs, masks, cameras):
        """
        Forward pass through the pre-trained 3DGS

        Args:
            rgbs (Nx3xHxW): Captured sparse view RGB images
            masks (Nx1xHxW): Sampling masks on the sparse views
            cameras (Cameras): NeRFStudio Cameras object of size N

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
            outputs = self.pipeline_pretrain.model(camera)
            loss_dict = self.pipeline_pretrain.model.get_loss_dict(
                outputs, batch, None
            )
            loss = sum(loss_dict.values())
            loss_accumulated += loss
        return loss_accumulated

    def refine_obj_pose_change(
        self, pose_init, rgbs, masks, cameras,
        lr=1e-3, epochs=500, patience=100
    ):
        """
        Refine object pose change to make the object pose pixel-perfect

        Args:
            pose_init (4x4): Object pose change initial value
            rgbs (Nx3xHxW): Captured sparse view RGB images
            masks (Nx1xHxW): Object masks on the sparse views
            cameras (Cameras): NeRFStudio Cameras object
            batch_size (int): Batch size for training
            epochs (int): Number of epochs
            patience (int): Number of epochs to wait for plateau
        
        Returns:
            pose_refined (4x4): Refined object pose change
        """
        from nerfstudio.cameras.lie_groups import exp_map_SO3xR3
        from nerfstudio.utils.poses import to4x4
        assert hasattr(self, "pipeline_pretrain"), \
            "Pre-training pipeline not loaded yet"
        assert pose_init.shape == (4, 4)
        cam_init = camera_clone(cameras)
        device = rgbs.device
        # Uncomment to debug
        # debug_masks(masks, self.debug_dir)
        # Make a pose update parameter
        pose_update = torch.nn.Parameter(
            torch.zeros((1, 6), device=device)
        )
        optimizer = torch.optim.Adam([pose_update], lr=lr)
        # Training loop
        best_loss, initial_loss = float("inf"), None
        plateau_count = 0
        with tqdm(total=epochs, desc="pose change opt") as pbar:
            for idx in range(epochs):
                optimizer.zero_grad()
                pose_update4x4 = to4x4(exp_map_SO3xR3(pose_update))
                pose_update4x4 = pose_update4x4.reshape(4, 4)
                # Map new cameras to corresponding pre-training cameras
                pretrain_cam = new_cams_to_pretrain(
                    cam_init, pose_init @ pose_update4x4
                )
                # Forward pass
                rgb_loss = self.pretrain_iteration(rgbs, masks, pretrain_cam)
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
        pose_refined = pose_init @ pose_update4x4.detach()
        return pose_refined

    def localize_wrt_posed_RGBD(
        self, rgb1, rgb2,  depth1, pose1, K1, K2, dist1, dist2,
        mask1=None, mask2=None, verbose=False
    ):
        """
        Localize a RGB image wrt a posed RGBD image
        using feature matching and SolvePnP-RANSAC

        Args:
            rgb1: (1x3xHxW) RGB image 1
            rgb2: (1x3xHxW) RGB image 2
            depth1 (1x1xHxW) Depth image 1
            pose1: (4x4 tensor) Camera pose wrt world for Image 1
            K1: (3x3 tensor) Intrinsics for camera 1
            K2: (3x3 tensor) Intrinsics for camera 2
            dist1: (4 tensor) distortion params for camera 1
            dist2: (4 tensor) distortion params for camera 2
            mask1: (1x1xHxW) Mask for image 1 (optional)
            mask2: (1x1xHxW) Mask for image 2 (optional)
            verbose (bool): Whether to print absolute pose estimation summary

        Returns:
            pose (4x4 tensor): Camera pose wrt world for Image 2
            num_inliers (float): Number of inliers
            num_matches (float): Number of matches
        """
        assert len(depth1.shape) == len(rgb1.shape) == len(rgb2.shape) == 4
        assert depth1.shape[1] == 1
        assert rgb1.shape[1] == rgb2.shape[1] == 3
        if mask1 is not None:
            assert mask1.shape[1] == 1
        if mask2 is not None:
            assert mask2.shape[1] == 1
        device = depth1.device
        _, _, H, W = depth1.shape
        # Image feature matching between rgb1 and rgb2
        kp1, kp2, matches = image_matching(
            rgb1, rgb2, mask1, mask2, flip=True
        )
        # Uncomment to debug
        # debug_matches(rgb1, rgb2, kp1, kp2, matches, self.debug_dir)
        kp1, kp2, matches = kp1[0], kp2[0], matches[0] 
        if matches.shape[0] < 4:
            print("Warn: Not enough matches!!")
            return None, 0.0, 0
        # NOTE: Don't undistort before reading depth
        #       since both rgb and depth images are distorted
        # Extract 3D positions of matched keypoints on image 1
        depths_at_kps1 = extract_depths_at_pixels(kp1, depth1)
        # Undistort keypoints on image 1
        kp1_und = undistort_points(
            kp1.unsqueeze(0), K1.unsqueeze(0), dist1.unsqueeze(0)
        ).squeeze(0)
        # Extract 3D positions of matched keypoints on image 1
        points_at_kps1 = depths_to_points(
            kp1_und, depths_at_kps1, pose1, K1
        )
        matched_pts3D = points_at_kps1[matches[:, 0]].cpu().numpy()
        # Extract 2D positions of matched keypoints on image 2
        # NOTE: No need to undistort since distort is considered in SolvePnP
        matched_pts2D = kp2[matches[:, 1]].cpu().numpy()
        # SolvePnP
        pycolmap_cam = pycolmap.Camera(
            model='OPENCV', width=W, height=H, params=[
                K2[0, 0], K2[1, 1], K2[0, -1], K2[1, -1], *dist2
            ]
        )
        ret = pycolmap.absolute_pose_estimation(
            matched_pts2D, matched_pts3D, pycolmap_cam,
            estimation_options={'ransac': {'max_error': 1.0, "min_num_trials": 10000}}, 
            refinement_options={'print_summary': verbose}
        )
        if not ret['success']:
            print("Warn: PnP failed!!")
            return None, 0.0, 0
        R_mat = torch.tensor(Quaternion(*ret['qvec']).rotation_matrix)
        tvec = torch.tensor(ret['tvec'])
        pose = torch.eye(4, device=device)
        pose[:3, :3], pose[:3, 3] = R_mat, tvec
        pose = pose.inverse()
        if verbose:
            print(f"Number of inliers: {ret['num_inliers']}/{len(matches)}")
            print(f"pose change:\n {pose}")
        return pose, ret["num_inliers"], len(matches)

    def main(
        self, transforms_json=None, configs=None,
        refine_pose=True, cam_path=None
    ):
        """
        Estimate object 2D masks, coarse 3D Bbox and pose change

        Args:
            transforms_json (Path or str):
                transforms.json for the post-reconfig training dataset
            save_masks (bool): If True, save the SAM-predicted 2D masks
            focus (bool): Whether to focus point sampling at the changed areas

        Returns:
            obj_3D_seg (list of Obj3DSeg): Object 3D segmentation
            pose_change: (list of 4x4 tensors) Object pose change
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
                "pose_refine_epochs": 500,
                "pose_refine_patience": 100,
                "move_out_2D_dilate": 41,
                "move_in_2D_dilate":31
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
                id_str = re.search(r'(\d+)(?!.*\d)', path.name)
                if not id_str:
                    raise ValueError("Train filenames must contain numbers")
                if "rgb_new" in path.as_posix():
                    sparse_view_file_ids.append(int(id_str.group()))
                    sparse_view_indices.append(ii)
                else:
                    pretrain_indices.append(ii)
                train_file_ids.append(int(id_str.group()))

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
        #   rgbs_render_sparse_view, rgbs_captured_sparse_view, self.debug_dir
        # )

        # Change detection on sparse views
        # TODO: Align the rendered and captured images
        # TODO: Handle multi-obj cases (1) obj association (2) multi-mask
        masks_changed_sparse = []
        for ii in range(len(sparse_view_file_ids)): 
            masks_changed = self.image_diff(
                rgbs_render_sparse_view[ii:ii+1],
                rgbs_captured_sparse_view[ii:ii+1]
            )
            masks_changed_sparse.append(masks_changed)
        # Uncomment to debug
        # masks_changed_tensor = torch.cat(masks_changed_sparse, dim=0)
        # save_masks(
        #     masks_changed_tensor / 255.0, [
        #         f"{self.debug_dir}/masks_changed{i}.png"
        #         for i in range(len(masks_changed_tensor))
        #     ]
        # )
        # Consensus between the sparse views on the number of changed masks
        num_masks_changed = statistics.mode(
            [len(m) for m in masks_changed_sparse]
        )
        masks_move_out_sparse_view, masks_move_in_sparse_view = [], []
        for ii, masks_changed in enumerate(masks_changed_sparse):
            # Compare SAM prediction confidences to determine move-in or -out
            masks_render, scores_render = sam_refine_masks(
                rgbs_render_sparse_view[ii:ii+1],
                masks_changed[:num_masks_changed],
                expand=configs["mask_refine_sparse_view"]
            )
            masks_capture, scores_capture = sam_refine_masks(
                rgbs_captured_sparse_view[ii:ii+1],
                masks_changed[:num_masks_changed],
                expand=configs["mask_refine_sparse_view"]
            )
            discarded_masks = []
            for jj in range(len(scores_render)):
                if scores_render[jj] > scores_capture[jj]:
                    masks_move_out_sparse_view.append(masks_render[jj])
                    discarded_masks.append((1, jj, scores_capture[jj]))
                    if scores_render[jj] < 0.95:
                        print(f"WARN: Bad move-out mask at sparse view {ii}")
                        print(f"score: {scores_render[jj]}")
                else:
                    masks_move_in_sparse_view.append(masks_capture[jj])
                    discarded_masks.append((0, jj, scores_render[jj]))
                    if scores_capture[jj] < 0.95:
                        print(f"WARN: Bad move-in mask at sparse view {ii}")
                        print(f"score: {scores_capture[jj]}")
            # In case there are overlapping masks, we need to:
            # Recycle discarded masks if we have less than needed masks
            discarded_masks.sort(key=lambda x: x[2], reverse=True)
            assert len(discarded_masks)+len(scores_render) >= num_masks_changed
            for _ in range(num_masks_changed - len(scores_render)):
                source, index, score = discarded_masks.pop(0)
                if source == 0:
                    masks_move_out_sparse_view.append(masks_render[index])
                else:
                    masks_move_in_sparse_view.append(masks_capture[index])
                if score < 0.95:
                    inout = ['in', 'out'][source]
                    print(f"WARN: Bad move-{inout} mask at sparse view {ii}")
                    print(f"score: {score}")
        masks_move_out_sparse_view = torch.stack(
            masks_move_out_sparse_view, dim=0
        )
        masks_move_in_sparse_view = torch.stack(
            masks_move_in_sparse_view, dim=0
        )
        # # Uncomment to debug
        # save_masks(
        #     masks_move_in_sparse_view, [
        #         f"{self.debug_dir}/masks_move_in{i}.png"
        #         for i in range(len(masks_move_in_sparse_view))
        #     ]
        # )
        # save_masks(
        #     masks_move_out_sparse_view, [
        #         f"{self.debug_dir}/masks_move_out{i}.png"
        #         for i in range(len(masks_move_out_sparse_view))
        #     ]
        # )
        # Estimate the object pose change from all the sparse views
        # Choose the one with the most inlier
        num_sparse_views = masks_move_out_sparse_view.size(0)
        num_inliers, num_matches = 0, 0
        pose_change = None
        for idx in tqdm(range(num_sparse_views), desc="pose estimation"):
            # Find the sparse view render with the largest number of matches
            _, _, matches = image_matching(
                rgbs_render_sparse_view,
                rgbs_captured_sparse_view[idx:idx+1].repeat(num_sparse_views, 1, 1, 1),
                masks_move_out_sparse_view,
                masks_move_in_sparse_view[idx:idx+1].repeat(num_sparse_views, 1, 1, 1),
                flip=True
            )
            matches_idx_ = [(len(mm), ii) for ii, mm in enumerate(matches)]
            idx_max = max(matches_idx_)[1]
            # Estimate the object pose change
            pose_change_i, num_inlier_i, num_match_i = \
            self.localize_wrt_posed_RGBD(
                rgbs_render_sparse_view[idx_max:idx_max+1],
                rgbs_captured_sparse_view[idx:idx+1],
                depths_sparse_view[idx_max:idx_max+1],
                cam_poses_sparse_view[idx_max],
                Ks_sparse_view[idx_max], Ks_sparse_view[idx],
                dist_params_sparse_view[idx_max], dist_params_sparse_view[idx],
                masks_move_out_sparse_view[idx_max:idx_max+1],
                masks_move_in_sparse_view[idx:idx+1],
            )
            if pose_change_i is None:
                continue
            # Equation in the paper
            pose_change_i =  pose_change_i.inverse()
            pose_change_i = cam_poses_sparse_view[idx] @ pose_change_i

            if configs["pose_change_break"] is not None and \
                idx == configs["pose_change_break"]:
                num_inliers = num_inlier_i
                num_matches = num_match_i
                pose_change = pose_change_i
                break
            if num_inlier_i > num_inliers:
                num_inliers = num_inlier_i
                num_matches = num_match_i
                pose_change = pose_change_i
        assert pose_change is not None, "Object pose change estimation failed!"
        print(f"pose_change: \n {pose_change.cpu().numpy()}")
        print(f"inlier_ratio: {num_inliers} / {num_matches}")

        # Refine object pose change
        if refine_pose:
            new_cameras = params_to_cameras(
                cam_poses_sparse_view, Ks_sparse_view, 
                dist_params_sparse_view, H, W
            )
            pose_change = self.refine_obj_pose_change(
                pose_change, rgbs_captured_sparse_view, 
                masks_move_in_sparse_view, new_cameras,
                lr=configs["pose_refine_lr"],
                epochs=configs["pose_refine_epochs"],
                patience=configs["pose_refine_patience"]
            )
            print(f"refined pose_change: \n {pose_change.cpu().numpy()}")


        # Convert sparse view depth images to point cloud wrt world 
        point_cloud = compute_point_cloud(
            depths_sparse_view, cam_poses_sparse_view, Ks_sparse_view,
            masks_move_out_sparse_view
        )
        point_cloud = point_cloud_filtering(
            point_cloud, configs["pcd_filtering"]
        )
        # debug_point_cloud(point_cloud, self.debug_dir)
        bbox3d = compute_3D_bbox(point_cloud)
        print(f"bbox3d: {bbox3d}")

        # Query SAM to obtain pre-training view 2D masks
        # Project the object point cloud to dense ald views
        point_cloud_proj, is_point_in_img = project_points(
            point_cloud, cam_poses_pretrain_view, Ks_pretrain_view,
            dist_params_pretrain_view, H, W
        )
        if not is_point_in_img.all():
            print("WARN: Some points are out of the pretraining images")
        # debug_point_prompts(
        #   color_images_pretrain_view, point_cloud_proj, self.debug_dir
        # )
        bbox2d = compute_2D_bbox(point_cloud_proj)
        # Slightly expand 2D bboxes to improve SAM predictions
        bbox2d = expand_2D_bbox(
            bbox2d, configs["pre_train_pred_bbox_expand"]
        )
        # SAM predict all masks
        masks, scores = sam_predict(color_images_pretrain_view, bbox2d)
        # Refine low-score masks
        for ii in tqdm(range(len(masks)), desc="SAM refine masks"):
            if scores[ii] < 0.95:
                masks[ii:ii+1], scores[ii:ii+1] = sam_refine_masks(
                    color_images_pretrain_view[ii:ii+1], masks[ii:ii+1],
                    expand=configs["pre_train_refine_bbox_expand"]
                )
        low_mask_scores = [x for x in scores if x < 0.95]
        if len(low_mask_scores) > 0:
            print(f"WARN: Got {len(low_mask_scores)} low score masks")
            print(f"Lowest mask score: {min(low_mask_scores)}")
        else:
            print("All masks look great!!")

        # Concat move-out masks on new sparse and old dense views
        masks_move_out = torch.empty(
            (N, *masks.shape[1:]), dtype=torch.bool
        ).to(device)
        is_sparse_view = torch.zeros(N, dtype=torch.bool).to(device)
        is_sparse_view[sparse_view_indices] = True
        masks_move_out[is_sparse_view] = masks_move_out_sparse_view
        masks_move_out[~is_sparse_view] = masks.to(device)
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

    change_det = ChangeDet(Path(args.config), Path(args.output))
    change_det.main(transforms_json=args.transform)