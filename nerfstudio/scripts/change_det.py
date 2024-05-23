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
    debug_point_cloud, debug_point_prompts
)
from nerfstudio.utils.effsam_utils import (
    effsam_predict, effsam_embedding, effsam_refine_masks,
    effsam_batch_predict, compute_2D_bbox, expand_2D_bbox
)
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils.img_utils import (
    extract_depths_at_pixels, image_align, filter_features_with_mask
)
from nerfstudio.utils.io import (
    read_dataset, read_imgs, read_transforms, save_masks, params_to_cameras
)
from nerfstudio.utils.obj_3d_seg import Object3DSeg, Obj3DFeats
from nerfstudio.utils.pcd_utils import (
    compute_3D_bbox, compute_point_cloud, expand_3D_bbox,
    point_cloud_filtering, nn_distance, pcd_size, bbox2voxel
)
from nerfstudio.utils.proj_utils import (
    depths_to_points, proj_check_3D_points, project_points
)
from nerfstudio.utils.render_utils import render_cameras
# from nerfstudio.utils.sam_utils import (
#     sam_embedding, sam_predict, sam_refine_masks
# )


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
    extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)
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

    def get_features_in_masks(self, rgbs, masks):
        """
        Extract SuperPoint descriptors in the masked regions

        Args:
            rgbs (Nx3xHxW): RGB images
            masks (N-list of Mx1xHxW): Image masks

        Returns:
            feats (N-list of M-list of TxC): SuperPoint descriptors
        """
        assert rgbs.shape[1] == 3 and len(rgbs) == len(masks)
        feats_all = []
        for i in range(len(rgbs)):
            feat_i = []
            for j in range(len(masks[i])):
                feat = self.extractor.extract(rgbs[i])
                # Ensure keypoints are within image
                feat["keypoints"].clamp_(min=0)
                # Filter keypoints using masks
                feat = filter_features_with_mask(feat, masks[i][j:j+1])
                feat_i.append(feat)
            feats_all.append(feat_i)
        return feats_all

    def match_move_out(self, rgbs, depths, masks, poses, Ks):
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
            pcd = point_cloud_filtering(pcd, 0.95)
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
                pcd = point_cloud_filtering(pcd, 0.95)
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
                    pcd_feats.append(Obj3DFeats(feats[i][k], pts3D))
        # Filter out object point clouds that appear in <25% of images
        pcds = [p for p, ct in zip(pcds, pcd_counts) if ct > N * 0.25]
        pcd_feats = [
            e for e, ct in zip(pcd_feats, pcd_counts) if ct > N * 0.25
        ]
        return pcds, pcd_feats

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
        # Uncomment to vis the pose regression process
        # blend = rgbs[-1] * 0.2 + outputs["rgb"] * masks[-1] * 0.8
        # blend = (blend.detach().cpu().numpy() * 255).astype(np.uint8)
        # Image.fromarray(blend).save(self.debug_dir + f"blend{debug_count}.png")
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
        #     rgbs_render_sparse_view, rgbs_captured_sparse_view, self.debug_dir
        # )

        # Change detection on sparse views
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
        masks_move_out_sparse_view = []
        for ii, masks_changed in enumerate(masks_changed_sparse):
            masks_render, scores_render = effsam_refine_masks(
                rgbs_render_sparse_view[ii:ii+1], masks_changed,
                expand=configs["mask_refine_sparse_view"]
            )
            # Move-out masks have SAM prediction scores > 0.95 on rendered image
            masks_out = torch.cat([
                masks_render[i:i+1] 
                for i, s in enumerate(scores_render) if s > 0.95
            ], dim=0)
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
            Ks_sparse_view[no_overlap_ind]
        )
        # Object pose change estimate
        feats = self.get_features_in_masks(
            rgbs_captured_sparse_view, 
            [m.any(dim=0, keepdim=True) for m in masks_changed_sparse]
        )
        pose_changes = []
        num_sparse_views = len(masks_move_out_sparse_view)
        for pcd_feat in pcd_feats:
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
                #     m3d_proj[:, :1, :], [m2d[:1, :]],
                #     torch.arange(1)[None, :, None].repeat(1, 1, 2),
                #     self.debug_dir
                # )
                if pose_change_i is None:
                    continue
                # Equation in the paper
                pose_change_i =  cam_poses_sparse_view[idx] @ pose_change_i.inverse()
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
            pose_changes.append(pose_change)
            assert pose_change is not None, "Object pose change estimation failed!"
            print(f"pose_change: \n {pose_change.cpu().numpy()}")
            print(f"inlier_ratio: {num_inliers} / {num_matches}")
        # # Uncomment to debug
        # debug_point_cloud(pcds[0], self.debug_dir)

        # Refine object pose change
        if refine_pose:
            new_cameras = params_to_cameras(
                cam_poses_sparse_view, Ks_sparse_view, 
                dist_params_sparse_view, H, W
            )
            # Move obj pcds and project to sparse views to obtain 2D bboxes
            bboxes2d = []
            for ii, (pcd, ps_chg) in enumerate(zip(pcds, pose_changes)):
                pcd_recfg = (ps_chg[:3, :3] @ pcds[ii].T + ps_chg[:3, 3:4]).T
                pcd_proj, is_point_in_img = project_points(
                    pcd_recfg, cam_poses_sparse_view, Ks_sparse_view,
                    dist_params_sparse_view, H, W
                )
                if not is_point_in_img.all():
                    print("WARN: Some points are out of the pretraining images")
                # debug_point_prompts(
                #     rgbs_captured_sparse_view, pcd_proj, self.debug_dir
                # )
                bbox2d = compute_2D_bbox(pcd_proj)
                bbox2d = expand_2D_bbox(
                    bbox2d, configs["pre_train_pred_bbox_expand"]
                )
                bboxes2d.append(bbox2d)
            bboxes2d = torch.stack(bboxes2d, dim=1) # NxMx4
            # SAM predict move-in masks on sparse-view captured images
            masks_move_in_sparse_view, scores = [], []
            for img, bbox2d in tqdm(
                zip(rgbs_captured_sparse_view, bboxes2d), desc="SAM predict"
            ):
                mask, score = effsam_batch_predict(img[None], bbox2d)
                masks_move_in_sparse_view.append(mask)
                scores.append(score)
            masks_move_in_sparse_view = torch.stack(
                masks_move_in_sparse_view, dim=1
            ) # MxNx1xHxW
            scores = [list(t) for t in zip(*scores)] # M-list of N-list
            for ii, pose_change in enumerate(pose_changes):
                obj_masks_move_in_sparse_view = masks_move_in_sparse_view[ii]
                high_score = [i for i, s in enumerate(scores[ii]) if s > 0.95]
                assert len(high_score) > 0, "No good move-in masks"
                high_score = torch.from_numpy(np.array(high_score).astype(int))
                pose_changes[ii] = self.refine_obj_pose_change(
                    pose_change, rgbs_captured_sparse_view[high_score],
                    obj_masks_move_in_sparse_view[high_score],
                    new_cameras[high_score],
                    lr=configs["pose_refine_lr"],
                    epochs=configs["pose_refine_epochs"],
                    patience=configs["pose_refine_patience"]
                )
                print(
                    f"refined pose_change: \n {pose_changes[ii].cpu().numpy()}"
                )

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
            Ks_pretrain_view, dist_params_pretrain_view, H, W
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

        # Compute finer object 3D segmentation
        # Initialize a 3D voxel grid
        for ii, pcd in enumerate(pcds):
            bbox3d = compute_3D_bbox(pcd)
            print(f"bbox3d: {bbox3d}")
            expanded_bbox3d = expand_3D_bbox(bbox3d, configs["bbox3d_expand"])
            voxel = bbox2voxel(bbox3d, configs["voxel_dim"], self.device)
            occ_grid = proj_check_3D_points(
                voxel, cam_poses_pretrain_view[high_score_inds[ii]], 
                Ks_pretrain_view[high_score_inds[ii]],
                dist_params_pretrain_view[high_score_inds[ii]],
                masks_move_out_pretrain_view[ii][high_score_inds[ii]],
                cutoff=0.99
            )
            obj3Dseg = Object3DSeg(
                *expanded_bbox3d, occ_grid, pose_change, bbox3d,
                configs["move_out_dilate"], 
                configs["mask3d_dilate_uniform"],
                configs["mask3d_dilate_top"]
            )
            # # Uncomment to debug
            # obj3Dseg.visualize(self.debug_dir)
        obj3Dseg.save(self.output_dir)
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