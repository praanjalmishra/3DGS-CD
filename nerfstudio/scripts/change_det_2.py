# 3DGS-based change detection
import argparse
import json
import os
import re
from pathlib import Path
import datetime

import open3d as o3d

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
        cv2.imwrite(f"{self.debug_dir}/similarity_map.png", similarity_map)
        # Threshold the SAM cosine similarity map
        thresh = cv2.threshold(
            similarity_map, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
        )[1]
        # remove the influence of the black region in the aligned image
        thresh = thresh * align_mask
        # Uncomment to debug
        cv2.imwrite(f"{self.debug_dir}/thresh.png", thresh)
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
        for i, mask in enumerate(masks):
            cv2.imwrite(
                f"{self.debug_dir}/mask_{i}.png",
                mask.squeeze().cpu().numpy()
            )
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
        if debug:
            debug_image_pairs(
                rgbs_render_sparse_view, rgbs_captured_sparse_view,
                self.debug_dir
            )

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
                
                # Get depth values
                z = depth[y, x]  # This will work correctly now
                # After getting the depth map
                depth_stats = {
                    "min": depth.min().item(),
                    "max": depth.max().item(),
                    "mean": depth.mean().item(),
                    "median": torch.median(depth[depth > 0]).item() if (depth > 0).any() else 0,
                    "non_zero_count": (depth > 0).sum().item()
                }
                print(f"Depth stats for view {ii}: {depth_stats}")
                
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
            # for idx, pts in enumerate(points_changed_sparse):
            #     if pts.shape[0]== 0:
            #         continue
            #     self.save_pcds(pts, self.debug_dir / f"changed_mask_{idx:03d}.ply")
            
            # save depth map 
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
                
                # Optional: Also save a color-mapped version for better visualization
                depth_color = cv2.applyColorMap(
                    (normalized_depth / 256).astype(np.uint8),  # Convert to 8-bit for color mapping
                    cv2.COLORMAP_JET
                )
                cv2.imwrite(str(self.debug_dir / f"depth_view_color_{i:03d}.png"), depth_color)


        if debug:
            masks_changed_tensor = torch.cat(masks_changed_sparse, dim=0)
            save_masks(
                masks_changed_tensor / 255.0, [
                    f"{self.debug_dir}/masks_changed{i}.png"
                    for i in range(len(masks_changed_tensor))
                ]
            )




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