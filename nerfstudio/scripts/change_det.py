# 3DGS-based change detection
import argparse
import cv2
import json
import numpy as np
import os
import pycolmap
import re
import statistics
import torch

from lightglue import viz2d
from pathlib import Path
from matplotlib import pyplot as plt
from nerfstudio.cameras.camera_paths import get_path_from_json
from nerfstudio.models.splatfacto import SplatfactoModel
from nerfstudio.pipelines.base_pipeline import Pipeline, VanillaPipeline
from nerfstudio.utils.debug_utils import (
    debug_masks, debug_images, debug_image_pairs, debug_matches
)
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils.img_utils import (
    undistort_images, image_align, image_matching, extract_depths_at_pixels
)
from nerfstudio.utils.io import (
    read_imgs, read_dataset, read_transforms, save_masks
)
from nerfstudio.utils.obj_3d_seg import Object3DSeg
from nerfstudio.utils.pcd_utils import (
    compute_point_cloud, point_cloud_filtering, compute_3D_bbox, expand_3D_bbox
)
from nerfstudio.utils.proj_utils import (
    undistort_points, project_points, depths_to_points, proj_check_3D_points
)
from nerfstudio.utils.render_utils import render_cameras
from nerfstudio.utils.sam_utils import (
    sam_embedding, sam_predict, sam_refine_masks,
    expand_2D_bbox, compute_2D_bbox
)
from pyquaternion import Quaternion
from tqdm import tqdm


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
        refine_pose=False, cam_path=None
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
        # if not self.output_dir.exists():
        #     self.output_dir.mkdir(parents=True)

        # Load pre-trained 3DGS
        assert os.path.isfile(self.load_config)
        _, self.pipeline_pretrain, _, _ = eval_setup(
            self.load_config, test_mode="inference"
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
    args = parser.parse_args()

    change_det = ChangeDet(Path(args.config), Path(args.output))
    change_det.main(transforms_json=args.transform)