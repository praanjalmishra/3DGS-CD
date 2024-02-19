# 3DGS-based change detection
import argparse
import cv2
import numpy as np
import os
import pycolmap
import re
import statistics
import torch

from lightglue import viz2d
from pathlib import Path
from matplotlib import pyplot as plt
from nerfstudio.models.splatfacto import SplatfactoModel
from nerfstudio.pipelines.base_pipeline import Pipeline, VanillaPipeline
from nerfstudio.utils.debug_utils import (
    debug_masks, debug_images, debug_image_pairs
)
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils.img_utils import (
    undistort_images, image_align, image_matching, extract_depths_at_pixels
)
from nerfstudio.utils.io import read_imgs, read_dataset, read_transforms
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
    sam_checkpoint = "/home/ziqi/Packages/Track-Anything/checkpoints/sam_vit_h_4b8939.pth"
    """Path to MobileSAM weights"""
    model_type = "vit_h"
    """MobileSAM model type"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    """Device"""

    # NOTE: This can be easily changed to mobile sam by changing lib name
    from segment_anything import SamPredictor, sam_model_registry
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device)
    sam.eval()

    predictor = SamPredictor(sam)

    def __init__(self, load_config: Path, output_dir: Path):
        # Path to the config.yml file of the pretrained 3DGS
        self.load_config = load_config
        # Path to save the output 3D segmentation
        self.output_dir = output_dir

    def main(self, transforms_json=None, refine_pose=False):
        pass

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