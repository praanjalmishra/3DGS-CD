# 3DGS-based change detection
import argparse
import numpy as np
import torch
from pathlib import Path

from nerfstudio.models.splatfacto import SplatfactoModel
from nerfstudio.pipelines.base_pipeline import Pipeline, VanillaPipeline
from nerfstudio.utils.debug_utils import debug_masks, debug_images
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils.render_utils import render_cameras



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