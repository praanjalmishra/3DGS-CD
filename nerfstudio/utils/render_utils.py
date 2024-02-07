# Simple rendering utilities for Splatfacto
import torch
import numpy as np


def render_cameras(pipeline, cameras, device="cpu"):
    """
    Splatfacto render at cameras

    Args:
        pipeline (SplatfactoPipeline): Splatfacto pipeline
        cameras (NeRFStudio Cameras): Cameras
        device (str): Device

    Returns:
        rgbs (Nx3xHxW Tensor): RGBs
        depths (Nx1xHxW Tensor): Depths
    """
    assert cameras.size > 0, "No cameras provided"
    with torch.no_grad():
        rgbs, depths = [], []
        for cam_idx in range(cameras.size):
            outputs = pipeline.model.get_outputs_for_camera(
                cameras[cam_idx:cam_idx + 1], obb_box=None
            )
            rgb, depth = outputs["rgb"], outputs["depth"]
            rgbs.append(rgb.cpu().numpy())
            depths.append(depth.cpu().numpy())
        rgbs = torch.tensor(
            np.array(rgbs), device=device
        ).permute(0, 3, 1, 2)
        depths = torch.tensor(
            np.array(depths), device=device
        ).permute(0, 3, 1, 2)
    return rgbs, depths