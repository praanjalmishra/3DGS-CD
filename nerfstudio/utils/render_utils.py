# Simple rendering utilities for Splatfacto
import os
import json
import torch
import numpy as np
from PIL import Image

from nerfstudio.cameras.camera_paths import get_path_from_json


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


def render_during_train(
    pipeline, cam_path, step, output_dir, render_every=100, loop=False
):
    """
    Render at camera path during training
    Render every 100 steps

    Args:
        pipeline (Pipeline): Vanilla Pipeline
        cam_path (List[Camera]): Camera path
        step (int): Step
        output_dir (str): Output directory
        render_every (int): Render frequency
        loop (bool): Use all poses in the camera path and loop around,
        if False just use the first pose
    """
    assert os.path.isfile(cam_path), f"{cam_path} not a file"
    assert os.path.isdir(output_dir), f"{output_dir} not a directory"
    with open(cam_path, "r", encoding="utf-8") as f:
        camera_path = json.load(f)
    if step % render_every == 0:
        with torch.no_grad():
            cameras = get_path_from_json(camera_path)
            if loop:
                cam = cameras[step % len(cameras):step % len(cameras)+1]
            else:
                cam = cameras[0:1]
            rgb = pipeline.model.get_outputs_for_camera(
                cam, obb_box=None
            )["rgb"].cpu().numpy() * 255
            Image.fromarray((rgb).astype(np.uint8)).save(
                f"{output_dir}/{step:05g}.png"
            )