import torch
import numpy as np

from efficient_sam.build_efficient_sam import (
    build_efficient_sam_vits, build_efficient_sam_vitt
)
from nerfstudio.utils.debug_utils import debug_bbox_prompts
from tqdm import tqdm

# Load EfficientSAM model
device = "cuda" if torch.cuda.is_available() else "cpu"
model_type = "vit_t"
if model_type == "vit_s":
    effsam = build_efficient_sam_vits()
elif model_type == "vit_t":
    effsam = build_efficient_sam_vitt()
else:
    raise ValueError("Invalid model type")
effsam.eval()
effsam.to(device)


def effsam_predict(rgbs, bboxes):
    """
    Query SAM model with bboxes prompts

    Args:
        rgbs: (N, 3, H, W) RGB images
        bboxes (N, 4): Bbox prompts (xyxy)

    Returns:
        masks (N, 1, H, W): Image masks
        scores (N-list): Confidence scores
    """
    if rgbs.shape[1] == 1:
        rgbs = rgbs.repeat(1, 3, 1, 1)
    elif rgbs.shape[1] == 3:
        pass
    else:
        raise ValueError("RGB images are of shape (N, 3, H, W)")
    assert bboxes.shape[-1] == 4, \
        "bbox prompts are of shape (N, 4)"
    assert bboxes.shape[0] == rgbs.shape[0], \
        "Image bbox batch mismatch"
    # Uncomment to debug
    # debug_bbox_prompts(rgbs, bboxes, "/home/ziqi/Desktop/test/")
    bbox_pts = bboxes.reshape(bboxes.shape[0], 1, -1, 2)
    # Make labels for bbox points: 2 for top-left, 3 for bottom-right
    pts_label = torch.tensor([2, 3]).to(bbox_pts.device).reshape(1, 1, -1)
    masks, scores = [], []
    for rgb, bbox_pt in tqdm(zip(rgbs, bbox_pts), desc="EffSAM"):
        rgb = rgb.to(device)
        bbox_pt = bbox_pt.to(device)
        logits, iou = effsam(rgb[None, ...], bbox_pt[None, ...], pts_label)
        sorted_ids = torch.argsort(iou, dim=-1, descending=True)
        iou = torch.take_along_dim(iou, sorted_ids, dim=2)
        logits = torch.take_along_dim(
            logits, sorted_ids[..., None, None], dim=2
        )
        mask = torch.ge(logits[0, 0, 0, :, :], 0)
        masks.append(mask)
        scores.append(iou[0, 0, 0].item())
    masks = torch.stack(masks, dim=0).unsqueeze(1)
    return masks, scores


def effsam_embedding(rgb):
    """
    Get pixel-aligned image embeddings
    @param rgb (HxWx3 np.array or 1x1xHxW tensor): Image
    @return features (1xCxHxW tensor): Pixel-aligned image embeddings
    """
    if isinstance(rgb, np.ndarray):
        if rgb.dtype == np.uint8:
            rgb = rgb.astype(np.float32) / 255.0
        rgb = torch.from_numpy(rgb).permute(2, 0, 1).to(device)[None, ...]
    elif isinstance(rgb, torch.Tensor):
        assert rgb.dim() == 4, "Input tensor should be 1x1xHxW"
        rgb = rgb.to(device)
    features = effsam.get_image_embeddings(rgb).detach()
    features = torch.nn.functional.interpolate(
        features, rgb.shape[-2:], mode="bilinear", align_corners=False
    )
    return features