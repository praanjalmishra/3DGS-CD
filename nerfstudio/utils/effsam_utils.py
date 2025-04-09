import numpy as np
import torch

from efficient_sam.build_efficient_sam import (
    build_efficient_sam_vits, build_efficient_sam_vitt
)
from nerfstudio.utils.debug_utils import (
    debug_bbox_prompts, debug_point_prompts
)
from tqdm import tqdm
import torch.nn.functional as F


# Load EfficientSAM model
device = "cuda" if torch.cuda.is_available() else "cpu"
model_type = "vit_s"
if model_type == "vit_s":
    effsam = build_efficient_sam_vits()
elif model_type == "vit_t":
    effsam = build_efficient_sam_vitt()
else:
    raise ValueError("Invalid model type")
effsam.eval()
effsam.to(device)


def compute_2D_bbox(points):
    """
    Compute bboxes for a batch of 2D points

    Args:
        points (NxMx2 Tensor): 2D points

    Returns:
        bboxes (Nx4 Tensor): 2D bboxes (xyxy)
    """
    assert len(points.shape) == 3
    mins, _ = torch.min(points, dim=1)
    maxs, _= torch.max(points, dim=1)
    bboxes = torch.cat((mins, maxs), dim=1)
    return bboxes


def expand_2D_bbox(bboxes, percent=0.05):
    """
    Expand 2D bboxes by a certain percentage

    Args:
        bboxes (Nx4 tensor): 2D bboxes
        H, W (int): Image height and width
        percent (float): percentage to expand (xyxy)

    Returns:
        expanded_bboxes (Nx4 tensor): expanded bboxes (xyxy)
    """
    assert bboxes.shape[-1] == 4
    bboxes = bboxes.float()
    # Calculate width and height of each box
    widths = bboxes[:, 2] - bboxes[:, 0]
    heights = bboxes [:, 3] - bboxes[:, 1]
    # Calculate the expansion for width and height
    expand_width = widths * percent
    expand_height = heights * percent
    # Adjust the bbax coordinates 
    expanded_bboxes = bboxes.clone()
    expanded_bboxes[:, 0] -= expand_width / 2
    expanded_bboxes[:, 1] -= expand_height / 2
    expanded_bboxes[:, 2] += expand_width / 2
    expanded_bboxes[:, 3] += expand_height / 2
    expanded_bboxes [:, 3] += expand_height / 2
    return expanded_bboxes


def effsam_predict(rgbs, bboxes=None, points=None):
    """
    Query SAM model with bboxes prompts

    Args:
        rgbs: (N, 3, H, W) RGB images
        bboxes (N, 4): Bbox prompts (xyxy)
        points (N, K, 2): Positive point prompts

    Returns:
        masks (N, 1, H, W): Image masks
        scores (N-list): Confidence scores
    """
    if rgbs.shape[1] == 1:
        rgbs = rgbs.repeat(1, 3, 1, 1)
    elif rgbs.shape[1] != 3:
        raise ValueError("RGB images must be (N, 3, H, W)")

    assert bboxes is None or bboxes.shape[-1] == 4, "bbox prompts must be (N, 4)"
    assert bboxes is None or bboxes.shape[0] == rgbs.shape[0], "Mismatch between image and bbox batches"

    device = rgbs.device
    pts, pts_labels = [], []

    if bboxes is not None:
        bbox_pts = bboxes.reshape(bboxes.shape[0], 1, -1, 2)
        labels = torch.tensor([2, 3], device=bbox_pts.device).reshape(1, 1, -1)
        pts.append(bbox_pts)
        pts_labels.append(labels)

    if points is not None:
        assert points.shape[0] == rgbs.shape[0]
        pts.append(points[:, None, :, :])
        pts_labels.append(torch.ones(1, 1, points.shape[1], device=device))

    pts = torch.cat(pts, dim=2)
    pts_label = torch.cat(pts_labels, dim=2)

    masks, scores = [], []

    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    for rgb, bbox_pt in tqdm(zip(rgbs, pts), desc="EffSAM"):
        try:
            rgb = rgb.to(device)
            bbox_pt = bbox_pt.to(device)

            with torch.no_grad():
                logits, iou = effsam(rgb[None, ...], bbox_pt[None, ...], pts_label)

        except torch.cuda.OutOfMemoryError:
            print("[EffSAM] CUDA OOM! Falling back to CPU.")
            torch.cuda.empty_cache()
            effsam.cpu()
            rgb = rgb.cpu()
            bbox_pt = bbox_pt.cpu()
            pts_label_cpu = pts_label.cpu()

            with torch.no_grad():
                logits, iou = effsam(rgb[None, ...], bbox_pt[None, ...], pts_label_cpu)

            effsam.to(device)  # Optionally move back to GPU for next

        sorted_ids = torch.argsort(iou, dim=-1, descending=True)
        iou = torch.take_along_dim(iou, sorted_ids, dim=2)
        logits = torch.take_along_dim(logits, sorted_ids[..., None, None], dim=2)
        mask = torch.ge(logits[0, 0, 0, :, :], 0)

        masks.append(mask)
        scores.append(iou[0, 0, 0].item())

    masks = torch.stack(masks, dim=0).unsqueeze(1)
    return masks, scores



def effsam_batch_predict(rgb, bboxes):
    """
    Multi-bbox batch predict with EfficientSAM

    Args:
        rgb (1, 3, H, W): RGB image
        bboxes (N, 4): Bbox prompts (xyxy)

    Returns:
        masks (N, 1, H, W): Image masks
        scores (N-list): Confidence scores
    """
    assert rgb.shape[:2] == (1, 3)
    assert len(bboxes.shape) == 2 and bboxes.shape[-1] == 4

    device = rgb.device
    bbox_pts = bboxes.reshape(1, bboxes.shape[0], 2, 2)
    labels = torch.tensor([2, 3], device=bbox_pts.device).view(1, 1, 2).repeat(1, bboxes.shape[0], 1)

    try:
        rgb = rgb.to(device)
        bbox_pts = bbox_pts.to(device)
        labels = labels.to(device)

        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        with torch.no_grad():
            logits, iou = effsam(rgb, bbox_pts, labels)

    except torch.cuda.OutOfMemoryError:
        print("[EffSAM] CUDA OOM in batch predict — falling back to CPU")
        torch.cuda.empty_cache()
        effsam.cpu()
        rgb = rgb.cpu()
        bbox_pts = bbox_pts.cpu()
        labels = labels.cpu()

        with torch.no_grad():
            logits, iou = effsam(rgb, bbox_pts, labels)

        effsam.to(device)

    sorted_ids = torch.argsort(iou, dim=-1, descending=True)
    iou = torch.take_along_dim(iou, sorted_ids, dim=2)
    logits = torch.take_along_dim(logits, sorted_ids[..., None, None], dim=2)

    masks = torch.ge(logits[0, :, 0, :, :], 0)
    masks = masks.unsqueeze(1)
    scores = iou[0, :, 0].tolist()

    return masks, scores



def effsam_embedding(rgb, upsample=True, target_size=(512, 512)):
    """
    Get pixel-aligned image embeddings
    @param rgb (HxWx3 np.array or 1x3xHxW tensor): Image
    @param upsample (bool): Whether to upsample the features
    @param target_size (tuple): Resize input to this before embedding
    @return features (1xCxHxW tensor): Pixel-aligned image embeddings
    """
    if isinstance(rgb, np.ndarray):
        if rgb.dtype == np.uint8:
            rgb = rgb.astype(np.float32) / 255.0
        rgb = torch.from_numpy(rgb).permute(2, 0, 1)  # [3, H, W]
    elif isinstance(rgb, torch.Tensor):
        assert rgb.dim() in [3, 4]
        if rgb.dim() == 4:
            rgb = rgb.squeeze(0)  # [3, H, W]

    import torchvision.transforms.functional as TF
    orig_size = rgb.shape[1:]  # (H, W)
    rgb_resized = TF.resize(rgb, target_size)  # [3, H', W']
    rgb_resized = rgb_resized.unsqueeze(0).to(device)

    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    try:
        with torch.no_grad():
            features = effsam.get_image_embeddings(rgb_resized).detach()
    except torch.cuda.OutOfMemoryError:
        print("[EffSAM] CUDA OOM during embedding — switching to CPU")
        effsam.cpu()
        rgb_resized = rgb_resized.cpu()
        with torch.no_grad():
            features = effsam.get_image_embeddings(rgb_resized).detach()
        effsam.to(device)

    # Upsample back to original resolution if needed
    if upsample:
        features = F.interpolate(
            features, size=orig_size, mode="bilinear", align_corners=False
        )
    return features



def get_effsam_embedding_in_masks(rgbs, masks):
    """
    Get pixel-aligned image embeddings in masks
    @param rgbs (Nx3xHxW): RGB images
    @param masks (N-list of Mx1xHxW): 2D object masks
    @return features (N-list of MxK): Per-view per-object embedding vectors
    """
    from nerfstudio.utils.img_utils import median_high_dim
    assert rgbs.shape[1] == 3
    device = rgbs.device
    embeddings = []
    for rgb, mask in zip(rgbs, masks):
        embedding = effsam_embedding(rgb[None], upsample=False)
        if len(mask) == 0:
            embeddings.append(torch.zeros(0, embedding.shape[1]).to(device))
            continue
        assert mask.shape[1] == 1
        mask = torch.nn.functional.interpolate(
            mask.float(), embedding.shape[-2:], mode="nearest"
        ).bool()
        embed_vecs = []
        for m in mask:
            embedding_map = embedding[0, :, m.squeeze()].permute(1, 0)
            embedding_vec = median_high_dim(embedding_map)
            embed_vecs.append(embedding_vec)
        embed_vecs = torch.stack(embed_vecs, dim=0)
        embeddings.append(embed_vecs)
    return embeddings


def effsam_refine_masks(rgb, masks, expand=0.1):
    """
    Use SAM to refine the masks on a RGB image

    Args:
        rgb: (1, 3, H, W) RGB image
        masks: (M, 1, H, W) Image masks
        expand (float): Expansion percentage for bbox

    Returns:
        masks_refined (M, 1, H, W): Refined image masks
        scores (M-list): Confidence scores
    """
    assert rgb.shape[1] == 3
    assert len(masks.shape) == 4

    bboxes = []
    for mask in masks:
        point_coords = torch.nonzero(mask.squeeze())[:, [1, 0]]  # (y, x) → (x, y)
        if point_coords.shape[0] == 0:
            # Skip empty masks
            bboxes.append(torch.tensor([[0, 0, 1, 1]], device=mask.device, dtype=torch.float))
            continue
        bbox = compute_2D_bbox(point_coords.unsqueeze(0))
        bbox = expand_2D_bbox(bbox, expand)
        bboxes.append(bbox)

    bboxes = torch.cat(bboxes, dim=0)

    try:
        masks_refined, scores = effsam_predict(rgb.repeat(masks.shape[0], 1, 1, 1), bboxes)
    except Exception as e:
        print("[EffSAM] Refinement failed:", e)
        masks_refined = masks
        scores = [0.0 for _ in range(masks.shape[0])]

    return masks_refined, scores
