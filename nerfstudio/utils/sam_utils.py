import numpy as np
import torch

from segment_anything import SamPredictor, sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
from tqdm import tqdm
from nerfstudio.utils.debug_utils import debug_bbox_prompts


"""Directory to save debug output"""
debug_dir = "/home/ziqi/Desktop/test/"
"""Path to MobileSAM weights"""
sam_checkpoint = "/home/ziqi/Packages/Track-Anything/checkpoints/sam_vit_h_4b8939.pth"
"""MobileSAM model type"""
model_type = "vit_h"
"""Device"""
device = "cuda" if torch.cuda.is_available() else "cpu"

# NOTE: This can be easily changed to mobile sam by changing lib name

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device)
sam.eval()

predictor = SamPredictor(sam)


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


def sam_embedding(rgb):
    """
    Get pixel-aligned image embeddings
    @param rgb (HxWx3 np.array): Image
    @return features (1xCxHxW tensor): Pixel-aligned image embeddings
    """
    if rgb.dtype == np.float32:
        rgb = (rgb * 255).astype(np.uint8)
    predictor.set_image(rgb)
    features = predictor.get_image_embedding()
    resize_transform = ResizeLongestSide(sam.image_encoder.img_size)
    # Reshape the image embedding to the original image size
    img_part_size = resize_transform.get_preprocess_shape(
        *rgb.shape[:2], sam.image_encoder.img_size
        # conanical input image size
    )
    features_original_size = sam.postprocess_masks(
        features, img_part_size, rgb.shape[:2]
    )
    return features_original_size


def sam_predict(rgbs, bboxes):
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
    bboxes = bboxes.cpu().numpy()
    masks, scores = [], []
    for rgb, bbox in tqdm(zip(rgbs, bboxes), desc="SAM"):
        rgb = rgb.permute(1, 2, 0).cpu().numpy()
        rgb = (rgb * 255).astype(np.uint8)
        predictor.set_image(rgb)
        mask, score, _ = predictor.predict(
            point_coords=None, point_labels=None,
            box=bbox, multimask_output=False
        )
        scores.append(score.item())
        masks.append(torch.from_numpy(mask))
    masks =  torch.cat(masks, dim=0).to(rgbs.device).unsqueeze(1)
    return masks, scores


def sam_refine_masks(rgb, masks, expand=0.1):
    """
    Use SAM to refine the masks on a RGB image

    Args:
        rgbs: (1, 3, H, W) RGB images
        masks: (M, 1, H, W) Image masks
        expand (float): How much we expand the extracted bbox as prompt (%) 

    Returns:
        masks_refined (M, 1, H, W): Refined image masks   
        scores (M-list): Confidence scores     
    """
    assert rgb.shape[1] == 3
    assert len(masks.shape) == 4
    bboxes = []
    for mask in masks:
        point_coords = torch.nonzero(mask.squeeze())[:, [1, 0]]
        bbox = compute_2D_bbox(point_coords.unsqueeze(0))
        bbox = expand_2D_bbox(bbox, expand)
        bboxes.append(bbox)
    bboxes = torch.cat(bboxes, dim=0)
    masks_refined, scores = sam_predict(
        rgb.repeat(masks.shape[0], 1, 1, 1), bboxes
    )
    return masks_refined, scores
