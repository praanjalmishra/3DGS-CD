# 2D image and mask operation functions
import cv2
import numpy as np
import torch

# from lightglue import LightGlue, SuperPoint
# from lightglue.utils import rbd
from scipy.ndimage import binary_dilation, generate_binary_structure
from torchvision.ops import RoIAlign, RoIPool
from tqdm import tqdm


def in_image(points2d, H, W):
    """
    Check if 2D points are in image

    Args:
        points2d (..., 2): 2D points
        H, W (int): Image height and width

    Returns:
        in_image_mask (...): Mask for points in image
    """
    assert points2d.shape[-1] == 2
    in_image_mask = (points2d[..., 0] >= 0) & (points2d[..., 0] < W) & \
        (points2d[..., 1] >= 0) & (points2d[..., 1] < H)
    return in_image_mask


def dilate_masks(masks, kernel_size=3):
    """
    Dilate binary masks using scipy's binary_dilation.
    
    Args:
        masks (Nx1xHxW): Binary masks
        kernel_size (int): Number of dilation iterations

    Returns:
        dilated_masks (Nx1xHxW): Dilated masks        
    """
    masks_np = masks.cpu().numpy()
    dilated_masks_np=[]
    for mask_np in tqdm(masks_np, desc="Dilate images"):
        dilated_mask_np = binary_dilation(
            mask_np[0], iterations=kernel_size,
            structure=generate_binary_structure(2, 3)
        )
        dilated_masks_np.append(dilated_mask_np)
    dilated_masks = torch.from_numpy(np.array(dilated_masks_np)).to(masks)
    return dilated_masks.unsqueeze(1)


def masks_to_focus(masks, kernel_ratio=0.15):
    """
    Dilate masks and only keep the new pixels masked out

    Args:
        masks (Nx1xHxW): Binary masks
        kernel_ratio (int): Ratio of the kernel size to the image size

    Returns:
        focus_masks (Nx1xHxW): Dilated masks
    """
    assert masks.shape[1] == 1, "Masks shape must be Nx1xHxW"
    masks_np = masks.cpu().numpy()
    kernel_size = int(masks.shape[-1] * kernel_ratio)
    focus_masks_np = []
    for mask_np in tqdm(masks_np, desc="Dilate to focus"):
        dilated_mask_np = binary_dilation(
            mask_np[0], iterations=kernel_size,
            structure=generate_binary_structure(2, 3)
        )
        # Keep only the new pixels white
        focus_mask_np = np.where(mask_np[0] == 1, 0, dilated_mask_np)
        # Invert the mask to keep only the new pixels masked out
        focus_mask_np = 1 - focus_mask_np
        focus_masks_np.append(focus_mask_np)
    focus_masks = torch.from_numpy(np.array(focus_masks_np)).to(masks)
    return focus_masks.unsqueeze(1)


def points2D_to_mask(points2D, valid2D, H, W):
    """
    Creates a mask from a set of 2D points.
    TODO: Make this more efficient
    
    Args:
        points2D (N, M, 2): 2D points
        valid2D (N, M): Mask for valid 2D points
        H (int): Height of the mask
        W (int): width of the mask

    Returns:
        mask (N, 1, H, W): Binary mask
    """
    assert points2D.shape[-1] == 2
    masks = []
    for idx, pts2D in enumerate(tqdm(points2D, desc="points to masks")):
        # Filter points that are outside the image
        valid_pts2D = pts2D[valid2D[idx]]
        mask = np.zeros((H, W), dtype=np.uint8)
        if len(valid_pts2D) != 0: # If no points in image, all-black mask
            valid_pts2D = valid_pts2D.cpu().numpy().astype(np.int32)
            hull = cv2.convexHull(valid_pts2D)
            cv2.fillPoly(mask, [hull], 1)
        mask = torch.from_numpy(mask).bool()[None, None, :, :]
        masks.append(mask)
    masks = torch.cat(masks, dim=0)
    return masks.to(points2D.device)


def points2D_to_point_masks(points2D, valid2D, H, W, kernel_size=3):
    """
    Creates a point mask from a set of 2D points.

    Args:
        points2D (N, M, 2): 2D points
        valid2D (N, M): Mask for valid 2D points
        H (int): Height of the mask
        W (int): width of the mask

    Returns:
        mask (N, 1, H, W): Binary mask
    """
    assert points2D.shape[-1] == 2
    masks = []
    for idx, pts2D in enumerate(tqdm(points2D, desc="points to pt masks")):
        # Filter points that are outside the image
        valid_pts2D = pts2D[valid2D[idx]]
        mask = np.zeros((H, W), dtype=np.uint8)
        if len(valid_pts2D) != 0: # If no points in image, all-black mask
            valid_pts2D = valid_pts2D.cpu().numpy().astype(np.int32)
            is_in_img = in_image(valid_pts2D, H, W)
            valid_pts2D = valid_pts2D[is_in_img]
            mask[valid_pts2D[:, 1], valid_pts2D[:, 0]] = 255
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = torch.from_numpy(mask).bool()[None, None, :, :]
        masks.append(mask)
    masks = torch.cat(masks, dim=0)
    return masks.to(points2D.device)

def filter_features_with_mask(feats, masks):
    """
    Filter out features that are not in the mask

    Args:
        feats (dict): SuperPoint feature positions and descriptors
        masks (1x1xHxW): Binary masks

    Returns:
        feats (dict): SuperPoints within the mask
    """
    keypoints = feats['keypoints']
    valid_indices = masks[
        0, 0, keypoints[..., 1].long(), keypoints[..., 0].long()
    ] > 0.5
    feats['keypoints']= keypoints[valid_indices].unsqueeze(0)
    feats['keypoint_scores'] = \
        feats['keypoint_scores'][valid_indices].unsqueeze(0)
    feats['descriptors'] = \
        feats['descriptors'][valid_indices].unsqueeze(0)
    return feats


# def image_matching(imgs1, imgs2, masks1=None, masks2=None):
#     """
#     Match imgs1 and imgs2 using their respective masks
#     and return the keypoints and matches.

#     Parameters:
#         imgs1, imgs2 (N, 3, H, W): Images
#         masks1, masks2 (N, 1, H, W or None): Masks

#     Returns:
#         kp_list1 (N-list of M1x2 tensor): Keypoints on image 1
#         kp_list2 (N-list of M2x2 tensor): Keypoints on image 2
#         matches_list (N-list of Mx2 tensor): Matched keypoint IDs
#     """
#     device = imgs1.device
#     # Load extractor and matcher modules
#     extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)
#     matcher = LightGlue(features='superpoint').eval().to(device)

#     # Initialize Lists to store results for each pair
#     kp_list1, kp_list2, matches_list = [], [], []
#     # Process each image pair individually
#     for i in range(imgs1.size(0)):
#         # Extract features
#         feats1 = extractor.extract(imgs1[i])
#         feats2 = extractor.extract(imgs2[i])
#         # Ensure keypoints are within image
#         feats1["keypoints"].clamp_(min=0)
#         feats2["keypoints"].clamp_(min=0)
#         # Filter keypoints using masks
#         if masks1 is not None:
#             feats1 = filter_features_with_mask(
#                 feats1, masks1[i:i+1].to(device)
#             )
#         if masks2 is not None:
#             feats2 = filter_features_with_mask(
#                 feats2, masks2[i:i+1].to(device)
#             )
#         # Match the extracted features
#         matches = matcher({'image0': feats1, 'image1': feats2})
#         feats1, feats2, matches = [
#             rbd(x) for x in [feats1, feats2, matches]
#         ]
#         # Append results to lists
#         kp_list1.append(feats1['keypoints'])
#         kp_list2.append(feats2['keypoints'])
#         matches_list.append(matches ['matches'])
#     return kp_list1, kp_list2, matches_list


# def image_align(img1, img2):
#     """
#     Align image2 to image1 using homography estimation

#     Args:
#         img1, img2 (1x3xHxW): Images to align

#     Returns:
#         img2_aligned (1x3xHxW): Aligned image
#         align_mask (1x1xHxW): Mask for the aligned image, True for valid pixels
#     """
#     H, W = img1.shape[-2:]
#     # Image alignment to account for slight misalignment
#     kp1, kp2, matches = image_matching(img1, img2)
#     assert matches[0].shape[0] > 4, "Need matches>4 to compute homo"
#     kp1 = kp1[0].cpu().numpy()
#     kp2 = kp2[0].cpu().numpy()
#     matches = matches[0].cpu().numpy()
#     src_pts = np.float32([kp1[m[0]] for m in matches])
#     dst_pts = np.float32([kp2[m[1]] for m in matches])
#     M, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
#     img2 = img2[0].permute(1, 2, 0).cpu().numpy()
#     img2_aligned = cv2.warpPerspective(img2, M, (W, H))
#     img2_aligned = torch.from_numpy(img2_aligned).permute(2, 0, 1).to(img1)
#     img2_aligned = img2_aligned.unsqueeze(0)
#     # Mask out the black region in the aligned image
#     align_mask = np.ones((H, W), dtype=np.uint8)
#     align_mask = cv2.warpPerspective(align_mask, M, (W, H))
#     align_mask = torch.from_numpy(align_mask).unsqueeze(0).unsqueeze(0).bool()
#     align_mask = align_mask.to(img1.device)
#     return img2_aligned, align_mask


def batch_crop_resize(
    img, rois, out_H, out_W, aligned=True, interpolation="bilinear"
):
    """
    Crop and resize images

    Args:
        img: [BxCxHxW] Batch of images
        rois: [Bx4] Region of Interest [[x1, y1, x2, y1], ...]
        out_H: [int] Output size height
        out_W: [int] Output size width

    Returns:
        cropped: [BxCxH'xW'] Cropped image batch
    """
    assert len(img.shape) >= 3 and img.shape[-3] == 3, \
        "Error: Image size must be (*, 3, H, W)"
    assert rois.shape[-1] == 4, "Error: Bboxes should be Bx4"
    roi_idx = torch.arange(rois.size(0)).view(-1, 1).to(rois)
    rois = torch.cat((roi_idx, rois), dim=-1)
    # Crop and resize
    output_size = (out_H, out_W)
    if interpolation == "bilinear":
        op = RoIAlign(output_size, 1.0, 0, aligned=aligned)
    elif interpolation == "nearest":
        op = RoIPool(output_size, 1.0)  #
    else:
        raise ValueError(f"Wrong interpolation type: {interpolation}")
    return op(img, rois)


# def get_interest_region(rgbs, masks, iteration=10):
#     """
#     Obtain regions of interest for a masked RGB image (iNeRF)

#     Args:
#         rgbs (Nx3xHxW): Captured RGB images
#         masks (Nx1xHxW): Binary masks
#         iteration (int): Number of dilation iterations

#     Returns:
#         masks_interest (Nx1xHxW): Interest region masks
#     """
#     device = rgbs.device
#     extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)

#     masks_interest = torch.zeros_like(masks).to(device)
#     for ii, (rgb, mask) in enumerate(zip(rgbs, masks)):
#         feats = extractor.extract(rgb)
#         # Ensure keypoints are within image
#         feats["keypoints"].clamp_(min=0)
#         # Filter out keypoints outside the mask
#         feats = filter_features_with_mask(feats, mask.unsqueeze(0).to(device))
#         # Get feature point masks
#         kpts = feats['keypoints'].squeeze(0).long()
#         masks_interest[ii, 0, kpts[..., 1], kpts[..., 0]] = 1
#     # Dilate the feature point masks
#     masks_interest = dilate_masks(masks_interest, kernel_size=iteration)
#     return masks_interest
