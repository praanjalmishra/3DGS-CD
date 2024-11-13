import numpy as np
import torch

from segment_anything import SamPredictor, sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
from tqdm import tqdm


"""Directory to save debug output"""
debug_dir = "/home/ziqil/Desktop/test/"
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


def split_masks(masks, threshold=1e-2):
    """
    Split disconnected masks in a batch of masks

    Args:
        masks (Nx1xHxW): Binary masks

    Returns:
        split_masks (Mx1xHxW): Split masks (M >= N)
    """
    assert len(masks.shape) == 4 and masks.shape[1] == 1
    masks_np = masks.cpu().numpy()
    H = masks.shape[-2]
    W = masks.shape[-1]
    split_masks_np = []
    for mask_np in tqdm(masks_np, desc="Split masks"):
        mask_np = mask_np[0]
        # Find connected components
        num_labels, labels = cv2.connectedComponents(mask_np.astype(np.uint8))
        for i in range(1, num_labels):
            # Split the masks
            component_mask = (labels == i).astype(np.uint8)
            if component_mask.sum() > threshold * H * W:
                split_masks_np.append(component_mask)
    if len(split_masks_np) == 0:
        return torch.zeros(0, 1, H, W).to(masks)
    split_masks = torch.from_numpy(np.array(split_masks_np)).to(masks)
    return split_masks.unsqueeze(1)


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


def auto_label(image_folder, mask_output_folder):
    images = glob(os.path.join(image_folder, '*.jpg')) + \
        glob(os.path.join(image_folder, '*.png')) + \
        glob(os.path.join(image_folder, '*.jpeg')) + \
        glob(os.path.join(image_folder, '*.JPG'))
    
    for image_path in tqdm(images, desc="Processing Images"):
        image_name = os.path.basename(image_path)
        image = cv2.imread(image_path)
        score = 0

        while score <= 0.95:
            bbox = cv2.selectROI("Image", image, fromCenter=False, showCrosshair=True)
            cv2.destroyAllWindows()

            # Assuming bbox is (x, y, width, height), convert to (x1, y1, x2, y2)
            bbox_xyxy = np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])

            # Placeholder for image processing and conversion to tensor, adjust as needed
            rgb_tensor = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0  # Adjust as necessary

            # Predict mask and score
            mask, score_list = sam_predict(rgb_tensor, torch.tensor([bbox_xyxy]))
            # mask = split_masks(mask)
            score = score_list[0]  # Assuming sam_predict returns a list of scores

            if score > 0.95:
                # Convert mask to overlay (simplified)
                mask_overlay = (mask[0].squeeze().cpu().numpy() * 255).astype(np.uint8)
                overlayed_image = cv2.addWeighted(image, 0.5, cv2.cvtColor(mask_overlay, cv2.COLOR_GRAY2BGR), 0.5, 0)

                # Display overlayed image
                cv2.imshow("Mask Overlay", overlayed_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

                # Save mask
                mask_save_path = os.path.join(mask_output_folder, image_name)
                mask_save_path = mask_save_path.replace(".jpg", ".png").replace(".jpeg", ".png")
                cv2.imwrite(mask_save_path, mask_overlay)
            else:
                print("Score below threshold, please redraw the bounding box.")


if __name__ == "__main__":
    import argparse, os, cv2
    from glob import glob
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--img_dir", "-i", type=str, required=True,
        help="Path to the folder containing images"
    )
    parser.add_argument(
        "--out", "-o", type=str, required=True, help="Folder to save masks"
    )
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    assert os.path.isdir(args.img_dir)

    auto_label(args.img_dir, args.out)