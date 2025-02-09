# Evaluate the NeRF rendered images against the captured images
import numpy as np
import torch

from nerfstudio.utils.img_utils import image_align, crop_imgs_w_masks
from nerfstudio.utils.io import read_imgs, read_masks
from nerfstudio.utils.debug_utils import debug_images
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


def psnr_masked(psnr_fun, image, rgb, mask):
    assert mask.dtype == torch.bool
    masked_psnr = psnr_fun(
        image[mask.expand_as(image)], rgb[mask.expand_as(rgb)]
    )
    return masked_psnr


def main(renders, captures, masks=None):
    """
    Align the renders and captures and compute their PSNR, SSIM, and LPIPS

    Args:
        renders (str-list): List of paths to the rendered images
        captures (str-list): List of paths to the captured images
        masks (str-list): List of paths to the masks
    """
    # Initialize the metrics
    psnr = PeakSignalNoiseRatio(data_range=1.0)
    ssim = structural_similarity_index_measure
    lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)
    # Read images
    assert len(renders) == len(captures), \
        "Number of renders and captures must match"
    renders = read_imgs(renders)
    captures = read_imgs(captures)
    # renders_align = renders
    renders_align = []
    for render, capture in zip(renders, captures):
        render_align, _ = image_align(capture.unsqueeze(0), render.unsqueeze(0))
        renders_align.append(render_align)
    renders_align = torch.cat(renders_align)
    # debug_images(renders_align, "/home/ziqi/Desktop/test/")
    psnr_v = psnr(captures, renders_align).item()
    ssim_v = ssim(captures, renders_align)
    lpips_v = lpips(captures, renders_align)
    if masks is not None:
        assert len(masks) == len(captures), \
            "Number of masks and captures must match"
        masks = read_masks(masks)
        captures_cropped = crop_imgs_w_masks(captures, masks)
        # debug_images(captures_cropped, "/home/ziqi/Desktop/test/")
        # import pdb; pdb.set_trace()
        renders_cropped = crop_imgs_w_masks(renders_align, masks)
        # debug_images(renders_cropped, "/home/ziqi/Desktop/test/")
        psnr_crop = psnr(captures_cropped, renders_cropped).item()
        ssim_crop = ssim(captures_cropped, renders_cropped)
        lpips_crop = lpips(captures_cropped, renders_cropped)
        psnr_mask = psnr_masked(psnr, captures, renders_align, masks)
        print(f"PSNR (masked): {psnr_mask:.4f}")
        print(f"PSNR (cropped): {psnr_crop:.4f}")
        print(f"SSIM (cropped): {ssim_crop:.4f}")
        print(f"LPIPS (cropped): {lpips_crop:.4f}")
    return float(psnr_v), float(ssim_v), float(lpips_v)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", "-r", type=str, nargs="+", required=True)
    parser.add_argument("--capture", "-c", type=str, nargs="+", required=True)
    parser.add_argument("--mask", "-m", type=str, nargs="+", default=None)
    args = parser.parse_args()

    psnr, ssim, lpips = main(args.render, args.capture, args.mask)
    print(f"PSNR: {psnr:.4f}")
    print(f"SSIM: {ssim:.4f}")
    print(f"LPIPS: {lpips:.4f}")