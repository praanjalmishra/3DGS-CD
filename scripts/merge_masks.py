from merge_nerf_data import combine_masks
import argparse, os
from glob import glob
from PIL import Image
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--masks1", "-m1", type=str, required=True, help="Path to first mask folder")
parser.add_argument("--masks2", "-m2", type=str, required=True, help="Path to second mask folder")
parser.add_argument("--out", "-o", type=str, required=True, help="Folder to save merged masks")
args = parser.parse_args()


def combine_masks(mask1_file, mask2_file, output_file):
    """Combine the white parts of two mask images.

    Args:
      mask1_file: String, path to the first mask image.
      mask2_file: String, path to the second mask image.
      output_file: String, path to the output mask image.
    """
    # Load the mask images
    mask1 = Image.open(mask1_file)
    mask1 = np.array(mask1)
    mask2 = Image.open(mask2_file)
    mask2 = np.array(mask2)

    # Ensure the mask arrays are in 2D if they're not
    if mask1.ndim > 2:
        mask1 = mask1.squeeze()
    if mask2.ndim > 2:
        mask2 = mask2.squeeze()

    # Combine the masks
    new_mask = np.logical_or(mask1==255, mask2==255).astype(int) * 255

    # Save the combined mask
    new_mask = Image.fromarray(new_mask.astype(np.uint8))
    new_mask.save(output_file)


mask_files1 = sorted(glob(f"{args.masks1}/*.png") + glob(f"{args.masks1}/*.jpg"))
mask_files2 = sorted(glob(f"{args.masks2}/*.png") + glob(f"{args.masks2}/*.jpg"))
assert len(mask_files1) == len(mask_files2)
for mask_file1, mask_file2 in zip(mask_files1, mask_files2):
    basename = os.path.basename(mask_file1)
    out_file = os.path.join(args.out, basename)
    combine_masks(mask_file1, mask_file2, out_file)

