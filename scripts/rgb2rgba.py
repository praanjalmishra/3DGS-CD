# Convert a folder of RGB(A) images and a folder of masks to RGBA images
from glob import glob
from PIL import Image
import os


def process_images(rgb_folder, mask_folder, output_folder):
    # Ensure output folder exists
    assert os.path.isdir(output_folder), "Output folder does not exist"
    rgb_img_files = sorted(glob(os.path.join(rgb_folder, "*.png")))
    mask_img_files = sorted(glob(os.path.join(mask_folder, "*.png")))

    # Iterate through all images in the RGB folder
    for rgb_img_path, mask_img_path in zip(rgb_img_files, mask_img_files):
        # Load the RGB(A) image and its corresponding mask
        rgb_image = Image.open(rgb_img_path).convert("RGBA")
        mask_image = Image.open(mask_img_path).convert("L")

        # Prepare the new alpha channel
        # Note: mask image is assumed to be in grayscale ('L' mode)
        # Convert mask to 'L' mode if it's not, ensuring compatibility
        alpha_channel = mask_image.point(lambda p: 255 if p > 0 else 0)

        # Combine original RGB channels with the new alpha channel
        rgb_channels = rgb_image.split()[:3]  # Ignore existing alpha if present
        new_image = Image.merge("RGBA", rgb_channels + (alpha_channel,))

        # Save the modified image
        rgb_img_basename = os.path.basename(rgb_img_path)
        output_path = os.path.join(output_folder, rgb_img_basename)
        new_image.save(output_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rgb", "-i", type=str,
        help="Path to the folder containing RGB images"
    )
    parser.add_argument(
        "--mask", "-m", type=str,
        help="Path to the folder containing mask images"
    )
    parser.add_argument(
        "--out", "-o", type=str, 
        help="Path to the output folder"
    )
    args = parser.parse_args()
    process_images(args.rgb, args.mask, args.out)