import argparse
import os
import sys
from PIL import Image, ImageChops

def parse_arguments():
    parser = argparse.ArgumentParser(description='Overlay masks onto RGB images.')
    parser.add_argument('--rgbs', '-r', required=True, type=str, help='Path to the folder containing RGB images.')
    parser.add_argument('--move_out', '-o', nargs='*', default=[], help='Paths to move-out mask folders.')
    parser.add_argument('--move_in', '-i', nargs='*', default=[], help='Paths to move-in mask folders.')
    parser.add_argument('--save', '-s', required=True, type=str, help='Path to the folder to save output images.')
    return parser.parse_args()

def get_sorted_file_list(folder):
    files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    files.sort()
    return files

def check_folder_files(file_lists):
    num_files = [len(files) for files in file_lists]
    if len(set(num_files)) != 1:
        print('Error: Folders do not contain the same number of files.')
        sys.exit(1)
    return num_files[0]

def main():
    args = parse_arguments()

    rgb_files = get_sorted_file_list(args.rgbs)
    file_lists = [rgb_files]

    # Process move-out mask folders if provided
    if args.move_out:
        move_out_mask_file_lists = [get_sorted_file_list(folder) for folder in args.move_out]
        file_lists.extend(move_out_mask_file_lists)
    else:
        move_out_mask_file_lists = []

    # Process move-in mask folders if provided
    if args.move_in:
        move_in_mask_file_lists = [get_sorted_file_list(folder) for folder in args.move_in]
        file_lists.extend(move_in_mask_file_lists)
    else:
        move_in_mask_file_lists = []

    # Check that all folders have the same number of files
    num_images = check_folder_files(file_lists)

    if not os.path.exists(args.save):
        os.makedirs(args.save)

    for i in range(num_images):
        # Load RGB image
        rgb_filename = rgb_files[i]
        rgb_path = os.path.join(args.rgbs, rgb_filename)
        rgb_image = Image.open(rgb_path).convert('RGBA')

        # Combine move-out masks if any
        move_out_mask = None
        if args.move_out:
            for idx, folder in enumerate(args.move_out):
                mask_filename = move_out_mask_file_lists[idx][i]
                mask_path = os.path.join(folder, mask_filename)
                mask = Image.open(mask_path).convert('L')
                if move_out_mask is None:
                    move_out_mask = mask
                else:
                    move_out_mask = ImageChops.lighter(move_out_mask, mask)

        # Combine move-in masks if any
        move_in_mask = None
        if args.move_in:
            for idx, folder in enumerate(args.move_in):
                mask_filename = move_in_mask_file_lists[idx][i]
                mask_path = os.path.join(folder, mask_filename)
                mask = Image.open(mask_path).convert('L')
                if move_in_mask is None:
                    move_in_mask = mask
                else:
                    move_in_mask = ImageChops.lighter(move_in_mask, mask)

        # Create overlays
        red_overlay = Image.new('RGBA', rgb_image.size, (255, 0, 0, 0))
        green_overlay = Image.new('RGBA', rgb_image.size, (0, 255, 0, 0))

        for y in range(rgb_image.size[1]):
            for x in range(rgb_image.size[0]):
                in_value = move_in_mask.getpixel((x, y)) if move_in_mask else 0
                out_value = move_out_mask.getpixel((x, y)) if move_out_mask else 0

                if in_value > 0:
                    # If move-in mask is present, color green
                    green_overlay.putpixel((x, y), (0, 255, 0, 128))
                elif out_value > 0:
                    # If only move-out mask is present, color red
                    red_overlay.putpixel((x, y), (255, 0, 0, 128))

        # Composite overlays onto the RGB image
        rgb_image = Image.alpha_composite(rgb_image, red_overlay)
        rgb_image = Image.alpha_composite(rgb_image, green_overlay)

        # Save the result
        output_path = os.path.join(args.save, rgb_filename)
        rgb_image.convert('RGB').save(output_path)

if __name__ == '__main__':
    main()
