import json
import math
import os
import re
import shutil
from typing import Any, Dict, List, Literal, Mapping, Optional

import numpy as np
from PIL import Image, ImageOps
from skimage.morphology import binary_dilation, square


def merge_json(
        json_file_old: str,
        json_file_new: str,
        output_file: str,
        new_view_indices: List[int] = [0, 5, 10, 15],
        split_fraction: float = 0.9,
        replace_old_views: bool = True,
        eval_on_new: bool = False,
        eval_on_all_pix: bool = False):
    """Merge two transforms.json files to generate data for object re-configs.

    Args:
        json_file_old: Path to the json file containing old view info.
        json_file_new: Path to the json file contianing new view info.
        new_view_indices: Indices of new images to use in training.
        output_file: Path to the output json file.
        split_fraction: number of training images / number of all images.
        replace_old_views: True to replace old views with new ones.
        eval_on_new: True to eval on new views, False to eval on old views.
        eval_on_all_pix: True to remove mask for eval images.
    """
    # load json files
    with open(json_file_old) as f1, open(json_file_new) as f2:
        data_old = json.load(f1)
        data_new = json.load(f2)

    assert all('mask_path' in frame for frame in data_old['frames']), \
            "Mask paths are missing for the old views!!"

    # update 'file_path' and 'mask_path' in data_new
    for frame in data_new['frames']:
        frame['file_path'] = frame['file_path'].replace('rgb', 'rgb_new')
        if 'mask_path' in frame:
            frame['mask_path'] = frame['mask_path'].replace('masks', 'masks_new')
        else:
            frame['mask_path'] = frame['file_path'].replace('rgb_new', 'masks_new')

    # Train-Eval split logic for the old views (borrowed from nerfstudio_dataparser.py)
    num_images = len(data_old['frames'])
    num_train_images = math.ceil(num_images * split_fraction)
    i_all = np.arange(num_images)
    i_train = np.linspace(0, num_images - 1, num_train_images, dtype=int)
    i_eval = np.setdiff1d(i_all, i_train)

    # Specify training and evaluation file names
    filenames = [f['file_path'] for f in data_old['frames']]
    data_old['train_filenames'] = [filenames[i] for i in i_train]
    data_old['val_filenames'] = [filenames[i] for i in i_eval]
    data_old['test_filenames'] = data_old['val_filenames']

    if replace_old_views:
        for idx in new_view_indices:
            new_view_base_filename = data_new['frames'][idx]['file_path'].split('/')[-1]
            # create old view filename using the id
            old_view_filename = 'rgb/' + new_view_base_filename
            if old_view_filename in data_old['train_filenames']:
                # replace the matching old view with the new view
                data_old['train_filenames'][data_old['train_filenames'].index(old_view_filename)] =\
                    data_new['frames'][idx]['file_path']
            else:
                # no matching old view found, replace the next available old view instead
                # wrap around if reached the end of the list
                new_view_id = int(re.search('frame_(\d+).png', new_view_base_filename).group(1))
                while old_view_filename not in data_old['train_filenames']:
                    new_view_id = (new_view_id + 1) % len(data_old['train_filenames'])
                    # replace the next available old view with the new view
                    old_view_filename = f"rgb/frame_{new_view_id:05d}.png"
                data_old['train_filenames'][data_old['train_filenames'].index(old_view_filename)] =\
                    data_new['frames'][idx]['file_path']
                print(
                    f"Warning: {new_view_base_filename} in old views is used for evaluation. "
                    f"Replacing {old_view_filename} with new view instead."
                )
    else:
        new_filenames = [
            data_new['frames'][idx]['file_path'] for idx in new_view_indices
        ]
        data_old['train_filenames'] = new_filenames + data_old['train_filenames']

    # Concatenate old and new views
    data_old['frames'] += data_new['frames']

    if eval_on_new:
        data_old['val_filenames'] = [
            fname.replace('rgb', 'rgb_new')
            for fname in data_old['val_filenames']
        ]
        data_old['test_filenames'] = [
            fname.replace('rgb', 'rgb_new')
            for fname in data_old['test_filenames']
        ]

    if eval_on_all_pix:
        for fname in data_old['val_filenames']:
            for frame in data_old['frames']:
                if frame['file_path'] == fname:
                    del frame['mask_path']

    with open(output_file, 'w') as fout:
        json.dump(data_old, fout, indent=4)


def combine_masks(mask1_file, mask2_file, output_file):
    """Combine the black parts of two mask images.

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
    new_mask = np.logical_and(mask1==255, mask2==255).astype(int) * 255

    # Save the combined mask
    new_mask = Image.fromarray(new_mask.astype(np.uint8))
    new_mask.save(output_file)


def invert_mask(mask_file, output_file):
    """Invert colors of a mask image.

    Args:
      mask_file: String, path to the mask image.
      output_file: String, path to the output mask image.
    """

    # Load the mask image
    mask = Image.open(mask_file)

    # Invert the colors of the mask image
    inverted_mask = ImageOps.invert(mask)

    # Save the inverted mask
    inverted_mask.save(output_file)


def dilate_mask(input_file, delta, output_file, grow_type=1, keep_only_new=False):
    """Expand masked or unmasked area in an image.

    Args:
      input_file: String, path to the input mask image.
      delta: Integer, value to expand the masked/unmasked area.
      output_file: String, path to the output image file.
      grow_type: int, 0 for dilating black area, 1 for dilating white area.
      keep_only_new: bool, keep only the newly added area unmasked
    """
    assert grow_type in [0, 1], "grow_type must be 0 or 1"

    # Load the mask image
    mask = Image.open(input_file)
    mask = np.array(mask)

    # Ensure the mask array is in 2D if it's not
    if mask.ndim > 2:
        mask = mask.squeeze()

    # Convert to binary mask (0-1)
    mask = np.where(mask > 0, 1, 0)

    if grow_type == 0:
        # Invert mask for dilating masked area
        mask = 1 - mask

    # Perform dilation
    dilated_mask = binary_dilation(mask, square(delta))

    if keep_only_new and grow_type == 1:
        # only the new area added by dilation is unmasked
        dilated_mask = np.where(mask == 1, 0, dilated_mask)

    if grow_type == 0:
        # Invert dilated mask back after dilation
        if keep_only_new:
            # only the new area added by dilation is unmasked
            dilated_mask = np.where(mask == 1, 0, dilated_mask)
        else:
            dilated_mask = 1 - dilated_mask


    # Scale to 0-255 and reshape to (H, W, 1)
    dilated_mask = (dilated_mask * 255).astype(np.uint8)
    dilated_mask = dilated_mask.reshape(mask.shape[0], mask.shape[1], 1)
    dilated_mask = dilated_mask.squeeze()

    # Save the dilated mask
    new_mask = Image.fromarray(dilated_mask)
    new_mask.save(output_file)


def extract_last_number(s):
    # finds the last group of digits in the string
    match = re.search(r'(\d+)(?!.*\d)', s)  
    if match:
        return int(match.group())
    else:
        raise ValueError(f"No number found in string {s}")


def main(folder_old: str, folder_new: str, folder_output: str,
         dilate_old: int=0, dilate_new: int=0,
         new_view_indices: List[int]=[0, 5, 10, 15], split_frac: float=0.9,
         replace_old_views: bool=True, eval_on_new: bool=False, 
         eval_on_all_pix: bool=False):
    """Merge folders and process masks as per the requirements.

    Args:
        folder_old: Path to the old folder.
        folder_new: Path to the new folder.
        folder_output: Path to the output folder.
        dilate_old: Number of pixels to dilate old view masks for focused sampling.
        dilate_new: Number of pixels to dilate new view masks.
    """
    # Check folder structure and number of images
    if not os.path.exists(os.path.join(folder_old, 'rgb')):
        raise ValueError('rgb folder not found in folder_old')

    if not os.path.exists(os.path.join(folder_new, 'rgb')):
        raise ValueError('rgb folder not found in folder_new')

    if not os.path.exists(os.path.join(folder_old, 'masks')):
        raise ValueError('masks folder not found in folder_old')

    if len(os.listdir(os.path.join(folder_old, 'rgb'))) != len(os.listdir(os.path.join(folder_new, 'rgb'))):
        raise ValueError('Number of images in rgb folders do not match')

    if len(os.listdir(os.path.join(folder_old, 'rgb'))) != len(os.listdir(os.path.join(folder_old, 'masks'))):
        raise ValueError(f'Number of images in {folder_old}/rgb and {folder_old}/masks folders do not match')

    if os.path.exists(os.path.join(folder_new, 'masks')):
        if len(os.listdir(os.path.join(folder_old, 'rgb'))) != len(os.listdir(os.path.join(folder_new, 'masks'))):
            raise ValueError('Number of images in rgb and masks folders do not match')

    # Create new directories in folder_output
    os.makedirs(os.path.join(folder_output, 'rgb'), exist_ok=True)
    os.makedirs(os.path.join(folder_output, 'rgb_new'), exist_ok=True)
    os.makedirs(os.path.join(folder_output, 'masks'), exist_ok=True)
    os.makedirs(os.path.join(folder_output, 'masks_new'), exist_ok=True)

    # Merge the json files
    json_old = os.path.join(folder_old, 'transforms.json')
    json_new = os.path.join(folder_new, 'transforms.json')
    json_output = os.path.join(folder_output, 'transforms.json')
    merge_json(
        json_old, json_new, json_output,
        new_view_indices, split_frac, replace_old_views, eval_on_new, 
        eval_on_all_pix
    )
    # Load the train_filenames from the json_output file
    with open(json_output) as f:
        data = json.load(f)
    train_file_ids = set(
        extract_last_number(name) for name in data['train_filenames']
    )

    # Combine masks from folder_old and folder_new and save to masks in folder_output
    if os.path.exists(os.path.join(folder_new, 'masks')):
        for filename in os.listdir(os.path.join(folder_new, 'masks')):
            mask_new_file = os.path.join(folder_new, 'masks', filename)
            output_file = os.path.join(folder_output, 'masks', filename)
            
            assert os.path.exists(os.path.join(folder_old, 'masks', filename)), \
                f'{filename} not found in {folder_old}/masks'
            mask_old_file = os.path.join(folder_old, 'masks', filename)
            combine_masks(mask_old_file, mask_new_file, output_file)

    # Copy the combined masks, invert them and save to masks_new in folder_output
    for filename in os.listdir(os.path.join(folder_output, 'masks')):
        source_file = os.path.join(folder_output, 'masks', filename)
        target_file = os.path.join(folder_output, 'masks_new', filename)
        shutil.copy2(source_file, target_file)

        # Invert the masks for masks_new
        invert_mask(target_file, target_file)

        # If dilate_new is set, dilate the inverted masks
        if dilate_new and extract_last_number(filename) in train_file_ids:
            dilate_mask(target_file, dilate_new, target_file, 1, False)

    # Copy rgb images from folder_old and folder_new to rgb and rgb_new respectively in folder_output
    for filename in os.listdir(os.path.join(folder_old, 'rgb')):
        source_file = os.path.join(folder_old, 'rgb', filename)
        target_file = os.path.join(folder_output, 'rgb', filename)
        shutil.copy2(source_file, target_file)

    for filename in os.listdir(os.path.join(folder_new, 'rgb')):
        source_file = os.path.join(folder_new, 'rgb', filename)
        target_file = os.path.join(folder_output, 'rgb_new', filename)
        shutil.copy2(source_file, target_file)

    # Dilate masks in masks in folder_output
    if dilate_old:
        for filename in os.listdir(os.path.join(folder_output, 'masks')):
            if extract_last_number(filename) in train_file_ids:
                input_file = os.path.join(folder_output, 'masks', filename)
                output_file = os.path.join(folder_output, 'masks', filename)
                dilate_mask(input_file, dilate_old, output_file, 0, True)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_old', "-fo", type=str, required=True)
    parser.add_argument('--folder_new', "-fn", type=str, required=True)
    parser.add_argument('--folder_output', "-o", type=str, required=True)
    parser.add_argument('--dilate_old', "-do", type=int, default=0)
    parser.add_argument('--dilate_new', "-dn", type=int, default=0)
    parser.add_argument(
        '--new_view_indices', "-n", nargs='+', type=int, default=[80, 85, 90, 95]
    )
    parser.add_argument('--split_frac', "-s", type=float, default=0.9)
    parser.add_argument('--dont_replace_old_views', "-dr", action='store_true')
    parser.add_argument('--eval_on_new', "-en", action='store_true')
    parser.add_argument('--eval_on_all_pix', "-ea", action='store_true')

    args = parser.parse_args()

    main(
        args.folder_old, args.folder_new, args.folder_output,
        args.dilate_old, args.dilate_new, args.new_view_indices,
        args.split_frac, not args.dont_replace_old_views,
        args.eval_on_new, args.eval_on_all_pix
    )