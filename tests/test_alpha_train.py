# Test alpha-transparent training for full object reconstruction
# MUST use kubric generated data

import argparse
import os
import shutil
import numpy as np

from glob import glob
from PIL import Image, ImageOps
from pathlib import Path
from nerfstudio.utils.io import load_from_json, write_to_json
from nerfstudio.utils.img_utils import invert_mask, rgb2rgba
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument(
    "--data", "-d", type=str, default="hdri29_cuben", help="data path name"
)
args = parser.parse_args()

data_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_recfg = Path(f"{data_base}/data/nerfstudio/{args.data}_reconfig")
data_move = Path(f"{data_base}/data/nerfstudio/{args.data}_move")
data = Path(f"{data_base}/data/nerfstudio/{args.data}")

# NOTE: This assumes the object canonical coordinate is at the origin!!
# Change this!
pose_change = np.array(
    [[1,  0, 0, 2.5 / 11.0], 
    [0, -1,  0, 2.5 / 11.0], 
    [0,  0, -1, 2.0 / 11.0], 
    [0,  0,  0, 1]]
)

# Move masks to reconfig data folder
if not os.path.exists(data_recfg / "obj_masks"):
    shutil.copytree(data / "masks", data_recfg / "obj_masks")
    mask_files = sorted(glob(f"{data_recfg}/obj_masks/*.png"))
    for i, mask_file in enumerate(mask_files):
        mask = invert_mask(mask_file, mask_file)

if not os.path.exists(data_recfg / "obj_masks_new"):
    shutil.copytree(data_move / "masks", data_recfg / "obj_masks_new")
    mask_files = sorted(glob(f"{data_recfg}/obj_masks_new/*.png"))
    for i, mask_file in enumerate(mask_files):
        mask = invert_mask(mask_file, mask_file)

if not os.path.exists(data_recfg / "rgba_new"):
    os.makedirs(data_recfg / "rgba_new")
    rgb_img_files = sorted(glob(os.path.join(f"{data_recfg}/rgb_new", "*.png")))
    mask_img_files = sorted(glob(os.path.join(f"{data_recfg}/obj_masks_new", "*.png")))
    rgb2rgba(rgb_img_files, mask_img_files, f"{data_recfg}/rgba_new")


if not os.path.exists(data_recfg / "rgba"):
    os.makedirs(data_recfg / "rgba")
    rgb_img_files = sorted(glob(os.path.join(f"{data_recfg}/rgb", "*.png")))
    mask_img_files = sorted(glob(os.path.join(f"{data_recfg}/obj_masks", "*.png")))
    rgb2rgba(rgb_img_files, mask_img_files, f"{data_recfg}/rgba")

if os.path.exists(data_recfg / "obj_masks"):
    shutil.rmtree(data_recfg / "obj_masks")
if os.path.exists(data_recfg / "obj_masks_new"):
    shutil.rmtree(data_recfg / "obj_masks_new")

tjson = load_from_json(Path(f"{data_recfg}/transforms.json"))
frames = tjson["frames"]
for frame in frames:
    # remove masks
    frame.pop("mask_path", None)
    # Update camera poses
    if "rgb_new" in frame["file_path"]:
        pose = np.array(frame["transform_matrix"])
        pose[0:3, 1:3] = -pose[0:3, 1:3] # OpenGL to OpenCV
        pose = np.linalg.inv(pose_change) @ pose
        pose[0:3, 1:3] = -pose[0:3, 1:3] # OpenCV to OpenGL
        frame["transform_matrix"] = pose.tolist()
    frame["file_path"] = frame["file_path"].replace("rgb", "rgba")

for i in range(len(tjson["train_filenames"])):
    tjson["train_filenames"][i] = tjson["train_filenames"][i].replace("rgb", "rgba")
for i in range(len(tjson["test_filenames"])):
    tjson["test_filenames"][i] = tjson["test_filenames"][i].replace("rgb", "rgba")
for i in range(len(tjson["val_filenames"])):
    tjson["val_filenames"][i] = tjson["val_filenames"][i].replace("rgb", "rgba")
    
write_to_json(Path(f"{data_recfg}/transforms_obj.json"), tjson)

print("Run the following command to reconstruct the full object model:")
print(
    f"ns-train splatfacto --pipeline.model.background-color random" + 
    f" --data {data_recfg}/transforms_obj.json"
)

