# Undistort NeRFStudio dataset
# Note: OpenCV undistort in nerfstudio has a bug so this may cause worse result
import argparse
import os

from pathlib import Path
from nerfstudio.utils.img_utils import undistort_images
from nerfstudio.utils.io import (
    read_transforms, load_from_json, save_imgs, write_to_json
)


parser = argparse.ArgumentParser("Undistort NeRFStudio dataset")
parser.add_argument(
    "--data_path", "-s", required=True, type=str,
    help="Path to data folder"
)
args = parser.parse_args()

assert os.path.exists(args.data_path + "/transforms.json")

train_imgs, train_files, _, _, _, train_cams = read_transforms(
    args.data_path + "/transforms.json", read_images=True, mode="train"
)
val_imgs, val_files, _, _, _, val_cams = read_transforms(
    args.data_path + "/transforms.json", read_images=True, mode="val"
)
other_imgs, other_files, _, _, _, other_cams = read_transforms(
    args.data_path + "/transforms.json", read_images=True, mode="other"
)

# Undistort images
train_cams, train_imgs = undistort_images(train_cams, train_imgs)
val_cams, val_imgs = undistort_images(val_cams, val_imgs)
other_cams, other_imgs = undistort_images(other_cams, other_imgs)

# Save images
save_imgs(train_imgs, [f"{args.data_path}/{fn}" for fn in train_files])
save_imgs(val_imgs, [f"{args.data_path}/{fn}" for fn in val_files])
save_imgs(other_imgs, [f"{args.data_path}/{fn}" for fn in other_files])

# Update intrinsics in transforms.json and transforms_pretrain.json
data = load_from_json(Path(args.data_path + "/transforms.json"))
H, W = val_imgs.shape[-2:]
intrin = train_cams.get_intrinsics_matrices().float().numpy()[0]
fx, fy, cx, cy = intrin[0, 0], intrin[1, 1], intrin[0, 2], intrin[1, 2]
data["w"], data["h"] = int(W), int(H)
data["flx"], data["fly"] = float(fx), float(fy)
data["cx"], data["cy"] = float(cx), float(cy)
data["k1"], data["k2"], data["p1"], data["p2"] = 0.0, 0.0, 0.0, 0.0
write_to_json(Path(args.data_path + "/transforms.json"), data)

data_pretrain = load_from_json(
    Path(args.data_path + "/transforms_pretrain.json")
)
data_pretrain["w"], data_pretrain["h"] = int(W), int(H)
data_pretrain["flx"], data_pretrain["fly"] = float(fx), float(fy)
data_pretrain["cx"], data_pretrain["cy"] = float(cx), float(cy)
data_pretrain["k1"], data_pretrain["k2"] = 0.0, 0.0
data_pretrain["p1"], data_pretrain["p2"] = 0.0, 0.0
write_to_json(
    Path(args.data_path + "/transforms_pretrain.json"), data_pretrain
)





