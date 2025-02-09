# Description: Run MTP on two folders of images to predict change masks
# NOTE: Must do before running this script
# 1. install and activate mtp conda env
# 2. Follow this issue to place mtp change detection files in open-cd
# 3. Place this file under open-cd and cd to open-cd
import argparse, os
from opencd.apis import OpenCDInferencer
from PIL import Image
import numpy as np
from glob import glob

parser = argparse.ArgumentParser()
parser.add_argument(
    "--imgs1", "-i1", type=str, required=True, help="Path to image folder 2"
)
parser.add_argument(
    "--imgs2", "-i2", type=str, required=True, help="Path to image folder 1"
)
parser.add_argument(
    "--out", "-o", type=str, help="Folder to save change masks",
    default=os.path.expanduser("~/Desktop/temp")
)
parser.add_argument(
    "--model", "-m", type=str, help="Path to pretrained MTP model",
    default="/home/ziqi/Packages/open-cd/checkpoints/cdd-rvsa-b-mae-mtp-epoch_200.pth"
)
parser.add_argument(
    "--config", "-c", type=str, help="Path to MTP config",
    default="/home/ziqi/Packages/open-cd/configs/mtp/cdd/rvsa-b-unet-256-mae-mtp_cdd.py"
)
args = parser.parse_args()

assert os.path.exists(args.out), f"Output folder {args.out} does not exist"


imgs1 = sorted(glob(f"{args.imgs1}/*.png") + glob(f"{args.imgs1}/*.jpg"))
imgs2 = sorted(glob(f"{args.imgs2}/*.png") + glob(f"{args.imgs2}/*.jpg"))
assert len(imgs1) == len(imgs2)


# Load and resize images to 256x256
imgs1 = [Image.open(img) for img in imgs1]
imgs2 = [Image.open(img) for img in imgs2]
img_size = imgs1[0].size
imgs1 = [np.array(img.resize((256, 256))) for img in imgs1]
imgs2 = [np.array(img.resize((256, 256))) for img in imgs2]
img_pairs = list(zip(imgs1, imgs2))
inferencer = OpenCDInferencer(
    weights=args.model, model=args.config,
    classes=('unchanged', 'changed'), palette=[[0, 0, 0], [255, 255, 255]]
)

res = inferencer(img_pairs, show=True, wait_time=1)

change_masks = res["predictions"]
# reshape back to original size
change_masks = [
    Image.fromarray((mask * 255).astype(np.uint8)).resize(
        img_size, resample=Image.NEAREST
    ) for mask in change_masks
]
# Save the change masks
for i, mask in enumerate(change_masks):
    mask.save(os.path.join(args.out, f"mask_{i}.png"))

