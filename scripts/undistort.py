# Undistort images with COLMAP since OpenCV undistortion in nerfstudio has a bug
# Input: NeRFStudio-style data folder
#  data
#  - images: folder containing images
#  - colmap: folder containing colmap results
# Output: COLMAP-style data folder with undistorted images
#  data
#  - images: folder containing undistorted images
#  - sparse: folder containing colmap results

import argparse
import os
import shutil

parser = argparse.ArgumentParser("Colmap converter")
parser.add_argument("--colmap_executable", default="", type=str)
parser.add_argument(
    "--data_path", "-s", required=True, type=str, 
    help="Path to data folder"
)
parser.add_argument(
    "--img_path", "-i", default="images", type=str,
    help="Relative path to folder containing images"
)
parser.add_argument(
    "--colmap_res", "-colmap", default="colmap", type=str,
    help="Relative path to folder containing colmap results"
)
args = parser.parse_args()

# Colmap executable
colmap_command = '"{}"'.format(args.colmap_executable) \
    if len(args.colmap_executable) > 0 else "colmap"

# Reorganize the colmap results folder if needed
assert os.path.exists(
    args.data_path + f"/{args.colmap_res}/sparse/0/cameras.bin"
)
assert os.path.exists(
    args.data_path + f"/{args.colmap_res}/sparse/0/images.bin"
)
assert os.path.exists(
    args.data_path + f"/{args.colmap_res}/sparse/0/points3D.bin"
)
database_moved = False
if not os.path.exists(args.data_path + f"/{args.colmap_res}/database.db"):
    assert(
        args.data_path + f"/{args.colmap_res}/sparse/0/database.db"
    )
    shutil.move(
        args.data_path + f"/{args.colmap_res}/sparse/0/database.db",
        args.data_path + f"/{args.colmap_res}/database.db"
    )
    database_moved = True


## Undistort our images into ideal pinhole intrinsics.
img_undist_cmd = (colmap_command + " image_undistorter \
    --image_path " + args.data_path + f"/{args.img_path} \
    --input_path " + args.data_path + f"/{args.colmap_res}/sparse/0 \
    --output_path " + args.data_path + "\
    --output_type COLMAP")
exit_code = os.system(img_undist_cmd)
if exit_code != 0:
    print(f"Mapper failed with code {exit_code}. Exiting.")
    exit(exit_code)

files = os.listdir(args.data_path + "/sparse")
os.makedirs(args.data_path + "/sparse/0", exist_ok=True)
# Copy each file from the source directory to the destination directory
for file in files:
    if file == '0':
        continue
    source_file = os.path.join(args.data_path, "sparse", file)
    destination_file = os.path.join(args.data_path, "sparse", "0", file)
    shutil.move(source_file, destination_file)


# Move database.db back
if os.path.exists(args.data_path + f"/{args.colmap_res}/database.db") \
    and database_moved:
    shutil.move(
        args.data_path + f"/{args.colmap_res}/database.db",
        args.data_path + f"/{args.colmap_res}/sparse/0/database.db"
    )

# Remove unnecessary files
if os.path.isfile(args.data_path + "/run-colmap-geometric.sh"):
    os.remove(args.data_path + "/run-colmap-geometric.sh")
if os.path.isfile(args.data_path + "/run-colmap-photometric.sh"):
    os.remove(args.data_path + "/run-colmap-photometric.sh")
if os.path.isdir(args.data_path + "/stereo"):
    shutil.rmtree(args.data_path + "/stereo")