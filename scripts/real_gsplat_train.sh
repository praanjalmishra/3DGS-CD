# Script to automate SplatFacto training on real data
#!/bin/bash
# Pre-requisites:
# 1. Install NeRFStudio
# 2. Install SAM & hloc into nerfstudio conda env
# 3. Download the following scripts
# 4. Move the scripts to corresponding locations or other user-specified dirs
# 5. Take images and organize them in the following folder structure:
#   - rgb
#     - frame_00000.png
#     - frame_00001.png
#     - ...
#   - rgb_new
#     - frame_00000.png
#     - frame_00001.png
#     - ...

# Set the default values for the arguments
DATA_FOLDER=${1:-"/home/ziqi/Desktop/test/WaterCube0"}
CUDA_VISIBLE_DEVICES=${2:-0}
TRAIN_IDX=${3:-"0 2 4 6"} # Indices of sparse images used for training
OUTPUT_FOLDER=${4:-"/home/ziqi/Packages/nerfstudio/outputs"}
KUBRIC_FOLDER=${5:-"/home/ziqi/Packages/kubric"}
NERFSTUDIO_FOLDER=${6:-"/home/ziqi/Packages/nerfstudio"}
MERGE_SCRIPT=${7:-"/home/ziqi/Packages/nerfstudio/scripts/merge_colmap_data.py"}
EDIT_SCRIPT=${8:-"/home/ziqi/Packages/nerfstudio/scripts/edit_nerf_data.py"}
UNDISTORT_SCRIPT=${9:-"/home/ziqi/Packages/nerfstudio/scripts/undistort_transforms.py"}


# ------------------ Setup Environment ------------------
# May need to build a conda env with dependencies required by the .py scripts
source ~/miniconda3/etc/profile.d/conda.sh
conda activate nerfgs
cd $NERFSTUDIO_FOLDER
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
current_time=$(date +%Y-%m-%d_%H%M%S)

# ------------------ Structure from Motion ------------------
ns-process-data images --data ${DATA_FOLDER}/rgb --output-dir ${DATA_FOLDER} \
  --sfm-tool hloc --feature-type sift --matcher-type adalam --num_downscales 0

# check if the previous command was successful
if [ $? -eq 0 ]
then
  echo "SFM completed successfully."
else
  echo "SFM failed." >&2
  exit 1
fi

# Rename transforms.json files
mv ${DATA_FOLDER}/transforms.json ${DATA_FOLDER}/transforms_colmap.json


# ------------------ Camera Localization ------------------
# TODO: read camera params from colmap
python $NERFSTUDIO_FOLDER/nerfstudio/process_data/reloc_utils.py \
  --image_dir ${DATA_FOLDER}/rgb_new \
  --colmap_path ${DATA_FOLDER}/colmap \
  --transforms_json transforms_colmap.json

# check if the previous command was successful
if [ $? -eq 0 ]
then
  echo "Camera localization completed successfully."
else
  echo "Camera localization failed." >&2
  exit 1
fi

# Rename transforms.json files
mv ${DATA_FOLDER}/rgb_new/transforms.json ${DATA_FOLDER}/transforms_reloc.json


# ------------------ Process training data ------------------
# Convert the train_idx string to an array
read -a train_idx_arr <<< $TRAIN_IDX
# Convert the SfM data to nerf data
python $MERGE_SCRIPT -jo ${DATA_FOLDER}/transforms_colmap.json \
  -jn ${DATA_FOLDER}/transforms_reloc.json \
  -oo ${DATA_FOLDER}/transforms_pretrain.json \
  -on ${DATA_FOLDER}/transforms.json \
  -n "${train_idx_arr[@]}"

# check if the previous command was successful
if [ $? -eq 0 ]
then
  echo "Colmap data conversion successful."
else
  echo "Colmap data conversion failed." >&2
  exit 1
fi

# Undistort images
python $UNDISTORT_SCRIPT -s ${DATA_FOLDER}

# check if the previous command was successful
if [ $? -eq 0 ]
then
  echo "Undistort images successful"
else
  echo "Undistort images failed." >&2
  exit 1
fi


# ------------------ Process data for training our method ------------------
# Add mask paths to the transforms.json file
python $EDIT_SCRIPT -i ${DATA_FOLDER}/transforms.json \
  -o ${DATA_FOLDER}/transforms.json -am

# check if the previous command was successful
if [ $? -eq 0 ]
then
  echo "Add mask paths completed successfully."
else
  echo "Add mask paths failed." >&2
  exit 1
fi
