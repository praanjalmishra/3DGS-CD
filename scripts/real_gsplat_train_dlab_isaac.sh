# Script to automate 3DGS-CD on real data
#!/bin/bash
# Pre-requisites:
# 1. Install NeRFStudio
# 2. Install EfficientSAM & hloc into your gscd conda env
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
DATA_FOLDER=${1:-"/local/home/pmishra/cvg/3dgscd/data/render_data"}
CUDA_VISIBLE_DEVICES=${2:-0}
TRAIN_IDX=${3:-"0 2 4 6"} # Indices of sparse images used for training
OUTPUT_FOLDER=${4:-"/local/home/pmishra/cvg/3dgscd/outputs"}
NERFSTUDIO_FOLDER=${5:-"/local/home/pmishra/cvg/3dgscd"}
MERGE_SCRIPT=${6:-"/local/home/pmishra/cvg/3dgscd/scripts/merge_colmap_data.py"}
EDIT_SCRIPT=${7:-"/local/home/pmishra/cvg/3dgscd/scripts/edit_nerf_data.py"}
UNDISTORT_SCRIPT=${8:-"/local/home/pmishra/cvg/3dgscd/scripts/undistort_transforms.py"}
CHANGE_DET_SCRIPT=${9:-"/local/home/pmishra/cvg/3dgscd/nerfstudio/scripts/change_det.py"}


# # ------------------ Setup Environment ------------------
# # May need to build a conda env with dependencies required by the .py scripts
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate gscd
# cd $NERFSTUDIO_FOLDER
# export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
# current_time=$(date +%Y-%m-%d_%H%M%S)

# # ------------------ Structure from Motion ------------------
# ns-process-data images --data ${DATA_FOLDER}/rgb --output-dir ${DATA_FOLDER} \
#   --sfm-tool hloc --feature-type sift --matcher-type adalam --num_downscales 0

# # check if the previous command was successful
# if [ $? -eq 0 ]
# then
#   echo "SFM completed successfully."
# else
#   echo "SFM failed." >&2
#   exit 1
# fi

# # Rename transforms.json files
# mv ${DATA_FOLDER}/transforms.json ${DATA_FOLDER}/transforms_colmap.json


# # ------------------ Camera Localization ------------------
# # TODO: read camera params from colmap
# python $NERFSTUDIO_FOLDER/nerfstudio/process_data/reloc_utils.py \
#   --image_dir ${DATA_FOLDER}/rgb_new \
#   --colmap_path ${DATA_FOLDER}/colmap \
#   --transforms_json transforms_colmap.json

# # check if the previous command was successful
# if [ $? -eq 0 ]
# then
#   echo "Camera localization completed successfully."
# else
#   echo "Camera localization failed." >&2
#   exit 1
# fi

# # Rename transforms.json files
# mv ${DATA_FOLDER}/rgb_new/transforms.json ${DATA_FOLDER}/transforms_reloc.json


# # ------------------ Process training data ------------------
# # Convert the train_idx string to an array
# read -a train_idx_arr <<< $TRAIN_IDX
# # Convert the SfM data to nerf data
# python $MERGE_SCRIPT -jo ${DATA_FOLDER}/transforms_colmap.json \
#   -jn ${DATA_FOLDER}/transforms_reloc.json \
#   -oo ${DATA_FOLDER}/transforms_pretrain.json \
#   -on ${DATA_FOLDER}/transforms.json \
#   -n "${train_idx_arr[@]}"

# # check if the previous command was successful
# if [ $? -eq 0 ]
# then
#   echo "Colmap data conversion successful."
# else
#   echo "Colmap data conversion failed." >&2
#   exit 1
# fi

# # Duplicate the rgb and rgb_new to rgb_distorted and rgb_new_distorted
# cp -r ${DATA_FOLDER}/rgb ${DATA_FOLDER}/rgb_distorted
# cp -r ${DATA_FOLDER}/rgb_new ${DATA_FOLDER}/rgb_new_distorted

# # Undistort images
# python $UNDISTORT_SCRIPT -s ${DATA_FOLDER}

# # check if the previous command was successful
# if [ $? -eq 0 ]
# then
#   echo "Undistort images successful"
# else
#   echo "Undistort images failed." >&2
#   exit 1
# fi


# # ------------------ Process data for training our method ------------------
# # Add mask paths to the transforms.json file
# python $EDIT_SCRIPT -i ${DATA_FOLDER}/transforms.json \
#   -o ${DATA_FOLDER}/transforms.json -am

# # check if the previous command was successful
# if [ $? -eq 0 ]
# then
#   echo "Add mask paths completed successfully."
# else
#   echo "Add mask paths failed." >&2
#   exit 1
# fi

# # Make the finetuning data by removing pre-train views in transforms.json file
# python $EDIT_SCRIPT -i ${DATA_FOLDER}/transforms.json \
#   -o ${DATA_FOLDER}/transforms_finetune.json --remove_pretrain

# # check if the previous command was successful
# if [ $? -eq 0 ]
# then
#   echo "Finetune data creation completed successfully."
# else
#   echo "Finetune data creation failed." >&2
#   exit 1
# fi


# # ------------------ Pre-Training ------------------
# while true; do
#   # Pre-trained NeRF: Train NeRFacto for the object-centric scene
#   ns-train splatfacto --vis viewer+tensorboard \
#     --experiment-name $(basename $DATA_FOLDER) \
#     --output-dir ${OUTPUT_FOLDER} \
#     --timestamp $current_time \
#     --steps_per_eval_all_images 1000 \
#     --pipeline.model.cull_alpha_thresh 0.005 \
#     --pipeline.model.continue_cull_post_densification=False \
#     --max-num-iterations 30000 \
#     --machine.num-devices 1 \
#     --viewer.quit-on-train-completion True \
#     nerfstudio-data --data ${DATA_FOLDER}/transforms_pretrain.json \
#     --auto-scale-poses=False --center-method none --orientation-method none \
#     --load-3D-points True \
#     --train_split_fraction 0.9

#   # check if the previous command was successful
#   if [ $? -ne 0 ]; then
#     echo "Training $(basename ${DATA_FOLDER}) failed." >&2
#     continue
#   fi

#   # If we reach here, pre-training was successful
#   break

# done

# # Detect 3D change in the scene

# python $CHANGE_DET_SCRIPT \
#   -c ${OUTPUT_FOLDER}/$(basename ${DATA_FOLDER})/splatfacto/${current_time}/config.yml \
#   -t ${DATA_FOLDER}/transforms.json \
#   -o ${DATA_FOLDER} \
#   -ckpt ${OUTPUT_FOLDER}/$(basename ${DATA_FOLDER})/splatfacto/${current_time}/nerfstudio_models  \


python $CHANGE_DET_SCRIPT \
  -c /local/home/pmishra/cvg/3dgscd/outputs/render_data/splatfacto/2025-05-01_174004/config.yml\
  -t ${DATA_FOLDER}/transforms.json \
  -o ${DATA_FOLDER} \
  -ckpt /local/home/pmishra/cvg/3dgscd/outputs/render_data/splatfacto/2025-05-01_174004/nerfstudio_models \
  -d  
