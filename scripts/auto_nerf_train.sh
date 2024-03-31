# Script to automate 2-NeRF training 
#!/bin/bash
# Pre-requisites:
# 1. Install Kubric and NeRFStudio (code.amzn.com)
# 2. Install SAM & hloc into nerfstudio conda env
# 3. Download the following scripts from here
# 4. Move the scripts to the Desktop or other user-specified folders
# 5. Move the keyframe.py script to the kubric/examples folder
# 6. Download the texture.jpg files from here to the kubric/output folder

# Set the default values for the arguments
HDRI_ID=${1:-5}
# choose from cube, cylinder, sphere, cone, torus, gear, torus_knot, sponge, 
# spot, teapot, suzanne
OBJECT_TYPE=${2:-"cube"}
TEXTURE_ID=${3:-0}
CUDA_VISIBLE_DEVICES=${4:-0}
FPS=${5:-5}
OUTPUT_FOLDER=${6:-"/home/ziqi/Packages/nerfstudio/outputs"}
KUBRIC_FOLDER=${7:-"/home/ziqi/Packages/kubric"}
NERFSTUDIO_FOLDER=${8:-"/home/ziqi/Packages/nerfstudio"}
MERGE_SCRIPT=${9:-"/home/ziqi/Packages/nerfstudio/scripts/merge_nerf_data.py"}
EDIT_SCRIPT=${10:-"/home/ziqi/Packages/nerfstudio/scripts/edit_nerf_data.py"}
CAM_PATH=${11:-"/home/ziqi/Packages/nerfstudio/scripts/cam_path_single.json"}
CAM_PATH_VIDEO=${12:-"/home/ziqi/Packages/nerfstudio/scripts/cam_path_video.json"}


# ------------------ Generate the data ------------------
# May need to build a conda env with dependencies required by the .py scripts
source ~/miniconda3/etc/profile.d/conda.sh
conda activate nerftest

# Generate synthetic data for an object-centric scene
cd $KUBRIC_FOLDER
docker run --rm --interactive --user $(id -u):$(id -g) \
  --volume "$(pwd):/kubric" kubricdockerhub/kubruntu /usr/bin/python3 \
  examples/keyframing.py -hid ${HDRI_ID} -m \
  -o output/hdri${HDRI_ID}_${OBJECT_TYPE} -obj ${OBJECT_TYPE} \
  -t output/texture${TEXTURE_ID}.jpg &
PID1=$!

# Generate synthetic data for the same scene but with object removed
docker run --rm --interactive --user $(id -u):$(id -g) \
  --volume "$(pwd):/kubric" kubricdockerhub/kubruntu /usr/bin/python3 \
  examples/keyframing.py -hid ${HDRI_ID} -m \
  -o output/hdri${HDRI_ID}_no_${OBJECT_TYPE} -obj None \
  -t output/texture${TEXTURE_ID}.jpg &
PID2=$!

# Generate synthetic data for the same scene but with object moved
docker run --rm --interactive --user $(id -u):$(id -g) \
  --volume "$(pwd):/kubric" kubricdockerhub/kubruntu /usr/bin/python3 \
  examples/keyframing.py -hid ${HDRI_ID} -m \
  -o output/hdri${HDRI_ID}_${OBJECT_TYPE}_move -obj ${OBJECT_TYPE} \
  -t output/texture${TEXTURE_ID}.jpg \
  -pos 2.5 2.5 1.0 -quat 1.0 0.0 0.0 0.0 & # NOTE: Change obj new pose here
PID3=$!

# Check the exit codes of each process
for pid in $PID1 $PID2 $PID3; do
    wait $pid
    EXIT_CODE=$?
    if [ $EXIT_CODE -ne 0 ]; then
        case $pid in
            $PID1)
                echo "hdri${HDRI_ID}_${OBJECT_TYPE} generation failed." >&2
                ;;
            $PID2)
                echo "hdri${HDRI_ID}_no_${OBJECT_TYPE} generation failed." >&2
                ;;
            $PID3)
                echo "hdri${HDRI_ID}_${OBJECT_TYPE}_move generation failed." >&2
                ;;
        esac
        exit 1
    fi
done

echo "Data generation completed successfully."


# ------------------ Post-process the data ------------------

# merge the object_type and no_object_type folders
python $MERGE_SCRIPT -fo "${KUBRIC_FOLDER}/output/hdri${HDRI_ID}_${OBJECT_TYPE}" \
  -fn "${KUBRIC_FOLDER}/output/hdri${HDRI_ID}_no_${OBJECT_TYPE}" \
  -o "${KUBRIC_FOLDER}/output/hdri${HDRI_ID}_${OBJECT_TYPE}_remove" -en

# check if the previous command was successful
if [ $? -eq 0 ]
then
  echo "Merge remove completed successfully."
else
  echo "Merge remove failed." >&2
  exit 1
fi

# now merge object_type and object_type_move to object_type_reconfig
python $MERGE_SCRIPT -fo "${KUBRIC_FOLDER}/output/hdri${HDRI_ID}_${OBJECT_TYPE}" \
  -fn "${KUBRIC_FOLDER}/output/hdri${HDRI_ID}_${OBJECT_TYPE}_move" \
  -o "${KUBRIC_FOLDER}/output/hdri${HDRI_ID}_${OBJECT_TYPE}_reconfig" -en

# check if the previous command was successful
if [ $? -eq 0 ]
then
  echo "Merge reconfig completed successfully."
else
  echo "Merge reconfig failed." >&2
  exit 1
fi

# if both merges were successful, proceed to edit the transform.json file in the object_type folder
python $EDIT_SCRIPT -i \
  "${KUBRIC_FOLDER}/output/hdri${HDRI_ID}_${OBJECT_TYPE}"/transforms.json \
  -o "${KUBRIC_FOLDER}/output/hdri${HDRI_ID}_${OBJECT_TYPE}"/transforms.json -rm

# check if the previous command was successful
if [ $? -eq 0 ]
then
  echo "Remove mask completed successfully."
else
  echo "Remove mask failed." >&2
  exit 1
fi

# NOTE: We don't remove the masks in the folder since it might be useful

# Remove the keyframe.blend file if it exists
if [ -f "${KUBRIC_FOLDER}/output/hdri${HDRI_ID}_${OBJECT_TYPE}"/keyframing.blend ]; then
    rm "${KUBRIC_FOLDER}/output/hdri${HDRI_ID}_${OBJECT_TYPE}"/keyframing.blend
    echo "Removed keyframing.blend file"
fi

# if both merges were successful, proceed to edit the transform.json file in the move folder
python $EDIT_SCRIPT -i \
  "${KUBRIC_FOLDER}/output/hdri${HDRI_ID}_${OBJECT_TYPE}_move"/transforms.json \
  -o "${KUBRIC_FOLDER}/output/hdri${HDRI_ID}_${OBJECT_TYPE}_move"/transforms.json -rm

# check if the previous command was successful
if [ $? -eq 0 ]
then
  echo "Remove mask in hdri${HDRI_ID}_${OBJECT_TYPE}_move  completed successfully."
else
  echo "Remove mask in hdri${HDRI_ID}_${OBJECT_TYPE}_move failed." >&2
  exit 1
fi

# NOTE: We don't remove the masks in the "..._move" folder since it could be useful for pose change est

# Remove the keyframe.blend file if it exists
if [ -f "${KUBRIC_FOLDER}/output/hdri${HDRI_ID}_${OBJECT_TYPE}_move"/keyframing.blend ]; then
    rm "${KUBRIC_FOLDER}/output/hdri${HDRI_ID}_${OBJECT_TYPE}_move"/keyframing.blend
    echo "Removed keyframing.blend file"
fi

# cp the folders back to the nerfstudio folder
for folder in "${OBJECT_TYPE}" "${OBJECT_TYPE}_reconfig" "${OBJECT_TYPE}_remove" "${OBJECT_TYPE}_move"; do
    FOLDER_NAME="hdri${HDRI_ID}_${folder}"
    cp -r ${KUBRIC_FOLDER}/output/${FOLDER_NAME} ${NERFSTUDIO_FOLDER}/data/nerfstudio/
done

# check if the previous command was successful
if [ $? -eq 0 ]
then
  echo "CP files to nerfstudio successfully."
else
  echo "CP files to nerfstudio failed." >&2
  exit 1
fi

# ------------------ Train NeRFacto ------------------

cd $NERFSTUDIO_FOLDER
conda activate nerftest
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}

current_time=$(date +%Y-%m-%d_%H%M%S)

while true; do
  # Pre-trained NeRF: Train NeRFacto for the object-centric scene
  ns-train nerfacto --vis viewer+tensorboard \
    --experiment-name hdri${HDRI_ID}_${OBJECT_TYPE} \
    --output-dir ${OUTPUT_FOLDER} \
    --timestamp $current_time \
    --steps_per_eval_all_images 500 \
    --pipeline.datamanager.train-num-rays-per-batch 4096 \
    --pipeline.datamanager.camera-optimizer.mode off \
    --pipeline.model.use_appearance_embedding=False \
    --pipeline.model.use_gradient_scaling True \
    --max-num-iterations 30000 \
    --machine.num-devices 1 \
    --viewer.quit-on-train-completion True \
    nerfstudio-data --data data/nerfstudio/hdri${HDRI_ID}_${OBJECT_TYPE} \
    --auto-scale-poses=False --center-method none --orientation-method none \
    --train_split_fraction 0.9

  # check if the previous command was successful
  if [ $? -ne 0 ]; then
    echo "Training hdri${HDRI_ID}_${OBJECT_TYPE} failed." >&2
    continue
  fi

  # Baseline: Train NeRFacto for the object-reconfigured scene
  ns-train nerfacto --vis viewer+tensorboard \
    --experiment-name hdri${HDRI_ID}_${OBJECT_TYPE}_reconfig \
    --output-dir ${OUTPUT_FOLDER} \
    --timestamp $current_time \
    --steps_per_eval_all_images 500 \
    --pipeline.datamanager.train-num-rays-per-batch 4096 \
    --pipeline.datamanager.camera-optimizer.mode off \
    --pipeline.model.use_appearance_embedding True \
    --pipeline.model.use_gradient_scaling True \
    --max-num-iterations 30000 \
    --machine.num-devices 1 \
    --viewer.quit-on-train-completion True \
    nerfstudio-data --data data/nerfstudio/hdri${HDRI_ID}_${OBJECT_TYPE}_reconfig \
    --auto-scale-poses=False --center-method none --orientation-method none \
    --train_split_fraction 0.9

  # check if the previous command was successful
  if [ $? -ne 0 ]; then
    echo "Training hdri${HDRI_ID}_${OBJECT_TYPE}_reconfig failed." >&2
    continue
  fi

  # Ground truth: Train NeRFacto for the object-moved scene
  # NOTE: Dense RGB observations for the reconfigured scene to provide GT
  ns-train nerfacto --vis viewer+tensorboard \
    --experiment-name hdri${HDRI_ID}_${OBJECT_TYPE}_move \
    --output-dir ${OUTPUT_FOLDER} \
    --timestamp $current_time \
    --steps_per_eval_all_images 500 \
    --pipeline.datamanager.train-num-rays-per-batch 4096 \
    --pipeline.datamanager.camera-optimizer.mode off \
    --pipeline.model.use_appearance_embedding=False \
    --pipeline.model.use_gradient_scaling True \
    --max-num-iterations 30000 \
    --machine.num-devices 1 \
    --viewer.quit-on-train-completion True \
    nerfstudio-data --data data/nerfstudio/hdri${HDRI_ID}_${OBJECT_TYPE}_move \
    --auto-scale-poses=False --center-method none --orientation-method none \
    --train_split_fraction 0.9

  # check if the previous command was successful
  if [ $? -ne 0 ]; then
    echo "Training hdri${HDRI_ID}_${OBJECT_TYPE}_move failed." >&2
    continue
  fi

  # # Train our model for the object-reconfigured scene
  # ns-train nerfacto2 --vis viewer+tensorboard \
  #   --experiment-name hdri${HDRI_ID}_${OBJECT_TYPE}_reconfig \
  #   --output-dir ${OUTPUT_FOLDER} \
  #   --timestamp $current_time \
  #   --steps_per_eval_all_images 500 \
  #   --pipeline.datamanager.train-num-rays-per-batch 4096 \
  #   --pipeline.datamanager.camera-optimizer.mode off \
  #   --pipeline.model.use_appearance_embedding False \
  #   --pipeline.model.use_gradient_scaling True \
  #   --pipeline.model.enable_reconfig True \
  #   --load-dir ${OUTPUT_FOLDER}/hdri${HDRI_ID}_${OBJECT_TYPE}/nerfacto/${current_time}/nerfstudio_models/ \
  #   --max-num-iterations 2000 \
  #   --machine.num-devices 1 \
  #   --viewer.quit-on-train-completion True \
  #   nerfstudio-data --data data/nerfstudio/hdri${HDRI_ID}_${OBJECT_TYPE}_reconfig \
  #   --auto-scale-poses=False --center-method none --orientation-method none \
  #   --train_split_fraction 0.9

  # # check if the previous command was successful
  # if [ $? -ne 0 ]; then
  #   echo "Training ours on hdri${HDRI_ID}_${OBJECT_TYPE}_reconfig failed." >&2
  #   continue
  # fi
  
  # If we reach here, all trainings were successful
  echo "All trainings completed successfully."
  break

done

# # ------------------ Render videos ------------------

# cd $NERFSTUDIO_FOLDER
# recfg_dir=${OUTPUT_FOLDER}/hdri${HDRI_ID}_${OBJECT_TYPE}_reconfig
# recfg_cfg=${recfg_dir}/nerfacto/${current_time}/config.yml
# recfg_ckpt_dir=${recfg_dir}/nerfacto/${current_time}/nerfstudio_models/
# recfg_dir=${OUTPUT_FOLDER}/hdri${HDRI_ID}_${OBJECT_TYPE}_reconfig
# recfg2_cfg=${recfg_dir}/nerfacto2/${current_time}/config.yml
# recfg2_ckpt_dir=${recfg_dir}/nerfacto2/${current_time}/nerfstudio_models/

# recfg_ckpts=$(ls $recfg_ckpt_dir | grep -oP '(?<=-)[0-9]+(?=\.)' | sort -n)
# recfg2_ckpts=$(ls $recfg2_ckpt_dir | grep -oP '(?<=-)[0-9]+(?=\.)' | sort -n)

# # Make folder to save renders
# render_target_dir=${NERFSTUDIO_FOLDER}/renders/${current_time}/

# ns-render camera-path --load-config ${recfg2_cfg} \
#   --camera-path-filename ${CAM_PATH_VIDEO} \
#   --output-path ${render_target_dir}/reconfig_2nerf.mp4

# # check if the previous command was successful
# if [ $? -eq 0 ]
# then
#   echo "Rendering of ours video completed successfully."
# else
#   echo "Rendering of ours video failed." >&2
#   exit 1
# fi

# ns-render camera-path --load-config ${recfg_cfg} \
#   --camera-path-filename ${CAM_PATH_VIDEO} \
#   --output-path ${render_target_dir}/reconfig_from_scratch.mp4

# # check if the previous command was successful
# if [ $? -eq 0 ]
# then
#   echo "Rendering of baseline video completed successfully."
# else
#   echo "Rendering of baseline video failed." >&2
#   exit 1
# fi

# move_dir=${OUTPUT_FOLDER}/hdri${HDRI_ID}_${OBJECT_TYPE}_move
# move_cfg=${move_dir}/nerfacto/${current_time}/config.yml

# ns-render camera-path --load-config ${move_cfg} \
#   --camera-path-filename ${CAM_PATH_VIDEO} \
#   --output-path ${render_target_dir}/reconfig_gt.mp4

# # check if the previous command was successful
# if [ $? -eq 0 ]
# then
#   echo "Rendering of GT video completed successfully."
# else
#   echo "Rendering of GT video failed." >&2
#   exit 1
# fi

# # Make folder to save renders
# mkdir -p $render_target_dir/renders
# mkdir -p $render_target_dir/renders2

# # Render images from a fixed view for all checkpoints
# for step in ${recfg_ckpts}; do
#   step_num=$((10#$step))

#   ns-render camera-path --load-config ${recfg_cfg} \
#   --camera-path-filename ${CAM_PATH} \
#   --output-path ${NERFSTUDIO_FOLDER}/renders/${current_time} \
#   --output-format images \
#   --load_step ${step_num}

#   mv ${NERFSTUDIO_FOLDER}/renders/${current_time}/00000.jpg \
#     $render_target_dir/renders/${step}.jpg
# done

# # Render images from a fixed view for all checkpoints
# for step in ${recfg2_ckpts}; do
#   step_num=$((10#$step))

#   ns-render camera-path --load-config ${recfg2_cfg} \
#   --camera-path-filename ${CAM_PATH} \
#   --output-path ${NERFSTUDIO_FOLDER}/renders/${current_time} \
#   --output-format images \
#   --load_step ${step_num}

#   mv ${NERFSTUDIO_FOLDER}/renders/${current_time}/00000.jpg \
#     $render_target_dir/renders2/${step}.jpg
# done

# # Render videos from images
# TEMP=${render_target_dir}/tmp
# mkdir -p $TEMP

# FOLDER1=${render_target_dir}/renders2
# FOLDER2=${render_target_dir}/renders 

# # Get the sorted list of images from folder1 and folder2
# IMGS_FOLDER1=($(ls $FOLDER1 | sort))
# IMGS_FOLDER2=($(ls $FOLDER2 | sort))

# LAST_IMG_FOLDER1=${IMGS_FOLDER1[-1]}

# index=1

# CONVERGED_ITERATION=500

# for img2 in "${IMGS_FOLDER2[@]}"; do
#   # Extract iteration number from folder2 image name
#   iteration=$(echo $img2 | grep -oE "[0-9]+" | sed 's/^0*//' | sed 's/^$/0/')

#   # Get corresponding image from folder1 if it exists, else use the last image
#   if [ $index -le ${#IMGS_FOLDER1[@]} ]; then
#       img1=${IMGS_FOLDER1[$((index-1))]}
#   else
#       img1=$LAST_IMG_FOLDER1
#   fi

#   # Combine images side by side using ffmpeg
#   ffmpeg -y -i $FOLDER1/$img1 -i $FOLDER2/$img2 \
#     -filter_complex "[0:v][1:v]hstack=inputs=2" \
#     $TEMP/temp_combined_${index}.jpg

#   # Add text using ffmpeg
#   if [ $iteration -ge $CONVERGED_ITERATION ]; then
#     ffmpeg -y -i $TEMP/temp_combined_${index}.jpg \
#       -vf "drawtext=text='Converged!!!':x=(w/2-tw-10):y=10:fontsize=100:fontcolor=red,drawtext=text='Iteration\: $iteration':x=w-tw-10:y=10:fontsize=100:fontcolor=white" \
#       $TEMP/combined_${index}.jpg
#   else
#     ffmpeg -y -i $TEMP/temp_combined_${index}.jpg \
#       -vf "drawtext=text='Iteration\: $iteration':x=w-tw-10:y=10:fontsize=100:fontcolor=white" \
#       $TEMP/combined_${index}.jpg
#   fi

#   index=$((index+1))
# done

# ffmpeg -framerate $FPS -i $TEMP/combined_%d.jpg \
#   -c:v libx264 -vf "fps=$FPS,format=yuv420p" \
#   ${render_target_dir}/combined.mp4 -y

# # rm -rf $TEMP