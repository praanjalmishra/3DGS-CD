# Pre-process iphone-captured images
# Prerequisite:
#   - Install ImageMagick: sudo apt-get install imagemagick

DATA_FOLDER=${1:-"/home/ziqi/Desktop/test/masks"}
NUM_DOWNSCALES=${2:-2}
NERFSTUDIO_FOLDER=${3:-"/home/ziqi/Packages/3dgscd"}

source ~/miniconda3/etc/profile.d/conda.sh
conda activate gscd

# ------------------ Convert image file type ------------------
# Convert JPG files to PNG files for pre-training images
mogrify -format png $DATA_FOLDER/rgb/*.JPG
rm $DATA_FOLDER/rgb/*.JPG
# Rename images as frame_{id:05g}.png
counter=0
for img in $(ls $DATA_FOLDER/rgb/IMG_*.png | sort); do
    new_name=$(printf "frame_%05d.png" $counter)
    mv "$img" "$DATA_FOLDER/rgb/$new_name"
    ((counter++))
done

# Convert JPG files to PNG files for new sparse view images
mogrify -format png $DATA_FOLDER/rgb_new/*.JPG
rm $DATA_FOLDER/rgb_new/*.JPG
# Rename images as frame_{id:05g}.png
counter=0
for img in $(ls $DATA_FOLDER/rgb_new/IMG_*.png | sort); do
    new_name=$(printf "frame_%05d.png" $counter)
    mv "$img" "$DATA_FOLDER/rgb_new/$new_name"
    ((counter++))
done

# ------------------ Downscale images ------------------

DOWNSCALED_VALUE=$((2**$NUM_DOWNSCALES))

# Downscale pre-training images
python $NERFSTUDIO_FOLDER/nerfstudio/process_data/process_data_utils.py \
  --img_dir $DATA_FOLDER/rgb \
  --num_downscales $NUM_DOWNSCALES

mv $DATA_FOLDER/rgb $DATA_FOLDER/rgb_original
mv $DATA_FOLDER/images_$DOWNSCALED_VALUE $DATA_FOLDER/rgb
rm -rf $DATA_FOLDER/images_*

# Downscale new sparse view images
python $NERFSTUDIO_FOLDER/nerfstudio/process_data/process_data_utils.py \
  --img_dir $DATA_FOLDER/rgb_new \
  --num_downscales $NUM_DOWNSCALES

mv $DATA_FOLDER/rgb_new $DATA_FOLDER/rgb_new_original
mv $DATA_FOLDER/images_$DOWNSCALED_VALUE $DATA_FOLDER/rgb_new
rm -rf $DATA_FOLDER/images_*
