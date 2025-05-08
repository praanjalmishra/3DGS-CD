#!/bin/bash
# Pre-process images with numerical names like 1.jpg, 2.jpg, etc.
# Prerequisite:
#   - Install ImageMagick: sudo apt-get install imagemagick

DATA_FOLDER=${1:-"/local/home/pmishra/cvg/3dgscd/data/dlab_data"}
NUM_DOWNSCALES=${2:-2}
NERFSTUDIO_FOLDER=${3:-"/local/home/pmishra/cvg/3dgscd"}

# Check if data directories exist
if [ ! -d "$DATA_FOLDER/rgb" ]; then
  echo "Creating directory $DATA_FOLDER/rgb"
  mkdir -p "$DATA_FOLDER/rgb"
fi

if [ ! -d "$DATA_FOLDER/rgb_new" ]; then
  echo "Creating directory $DATA_FOLDER/rgb_new"
  mkdir -p "$DATA_FOLDER/rgb_new"
fi

# Check if ImageMagick is installed
if ! command -v mogrify &> /dev/null; then
  echo "ImageMagick not found. Please install it using: sudo apt-get install imagemagick"
  exit 1
fi

# Setup environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate gscd

# ------------------ Convert image file type ------------------
# Check if there are any JPG/JPEG files in rgb directory
echo "Checking for images in $DATA_FOLDER/rgb"
ls -l $DATA_FOLDER/rgb/

echo "Converting JPG/JPEG files to PNG files for pre-training images"
# Use find command which is more reliable for finding files
find $DATA_FOLDER/rgb -name "*.jpg" -o -name "*.jpeg" | while read img; do
  echo "Converting $img to PNG"
  base=$(basename "$img" | sed 's/\.[^.]*$//')
  dir=$(dirname "$img")
  convert "$img" "$dir/$base.png"
  rm "$img"
done

# Rename images as frame_{id:05g}.png - using find again
counter=0
# Create a temporary file with all PNG files sorted numerically
find $DATA_FOLDER/rgb -name "*.png" | sort -V > /tmp/png_files.txt
# Process each file in the sorted list
while read img; do
  new_name=$(printf "frame_%05d.png" $counter)
  echo "Renaming $img to $DATA_FOLDER/rgb/$new_name"
  mv "$img" "$DATA_FOLDER/rgb/$new_name"
  ((counter++))
done < /tmp/png_files.txt
rm /tmp/png_files.txt

# Check if no images were found
if [ $counter -eq 0 ]; then
  echo "No PNG images were created in $DATA_FOLDER/rgb"
  echo "Please make sure your images are in this directory"
  exit 1
fi

# Repeat for rgb_new directory
echo "Checking for images in $DATA_FOLDER/rgb_new"
ls -l $DATA_FOLDER/rgb_new/

echo "Converting JPG/JPEG files to PNG files for new sparse view images"
find $DATA_FOLDER/rgb_new -name "*.jpg" -o -name "*.jpeg" | while read img; do
  echo "Converting $img to PNG"
  base=$(basename "$img" | sed 's/\.[^.]*$//')
  dir=$(dirname "$img")
  convert "$img" "$dir/$base.png"
  rm "$img"
done

counter=0
# Create a temporary file with all PNG files sorted numerically
find $DATA_FOLDER/rgb_new -name "*.png" | sort -V > /tmp/png_files.txt
# Process each file in the sorted list
while read img; do
  new_name=$(printf "frame_%05d.png" $counter)
  echo "Renaming $img to $DATA_FOLDER/rgb_new/$new_name"
  mv "$img" "$DATA_FOLDER/rgb_new/$new_name"
  ((counter++))
done < /tmp/png_files.txt
rm /tmp/png_files.txt

# Check if no images were found
if [ $counter -eq 0 ]; then
  echo "No PNG images were created in $DATA_FOLDER/rgb_new"
  echo "Please make sure your images are in this directory"
  exit 1
fi

# ------------------ Downscale images ------------------
DOWNSCALED_VALUE=$((2**$NUM_DOWNSCALES))
echo "Downscaling images by a factor of $DOWNSCALED_VALUE"

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

echo "Image processing complete!"