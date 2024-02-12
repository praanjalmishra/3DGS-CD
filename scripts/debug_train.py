# Visualize the training data for our method
import argparse
import json
import cv2
import os


def process_and_save(json_file, save_folder):
    """
    Process the RGB images with masks and save them to the specified folder.
    Only processes images listed in train_filenames.
    """
    base_folder = os.path.dirname(json_file)
    with open(json_file, 'r') as f:
        data = json.load(f)

    train_files_set = set(data["train_filenames"])
    for frame in data["frames"]:
        rgb_file = frame["file_path"]
        mask_file = frame["mask_path"]

        # Only process if the RGB filename is in train_filenames
        if rgb_file in train_files_set:
            rgb_path = os.path.join(base_folder, rgb_file)
            mask_path = os.path.join(base_folder, mask_file)

            rgb_image = cv2.imread(rgb_path)
            mask_image = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            masked_image = cv2.bitwise_and(
                rgb_image, rgb_image, mask=mask_image
            )
            
            # Save the masked image to the specified folder
            output_path = os.path.join(save_folder, os.path.basename(rgb_file))
            cv2.imwrite(output_path, masked_image)


def main(args):
    assert os.path.isdir(args.save), "Save folder does not exist."
    assert os.path.isfile(args.json), "JSON file does not exist."

    process_and_save(args.json, args.save)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Debug NeRFacto2 training data"
    )
    parser.add_argument(
        "-j", "--json", type=str, required=True,
        help="Path to the transforms.json file."
    )
    parser.add_argument(
        "-s", "--save", type=str, required=True,
        help="Path to the folder to save the masked RGB images."
    )
    args = parser.parse_args()

    main(args)