'''
This script takes as input a dataset in nnunet format and augments the 
training set with resized versions of the data at different scales. This 
could be seen as a very aggressive offline data augmentation, with the 
aim of regularizing the model output across resolutions.
'''

import argparse
import shutil
from pathlib import Path
import re

RESIZE_FACTORS = [0.5, 2.0, 4.0]


def get_associated_gt(image_path: Path) -> Path:
    """
    Given the path to an image, return the path to its associated ground truth label.
    """
    gt_path = image_path.parent.parent / "labelsTr" / image_path.name.replace("_0000.png", ".png")
    assert gt_path.exists(), f"Ground truth for {image_path} does not exist."
    return gt_path

def get_case_id_from_path(image_path: Path) -> int:
    """
    Given the path to an image, return the case ID (the part of the filename before the last underscore).
    """
    return int("_".join(image_path.stem.split("_")[:-1]))

def main():
    """
    Main function to run the multiresolution preprocessor.
    """
    parser = argparse.ArgumentParser(
        description="Multiresolution Preprocessor for nnUNet datasets"
    )
    parser.add_argument(
        "--nnunetraw_dir",
        type=str,
        required=True,
        help="Path to the nnUNet_raw directory containing the dataset.",
    )
    parser.add_argument(
        "--target_id",
        type=int,
        required=True,
        help="ID of the dataset to be processed (e.g., 1 for 'Dataset001').",
    )
    parser.add_argument(
        "--name",
        default="Dataset044_TEM_UNMYELINATED_AGG_MULTIRES",
        help="Name of the aggregated dataset. Default: 'Dataset044_TEM_UNMYELINATED_AGG_MULTIRES'",
    )
    parser.add_argument(
        "--description",
        default="Combined multi-resolution datasets for myelinated and unmyelinated axon segmentation",
        help="Description of the aggregated dataset. Default: 'Combined multi-resolution datasets for myelinated and unmyelinated axon segmentation'",
    )
    args = parser.parse_args()

    datadir = list(Path(args.nnunetraw_dir).glob(f"Dataset{args.target_id:03d}*"))
    datadir = datadir[0] if datadir else None
    assert datadir.exists(), f"Directory {args.nnunetraw_dir} does not exist."
    assert (datadir / "imagesTr").exists(), f"Directory {args.nnunetraw_dir}/imagesTr does not exist."
    assert (datadir / "labelsTr").exists(), f"Directory {args.nnunetraw_dir}/labelsTr does not exist."

    # copy the original dataset to a new directory
    new_dataset_dir = datadir.parent / args.name
    new_dataset_dir.mkdir(parents=True, exist_ok=False)
    shutil.copytree(datadir / "imagesTr", new_dataset_dir / "imagesTr")
    shutil.copytree(datadir / "labelsTr", new_dataset_dir / "labelsTr")
    shutil.copy(datadir / "dataset.json", new_dataset_dir / "dataset.json")

    # iterate over all images
    training_images = list((new_dataset_dir / "imagesTr").glob("*.png"))
    current_id = 1 + max([get_case_id_from_path(img) for img in training_images])
    for image_path in training_images:
        gt_path = get_associated_gt(image_path)
        print(f"Processing case ID {get_case_id_from_path(image_path)} and adding {len(RESIZE_FACTORS)} resized versions.")
        for factor in RESIZE_FACTORS:
            # find the new image and ground truth paths
            new_gt_fname = Path(re.sub(r"\d\d\d\.png$", f"{current_id:03d}.png", str(gt_path)))
            new_image_fname = str(new_gt_fname).replace(".png", "_0000.png")
            # resize the image and ground truth
            # resize_and_save_image(image_path, new_image_path, factor)
            # resize_and_save_image(gt_path, new_gt_path, factor)

            current_id += 1

if __name__ == "__main__":
    main()