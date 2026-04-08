'''This file provides utilities to preprocess the dataset into a format suitable 
for Cellpose training and inference.

Author: Armand Collin
'''

from pathlib import Path
import argparse
import shutil

from PIL import Image
from skimage import measure
import numpy as np

CELLPOSE_MASK_SUFFIX = '_seg-cellpose'

def convert_axonmyelin_mask_to_cellpose(mask_path: Path, output_path: Path):
    """
    Convert an axon-myelin segmentation mask to a Cellpose-compatible mask.
    
    In the axon-myelin mask:
    - Background: 0
    - Myelin: 127
    - Axon: 255
    
    In the Cellpose mask:
    - Instance segmentation
    - Background: 0
    - Cell 1: 1 (axon and myelin combined)
    - Cell 2: 2
    - ... and so on for each cell instance.
    
    Parameters:
    - mask_path: Path to the input axon-myelin mask.
    - output_path: Path to save the converted Cellpose mask.
    """
    mask = Image.open(str(mask_path)).convert("L")
    mask = np.array(mask)
    cellpose_mask = (mask > 0)  # Set axon and myelin to 1, background to 0
    cellpose_mask = measure.label(cellpose_mask, connectivity=1)  # Label connected components

    # Ensure the mask is cast to a supported data type
    cellpose_mask = cellpose_mask.astype(np.uint16)

    cellpose_mask_img = Image.fromarray(cellpose_mask, mode='I;16')
    cellpose_mask_img.save(str(output_path))

def find_image_mask_pairs_from_bids(dir_path: Path, mask_suffix: str = '_seg-axonmyelin-manual.png') -> list[tuple[Path, Path]]:
    """
    Find all image and corresponding mask file paths in a BIDS-like directory structure.
    
    Parameters:
    - dir_path: Path to the root directory to search (should contain subdirectories like 'sub-*/micr/').
    - mask_suffix: Suffix used to identify mask files.
    
    Returns:
    - List of tuples containing (image_path, mask_path).
    """
    image_mask_pairs = []
    mask_files = list(dir_path.glob(f'derivatives/labels/sub-*/micr/*{mask_suffix}'))
    for mask_file in mask_files:
        image_filename = mask_file.name.replace(mask_suffix, '.png')
        image_file = list(dir_path.glob(f'sub-*/micr/{image_filename}'))[0]
        if image_file.exists():
            image_mask_pairs.append((image_file, mask_file))
        else: 
            # in case images are in TIFF format instead of PNG
            image_file_tiff = mask_file.with_name(mask_file.name.replace(mask_suffix, '.tif'))
            if image_file_tiff.exists():
                image_mask_pairs.append((image_file_tiff, mask_file))
            else:
                print(f'Warning: No corresponding image found for mask {mask_file}')
    return image_mask_pairs

def preprocess_dataset(data_dir: Path, output_dir: Path, target_class: str = 'myelinated'):
    
    suffix = '_seg-axonmyelin-manual' if target_class == 'myelinated' else '_seg-uaxon-manual'
    pairs = find_image_mask_pairs_from_bids(data_dir, suffix)

    output_dir = output_dir / target_class
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for image_path, mask_path in pairs:
        output_image_path = output_dir / image_path.name
        output_mask_path = output_dir / (mask_path.name.replace(suffix, CELLPOSE_MASK_SUFFIX))

        shutil.copy(image_path, output_image_path)
        convert_axonmyelin_mask_to_cellpose(mask_path, output_mask_path)

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description="Preprocess dataset for Cellpose")
    ap.add_argument('data_dir', type=str, help="Path to the dataset (split into train/ and test/ directories)")
    ap.add_argument('-t', '--target-class', type=str, default=None, help="Type of mask to process. Either 'myelinated' or 'unmyelinated'. Defaults to both.")
    ap.add_argument('-o', '--output_dir', type=str, default=None, help="Path to save the preprocessed data.")
    args = ap.parse_args()
    data_dir = Path(args.data_dir)
    target_class = args.target_class
    output_dir = Path(args.output_dir) if args.output_dir else Path('.') / 'cellpose_preprocessed'

    if target_class is None:
        for cls in ['myelinated', 'unmyelinated']:
            preprocess_dataset(data_dir, output_dir, cls)
    elif target_class not in ['myelinated', 'unmyelinated']:
        raise ValueError("Invalid target class. Must be either 'myelinated' or 'unmyelinated'.")
    else:
        preprocess_dataset(data_dir, output_dir, target_class)