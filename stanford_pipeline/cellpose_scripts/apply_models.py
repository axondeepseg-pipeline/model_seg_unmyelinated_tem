"""
Apply both myelinated and unmyelinated finetuned models to a given folder of 
images.
"""

from cellpose import models, io
from pathlib import Path
import numpy as np
import argparse
import logging
import torch
import cv2


def visualize_instance_seg(mask: np.ndarray) -> np.ndarray:
    """
    Convert an instance segmentation mask (where each instance has a unique integer label) 
    into an RGB image where each instance is colored differently.

    Parameters:
    - mask: 2D numpy array of shape (H, W) with integer labels for each instance.

    Returns:
    - A 3D numpy array of shape (H, W, 3) representing the RGB image.
    """
    img_normalized = cv2.normalize(mask, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)
    rgb_image = cv2.applyColorMap(img_normalized, cv2.COLORMAP_SPRING)
    rgb_image[mask == 0] = [0, 0, 0]  # Set background to black
    return rgb_image

def main(model_dir: Path, input_dir: Path):
    finetuned = {
        'myelinated': model_dir / 'cpsam_finetuned_myelinated',
        'unmyelinated': model_dir / 'cpsam_finetuned_unmyelinated'
    }

    m = torch.load(finetuned['myelinated'], map_location='cpu')
    myelinated_diam_labels = m['diam_labels']
    del m
    print('Diameter parameter for myelinated axon model:', myelinated_diam_labels.item())

    internal_logger = logging.getLogger(models.__name__)
    internal_logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(name)s: %(message)s')
    handler.setFormatter(formatter)
    internal_logger.addHandler(handler)

    image_paths = list(input_dir.glob('*.png'))
    # process folder one model at a time to avoid loading both on gpu at once
    for target_class, model_path in finetuned.items():
        print(f'Applying {target_class} model...')
        model = models.CellposeModel(gpu=True, pretrained_model=str(model_path))
        for img_path in image_paths:
            img = io.imread(str(img_path))
            masks, _, _ = model.eval(img)
            
            # transform instance segmentation mask into something we can visualize in RGB (each instance gets a unique color)
            print(f'Creating visuals of {target_class} predictions for {img_path.name}...')
            rgb_mask = visualize_instance_seg(masks)
            output_mask_path = input_dir / f'{img_path.stem}_pred-{target_class}.png'

            io.imsave(str(output_mask_path), masks.astype(np.uint16))
            io.imsave(str(input_dir / f'{img_path.stem}_pred-{target_class}_rgb.png'), rgb_mask)
        del model


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description="Apply axon counter models to a folder of images")
    ap.add_argument('-m', '--model_dir', type=Path, required=True, help="Path to the directory containing the finetuned models")
    ap.add_argument('-i', '--input_dir', type=Path, required=True, help="Path to the directory containing the input images")
    args = { k: Path(v) for k, v in vars(ap.parse_args()).items() }

    main(args['model_dir'], args['input_dir'])
