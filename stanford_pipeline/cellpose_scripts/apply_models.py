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


def main(model_dir: Path, input_dir: Path):
    finetuned = {
        'myelinated': model_dir / 'cpsam_finetuned_myelinated',
        'unmyelinated': model_dir / 'cpsam_finetuned_unmyelinated'
    }

    m = torch.load(finetuned['myelinated'], map_location='cpu')
    myelinated_diam_labels = m['diam_labels']
    del m

    internal_logger = logging.getLogger(models.__name__)
    internal_logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(name)s: %(message)s')
    handler.setFormatter(formatter)
    internal_logger.addHandler(handler)

    # process folder one model at a time to avoid loading both on gpu at once
    for target_class, model_path in finetuned.items():
        print(f'Applying {target_class} model...')
        model = models.CellposeModel(gpu=True, pretrained_model=str(model_path))
        for img_path in input_dir.glob('*.png'):
            img = io.imread(str(img_path))
            diam = myelinated_diam_labels if target_class == 'myelinated' else None
            masks, _, _ = model.eval(img, diameter=diam)
            print(model.net['diam_labels'])
            output_mask_path = input_dir / f'{img_path.stem}_pred-{target_class}.png'
            io.imsave(str(output_mask_path), masks.astype(np.uint16))
        del model
    

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description="Apply axon counter models to a folder of images")
    ap.add_argument('-m', '--model_dir', type=Path, required=True, help="Path to the directory containing the finetuned models")
    ap.add_argument('-i', '--input_dir', type=Path, required=True, help="Path to the directory containing the input images")
    args = { k: Path(v) for k, v in vars(ap.parse_args()).items() }

    main(args['model_dir'], args['input_dir'])