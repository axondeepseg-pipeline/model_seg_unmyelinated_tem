"""
Fine-tune a Cellpose model on the Stanford SRF dataset.


Authors: Pauline Goue, Armand Collin
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from skimage import measure  
from cellpose import models, train, metrics
import logging


CELLPOSE_MASK_SUFFIX = '_seg-cellpose.png'


def load_image_mask_pairs(data_dir: Path) -> tuple[list[str], list[str]]:
    """
    Load image and corresponding mask file paths from the specified directory.

    Parameters:
    - data_dir: Path to the directory containing the preprocessed images and masks.

    Returns:
    - A tuple containing two lists: (image_files, mask_files)
    """
    mask_files = [str(f) for f in data_dir.glob(f'*{CELLPOSE_MASK_SUFFIX}')]
    image_files = [f.replace(CELLPOSE_MASK_SUFFIX, '.png') for f in mask_files]
    return image_files, mask_files

def main(data_dir: Path, test_dir: Path, output_dir: Path):
    """
    Main function to fine-tune a Cellpose model on the provided dataset.

    Parameters:
    - data_dir: Path to the directory created by prepare_data.py
    - output_dir: Path to the directory where the fine-tuned model and evaluation results will be saved
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    target_class = 'unmyelinated' if 'unmyelinated' in data_dir.name else 'myelinated'
    image_files, label_files = load_image_mask_pairs(data_dir)
    test_image_files, test_label_files = load_image_mask_pairs(test_dir)

    model = models.CellposeModel(gpu=True, pretrained_model="cpsam")

    internal_logger = logging.getLogger(train.__name__)
    internal_logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(name)s: %(message)s')
    handler.setFormatter(formatter)
    internal_logger.addHandler(handler)

    new_model_path, train_losses, test_losses = train.train_seg(
        model.net,
        train_files=image_files,
        train_labels_files=label_files,
        test_files=test_image_files,
        test_labels_files=test_label_files,
        normalize=True,
        n_epochs=500,
        learning_rate=1e-4,  
        save_path=str(output_dir),
        save_every=50,
        model_name=f"cpsam_finetuned_{target_class}"
    )
    print(f"Fine-tuned model saved at: {new_model_path}")

    # skip eval for now

    # pred_masks = []
    # finetuned_model = models.CellposeModel(gpu=True, pretrained_model=new_model_path)
    # for img in test_imgs:
    #     m, _, _ = finetuned_model.eval(img, diameter=None, channels=[0, 0])
    #     pred_masks.append(m)


    # ap, _, _, _ = metrics.average_precision(test_masks, pred_masks)


    # print(f"AP @ IoU 50% : {ap[:, 0].mean():.3f}   (target > 0.75)")
    # print(f"AP @ IoU 75% : {ap[:, 1].mean():.3f}   (target > 0.60)")
    # print(f"AP @ IoU 90% : {ap[:, 2].mean():.3f}")


    # n = min(4, len(test_imgs))
    # fig, axes = plt.subplots(n, 3, figsize=(13, 4 * n))
    # for i in range(n):
    #     axes[i, 0].imshow(test_imgs[i],  cmap="gray");    axes[i, 0].set_title("Raw TEM");       axes[i, 0].axis("off")
    #     axes[i, 1].imshow(test_masks[i], cmap="tab20b");  axes[i, 1].set_title("Ground truth");  axes[i, 1].axis("off")
    #     axes[i, 2].imshow(pred_masks[i], cmap="tab20b");  axes[i, 2].set_title(f"Predicted  AP@50={ap[i,0]:.2f}"); axes[i, 2].axis("off")
    # plt.tight_layout()
    # plt.savefig(str(output_dir / "eval_grid.png"), dpi=150)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Fine-tune Cellpose model on axon dataset")
    ap.add_argument('data_dir', type=str, help="Path to the preprocessed training set (output of prepare_data.py)")
    ap.add_argument('test_dir', type=str, help="Path to the preprocessed testing set (output of prepare_data.py).")
    ap.add_argument('output_dir', type=str, help="Path to save the fine-tuned model and evaluation results")
    args = ap.parse_args()

    main(Path(args.data_dir), Path(args.test_dir), Path(args.output_dir))
