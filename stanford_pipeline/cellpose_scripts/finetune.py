import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from skimage import measure  
from cellpose import models, train, metrics


CELLPOSE_MASK_SUFFIX = '_seg-cellpose.png'


def main(data_dir: Path, output_dir: Path):
    """
    Main function to fine-tune a Cellpose model on the provided dataset.

    Parameters:
    - data_dir: Path to the directory created by prepare_data.py
    - output_dir: Path to the directory where the fine-tuned model and evaluation results will be saved
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    label_files = [str(f) for f in data_dir.glob(f'*{CELLPOSE_MASK_SUFFIX}')]
    image_files = [f.replace(CELLPOSE_MASK_SUFFIX, '.png') for f in label_files]

    model = models.CellposeModel(gpu=True, pretrained_model="cpsam")

    new_model_path = train.train_seg(
        model.net,
        train_files=image_files,
        train_labels_files=label_files,
        normalize=True,
        n_epochs=500,
        learning_rate=1e-4,  
        save_path=str(output_dir),
        save_every=25,
    )

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
    ap = argparse.ArgumentParser(description="Fine-tune Cellpose model on custom dataset")
    ap.add_argument('data_dir', type=str, help="Path to the preprocessed dataset (output of prepare_data.py)")
    ap.add_argument('output_dir', type=str, help="Path to save the fine-tuned model and evaluation results")
    args = ap.parse_args()

    main(Path(args.data_dir), Path(args.output_dir))