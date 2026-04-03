import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from skimage import measure  
from sklearn.model_selection import train_test_split
from cellpose import models, train, metrics

ROOT       = Path("/scratch/users/pgoue/cellpose_img/001350")
LABELS      = ROOT / "derivatives/labels"
OUTPUT_DIR = "/scratch/users/pgoue/cellpose_img/models"
OUTPUT_NAME = "cpsam_uaxon"
os.makedirs(OUTPUT_DIR, exist_ok=True)

images, masks = [], []

for mask_path in sorted(LABELS.glob("sub-*/micr/*_seg-uaxon-manual.png")):
    sub    = mask_path.parts[-3]          
    stem   = mask_path.name             
    img_name = stem.replace("_seg-uaxon-manual", "")  
    img_path = ROOT / sub / "micr" / img_name

    img  = np.array(Image.open(img_path).convert("L"))
    mask = np.array(Image.open(mask_path))

    if mask.max() == 255:
        from skimage import measure
        mask = measure.label(mask > 127).astype(np.uint16)

    images.append(img)
    masks.append(mask)

idx = list(range(len(images)))
train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=42)

train_imgs  = [images[i] for i in train_idx]
train_masks = [masks[i]  for i in train_idx]
test_imgs   = [images[i] for i in test_idx]
test_masks  = [masks[i]  for i in test_idx]
print(f"Train: {len(train_imgs)} | Test: {len(test_imgs)}")

model = models.CellposeModel(
    gpu=True,
    pretrained_model=os.path.expanduser("~/.cellpose/models/cpsam")
)

new_model_path = train.train_seg(
    model.net,
    train_data=train_imgs,
    train_labels=train_masks,
    test_data=test_imgs,
    test_labels=test_masks,
    normalize=True,
    n_epochs=500,
    learning_rate=1e-4,  
    save_path=OUTPUT_DIR,
    save_every=25,
)

pred_masks = []
finetuned_model = models.CellposeModel(gpu=True, pretrained_model=new_model_path)
for img in test_imgs:
    m, _, _ = finetuned_model.eval(img, diameter=None, channels=[0, 0])
    pred_masks.append(m)


ap, _, _, _ = metrics.average_precision(test_masks, pred_masks)


print(f"AP @ IoU 50% : {ap[:, 0].mean():.3f}   (target > 0.75)")
print(f"AP @ IoU 75% : {ap[:, 1].mean():.3f}   (target > 0.60)")
print(f"AP @ IoU 90% : {ap[:, 2].mean():.3f}")


n = min(4, len(test_imgs))
fig, axes = plt.subplots(n, 3, figsize=(13, 4 * n))
for i in range(n):
    axes[i, 0].imshow(test_imgs[i],  cmap="gray");    axes[i, 0].set_title("Raw TEM");       axes[i, 0].axis("off")
    axes[i, 1].imshow(test_masks[i], cmap="tab20b");  axes[i, 1].set_title("Ground truth");  axes[i, 1].axis("off")
    axes[i, 2].imshow(pred_masks[i], cmap="tab20b");  axes[i, 2].set_title(f"Predicted  AP@50={ap[i,0]:.2f}"); axes[i, 2].axis("off")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "eval_grid.png"), dpi=150)