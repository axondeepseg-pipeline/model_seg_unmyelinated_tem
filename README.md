# model_seg_unmyelinated_tem
Model for unmyelinated axon segmentation in TEM images, based on nnUNetv2.

## Model overview
**TODO** image and description

## Usage
To use the checkpoints provided in this repository, please see this temporary solution: https://github.com/axondeepseg/nn-axondeepseg. 

## Data
The datasets can be found in the NeuroPoly internal data server. Look for `datasets/data_axondeepseg_sickkids` and `data_axondeepseg_stanford`.

**TODO** show how to use script to preprocess data


## Training
For more detailed information on the training pipelines, see the `sickkids_pipeline` and `stanford_pipeline` folders.

# Using the Cellpose model for unmyelinated axon segmentation

This model automatically detects and outlines **unmyelinated axons** in TEM (transmission electron microscopy) images. It is based on [Cellpose](https://cellpose.readthedocs.io/en/latest/) and was fine-tuned on manually annotated TEM data from the Stanford dataset.

---

## What you need

### 1. The trained model file

Download the model file (`cpsam_uaxon`) from the [Releases page](https://github.com/YOUR-ORG/YOUR-REPO/releases) of this repository and save it somewhere on your computer, for example:

```
~/models/cpsam_uaxon
```

### 2. Your TEM images

Images must be in `.png` format. Organize them as one folder per subject, with images inside a `micr/` subfolder:

```
your_dataset/
├── sub-01/
│   └── micr/
│       ├── image1.png
│       └── image2.png
├── sub-02/
│   └── micr/
│       └── ...
```

### 3. Python environment

Install the required packages:

```bash
pip install cellpose tqdm scikit-image torch
```

---

## How to run

Open `cpsam_SRF.py` in a text editor and update the two lines at the top:

```python
ROOT_DIR = "/path/to/your_dataset"        # ← folder containing your sub-* folders
```

Also update the path to your model file inside the script:

```python
pretrained_model = "/path/to/cpsam_uaxon"
```

Then run the script from a terminal:

```bash
python cpsam_SRF.py
```

---

## Output

For each image, the script creates a segmentation mask saved as `<image_name>_seg-cellpose.png` inside an `output/` folder next to each subject's images:

```
your_dataset/
└── sub-01/
    ├── micr/
    │   └── image1.png
    └── output/
        └── image1_seg-cellpose.png    ← segmentation result
```

Each output mask is a grayscale image where each detected axon is assigned a unique integer value (1, 2, 3, …). Background pixels are 0. This format is compatible with ImageJ/Fiji and AxonDeepSeg.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `No module named cellpose` | Run `pip install cellpose` |
| `CUDA out of memory` | Your GPU memory is insufficient — add `gpu=False` in the script to use CPU instead (slower) |
| No output files created | Check that your images are `.png` and that the `micr/` subfolder exists |
| Model file not found | Verify the path to `cpsam_uaxon` in the script |
