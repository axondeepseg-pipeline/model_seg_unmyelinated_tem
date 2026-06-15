# Segment all axons in EM images

![seg-animation](assets/unmyelinated-model.gif)

Model for axon segmentation in TEM images. Both myelinated and unmyelinated axons are processed, allowing users to compute the percentage of myelinated axons without manual counting.

## Repo overview
Two model architectures were considered for this project:
- `scripts/nnunet/` contains the material to train a nnUNetv2 model (package version 2.2.1).
- `scripts/cellpose` contains scripts to train and evaluate a Cellpose4 model (cellpose-SAM).

Model weights are available as release assets.

## Usage
The nnunet models can be downloaded and used in the [AxonDeepSeg](https://github.com/axondeepseg/axondeepseg) software. Use the following AxonDeepSeg command to download the model locally:
```
download_model -m unmyelinated-TEM
```

## Data
The main dataset used in this study is available for download here: https://doi.org/10.48324/dandi.001350/0.250511.1527