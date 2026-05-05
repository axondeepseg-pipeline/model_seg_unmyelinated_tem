# model_seg_unmyelinated_tem
Model for unmyelinated axon segmentation in TEM images, based on nnUNetv2.

## Model overview
The dataset we used contained files in BIDS format. Images were preprocessed to match nnUNetv2's input format requirements, by combining segmentation classes into a single 5-class label map. The classes are : (unmyelinated + myelinated) axon, myelin, nuclei and process.

## Installation

Clone the github repo in your working directory : 
```
git clone https://github.com/axondeepseg-pipeline/model_seg_unmyelinated_tem.git cd model_seg_unmyelinated_tem
```

Then create the dedicated environment :
```
conda env create -f environment.yml
```

## Usage
To use the checkpoints provided in this repository, please see this temporary solution: https://github.com/axondeepseg/nn-axondeepseg. 

## Data
The datasets can be found in the NeuroPoly internal data server. Look for `datasets/data_axondeepseg_sickkids` and `data_axondeepseg_stanford`.

**TODO** show how to use script to preprocess data


## Training
For more detailed information on the training pipelines, see the `sickkids_pipeline` and `stanford_pipeline` folders.

# Using the Cellpose model for unmyelinated axon segmentation

We also compared our model's results with with [Cellpose](https://cellpose.readthedocs.io/en/latest/), and fine-tuned Cellpose's 'cpsam' model on manually annotated data from the SRF dataset. To install Cellpose, refer to Cellpose's [github](https://github.com/MouseLand/cellpose) and [documentation] (https://cellpose.readthedocs.io/en/latest/installation.html). 
