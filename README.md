# model_seg_unmyelinated_tem
Model for unmyelinated axon segmentation in TEM images, based on nnUNetv2.

## Installation

**Clone the github repo** in your working directory : 
git clone https://github.com/axondeepseg-pipeline/model_seg_unmyelinated_tem.git
cd model_seg_unmyelinated_tem

**Create the environment**
conda env create -f environment.yml

## Model overview
**TODO** image and description

- **Cellpose** : To use Cellpose for mask inference, the files in the working directory should have a name ending with the suffix : '_seg-cellpose'. Then, the data should be prepared to apply and finetune the model, using the associated files.

## Usage
To use the checkpoints provided in this repository, please see this temporary solution: https://github.com/axondeepseg/nn-axondeepseg. 

## Data
The datasets can be found in the NeuroPoly internal data server. Look for `datasets/data_axondeepseg_sickkids` and `data_axondeepseg_stanford`.

**TODO** show how to use script to preprocess data


## Training
For more detailed information on the training pipelines, see the `sickkids_pipeline` and `stanford_pipeline` folders.
