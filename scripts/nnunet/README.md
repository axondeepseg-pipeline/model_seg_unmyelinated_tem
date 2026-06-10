# How to use these scripts
This section explains how to use the content of this folder.

## Pipeline
The whole pipeline can be reproduced using the scripts `setup_nnunet.sh` and 
`train_nnunet.sh`. For a more granular control over the process, you can also 
invoke the commands in these scripts yourself.

## Data
Initially, this study used the dataset located here: https://dandiarchive.org/dandiset/001350/0.250511.1527.
As of today (_2026-06-10_), we added a second dataset to improve generalization (not yet 
public).

Datasets are initially in BIDS format. To convert the data into a format compatible 
with nnunetv2, the script `convert_bids_to_nnunetv2_format.py`.

### Combining datasets
First, create the following directory structure:
```bash
mkdir MY_FOLDER MY_FOLDER/nnUNet_raw MY_FOLDER/nnUNet_preprocessed
tree MY_FOLDER/
MY_FOLDER/
├── nnUNet_preprocessed
└── nnUNet_raw

3 directories, 0 files
```

Then, insert all your datasets in `MY_FOLDER/nnUNet_raw`. The final step to combine 
datasets is to call the script `aggregate_data.py`:
```bash
$ python aggregate_data.py -h
usage: aggregate_data.py [-h] [--nnunet_dir NNUNET_DIR] [--dataset_ids [DATASET_IDS ...]] [--name NAME] [--description DESCRIPTION] [--k K]

options:
  -h, --help            show this help message and exit
  --nnunet_dir NNUNET_DIR
                        Path to nnunet directory, where the existing datasets are stored and where the new dataset will be created.
  --dataset_ids [DATASET_IDS ...]
                        List of dataset indices to be combined. Default: all datasets
  --name NAME           Name of the aggregated dataset. Default: 'Dataset043_TEM_UNMYELINATED_AGG'
  --description DESCRIPTION
                        Description of the aggregated dataset. Default: 'Combined datasets for myelinated and unmyelinated axon segmentation'
  --k K                 Number of folds for cross-validation. Default: 5

```

## Preprocessing
For the rest of the process, you will need to setup the 3 nnunet environment variables.
It should look something like this:
```bash
RESULTS_DIR=$(realpath MY_FOLDER)
export nnUNet_raw="$RESULTS_DIR/nnUNet_raw"
export nnUNet_preprocessed="$RESULTS_DIR/nnUNet_preprocessed"
export nnUNet_results="$RESULTS_DIR/nnUNet_results"
```

Use nnunet's preprocessing utility directly:
```bash
nnUNetv2_plan_and_preprocess -d <DATASET_ID> --verify_dataset_integrity -c "2d"
```

## Training
Again, use nnunet's training utility:
```bash
nnUNetv2_train <DATASET_ID> "2d" <FOLD_ID>
```