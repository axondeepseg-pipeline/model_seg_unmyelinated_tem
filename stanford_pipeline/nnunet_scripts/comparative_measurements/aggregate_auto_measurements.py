'''This script reads image-level measurements from CSV files and aggregates them 
into a single CSV file, with a column indicating image fname.
'''

import argparse
import pandas as pd
from pathlib import Path


ORIGINAL_PX_SIZE = 0.00493 / 4  # 0.00493 is for the resized images, but they were initially 4x larger
CTRL_SUBJECTS = {
    '366A': 'CTL1', 
    '368A': 'CTL2', 
    '369B': 'CTL3', 
    '370': 'CTL4', 
    '371': 'CTL5'
}
EXP_SUBJECTS = {
    '367A': 'EXP1', 
    '372': 'EXP2',
    '373C': 'EXP3',
    '374': 'EXP4',
    '375': 'EXP5'
}


def main():
    ap = argparse.ArgumentParser(description="Aggregate image-level ADS measurements into a single CSV file.")
    ap.add_argument("input_dir", help="Directory containing the CSV files with image-level measurements.")

    input_dir = ap.parse_args().input_dir
    all_dfs = []
    for csv_file in Path(input_dir).glob("*_matched_gratios.csv"):
        df = pd.read_csv(csv_file)
        df["fname"] = csv_file.stem  # Add a column with the filename (without extension)
        all_dfs.append(df)
    all_dfs = pd.concat(all_dfs, ignore_index=True)
    all_dfs.drop(columns=['fname_and_coords'], inplace=True)

    matched_df = pd.DataFrame()
    controls = list(CTRL_SUBJECTS.keys())
    all_subjects = controls + list(EXP_SUBJECTS.keys())
    for sub in all_subjects:
        mask = all_dfs["fname"].str.startswith(sub)
        subject_slice = all_dfs[mask]
        is_control_subject = sub in controls

        # -----------------------------#
        # Matched measurements cleanup #
        # -----------------------------#
        subject_slice = subject_slice.drop(columns=['coords'])
        subject_slice['manual_axon_diam'] = subject_slice['axon_diam'] * ORIGINAL_PX_SIZE
        subject_slice['manual_myelin_thickness'] = subject_slice['myelin_thickness'] * ORIGINAL_PX_SIZE
        subject_slice['subject'] = sub
        subject_slice['group'] = 'CTL' if is_control_subject else 'EXP'
        matched_df = pd.concat([matched_df, subject_slice], ignore_index=True)

    matched_df.to_csv("all_matched_measurements.csv", index=False)


if __name__ == "__main__":
    main()