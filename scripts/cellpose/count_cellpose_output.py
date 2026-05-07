from cellpose import io
import argparse
from pathlib import Path
import pandas as pd


def count_axons_in_cellpose_output(mask_path: str) -> int:
    """
    Count the number of cells in a Cellpose output mask.
    Parameters:
    - mask_path: Path to the Cellpose output mask (should be a .png file where each cell instance is labeled with a unique integer).
    Returns:
    - The number of cells in the mask.
    """
    mask = io.imread(mask_path)
    count = max(mask.flatten())
    print(f"Counted {count} cells in mask {mask_path}")
    return count

def main(mask_dir: Path):
    myelinated_masks = list(mask_dir.glob('*_pred-myelinated.png'))
    assert all([Path(str(p).replace('myelinated', 'unmyelinated')).exists() for p in myelinated_masks]) , \
        "Each myelinated mask must have a corresponding unmyelinated mask."
    
    counts = pd.DataFrame(columns=['image', 'myelinated_count', 'unmyelinated_count'])
    for myelinated_mask in myelinated_masks:
        unmyelinated_mask = Path(str(myelinated_mask).replace('myelinated', 'unmyelinated'))
        myelinated_count = count_axons_in_cellpose_output(str(myelinated_mask))
        unmyelinated_count = count_axons_in_cellpose_output(str(unmyelinated_mask))
        counts.loc[len(counts)] = [
            myelinated_mask.stem.replace('_pred-myelinated', ''), 
            myelinated_count, 
            unmyelinated_count
        ]
        # new_row = pd.DataFrame({
        #     'image': myelinated_mask.stem.replace('_pred-myelinated', ''),
        #     'myelinated_count': myelinated_count,
        #     'unmyelinated_count': unmyelinated_count
        # })
        # counts = pd.concat([counts, new_row], ignore_index=True)
    counts.to_csv(mask_dir / 'axon_counts.csv', index=False)
    print(f"Counts saved to {mask_dir / 'axon_counts.csv'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Count the number of cells in a Cellpose output mask.')
    parser.add_argument('-i', type=str, required=True, help='Path to the directory containing cellpose output masks.')
    args = parser.parse_args()
    
    main(Path(args.i))