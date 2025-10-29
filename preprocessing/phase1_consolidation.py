import pandas as pd
import os
import json


def main():
    # Configuration - adjust paths to go up to data folder
    DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "Data")
    OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "Data")

    print(f"Data directory: {DATA_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")

    # Load all 4 indices
    print("\n Loading all dataset indices...")

    try:
        rsna_index = pd.read_csv(f"{DATA_DIR}/D_rsna_pneumonia/index.csv")
        print(f"   RSNA loaded: {len(rsna_index)} images")
    except Exception as e:
        print(f"   Error loading RSNA: {e}")
        rsna_index = None

    try:
        indiana_index = pd.read_csv(f"{DATA_DIR}/C_indiana_small/index.csv")
        print(f"   Indiana loaded: {len(indiana_index)} images")
    except Exception as e:
        print(f"   Error loading Indiana: {e}")
        indiana_index = None

    try:
        nih_index = pd.read_csv(f"{DATA_DIR}/B_nih_adult/index.csv")
        print(f"   NIH loaded: {len(nih_index)} images")
    except Exception as e:
        print(f"   Error loading NIH: {e}")
        nih_index = None

    try:
        pediatric_index = pd.read_csv(f"{DATA_DIR}/A_pediatric/index.csv")
        print(f"   Pediatric loaded: {len(pediatric_index)} images")
    except Exception as e:
        print(f"   Error loading Pediatric: {e}")
        pediatric_index = None

    # Store in dictionary
    datasets = {}
    if rsna_index is not None:
        datasets["RSNA"] = rsna_index
    if indiana_index is not None:
        datasets["Indiana"] = indiana_index
    if nih_index is not None:
        datasets["NIH"] = nih_index
    if pediatric_index is not None:
        datasets["Pediatric"] = pediatric_index

    if not datasets:
        print(" No datasets loaded! Check your file paths.")
        exit()

    # Display summary statistics
    print("\n" + "="*70)
    print("DATASET SUMMARY")
    print("="*70)

    total_images = 0
    for name, df in datasets.items():
        print(f"\n{name}:")
        print(f"  Total images: {len(df):,}")
        print(f"  Hospital ID: {df['hospital_id'].unique()[0]}")
        print(f"  Labels: {list(df['label'].unique())}")
        print(f"  Label distribution:")
        for label, count in df['label'].value_counts().items():
            print(f"    - {label}: {count:,}")
        print(f"  Train/Val/Test split:")
        for split, count in df['split'].value_counts().items():
            print(f"    - {split}: {count:,}")
        total_images += len(df)

    # Merge all datasets
    print("\n" + "="*70)
    print("MERGING ALL DATASETS")
    print("="*70)

    global_index = pd.concat(
        [df for df in datasets.values()],
        ignore_index=True
    )

    print(f"\n Global index created!")
    print(f"   Total images: {len(global_index):,}")

    print(f"\nLabel distribution across ALL hospitals:")
    label_dist = global_index.groupby(['hospital_id', 'label']).size().unstack(fill_value=0)
    print(label_dist)

    print(f"\nSplit distribution across ALL hospitals:")
    split_dist = global_index.groupby(['hospital_id', 'split']).size().unstack(fill_value=0)
    print(split_dist)

    # Save global index
    global_index.to_csv(f"{OUTPUT_DIR}/global_index.csv", index=False)
    print(f"\n Saved: {OUTPUT_DIR}/global_index.csv")

    # Create and save label map
    label_map = {
        "Normal": 0,
        "Pneumonia": 1,
        "Other": 2
    }

    with open(f"{OUTPUT_DIR}/global_label_map.json", 'w') as f:
        json.dump(label_map, f, indent=2)

    print(f" Saved: {OUTPUT_DIR}/global_label_map.json")

    # Final statistics
    print("\n" + "="*70)
    print("FINAL STATISTICS")
    print("="*70)
    print(f"\nTotal samples: {len(global_index):,}")
    print(f"Number of hospitals: {global_index['hospital_id'].nunique()}")
    print(f"Hospitals: {list(global_index['hospital_id'].unique())}")

    print(f"\nOverall label distribution:")
    for label in sorted(global_index['label'].unique()):
        count = (global_index['label'] == label).sum()
        percentage = (count / len(global_index)) * 100
        print(f"  {label}: {count:,} ({percentage:.2f}%)")

    print(f"\n Phase 1 Step 1 Complete!")


if __name__ == "__main__":
    main()
