import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split


def main():
    DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "Data")

    print("="*70)
    print("PHASE 1 STEP 3: CREATE IID AND NON-IID PARTITIONS")
    print("="*70)

    # Load global index
    global_index = pd.read_csv(f"{DATA_DIR}/global_index.csv")

    print(f"\nLoaded global index: {len(global_index)} images")

    # ============================================================================
    # PARTITION 1: NON-IID (Natural - Keep hospitals separate)
    # ============================================================================
    print("\n" + "-"*70)
    print("PARTITION 1: NON-IID (Realistic Healthcare Scenario)")
    print("-"*70)

    non_iid_partition = global_index.copy()
    non_iid_partition['partition_type'] = 'non_iid'
    non_iid_partition['client_id'] = non_iid_partition['hospital_id']  # Each hospital is a client

    print(f"\nNon-IID partitioning (each hospital keeps its data):")
    for hospital in non_iid_partition['hospital_id'].unique():
        hospital_data = non_iid_partition[non_iid_partition['hospital_id'] == hospital]
        print(f"\n  {hospital}:")
        print(f"    Total samples: {len(hospital_data):,}")
        print(f"    Label distribution:")
        for label, count in hospital_data['label'].value_counts().items():
            pct = (count / len(hospital_data)) * 100
            print(f"      - {label}: {count:,} ({pct:.2f}%)")

    # ============================================================================
    # PARTITION 2: IID (Balanced - Redistribute to equalize distributions)
    # ============================================================================
    print("\n" + "-"*70)
    print("PARTITION 2: IID (Balanced Baseline)")
    print("-"*70)

    # Pool all images and redistribute equally across 4 hospitals
    np.random.seed(42)
    iid_partition = global_index.copy()

    # Shuffle globally
    iid_partition = iid_partition.sample(frac=1, random_state=42).reset_index(drop=True)

    # Divide into 4 equal parts (one per hospital)
    n_hospitals = 4
    samples_per_hospital = len(iid_partition) // n_hospitals

    hospital_ids = ['D_rsna_pneumonia', 'C_indiana_small', 'B_nih_adult', 'D_pediatric_pneumonia']
    client_id_mapping = []

    for i, hospital in enumerate(hospital_ids):
        start_idx = i * samples_per_hospital
        if i == n_hospitals - 1:  # Last hospital gets remaining samples
            end_idx = len(iid_partition)
        else:
            end_idx = start_idx + samples_per_hospital

        iid_partition.loc[start_idx:end_idx - 1, 'hospital_id'] = hospital
        iid_partition.loc[start_idx:end_idx - 1, 'client_id'] = i

    iid_partition['partition_type'] = 'iid'

    print(f"\nIID partitioning (balanced redistribution):")
    for hospital in iid_partition['hospital_id'].unique():
        hospital_data = iid_partition[iid_partition['hospital_id'] == hospital]
        print(f"\n  {hospital}:")
        print(f"    Total samples: {len(hospital_data):,}")
        print(f"    Label distribution:")
        for label, count in hospital_data['label'].value_counts().items():
            pct = (count / len(hospital_data)) * 100
            print(f"      - {label}: {count:,} ({pct:.2f}%)")

    # ============================================================================
    # SAVE PARTITIONS
    # ============================================================================
    print("\n" + "="*70)
    print("SAVING PARTITIONS")
    print("="*70)

    # Save non-IID
    non_iid_partition.to_csv(f"{DATA_DIR}/partition_non_iid.csv", index=False)
    print(f" Saved: {DATA_DIR}/partition_non_iid.csv")

    # Save IID
    iid_partition.to_csv(f"{DATA_DIR}/partition_iid.csv", index=False)
    print(f" Saved: {DATA_DIR}/partition_iid.csv")

    # Create summary statistics file
    summary = {
        "total_samples": len(global_index),
        "num_hospitals": 4,
        "num_classes": 3,
        "classes": ["Normal", "Pneumonia", "Other"],
        "label_distribution": {
            "Normal": int((global_index['label'] == 'Normal').sum()),
            "Pneumonia": int((global_index['label'] == 'Pneumonia').sum()),
            "Other": int((global_index['label'] == 'Other').sum()),
        },
        "hospitals": {
            "D_rsna_pneumonia": {
                "samples": int((global_index['hospital_id'] == 'D_rsna_pneumonia').sum()),
                "type": "Large, Pneumonia-specialized"
            },
            "C_indiana_small": {
                "samples": int((global_index['hospital_id'] == 'C_indiana_small').sum()),
                "type": "Small, Low-resource"
            },
            "B_nih_adult": {
                "samples": int((global_index['hospital_id'] == 'B_nih_adult').sum()),
                "type": "Large, Adult, Diverse"
            },
            "D_pediatric_pneumonia": {
                "samples": int((global_index['hospital_id'] == 'D_pediatric_pneumonia').sum()),
                "type": "Pediatric, High pneumonia rate"
            }
        }
    }

    import json
    with open(f"{DATA_DIR}/dataset_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f" Saved: {DATA_DIR}/dataset_summary.json")

    print("\n" + "="*70)
    print("PHASE 1 STEP 3 COMPLETE!")
    print("="*70)
    print("\n You now have:")
    print("   1. partition_non_iid.csv - Realistic healthcare scenario")
    print("   2. partition_iid.csv - Balanced baseline for comparison")
    print("   3. dataset_summary.json - Dataset statistics")


if __name__ == "__main__":
    main()
