import pandas as pd
import os


def main():
    DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "Data")

    print("="*70)
    print("PHASE 1 STEP 2: STANDARDIZE LABELS")
    print("="*70)

    # Load global index
    global_index = pd.read_csv(f"{DATA_DIR}/global_index.csv")

    print(f"\nBefore standardization:")
    print(f"Unique labels: {global_index['label'].unique()}")
    print(f"Label counts:\n{global_index['label'].value_counts()}")

    # Standardize labels: convert all to lowercase then capitalize first letter
    # This converts NORMAL -> Normal, PNEUMONIA -> Pneumonia
    global_index['label'] = global_index['label'].str.lower().str.capitalize()

    print(f"\nAfter standardization:")
    print(f"Unique labels: {global_index['label'].unique()}")
    print(f"Label counts:\n{global_index['label'].value_counts()}")

    print(f"\nLabel distribution across hospitals:")
    print(global_index.groupby(['hospital_id', 'label']).size().unstack(fill_value=0))

    # Save standardized global index
    global_index.to_csv(f"{DATA_DIR}/global_index.csv", index=False)

    print(f"\n Standardized labels saved to: {DATA_DIR}/global_index.csv")

    # Display final statistics
    print("\n" + "="*70)
    print("FINAL STANDARDIZED STATISTICS")
    print("="*70)

    print(f"\nTotal samples: {len(global_index):,}")
    print(f"\nGlobal label distribution:")
    for label in sorted(global_index['label'].unique()):
        count = (global_index['label'] == label).sum()
        percentage = (count / len(global_index)) * 100
        print(f"  {label}: {count:,} ({percentage:.2f}%)")

    print(f"\n Phase 1 Step 2 Complete!")


if __name__ == "__main__":
    main()
