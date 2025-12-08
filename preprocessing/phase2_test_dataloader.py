"""Load a real batch from each requested hospital to check paths and labels."""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data_loader import HospitalDataManager


def main(default_hospital=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hospital', default=default_hospital)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'Data')
    args = parser.parse_args()
    partition = pd.read_csv(args.data_dir / 'partition_non_iid.csv', low_memory=False)
    labels = json.loads((args.data_dir / 'global_label_map.json').read_text())
    hospitals = [args.hospital] if args.hospital else partition['hospital_id'].unique()
    for hospital in hospitals:
        manager = HospitalDataManager(hospital, partition, labels, num_workers=0)
        loader = manager.get_dataloader('train')
        if loader is None:
            raise ValueError(f'No training split for {hospital}')
        images, targets, paths = next(iter(loader))
        print(f'{hospital}: {tuple(images.shape)}, labels={targets.tolist()[:8]}')


if __name__ == '__main__':
    main()
