import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from typing import Tuple, List, Dict

class MedicalImageDataset(Dataset):
    """PyTorch Dataset for medical images with standardized preprocessing"""

    def __init__(
        self,
        image_paths: List[str],
        labels: List[int],
        image_size: int = 224,
        normalize: bool = True
    ):
        # Resolve image paths to actual file locations
        if len(image_paths) != len(labels):
            raise ValueError("Each image path must have one label")
        self.image_paths = self._resolve_paths(image_paths)
        self.labels = labels
        self.image_size = image_size

        # Define transforms
        self.transforms = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Grayscale(num_output_channels=3),
        ])

        if normalize:
            self.transforms.transforms.append(
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            )

    def _resolve_paths(self, image_paths: List[str]) -> List[str]:
        """
        Resolve image paths from partition CSV to actual file locations.
        Handles path mismatches between CSV paths and extracted folder structure.
        """
        resolved_paths = []
        base_dir = os.path.dirname(os.path.dirname(__file__))

        for path in image_paths:
            resolved = None

            # Try original path first (absolute or relative)
            if os.path.exists(path):
                resolved = path
            else:
                # Try relative to project root
                abs_path = os.path.join(base_dir, path)
                if os.path.exists(abs_path):
                    resolved = abs_path
                else:
                    # For NIH: map nih/images-224/images-224/*.png to preprocessed_nih/images/*.png
                    if 'nih/images-224/images-224/' in path or 'B_nih_adult' in path:
                        filename = os.path.basename(path)
                        nih_path = os.path.join(base_dir, 'preprocessed_nih', 'images', filename)
                        if os.path.exists(nih_path):
                            resolved = nih_path

                    # For RSNA: map rsna/images-224/images-224/*.png to preprocessed_rsna/images/*.png
                    elif 'rsna/images-224/images-224/' in path or 'D_rsna_pneumonia' in path:
                        filename = os.path.basename(path)
                        rsna_path = os.path.join(base_dir, 'preprocessed_rsna', 'images', filename)
                        if os.path.exists(rsna_path):
                            resolved = rsna_path

                    # For Pediatric: map pediatric/images-224/images-224/*.png to preprocessed_pediatric/images/*.png
                    elif 'pediatric/images-224/images-224/' in path or 'A_pediatric/images-224/images-224/' in path or 'D_pediatric_pneumonia' in path:
                        filename = os.path.basename(path)
                        pediatric_path = os.path.join(base_dir, 'preprocessed_pediatric', 'images', filename)
                        if os.path.exists(pediatric_path):
                            resolved = pediatric_path

                    # For Indiana: map indiana/images-224/images-224/*.png to preprocessed_indiana/images/*.png
                    elif 'indiana/images-224/images-224/' in path or 'C_indiana_small' in path:
                        filename = os.path.basename(path)
                        indiana_path = os.path.join(base_dir, 'preprocessed_indiana', 'images', filename)
                        if os.path.exists(indiana_path):
                            resolved = indiana_path

            # Use resolved path if found, otherwise keep original (will error gracefully)
            resolved_paths.append(resolved if resolved else path)

        return resolved_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            image_path = self.image_paths[idx]
            image = Image.open(image_path).convert('RGB')
            image = self.transforms(image)
            label = self.labels[idx]
            return image, label, self.image_paths[idx]
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"Could not load image: {self.image_paths[idx]}") from exc


class HospitalDataManager:
    """Manages data for a single hospital (client) in federated learning"""

    def __init__(
        self,
        hospital_id: str,
        partition_df: pd.DataFrame,
        label_map: Dict[str, int],
        image_size: int = 224,
        batch_size: int = 32,
        num_workers: int = 0
    ):
        self.hospital_id = hospital_id
        self.label_map = label_map
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Filter data for this hospital
        self.hospital_data = partition_df[
            partition_df['hospital_id'] == hospital_id
        ].reset_index(drop=True)

        print(f"\n{'='*70}")
        print(f"Hospital: {hospital_id}")
        print(f"{'='*70}")
        print(f"Total samples: {len(self.hospital_data):,}")

        # Create dataloaders for train/val/test
        self.dataloaders = {}
        self._create_dataloaders()

    def _create_dataloaders(self):
        """Create PyTorch dataloaders for each split"""

        for split in ['train', 'val', 'test']:
            split_data = self.hospital_data[
                self.hospital_data['split'] == split
            ]

            if len(split_data) == 0:
                print(f"  {split}: 0 samples")
                self.dataloaders[split] = None
                continue

            # Prepare data
            image_paths = split_data['image_path'].values.tolist()
            mapped_labels = split_data['label'].map(self.label_map)
            if mapped_labels.isna().any():
                unknown = split_data.loc[mapped_labels.isna(), 'label'].unique().tolist()
                raise ValueError(f"Unmapped labels for {self.hospital_id}: {unknown}")
            labels = mapped_labels.astype(int).values

            # Create dataset and dataloader
            dataset = MedicalImageDataset(
                image_paths=image_paths,
                labels=labels,
                image_size=self.image_size,
                normalize=True
            )

            dataloader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=(split == 'train'),
                num_workers=self.num_workers,
                pin_memory=True
            )

            self.dataloaders[split] = dataloader

            # Print statistics
            label_counts = split_data['label'].value_counts()
            print(f"\n  {split.upper()}: {len(split_data):,} samples")
            for label, count in label_counts.items():
                pct = (count / len(split_data)) * 100
                print(f"    - {label}: {count:,} ({pct:.2f}%)")

    def get_dataloader(self, split: str = 'train'):
        """Get dataloader for a specific split"""
        return self.dataloaders.get(split)

    def get_all_dataloaders(self):
        """Get all dataloaders"""
        return self.dataloaders

    def get_local_data_size(self, split: str = 'train') -> int:
        """Get number of samples in a split"""
        split_data = self.hospital_data[
            self.hospital_data['split'] == split
        ]
        return len(split_data)

    def get_label_distribution(self, split: str = 'train') -> Dict[str, int]:
        """Get label distribution for a split"""
        split_data = self.hospital_data[
            self.hospital_data['split'] == split
        ]
        return split_data['label'].value_counts().to_dict()
