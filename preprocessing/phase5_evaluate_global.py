#!/usr/bin/env python3
"""
PHASE 5: EVALUATE FINAL GLOBAL MODEL (3-CLASS)
==============================================
Evaluates an explicitly supplied checkpoint on the hospital test split.

Classes: NORMAL=0, PNEUMONIA=1, OTHER=2
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import os
import argparse
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from pathlib import Path

# -------------------
# CONFIG
# -------------------

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from preprocessing.phase5_common import HOSPITAL_CONFIGS, ROOT, label_to_class, image_filename

BATCH_SIZE = 32
IMAGE_SIZE = 224
NUM_WORKERS = 0  # Set to 0 for Windows compatibility
NUM_CLASSES = 3  # NORMAL, PNEUMONIA, OTHER

# -------------------
# DATASET
# -------------------

class PneumoniaDataset(Dataset):
    def __init__(self, csv_file, images_folder, split='test', transform=None):
        self.df = pd.read_csv(csv_file)
        self.df = self.df[self.df['split'] == split].reset_index(drop=True)
        self.images_folder = images_folder
        self.transform = transform
        print(f"Loaded {len(self.df)} {split} samples")

    def _map_label_to_int(self, label_str):
        return label_to_class(label_str)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.images_folder, image_filename(row['image_path']))
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = self._map_label_to_int(row['label'])
        return image, label

# -------------------
# TRANSFORMS
# -------------------

test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# -------------------
# MODEL
# -------------------

def create_model(num_classes=3):
    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

# -------------------
# MAIN
# -------------------

def main():
    parser = argparse.ArgumentParser(description='Phase 5: Evaluate Global Model (3-class)')
    parser.add_argument('--hospital', type=str, default='nih',
                       choices=['rsna', 'pediatric', 'nih'],
                       help='Target hospital for evaluation (default: nih)')
    parser.add_argument('--model', type=Path, required=True,
                        help='Exact checkpoint to evaluate; no round is assumed')
    parser.add_argument('--csv', type=Path, help='Override the hospital manifest')
    parser.add_argument('--images-dir', type=Path, help='Override the image folder')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs' / 'phase5' / 'results')
    args = parser.parse_args()

    hospital_config = dict(HOSPITAL_CONFIGS[args.hospital])
    hospital_config['csv_file'] = args.csv or hospital_config['csv_file']
    hospital_config['images_folder'] = args.images_dir or hospital_config['images_folder']
    print(f"Evaluating on: {hospital_config['display_name']}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")

    # Data
    dataset = PneumoniaDataset(hospital_config['csv_file'], hospital_config['images_folder'],
                              split='test', transform=test_transform)
    if len(dataset) == 0:
        raise ValueError('The test split contains no images')
    loader = DataLoader(dataset, batch_size=BATCH_SIZE,
                        shuffle=False, num_workers=NUM_WORKERS)

    # Model
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Global model not found: {args.model}")
    print(f"Loading model: {args.model}")

    model = create_model(num_classes=NUM_CLASSES)
    state = torch.load(args.model, map_location=device, weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise ValueError(f'Incomplete checkpoint: missing={missing}, unexpected={unexpected}')

    model = model.to(device)
    model.eval()

    # Evaluation
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(1).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # Info about which labels are present
    present_labels = np.unique(np.concatenate([all_labels, all_preds]))
    print("Unique labels present:", present_labels)

    labels = [0, 1, 2]
    target_names = ['NORMAL (0)', 'PNEUMONIA (1)', 'OTHER (2)']

    print(f"\nClassification report ({hospital_config['display_name']} test, 3-class):")
    print(classification_report(
        all_labels,
        all_preds,
        labels=labels,
        target_names=target_names,
        zero_division=0
    ))

    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(all_labels, all_preds, labels=labels))

    # Save to file
    os.makedirs(args.output_dir, exist_ok=True)
    result_file = args.output_dir / f'phase5_{args.hospital}_test_report.txt'
    with open(result_file, 'w', encoding='utf-8') as f:
        f.write(f'Checkpoint: {args.model}\n')
        f.write(f'Manifest: {hospital_config["csv_file"]}\n')
        f.write(f"Classification report ({hospital_config['display_name']} test, 3-class):\n")
        f.write(classification_report(
            all_labels,
            all_preds,
            labels=labels,
            target_names=target_names,
            zero_division=0
        ))
        f.write("\nConfusion matrix (rows=true, cols=pred):\n")
        f.write(str(confusion_matrix(all_labels, all_preds, labels=labels)))

    print(f"\nSaved detailed report to {result_file}")

if __name__ == "__main__":
    main()
