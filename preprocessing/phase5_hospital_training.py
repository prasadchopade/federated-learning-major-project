#!/usr/bin/env python3
"""
PHASE 5: HOSPITAL LOCAL TRAINING SCRIPT (3-CLASS VERSION)
=========================================================
This script is used by ALL hospitals (RSNA, Pediatric, NIH)
Each hospital trains their local model using the global model as starting point.

Label mapping: NORMAL=0, PNEUMONIA=1, OTHER=2

Usage:
    python -m preprocessing.phase5_hospital_training --hospital rsna --round 1 --global-model INITIAL.pt
    python -m preprocessing.phase5_hospital_training --hospital pediatric --round 2 --global-model ROUND1.pt
    python -m preprocessing.phase5_hospital_training --hospital nih --round 3 --global-model ROUND2.pt
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
import os
import argparse
from pathlib import Path
import time
from tqdm import tqdm

# ========================================
# CONFIGURATION
# ========================================

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from preprocessing.phase5_common import HOSPITAL_CONFIGS, ROOT, label_to_class, image_filename

# Training hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
EPOCHS_PER_ROUND = 3  # 3 epochs per federated round
NUM_WORKERS = 0  # Windows: set to 0 to avoid multiprocessing issues. Change to 4 on Linux/Mac
IMAGE_SIZE = 224
NUM_CLASSES = 3  # NORMAL, PNEUMONIA, OTHER

# Paths
DEFAULT_OUTPUT_DIR = ROOT / 'outputs' / 'phase5'

# ========================================
# DATASET CLASS
# ========================================

class PneumoniaDataset(Dataset):
    def __init__(self, csv_file, images_folder, split='train', transform=None):
        """
        Args:
            csv_file: Path to CSV with columns: image_path, hospital_id, label, split
            images_folder: Root folder containing images
            split: 'train', 'val', or 'test'
            transform: Torchvision transforms
        """
        self.df = pd.read_csv(csv_file)
        self.df = self.df[self.df['split'] == split].reset_index(drop=True)
        self.images_folder = images_folder
        self.transform = transform

        print(f"  Loaded {len(self.df)} {split} samples")

        # Print unique labels for debugging
        unique_labels = self.df['label'].unique()
        print(f"  Unique labels in data: {unique_labels}")

    def __len__(self):
        return len(self.df)

    def _map_label_to_int(self, label_str):
        return label_to_class(label_str)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load image
        img_path = os.path.join(self.images_folder, image_filename(row['image_path']))
        image = Image.open(img_path).convert('RGB')

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        # Get label
        label = self._map_label_to_int(row['label'])

        return image, label

# ========================================
# DATA TRANSFORMS
# ========================================

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ========================================
# MODEL CREATION
# ========================================

def create_model(num_classes=3):
    """Create ResNet50 model for 3-class classification"""
    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

# ========================================
# TRAINING FUNCTION
# ========================================

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc='Training')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix({'loss': f'{running_loss/(pbar.n+1):.3f}',
                         'acc': f'{100.*correct/total:.1f}%'})

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

# ========================================
# VALIDATION FUNCTION
# ========================================

def validate(model, val_loader, criterion, device):
    """Validate model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    val_loss = running_loss / len(val_loader)
    val_acc = 100. * correct / total
    return val_loss, val_acc

# ========================================
# MAIN TRAINING FUNCTION
# ========================================

def train_hospital(hospital_name, round_num, *, global_model_path, csv_file=None, images_folder=None, output_dir=DEFAULT_OUTPUT_DIR):
    """
    Train a single hospital for one federated round

    Args:
        hospital_name: 'rsna', 'pediatric', or 'nih'
        round_num: Current round number (1-6)
    """
    print("=" * 80)
    print("PHASE 5: FEDERATED LEARNING - HOSPITAL TRAINING (3-CLASS)")
    print("=" * 80)
    print(f"\nHospital: {HOSPITAL_CONFIGS[hospital_name]['display_name']}")
    print(f"Round: {round_num}/6")
    print(f"Classes: NORMAL=0, PNEUMONIA=1, OTHER=2")

    # Device (prioritize CUDA > MPS > CPU)
    if torch.cuda.is_available():
        device = torch.device('cuda')
        device_name = torch.cuda.get_device_name(0)
        device_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        device_name = 'Apple Metal Performance Shaders'
        device_memory = None
    else:
        device = torch.device('cpu')
        device_name = 'CPU'
        device_memory = None

    print(f"Device: {device} ({device_name})")
    if device_memory:
        print(f"GPU Memory: {device_memory:.1f} GB")

    # Get hospital config
    config = dict(HOSPITAL_CONFIGS[hospital_name])
    config['csv_file'] = csv_file or config['csv_file']
    config['images_folder'] = images_folder or config['images_folder']
    weights_dir = Path(output_dir) / 'hospital_weights'
    results_dir = Path(output_dir) / 'results'

    # ========================================
    # STEP 1: Load Data
    # ========================================
    print(f"\n[STEP 1] Loading {config['display_name']} dataset...")

    train_dataset = PneumoniaDataset(
        csv_file=config['csv_file'],
        images_folder=config['images_folder'],
        split='train',
        transform=train_transform
    )

    val_dataset = PneumoniaDataset(
        csv_file=config['csv_file'],
        images_folder=config['images_folder'],
        split='val',
        transform=val_transform
    )

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise ValueError('Both train and val splits must contain images')

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                             shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                           shuffle=False, num_workers=NUM_WORKERS)

    print(f"   Train samples: {len(train_dataset)}")
    print(f"   Val samples: {len(val_dataset)}")

    # ========================================
    # STEP 2: Load Global Model
    # ========================================
    print(f"\n[STEP 2] Loading global model from Round {round_num-1}...")

    print(f"  Loading supplied global checkpoint: {global_model_path}")

    if not os.path.exists(global_model_path):
        raise FileNotFoundError(f"Global model not found: {global_model_path}")

    # Create model and load weights
    model = create_model(num_classes=NUM_CLASSES)

    # Load with strict=False to handle missing BatchNorm running stats
    missing_keys, unexpected_keys = model.load_state_dict(
        torch.load(global_model_path, map_location=device, weights_only=True),
        strict=False
    )

    allowed_suffixes = ('running_mean', 'running_var', 'num_batches_tracked')
    invalid_missing = [key for key in missing_keys if not key.endswith(allowed_suffixes)]
    if invalid_missing or unexpected_keys:
        raise ValueError(f'Incompatible checkpoint: missing={invalid_missing}, unexpected={unexpected_keys}')

    if missing_keys:
        print(f"    Missing keys (will be initialized): {len(missing_keys)} keys")
        print(f"      (This is normal for BatchNorm running_mean/var)")

    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Loaded global model ({total_params:,} parameters)")

    # ========================================
    # STEP 3: Train Locally
    # ========================================
    print(f"\n[STEP 3] Training on local {config['display_name']} data ({EPOCHS_PER_ROUND} epochs)...")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(1, EPOCHS_PER_ROUND + 1):
        print(f"\n  Epoch {epoch}/{EPOCHS_PER_ROUND}:")

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        print(f"    Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"    Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

    print(f"\n   Training complete! Final val accuracy: {val_acc:.2f}%")

    # ========================================
    # STEP 4: Save Local Weights
    # ========================================
    print(f"\n[STEP 4] Saving local weights...")

    output_dir = weights_dir / f"round_{round_num}"
    os.makedirs(output_dir, exist_ok=True)

    output_path = f"{output_dir}/{hospital_name}_round{round_num}_weights.pt"
    torch.save(model.state_dict(), output_path)

    file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
    print(f"   Saved: {output_path} ({file_size:.1f} MB)")

    print(f"Local checkpoint ready for the coordinator: {output_path}")

    # Save training log
    log_path = results_dir / f"{hospital_name}_round{round_num}_log.txt"
    os.makedirs(results_dir, exist_ok=True)
    with open(log_path, 'w') as f:
        f.write(f"Hospital: {config['display_name']}\n")
        f.write(f"Round: {round_num}\n")
        f.write(f"Final Val Accuracy: {val_acc:.2f}%\n")
        f.write(f"Final Val Loss: {val_loss:.4f}\n")
        f.write(f"Classes: NORMAL=0, PNEUMONIA=1, OTHER=2\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Phase 5: Hospital Training (3-class)')
    parser.add_argument('--hospital', type=str, required=True,
                       choices=['rsna', 'pediatric', 'nih'],
                       help='Hospital name (rsna/pediatric/nih)')
    parser.add_argument('--round', type=int, required=True,
                       help='Round number (1-6)')

    parser.add_argument('--global-model', type=Path, required=True,
                        help='Shared starting checkpoint for this round')
    parser.add_argument('--csv', type=Path, help='Override the hospital manifest')
    parser.add_argument('--images-dir', type=Path, help='Override the image folder')
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    # Validate round number
    if not 1 <= args.round <= 6:
        raise ValueError("Round number must be between 1 and 6")

    # Run training
    start_time = time.time()
    train_hospital(args.hospital, args.round, global_model_path=args.global_model,
                   csv_file=args.csv, images_folder=args.images_dir, output_dir=args.output_dir)
    elapsed = time.time() - start_time

    print(f"\n  Total time: {elapsed/60:.1f} minutes")
