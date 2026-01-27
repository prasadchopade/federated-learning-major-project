#!/usr/bin/env python3
"""
Phase 4: RL-based Client Selection using Q-Learning (NIH-only version)
========================================================================

This script learns client participation schedules using Q-learning
with simulated heterogeneous virtual clients from the NIH dataset.

We partition NIH into 3 virtual clients using Dirichlet distribution
to simulate non-IID multi-hospital scenario.

Label mapping:
    NORMAL = 0
    PNEUMONIA = 1
    OTHER = 2

Usage:
    python -m preprocessing.phase4_modified --episodes 30 --local_epochs 2
"""

import os
import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from preprocessing.phase5_common import ROOT, label_to_class, image_filename
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import models, transforms
from PIL import Image
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt

# ============================================================
# Device
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available()
                      else "cpu")

print(f"Using device: {device}")

# ============================================================
# DATASET
# ============================================================

class PneumoniaDataset(Dataset):
    """NIH Dataset with unified 3-class mapping"""

    def __init__(self, csv_path, images_dir, split='train', transform=None):
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df['split'] == split].reset_index(drop=True)
        self.images_dir = images_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def _map_label_to_class(self, label_str):
        return label_to_class(label_str)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        filename = image_filename(row['image_path'])
        img_path = os.path.join(self.images_dir, filename)

        image = Image.open(img_path).convert('RGB')
        label = self._map_label_to_class(row['label'])

        if self.transform:
            image = self.transform(image)

        return image, label

# ============================================================
# NON-IID SPLIT USING DIRICHLET
# ============================================================

def create_non_iid_splits(dataset, num_clients=3, alpha=0.5):

    labels = dataset.df["label"].map(label_to_class).tolist()
    num_classes = 3

    class_indices = {c: [] for c in range(num_classes)}
    for idx, label in enumerate(labels):
        class_indices[label].append(idx)

    client_indices = {i: [] for i in range(num_clients)}

    for c in range(num_classes):
        indices = class_indices[c]
        np.random.shuffle(indices)

        proportions = np.random.dirichlet([alpha] * num_clients)
        proportions = (np.cumsum(proportions) * len(indices)).astype(int)[:-1]

        splits = np.split(indices, proportions)

        for client_idx, split in enumerate(splits):
            client_indices[client_idx].extend(split.tolist())

    for client_idx in client_indices:
        np.random.shuffle(client_indices[client_idx])

    return client_indices

# ============================================================
# MODEL
# ============================================================

def create_model():
    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 3)
    return model.to(device)

# ============================================================
# TRAINING
# ============================================================

def train_local_model(model, train_loader, epochs=2, lr=1e-4):

    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for _ in range(epochs):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    return model.state_dict()

def evaluate_model(model, val_loader):

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100.0 * correct / total if total > 0 else 0.0

# ============================================================
# AGGREGATION
# ============================================================

def aggregate_weights(local_weights_list, client_weights):

    global_dict = {}

    for key in local_weights_list[0].keys():
        global_dict[key] = sum(
            w * local_weights[key]
            for w, local_weights in zip(client_weights, local_weights_list)
        )

    return global_dict

# ============================================================
# ACTION SPACE
# ============================================================

ACTIONS = [
    [1, 1, 1],  # all
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 1, 0],
    [1, 0, 1],
    [0, 1, 1],
]

# ============================================================
# Q-LEARNING AGENT
# ============================================================

class QLearningAgent:

    def __init__(self, num_rounds=6, num_actions=7,
                 alpha=0.1, gamma=0.9, epsilon=0.3):

        self.num_rounds = num_rounds
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = defaultdict(lambda: np.zeros(num_actions))

    def select_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.num_actions)
        else:
            return np.argmax(self.Q[state])

    def update(self, state, action, reward, next_state):
        current_q = self.Q[state][action]
        max_next_q = np.max(self.Q[next_state])
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.Q[state][action] = new_q

    def get_best_policy(self):
        policy = {}
        for state in range(1, self.num_rounds + 1):
            policy[state] = int(np.argmax(self.Q[state]))
        return policy

# ============================================================
# FEDERATED SIMULATION
# ============================================================

def run_federated_learning(train_loaders,
                           val_loader_global,
                           action_sequence,
                           adaptive_weights,
                           num_rounds=6,
                           local_epochs=2):

    global_model = create_model()

    for round_num in range(1, num_rounds + 1):

        action = action_sequence[round_num - 1]
        selected_clients = [i for i in range(3) if ACTIONS[action][i] == 1]

        if not selected_clients:
            continue

        local_weights_list = []
        local_client_weights = []

        for client_idx in selected_clients:

            client_model = create_model()
            client_model.load_state_dict(global_model.state_dict())

            local_weights = train_local_model(
                client_model,
                train_loaders[client_idx],
                epochs=local_epochs
            )

            local_weights_list.append(local_weights)
            local_client_weights.append(adaptive_weights[client_idx])

        total = sum(local_client_weights)
        normalized_weights = [w / total for w in local_client_weights]

        global_weights = aggregate_weights(local_weights_list, normalized_weights)
        global_model.load_state_dict(global_weights)

    final_accuracy = evaluate_model(global_model, val_loader_global)
    return final_accuracy

# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=30)
    parser.add_argument('--local_epochs', type=int, default=2)
    parser.add_argument('--dirichlet_alpha', type=float, default=0.5)
    parser.add_argument('--csv', type=Path, default=ROOT / 'preprocessed_nih' / 'B_nih_adult_index.csv')
    parser.add_argument('--images-dir', type=Path, default=ROOT / 'preprocessed_nih' / 'images')
    args = parser.parse_args()
    if args.episodes < 1 or args.local_epochs < 1 or args.dirichlet_alpha <= 0:
        parser.error('Episodes, local epochs, and Dirichlet alpha must be positive')

    print("=" * 80)
    print("NIH RL-BASED CLIENT SELECTION")
    print("=" * 80)

    csv_path = args.csv
    images_dir = args.images_dir

    if not os.path.exists(csv_path):
        print("ERROR: NIH CSV not found.")
        sys.exit(1)

    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    train_dataset = PneumoniaDataset(csv_path, images_dir, 'train', transform_train)
    val_dataset = PneumoniaDataset(csv_path, images_dir, 'val', transform_val)

    client_splits = create_non_iid_splits(
        train_dataset,
        num_clients=3,
        alpha=args.dirichlet_alpha
    )

    if any(not indices for indices in client_splits.values()):
        raise ValueError('A sampled virtual client is empty; change the Dirichlet alpha')

    train_loaders = {
        i: DataLoader(Subset(train_dataset, idxs),
                      batch_size=32,
                      shuffle=True)
        for i, idxs in client_splits.items()
    }

    val_loader_global = DataLoader(val_dataset,
                                   batch_size=32,
                                   shuffle=False)

    adaptive_weights = {0: 0.35, 1: 0.30, 2: 0.35}

    agent = QLearningAgent()

    episode_rewards = []

    for episode in range(args.episodes):

        action_sequence = [
            agent.select_action(round_num)
            for round_num in range(1, 7)
        ]

        final_accuracy = run_federated_learning(
            train_loaders,
            val_loader_global,
            action_sequence,
            adaptive_weights,
            num_rounds=6,
            local_epochs=args.local_epochs
        )

        episode_rewards.append(final_accuracy)

        for round_num in range(1, 7):
            action = action_sequence[round_num - 1]
            next_state = round_num + 1 if round_num < 6 else 6
            reward = final_accuracy if round_num == 6 else 0
            agent.update(round_num, action, reward, next_state)

        print(f"Episode {episode+1}/{args.episodes} - Acc: {final_accuracy:.2f}%")

    best_policy = agent.get_best_policy()

    print("\nLearned Schedule:")
    for r, a in best_policy.items():
        clients = [i for i in range(3) if ACTIONS[a][i] == 1]
        print(f"Round {r}: Clients {clients}")

if __name__ == "__main__":
    main()
