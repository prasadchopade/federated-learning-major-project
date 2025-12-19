import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from .models import get_resnet50_model, get_model_weights, set_model_weights


class FederatedClient:
    """
    Represents a single hospital (client) in federated learning
    """

    def __init__(self, hospital_id, train_dataloader, val_dataloader=None, device='cpu'):
        """
        Args:
            hospital_id: Hospital identifier (e.g., 'D_rsna_pneumonia')
            train_dataloader: PyTorch DataLoader for training
            val_dataloader: PyTorch DataLoader for validation
            device: 'cpu' or 'cuda'
        """
        self.hospital_id = hospital_id
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device

        # Initialize model
        self.model = get_resnet50_model(num_classes=3, pretrained=True)
        self.model.to(device)

        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        # Metrics
        self.training_loss = []
        self.validation_accuracy = []

    def local_training(self, epochs=1):
        """
        Train model locally on hospital data

        Args:
            epochs: Number of local training epochs

        Returns:
            avg_loss: Average training loss
        """
        self.model.train()
        total_loss = 0
        num_batches = 0

        for epoch in range(epochs):
            for batch_idx, (images, labels, _) in enumerate(self.train_dataloader):
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                num_batches += 1

        if num_batches == 0:
            raise ValueError(f"No training batches for {self.hospital_id}")
        avg_loss = total_loss / num_batches
        self.training_loss.append(avg_loss)

        return avg_loss

    def evaluate(self):
        """
        Evaluate model on validation data

        Returns:
            accuracy: Validation accuracy
        """
        if self.val_dataloader is None:
            return None

        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels, _ in self.val_dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total if total > 0 else 0
        self.validation_accuracy.append(accuracy)

        return accuracy

    def get_model_weights(self):
        """
        Get current model weights

        Returns:
            weights: Dictionary of model parameters
        """
        return get_model_weights(self.model)

    def set_model_weights(self, weights):
        """
        Set model weights (from global model)

        Args:
            weights: Dictionary of model parameters
        """
        set_model_weights(self.model, weights)

    def get_data_size(self):
        """Get number of training samples"""
        return len(self.train_dataloader.dataset)

    def get_local_accuracy(self):
        """Get last validation accuracy"""
        if len(self.validation_accuracy) > 0:
            return self.validation_accuracy[-1]
        return 0.0
