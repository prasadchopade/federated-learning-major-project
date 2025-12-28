import torch
import numpy as np
from .models import get_resnet50_model, get_model_weights, set_model_weights


class FederatedServer:
    """
    Central server that coordinates federated learning
    """

    def __init__(self, num_classes=3, device='cpu'):
        """
        Args:
            num_classes: Number of output classes
            device: 'cpu' or 'cuda'
        """
        self.device = device
        self.global_model = get_resnet50_model(num_classes=num_classes, pretrained=True)
        self.global_model.to(device)

        # Metrics tracking
        self.round_accuracies = []
        self.round_losses = []

    def get_global_model_weights(self):
        """
        Get current global model weights to send to clients

        Returns:
            weights: Dictionary of model parameters
        """
        return get_model_weights(self.global_model)

    def federated_averaging(self, client_weights_list, client_data_sizes):
        """
        FedAvg: Average model weights from all clients, weighted by data size

        Args:
            client_weights_list: List of weight dictionaries from each client
            client_data_sizes: List of data sizes for each client (for weighting)

        Returns:
            averaged_weights: Aggregated model weights
        """
        if not client_weights_list or len(client_weights_list) != len(client_data_sizes):
            raise ValueError("Provide one data size per client and at least one client")
        if any(not np.isfinite(size) or size <= 0 for size in client_data_sizes):
            raise ValueError("Client data sizes must be finite and positive")
        total_data_size = sum(client_data_sizes)

        # Initialize averaged weights
        averaged_weights = {}

        # Get all parameter names
        param_names = list(client_weights_list[0].keys())
        if any(set(weights) != set(param_names) for weights in client_weights_list):
            raise ValueError("Client state dictionaries must have matching keys")

        # Average each parameter
        for param_name in param_names:
            tensors = [weights[param_name] for weights in client_weights_list]
            if any(t.shape != tensors[0].shape or t.dtype != tensors[0].dtype for t in tensors):
                raise ValueError(f"Incompatible client tensor: {param_name}")
            # BatchNorm counters are integer buffers, not parameters to average.
            if not tensors[0].is_floating_point():
                averaged_weights[param_name] = torch.stack(tensors).amax(dim=0)
                continue
            weighted_sum = None

            for client_idx, weights in enumerate(client_weights_list):
                weight = weights[param_name]
                data_size = client_data_sizes[client_idx]
                weight_factor = data_size / total_data_size

                if weighted_sum is None:
                    weighted_sum = weight * weight_factor
                else:
                    weighted_sum += weight * weight_factor

            averaged_weights[param_name] = weighted_sum

        return averaged_weights

    def update_global_model(self, averaged_weights):
        """
        Update global model with averaged weights

        Args:
            averaged_weights: Dictionary of aggregated weights
        """
        set_model_weights(self.global_model, averaged_weights)

    def federated_round(self, clients, local_epochs=1):
        """
        Execute one federated learning round

        Steps:
        1. Send global model to all clients
        2. Each client trains locally
        3. Collect updated weights
        4. Aggregate weights
        5. Update global model

        Args:
            clients: List of FederatedClient objects
            local_epochs: Number of local training epochs per client

        Returns:
            round_stats: Dictionary with round statistics
        """
        print(f"\n{'='*70}")
        print(f"Federated Learning Round")
        print(f"{'='*70}")

        # Step 1: Send global model to all clients
        global_weights = self.get_global_model_weights()
        for client in clients:
            client.set_model_weights(global_weights)

        # Step 2 & 3: Local training and collect weights
        client_weights_list = []
        client_data_sizes = []
        client_accuracies = []

        for client in clients:
            print(f"\n{client.hospital_id}: Training locally...")

            # Local training
            train_loss = client.local_training(epochs=local_epochs)

            # Evaluation
            accuracy = client.evaluate()

            # Collect
            client_weights_list.append(client.get_model_weights())
            client_data_sizes.append(client.get_data_size())
            client_accuracies.append(accuracy if accuracy is not None else 0.0)

            print(f"  Loss: {train_loss:.4f}")
            print(f"  Accuracy: {accuracy:.4f}" if accuracy is not None else "  Accuracy: N/A")

        # Step 4: Aggregate weights (FedAvg)
        averaged_weights = self.federated_averaging(client_weights_list, client_data_sizes)

        # Step 5: Update global model
        self.update_global_model(averaged_weights)

        # Compute statistics
        avg_accuracy = np.mean(client_accuracies)
        self.round_accuracies.append(avg_accuracy)

        round_stats = {
            'avg_accuracy': avg_accuracy,
            'client_accuracies': client_accuracies,
            'client_names': [c.hospital_id for c in clients]
        }

        print(f"\n{'='*70}")
        print(f"Round Summary:")
        for idx, client in enumerate(clients):
            print(f"  {client.hospital_id}: {client_accuracies[idx]:.4f}")
        print(f"  Average Accuracy: {avg_accuracy:.4f}")
        print(f"{'='*70}")

        return round_stats

    def get_global_model(self):
        """Get the global model"""
        return self.global_model
