import sys
import os
import json
import argparse
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_loader import HospitalDataManager
from src.client import FederatedClient
from src.server import FederatedServer
import pandas as pd
import torch

def main():
    """
    Main federated learning training script
    """

    print("="*70)
    print("PHASE 3: FEDERATED LEARNING WITH FedAvg (Weights Saved)")
    print("="*70)

    # Project structure
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    # Configuration - handle both Data/data casing
    data_dir_candidates = [
        os.path.join(project_root, "Data"),
        os.path.join(project_root, "data"),
    ]
    DATA_DIR = next((path for path in data_dir_candidates if os.path.exists(path)), data_dir_candidates[-1])

    PARTITION_PATH = os.path.join(DATA_DIR, "partition_non_iid.csv")
    LABEL_MAP_PATH = os.path.join(DATA_DIR, "global_label_map.json")

    print(f"\nProject root: {project_root}")
    print(f"Data directory: {DATA_DIR}")

    parser = argparse.ArgumentParser(description='Early FedAvg baseline')
    parser.add_argument('--hospital', action='append', help='Repeat to select multiple hospitals')
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--local-epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--data-dir', type=Path, default=Path(DATA_DIR))
    parser.add_argument('--output-dir', type=Path, default=Path(project_root) / 'outputs' / 'phase3')
    args = parser.parse_args()
    if min(args.rounds, args.local_epochs, args.batch_size) <= 0:
        parser.error('Rounds, local epochs, and batch size must be positive')
    PARTITION_PATH = args.data_dir / 'partition_non_iid.csv'
    LABEL_MAP_PATH = args.data_dir / 'global_label_map.json'
    args.output_dir.mkdir(parents=True, exist_ok=True)
    NUM_ROUNDS = args.rounds
    LOCAL_EPOCHS = args.local_epochs
    BATCH_SIZE = args.batch_size

    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")

    # Load data
    print("\n" + "="*70)
    print("LOADING DATA")
    print("="*70)

    partition_df = pd.read_csv(PARTITION_PATH, low_memory=False)
    with open(LABEL_MAP_PATH) as f:
        label_map = json.load(f)

    # Create clients (hospitals)
    hospital_ids = args.hospital or ['B_nih_adult']

    clients = []

    for hospital_id in hospital_ids:
        print(f"\nInitializing client: {hospital_id}")

        manager = HospitalDataManager(
            hospital_id=hospital_id,
            partition_df=partition_df,
            label_map=label_map,
            image_size=224,
            batch_size=BATCH_SIZE,
            num_workers=0
        )

        train_loader = manager.get_dataloader('train')
        val_loader = manager.get_dataloader('val')

        client = FederatedClient(
            hospital_id=hospital_id,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            device=device
        )

        clients.append(client)

    # Initialize Federated Server
    print("\n" + "="*70)
    print("INITIALIZING FEDERATED SERVER")
    print("="*70)

    server = FederatedServer(num_classes=3, device=device)
    print(f"Global model initialized on {device}")

    # Federated Learning Rounds
    print("\n" + "="*70)
    print(f"STARTING {NUM_ROUNDS} FEDERATED LEARNING ROUNDS")
    print("="*70)

    for round_num in range(NUM_ROUNDS):
        print(f"\n\n{'#'*70}")
        print(f"# ROUND {round_num + 1}/{NUM_ROUNDS}")
        print(f"{'#'*70}")

        server.federated_round(clients, local_epochs=LOCAL_EPOCHS)

    # Final Results
    print("\n" + "="*70)
    print("FEDERATED LEARNING COMPLETE")
    print("="*70)

    print(f"\nFinal Accuracies per Hospital:")
    for client in clients:
        print(f"  {client.hospital_id}: {client.get_local_accuracy():.4f}")

    if server.round_accuracies:
        avg_accuracy = sum(server.round_accuracies) / len(server.round_accuracies)
        print(f"\nAverage Accuracy over all rounds: {avg_accuracy:.4f}")

    # Save results (JSON)
    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)

    results = {
        'total_rounds': NUM_ROUNDS,
        'local_epochs': LOCAL_EPOCHS,
        'round_accuracies': server.round_accuracies,
        'client_names': hospital_ids,
        'final_accuracies': {client.hospital_id: client.get_local_accuracy() for client in clients}
    }

    results_path = args.output_dir / 'phase3_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f" Saved metrics: {results_path}")

    # Save model weights for each client and global model
    print("\n" + "="*70)
    print("SAVING MODEL WEIGHTS")
    print("="*70)

    for client in clients:
        weights = client.get_model_weights()
        weights_filename = args.output_dir / f"{client.hospital_id}_weights_final.pt"
        torch.save(weights, weights_filename)
        print(f" Saved weights: {weights_filename}")

    global_weights = server.get_global_model_weights()
    global_filename = args.output_dir / "global_model_phase3.pt"
    torch.save(global_weights, global_filename)
    print(f" Saved global model: {global_filename}")

    print("\n" + "="*70)
    print(" READY FOR PHASE 4!")
    print("="*70)

if __name__ == "__main__":
    main()
