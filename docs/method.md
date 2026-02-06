# Method

The project studies three simulated hospital clients with different dataset sizes, patient populations, and annotation schemes. The shared task has three classes: Normal (0), Pneumonia (1), and Other (2).

## Local training

| Setting | Final study |
| :--- | :--- |
| Architecture | ResNet-50 with a three-output classifier |
| Initial architecture weights | Random initialization; no ImageNet pretraining |
| Input | 224 × 224 RGB images |
| Normalization | ImageNet channel means and standard deviations |
| Training augmentation | Horizontal flips, rotations up to 10°, brightness and contrast jitter |
| Optimizer | Adam |
| Learning rate | 0.0001 |
| Batch size | 32 |
| Local epochs per round | 3 |
| Communication rounds | 6 |
| Loss | Cross-entropy |

Each selected hospital receives the same starting global model for a round, trains it on its own images, and returns its local weights. The hospital training script requires the starting checkpoint explicitly.

## Aggregation

The final method gives each hospital a fixed importance coefficient:

| Hospital | Coefficient |
| :--- | ---: |
| RSNA | 0.321 |
| Pediatric | 0.290 |
| NIH | 0.389 |

For the selected set of hospitals `S`, the server computes:

```text
next_global = sum(alpha[h] * local_weights[h] for h in S) / sum(alpha[h] for h in S)
```

The weights are manually chosen and stay fixed across rounds. They do not scale directly with dataset size.

Hospital participation follows a deterministic schedule. The term “reinforcement learning” in the submitted title refers to the scheduling motivation and related experiments; the final study does not claim a learned RL policy.

## Implementation coverage

The available `src/server.py` implements the earlier, sample-size-weighted FedAvg baseline. It is separate from the final weighted hospital coordinator described above. That final coordinator is absent from the supplied folder.

`preprocessing/phase4_modified.py` is an NIH-only Q-learning experiment with three virtual clients. `preprocessing/phase5_hospital_training.py` handles local hospital training, and `preprocessing/phase5_evaluate_global.py` evaluates a supplied checkpoint. See [the experiment comparison](../PHASE_COMPARISON.md).

## Limits

Only model weights are exchanged in the described workflow. Secure aggregation and differential privacy were not implemented. The project is a research prototype.

Mapping NIH's multiple findings into three classes loses detail. Evaluation needs per-class precision, recall, F1, and confusion matrices as well as accuracy. A single average can obscure weak performance at an individual client.
