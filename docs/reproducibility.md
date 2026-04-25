# Reproducibility and available artifacts

This page lists the code and artifacts needed to run the training and evaluation workflow.

| Component | Availability |
| :--- | :--- |
| Metadata consolidation and IID/non-IID partition scripts | Included |
| ResNet-50, hospital data loader, and FedAvg baseline | Included |
| NIH Q-learning experiment | Included |
| Hospital local-training and checkpoint-evaluation scripts | Included |
| Early baseline metrics and NIH local validation logs | Included |
| Final hospital weighting and scheduling description | Documented in the method guide |
| Final multi-hospital coordination/aggregation script | Not present in the source folder |
| Original initial global and final round-6 checkpoints | Not present in the source folder |
| Medical images and patient-level manifests | Local only; excluded from Git |
| Intermediate model checkpoints | Local only; excluded from Git |
| Final test predictions and per-hospital final reports | Not present in the source folder |

## Changes made for the repository import

- Fixed package imports and the data check that referred to a missing manager class.
- Resolved default paths from the repository root and added explicit data/checkpoint arguments.
- Separated new output from archived experiment logs.
- Replaced silent missing-image fallbacks with errors.
- Preserved BatchNorm buffers during model state transfer and handled integer counters separately in baseline FedAvg.
- Corrected the boundary indexing in the early IID metadata partition.
- Made original label variants consistent across the hospital scripts.
- Updated ResNet initialization to torchvision's weights API.

These changes make the available code easier to inspect and run. Historical results are retained without modification. The final training study was not rerun during publication.

## Evaluation limits

The three sources simulate hospital clients. The scripts alone do not establish clinical validity or formal privacy protection. The archive also does not establish that all train/test partitions are patient-disjoint or that NIH and RSNA records are disjoint across clients. A fresh evaluation needs those checks, fixed random seeds, exact checkpoints, and per-class metrics.

The Q-learning script is a separate exploratory experiment. It should not be used to claim that a learned policy produced the final hospital results.
