# Running the available code

Run commands from the repository root after installing `requirements.txt`. The commands below use the archived scripts with portable input/output arguments. Data and checkpoints must be supplied locally.

## Local checks

```bash
python -m unittest discover -s tests -v
python -m preprocessing.phase5_hospital_training --help
python -m preprocessing.phase5_evaluate_global --help
```

Tests use temporary synthetic images and small models. They do not download data or pretrained weights.

## Train one hospital for one round

```bash
python -m preprocessing.phase5_hospital_training --hospital nih --round 1 --global-model /path/to/initial_global.pt
```

For each selected hospital, pass the same starting checkpoint for that round. The original coordinator must combine the returned weights before the next round begins. For round 2, for example, supply the aggregated round-1 checkpoint, not that hospital's previous local checkpoint.

Override local data paths when needed:

```bash
python -m preprocessing.phase5_hospital_training --hospital nih --round 2 --global-model /path/to/global_round1.pt --csv /path/to/index.csv --images-dir /path/to/images --output-dir outputs/phase5
```

The script uses three local epochs, Adam at `1e-4`, batch size 32, and a three-class ResNet-50. It prefers CUDA, then Apple MPS, then CPU. Output defaults to:

```text
outputs/phase5/
    hospital_weights/round_1/nih_round1_weights.pt
    results/nih_round1_log.txt
```

The trainer accepts older parameter-only checkpoints that omit BatchNorm running buffers and prints the missing-buffer count. Other incompatible keys raise an error. Such initialization is relevant to the earlier baseline; it is not evidence of reproducing the final study.

## Evaluate a checkpoint

```bash
python -m preprocessing.phase5_evaluate_global --hospital nih --model /path/to/global_round6.pt
```

The evaluator requires an explicit model path and a complete state dictionary. It reports all three classes, the confusion matrix, and support counts. It saves the checkpoint and manifest paths with the report under `outputs/phase5/results/`. `--csv`, `--images-dir`, and `--output-dir` are available here too.

The final round-6 checkpoint is not bundled. Supplying an earlier checkpoint evaluates that earlier model only.

## Earlier data preparation and FedAvg baseline

With the prepared Phase 1 indices described in [dataset setup](datasets.md):

```bash
python -m preprocessing.phase1_consolidation
python -m preprocessing.phase1_standardize_labels
python -m preprocessing.phase1_create_partitions
python -m preprocessing.phase2_test_nih
python -m preprocessing.phase3_federated_training --hospital B_nih_adult --rounds 3 --local-epochs 1
```

The Phase 1 commands write metadata under `Data/`; rerunning them replaces the corresponding generated CSVs. Keep original experiment manifests separately. The baseline defaults to ImageNet-pretrained ResNet-50 and may download its weights on first use. It uses the earlier training configuration, not the final hospital protocol.

Repeat `--hospital` to include multiple clients in the baseline. Checkpoints and JSON metrics go to `outputs/phase3/`. The early `src/server.py` weights updates by local sample count.

## NIH Q-learning experiment

```bash
python -m preprocessing.phase4_modified --episodes 30 --local_epochs 2 --dirichlet_alpha 0.5
```

This partitions NIH training records into three virtual clients, tries seven nonempty client subsets, and prints a Q-learning-derived schedule. It can require many model-training runs. It is exploratory code and has not been rerun for this repository import. The final three-dataset study uses a separate deterministic schedule.
