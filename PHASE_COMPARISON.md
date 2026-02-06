# Earlier experiments and the final hospital study

| Aspect | Phase 3 | Phase 4 | Phase 5 / final study |
| :--- | :--- | :--- | :--- |
| Purpose | FedAvg baseline and single-client checks | Explore client selection | Train and evaluate the hospital federation |
| Data | Hospital partitions; saved RSNA and NIH runs | NIH split into three virtual clients | RSNA, Pediatric, and NIH |
| Backbone | ImageNet-pretrained ResNet-50 | ResNet-50 without pretraining | ResNet-50 without ImageNet pretraining, initialized from the supplied global checkpoint |
| Aggregation | Sample-count-weighted FedAvg | Fixed virtual-client weights: 0.35, 0.30, 0.35 | Fixed hospital weights: 0.321, 0.290, 0.389 |
| Selection | All configured clients | Q-learning over seven nonempty subsets | Deterministic participation schedule |
| Available code | `src/server.py` and the Phase 3 entry point | `preprocessing/phase4_modified.py` | Hospital training and evaluation scripts; final coordinator absent |

## Phase 4: NIH Q-learning experiment

The available script uses NIH data, not RSNA. It creates three virtual clients with a Dirichlet partition, then uses the round number as the Q-learning state. Actions select one, two, or all three clients. Final episode validation accuracy supplies the reward.

The script prints the selected policy. No trained Q-table or learned-policy artifact is included. It is an exploratory experiment, separate from the final three-dataset result table.

## Phase 5: hospital training

A selected hospital receives a global checkpoint, trains locally for three epochs, and saves a local state dictionary. The coordinator must collect and aggregate the selected updates before supplying the next global checkpoint. The evaluation script measures an explicitly supplied model on one hospital's test split.

The final hospital study uses fixed importance weights and a deterministic participation schedule. The local trainer does not learn or enforce that schedule. The coordinator used for the final experiment is not in the supplied source folder.

See [the method](docs/method.md), [run guide](docs/running.md), and [results](docs/results.md).
