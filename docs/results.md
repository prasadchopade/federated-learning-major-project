# Results

## Test accuracy

| Dataset | Local training | Standard FedAvg | Weighted + scheduled FL |
| :--- | ---: | ---: | ---: |
| RSNA | 81.03% | 79.50% | **84.80%** |
| Pediatric | 81.49% | 79.20% | **87.50%** |
| NIH ChestX-ray14 | 67.29% | **72.80%** | 71.20% |

## RSNA scheduling comparison

| Participation strategy | Test accuracy |
| :--- | ---: |
| Full participation | 77.74% |
| Scheduled participation | **83.03%** |

[Machine-readable results](../results/metrics.json).
