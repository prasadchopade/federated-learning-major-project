# Dataset setup

Download the images from their original publishers and follow each dataset's terms. The repository does not distribute medical images, patient-level manifests, or model checkpoints.

| Dataset | Source | Collection size |
| :--- | :--- | ---: |
| RSNA Pneumonia Detection | [RSNA challenge downloads and attribution](https://www.rsna.org/artificial-intelligence/ai-image-challenge/rsna-pneumonia-detection-challenge-2018) | 26,684 |
| Pediatric chest X-rays | [Kermany, Zhang, and Goldbaum dataset](https://data.mendeley.com/datasets/rscbjbr9sj/2) | 5,856 |
| NIH ChestX-ray14 | [Dataset paper and download reference](https://arxiv.org/abs/1705.02315) | 112,120 |

These collection sizes differ from the number of training examples in an individual split. The RSNA challenge uses images from NIH's chest X-ray collection; the [RSNA page](https://www.rsna.org/artificial-intelligence/ai-image-challenge/rsna-pneumonia-detection-challenge-2018) provides a mapping to the original NIH images. Cross-client and patient-level overlap must be checked when preparing a new evaluation.

## Hospital script layout

Paths are resolved from the repository root. `--csv` and `--images-dir` override the defaults.

```text
preprocessed_rsna/
    index.csv
    images/
Data/A_pediatric/
    index.csv
    images/
preprocessed_nih/
    B_nih_adult_index.csv
    images/
```

Each manifest must have `image_path`, `label`, and `split` columns. `split` must be `train`, `val`, or `test`. Additional fields such as `hospital_id` and `patient_id` can remain local. Each image basename must be unique within its hospital image directory.

An example of the schema, using fictional filenames:

```csv
image_path,hospital_id,label,split
images/example_normal.png,B_nih_adult,Normal,train
images/example_pneumonia.png,B_nih_adult,Pneumonia,val
images/example_other.png,B_nih_adult,Other,test
```

This snippet illustrates columns only. It is not a dataset or a suitable train/validation/test split.

## Labels and transforms

The normalized project labels are `Normal`, `Pneumonia`, and `Other`. Hospital scripts also accept `No Finding` as Normal, bacterial/viral pneumonia as Pneumonia, and RSNA's `Lung Opacity` as Pneumonia. `No Lung Opacity / Not Normal` maps to Other. Other pathology labels map to Other.

Images are opened as RGB, resized to 224 × 224, and normalized with means `[0.485, 0.456, 0.406]` and standard deviations `[0.229, 0.224, 0.225]`. Training adds augmentation; evaluation does not.

The scripts expect prepared images and manifests. The original raw-image conversion and patient-split generation pipeline is not included. Preserve the original experiment manifests when comparing results; newly generated splits describe a new experiment.

## Earlier four-client preparation

The Phase 1 scripts additionally support an Indiana client. They read already prepared indices at:

```text
Data/A_pediatric/index.csv
Data/B_nih_adult/index.csv
Data/C_indiana_small/index.csv
Data/D_rsna_pneumonia/index.csv
```

They consolidate metadata, normalize label capitalization, and generate IID and natural hospital partitions. They do not copy images between hospitals or create patient-level train/test splits. The IID partition is an experimental redistribution of metadata.

`Data/dataset_summary.json` is preserved from this earlier four-client setup (152,126 records). Indiana is not part of the final three-client study. The legacy pediatric ID is `D_pediatric_pneumonia` even though the source directory is `A_pediatric`.

The baseline loader resolves original manifest paths to the local `preprocessed_*` image folders. Missing or unreadable images raise an error instead of becoming fabricated Normal examples.
