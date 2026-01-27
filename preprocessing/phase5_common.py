"""Shared paths and label mapping for the hospital training scripts."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

HOSPITAL_CONFIGS = {
    "rsna": {
        "csv_file": ROOT / "preprocessed_rsna" / "index.csv",
        "images_folder": ROOT / "preprocessed_rsna" / "images",
        "display_name": "RSNA Pneumonia",
    },
    "pediatric": {
        "csv_file": ROOT / "Data" / "A_pediatric" / "index.csv",
        "images_folder": ROOT / "Data" / "A_pediatric" / "images",
        "display_name": "Pediatric",
    },
    "nih": {
        "csv_file": ROOT / "preprocessed_nih" / "B_nih_adult_index.csv",
        "images_folder": ROOT / "preprocessed_nih" / "images",
        "display_name": "NIH Adult",
    },
}


def label_to_class(label):
    """Map normalized project labels and the original dataset label variants."""
    value = str(label).strip().upper()
    if value in {"NORMAL", "NO FINDING", "NO FINDINGS"}:
        return 0
    if value in {"NO LUNG OPACITY / NOT NORMAL", "NOT NORMAL", "OTHER"}:
        return 2
    if value == "LUNG OPACITY" or any(word in value for word in ("PNEUMONIA", "BACTERIAL", "VIRAL")):
        return 1
    return 2


def image_filename(path):
    """Read manifest basenames written on either Windows or Unix."""
    return str(path).replace("\\", "/").rsplit("/", 1)[-1]
