"""Check one batch from the NIH hospital manifest."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from preprocessing.phase2_test_dataloader import main

if __name__ == '__main__':
    main(default_hospital='B_nih_adult')
