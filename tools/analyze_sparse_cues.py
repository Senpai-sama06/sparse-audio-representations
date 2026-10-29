"""Compatibility entry point; prefer run.py study --config configs/study_sparse_cues.yaml."""
from pathlib import Path
import sys

from lrac.cli import main


if __name__ == "__main__":
    config = Path(__file__).resolve().parents[1] / "configs/study_sparse_cues.yaml"
    main(["study", "--config", str(config), *sys.argv[1:]])
