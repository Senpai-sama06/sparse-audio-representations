"""Small filesystem helpers shared by research workflows."""
from pathlib import Path
import csv
import json
import os
import torch


def write_json(path, obj):
    """Write JSON atomically, creating the destination directory as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def write_csv(path, rows):
    """Write a nonempty table of dictionaries with a shared explicit schema."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def atomic_checkpoint(path, payload):
    """Save a PyTorch checkpoint through a temporary file and atomic replace."""
    path = Path(path)
    temp = path.with_suffix(".tmp")
    torch.save(payload, temp)
    os.replace(temp, path)
