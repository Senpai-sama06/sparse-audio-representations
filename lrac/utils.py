from pathlib import Path
import hashlib
import csv
import json
import os
import random
import numpy as np
import torch


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def write_csv(path, rows):
    """Write a nonempty table of dictionaries with a shared, explicit schema."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_device(name="auto"):
    return torch.device("cuda" if torch.cuda.is_available() else "cpu") if name == "auto" else torch.device(name)


def environment():
    import importlib.metadata as md
    import platform
    result = {"python": platform.python_version(), "platform": platform.platform(), "torch": torch.__version__, "cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available()}
    for package in ["numpy", "scipy", "soundfile", "librosa", "scikit-learn", "PyYAML", "matplotlib", "pytest", "pesq", "pystoi"]:
        try:
            result[package] = md.version(package)
        except md.PackageNotFoundError:
            result[package] = None
    if torch.cuda.is_available():
        result["gpu"] = torch.cuda.get_device_name(0)
    return result


def source_fingerprint(root):
    root = Path(root)
    files = sorted((root / "lrac").rglob("*.py"))
    files += [p for p in [root / "pyproject.toml", root / "requirements-lock.txt"] if p.exists()]
    hashes = {str(p.relative_to(root)).replace("\\", "/"): file_hash(p) for p in files}
    return {"files": hashes, "combined_sha256": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()}


def atomic_checkpoint(path, payload):
    path = Path(path)
    temp = path.with_suffix(".tmp")
    torch.save(payload, temp)
    os.replace(temp, path)
