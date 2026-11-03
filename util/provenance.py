"""Hashing and environment records for reproducible research runs."""
from pathlib import Path
import hashlib
import json
import torch


def file_hash(path):
    """Return a streaming SHA-256 digest for a file."""
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def environment():
    """Capture the runtime and installed library versions for a run record."""
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
    """Hash project Python sources and dependency lockfiles for run provenance."""
    root = Path(root)
    files = sorted((root / "approaches").rglob("*.py"))
    files += sorted((root / "util").rglob("*.py"))
    files += [p for p in [root / "pyproject.toml", root / "requirements-lock.txt"] if p.exists()]
    hashes = {str(p.relative_to(root)).replace("\\", "/"): file_hash(p) for p in files}
    return {"files": hashes, "combined_sha256": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()}
