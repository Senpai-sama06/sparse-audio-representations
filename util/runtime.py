"""Reproducibility and device helpers shared by model workflows."""
import random
import numpy as np
import torch


def seed_all(seed):
    """Seed Python, NumPy and PyTorch and enable deterministic cuDNN behavior."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_device(name="auto"):
    """Resolve ``auto`` to CUDA when available, otherwise use the requested device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu") if name == "auto" else torch.device(name)
