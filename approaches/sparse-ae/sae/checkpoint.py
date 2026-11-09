"""Atomic artifacts and independent best-weight / early-stop tracking."""
from dataclasses import asdict, dataclass
import math
from pathlib import Path

import torch


def atomic_save(value, path):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


@dataclass
class Stopper:
    patience: int = 8
    min_delta: float = 1e-4
    min_epochs: int = 1
    best: float = math.inf
    anchor: float = math.inf
    best_epoch: int = 0
    bad_epochs: int = 0

    def update(self, value, epoch):
        if not math.isfinite(value):
            raise FloatingPointError("Nonfinite validation monitor")
        improved = value < self.best
        if improved:
            self.best, self.best_epoch = value, epoch
        # Accumulated small improvements can eventually exceed min_delta.
        if value < self.anchor - self.min_delta:
            self.anchor, self.bad_epochs = value, 0
        else:
            self.bad_epochs += 1
        if epoch < self.min_epochs:
            self.bad_epochs = 0
        stop = self.patience > 0 and epoch >= self.min_epochs and self.bad_epochs >= self.patience
        return improved, stop

    def state(self):
        return asdict(self)
