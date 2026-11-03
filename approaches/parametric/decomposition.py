"""Adapters for EMD and the pinned VMD reference implementation.

This baseline retains sampled modes. It does not fit parametric functions or
encode a compressed representation. Decomposition and coding errors must remain
separate when those stages are added later.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import math

import numpy as np


@dataclass
class Decomposition:
    """Modes shaped (modes, samples), a residual, and backend diagnostics."""

    modes: np.ndarray
    residual: np.ndarray
    diagnostics: dict = field(default_factory=dict)

    def reconstruct(self, *, include_residual=False):
        values = self.modes.sum(axis=0)
        return values + self.residual if include_residual else values

    def validate(self, samples):
        if self.modes.ndim != 2 or self.modes.shape[1] != samples:
            raise ValueError("Backend returned modes with the wrong sample count")
        if self.residual.shape != (samples,):
            raise ValueError("Backend returned a residual with the wrong sample count")
        if not np.isfinite(self.modes).all() or not np.isfinite(self.residual).all():
            raise FloatingPointError("Decomposition returned nonfinite values")
        return self


class ModeDecomposer(ABC):
    """One independently fitted decomposition per mono float64 frame."""

    name: str

    @abstractmethod
    def decompose(self, signal):
        """Return sampled modes and a separately identified residual."""

    @staticmethod
    def signal(values):
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 1 or len(values) < 4 or not np.isfinite(values).all():
            raise ValueError("Expected at least four finite mono samples")
        return values


class EMDDecomposer(ModeDecomposer):
    """PyEMD sifting with native IMFs and residue kept separate.

    Summing IMFs plus the native residue is a reconstruction control, generally
    near exact. It cannot establish the quality of a compressed representation.
    """

    name = "emd"

    def __init__(self, max_imfs=-1, max_iterations=1000):
        if isinstance(max_imfs, bool) or not isinstance(max_imfs, int) or (max_imfs != -1 and max_imfs < 1):
            raise ValueError("max_imfs must be -1 (all) or a positive integer")
        if isinstance(max_iterations, bool) or not isinstance(max_iterations, int) or max_iterations < 1:
            raise ValueError("max_iterations must be a positive integer")
        try:
            from PyEMD import EMD
        except ImportError as error:
            raise ImportError('Install the decomposition extras: pip install -e ".[parametric]"') from error
        self._emd_class = EMD
        self.max_imfs, self.max_iterations = max_imfs, max_iterations

    def decompose(self, signal):
        x = self.signal(signal)
        backend = self._emd_class(MAX_ITERATION=self.max_iterations)
        backend.emd(x, max_imf=self.max_imfs)
        modes, residual = backend.get_imfs_and_residue()
        modes = np.asarray(modes, dtype=np.float64).reshape(-1, len(x))
        return Decomposition(modes, np.asarray(residual, dtype=np.float64), {
            "residual_kind": "native_emd_residue",
        }).validate(len(x))


class VMDDecomposer(ModeDecomposer):
    """Fixed-K VMD using vmdpy 0.2 and deterministic uniform initialization.

    The residual here is explicitly input minus the returned mode sum. Adding
    that raw residual is an oracle control, not a VMD compression result. Odd
    frames are extended by one sample because this reference drops odd tails.
    """

    name = "vmd"

    def __init__(self, modes=8, alpha=2000.0, tau=0.0, tolerance=1e-7):
        if isinstance(modes, bool) or not isinstance(modes, int) or modes < 1:
            raise ValueError("VMD modes must be a positive integer")
        if not all(math.isfinite(v) for v in (alpha, tau, tolerance)) or alpha <= 0 or tau < 0 or tolerance <= 0:
            raise ValueError("Require finite alpha > 0, tau >= 0 and tolerance > 0")
        try:
            from vmdpy import VMD
        except ImportError as error:
            raise ImportError('Install the decomposition extras: pip install -e ".[parametric]"') from error
        self._vmd = VMD
        self.modes, self.alpha, self.tau, self.tolerance = modes, alpha, tau, tolerance

    def decompose(self, signal):
        x = self.signal(signal)
        if not np.any(x):
            return Decomposition(np.zeros((self.modes, len(x))), np.zeros_like(x), {
                "residual_kind": "input_minus_mode_sum", "silent_frame": True,
            })
        padded = np.pad(x, (0, len(x) % 2), mode="edge")
        values, _, omega = self._vmd(padded, self.alpha, self.tau, self.modes, False, 1, self.tolerance)
        values = np.asarray(values, dtype=np.float64)[:, :len(x)]
        return Decomposition(values, x - values.sum(axis=0), {
            "residual_kind": "input_minus_mode_sum",
            "frequency_history_rows": len(omega),
            "center_frequencies_cycles_per_sample": omega[-1].tolist(),
        }).validate(len(x))
