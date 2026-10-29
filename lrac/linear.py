"""Matrix-free, unit-norm cosine/impulse synthesis for numerical experiments.

These whole-clip transforms have no window, padding or fitted parameters. They
are separate from the framed PyTorch models used by the neural training pipeline.
"""
from dataclasses import dataclass

import numpy as np
from scipy.fft import dct, idct


@dataclass(frozen=True)
class CosineDictionary:
    """Q for ``dct``, or [I, Q] for ``impulse_dct``; Q is orthonormal IDCT-II."""

    samples: int
    kind: str = "impulse_dct"

    def __post_init__(self):
        if self.kind not in {"dct", "impulse_dct"} or self.samples < 1:
            raise ValueError("Expected a positive length and dct or impulse_dct")

    @property
    def size(self):
        return self.samples * (2 if self.kind == "impulse_dct" else 1)

    @property
    def frame_bound(self):
        # Phi Phi.T is I or 2I. This is the exact gradient Lipschitz constant
        # for 0.5 ||x - Phi z||^2, without estimating a large matrix norm.
        return 2.0 if self.kind == "impulse_dct" else 1.0

    def synthesis(self, coefficients):
        """Reconstruct a waveform using coefficients and this shared dictionary."""
        z = np.asarray(coefficients, dtype=np.float64)
        if z.shape != (self.size,):
            raise ValueError("Coefficient vector has the wrong dictionary size")
        if self.kind == "dct":
            return idct(z, type=2, norm="ortho", workers=1)
        return z[:self.samples] + idct(z[self.samples:], type=2, norm="ortho", workers=1)

    def adjoint(self, waveform):
        """Apply Phi.T; for the overcomplete dictionary this is not its inverse."""
        x = np.asarray(waveform, dtype=np.float64)
        if x.shape != (self.samples,):
            raise ValueError("Waveform has the wrong length")
        cosine = dct(x, type=2, norm="ortho", workers=1)
        return np.concatenate((x, cosine)) if self.kind == "impulse_dct" else cosine
