"""Independent frame decomposition followed by normalized overlap-add."""
from dataclasses import dataclass
import math
import time

import numpy as np

from approaches.parametric.decomposition import ModeDecomposer


@dataclass
class Reconstruction:
    """Two full-length reconstructions and per-frame decomposition records."""

    modes: np.ndarray
    modes_plus_residual_control: np.ndarray
    frames: list[dict]
    decomposition_seconds: float


class WindowedReconstructor:
    """Preserve sample count and amplitude through frame boundaries.

    Frames contain unwindowed samples with reflected boundary context. A
    strictly positive Hann synthesis window is applied after decomposition and
    divided by its accumulated weights. Reflection uses only the selected clip.
    Each frame is independently decomposed; modes are summed before stitching,
    so no cross-frame mode correspondence is assumed.
    """

    def __init__(self, window_samples, overlap=0.5):
        if isinstance(window_samples, bool) or not isinstance(window_samples, int) or window_samples < 4:
            raise ValueError("window_samples must be an integer >= 4")
        if not math.isfinite(overlap) or not 0 <= overlap < 1:
            raise ValueError("overlap must be a finite fraction in [0, 1)")
        self.window_samples = window_samples
        self.hop_samples = max(1, round(window_samples * (1 - overlap)))
        self.weights = np.hanning(window_samples + 2)[1:-1] if overlap else np.ones(window_samples)

    def reconstruct(self, signal, decomposer: ModeDecomposer, on_frame=None):
        """Reconstruct a clip, optionally persisting each frame's sampled modes."""
        x = np.asarray(signal, dtype=np.float64)
        if x.ndim != 1 or not len(x) or not np.isfinite(x).all():
            raise ValueError("Expected nonempty finite mono audio")
        size = self.window_samples
        padded = np.pad(x, (size, size), mode="reflect" if len(x) > 1 else "edge")
        summed, control, weight_sum = np.zeros_like(x), np.zeros_like(x), np.zeros_like(x)
        frames, total_seconds = [], 0.0
        for index, start in enumerate(range(-(size // 2), len(x), self.hop_samples)):
            frame = padded[start + size:start + 2 * size]
            began = time.perf_counter()
            result = decomposer.decompose(frame).validate(size)
            elapsed = time.perf_counter() - began
            total_seconds += elapsed
            left, right = max(0, start), min(len(x), start + size)
            source = slice(left - start, right - start)
            weights = self.weights[source]
            summed[left:right] += result.reconstruct()[source] * weights
            control[left:right] += result.reconstruct(include_residual=True)[source] * weights
            weight_sum[left:right] += weights
            energy = float(frame @ frame)
            record = {
                "frame": index, "start_sample": start, "mode_count": len(result.modes),
                "residual_nmse": float(result.residual @ result.residual) / energy if energy else None,
                "seconds": elapsed, "diagnostics": result.diagnostics,
            }
            frames.append(record)
            if on_frame is not None:
                on_frame(index, start, result)
        if np.any(weight_sum <= 0):
            raise RuntimeError("Frame synthesis left uncovered samples")
        return Reconstruction(summed / weight_sum, control / weight_sum, frames, total_seconds)
