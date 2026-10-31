"""Shared full-utterance WORLD cache, followed by frame-aligned training crops.

The cache is immutable per source ID/seed. It permits training to begin before
the entire corpus is prepared while a background producer fills the same cache.
Both variants see the same waveform/F0 perturbation for each utterance.
"""
from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
import time

import numpy as np
import pyworld
from scipy.signal import resample_poly
import soundfile as sf
import torch
from torch.utils.data import Dataset


@contextmanager
def cache_lock(path):
    """Serialize per-utterance creation across the producer and both trainers."""
    with path.open("a+b") as stream:
        stream.seek(0, 2)
        if not stream.tell():
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            while True:
                try:
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    time.sleep(.2)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def cached_utterance(row, cache, seed=0):
    """24-kHz WORLD flat+shift, preserving voicing and duration before cropping.

    Harvest/StoneMask, CheapTrick, D4C and 256-sample F0 frames follow the official
    preprocessing. Random shifts are in [-5,+5] semitones. Cache identity includes
    source identity and seed; the campaign separately fingerprints its manifests.
    """
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    identity = hashlib.sha256(f"{row['id']}:{seed}:pitchflower24k-v1".encode()).hexdigest()
    path = cache/f"{identity}.npz"
    if not path.exists():
        with cache_lock(cache/f"{identity}.lock"):
            if not path.exists():
                x, sr = sf.read(row["path"], dtype="float64")
                if x.ndim != 1 or sr != 16000 or not np.isfinite(x).all():
                    raise ValueError("Expected finite native mono 16-kHz LibriSpeech")
                original = resample_poly(x, 3, 2)
                f0, t = pyworld.harvest(original, 24000, frame_period=256/24)
                f0 = pyworld.stonemask(original, f0, t, 24000)
                spectral = pyworld.cheaptrick(original, f0, t, 24000)
                aperiodicity = pyworld.d4c(original, f0, t, 24000)
                voiced = f0 > 0
                rng = np.random.default_rng(int(identity[:16], 16))
                shift = float(rng.uniform(-5, 5))
                flat_f0 = np.where(voiced, f0[voiced].mean()*2**(shift/12) if voiced.any() else 0., 0.)
                flattened = pyworld.synthesize(flat_f0, spectral, aperiodicity, 24000, frame_period=256/24)
                flattened = np.pad(flattened[:len(original)], (0, max(0, len(original)-len(flattened))))
                log_f0 = np.zeros_like(f0)
                log_f0[voiced] = np.log2(f0[voiced])
                temp = path.with_suffix(f".{os.getpid()}.tmp.npz")
                # Store FLOAT arrays, avoiding the reference script's implicit
                # PCM16 clipping of resynthesized WORLD audio.
                np.savez_compressed(temp, original=original.astype("float32"),
                    perturbed=flattened.astype("float32"), pitch=log_f0.astype("float32"), shift=shift)
                temp.replace(path)
    return path


class PitchDataset(Dataset):
    """Crop both waveforms and F0 at the same hop boundary; no speaker labels fed in."""
    def __init__(self, rows, cache, samples=36096, seed=0, training=True):
        self.rows, self.cache, self.samples, self.seed, self.training = rows, cache, samples, seed, training

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        path = cached_utterance(self.rows[index], self.cache, self.seed)
        with np.load(path, allow_pickle=False) as data:
            original, perturbed, pitch = data["original"], data["perturbed"], data["pitch"]
        frames = self.samples//256+1
        if len(original) < self.samples:
            original = np.tile(original, int(np.ceil(self.samples/len(original))))
            perturbed = np.tile(perturbed, int(np.ceil(self.samples/len(perturbed))))
            pitch = np.tile(pitch, int(np.ceil(frames/len(pitch))))
        maximum = min((len(original)-self.samples)//256, max(0, len(pitch)-frames))
        first_frame = int(torch.randint(maximum+1, (1,))) if self.training else 0
        offset = first_frame*256
        a = original[offset:offset+self.samples].copy()
        b = perturbed[offset:offset+self.samples].copy()
        # Independent crop peak normalization matches the released DatasetVC;
        # this is a training convention, not a transmitted free gain for a codec.
        a /= max(float(np.abs(a).max()), 1e-8)
        b /= max(float(np.abs(b).max()), 1e-8)
        return torch.from_numpy(a), torch.from_numpy(b), torch.from_numpy(pitch[first_frame:first_frame+frames].copy())
