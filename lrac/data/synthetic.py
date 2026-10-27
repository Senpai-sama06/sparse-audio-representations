"""Seeded signal families with explicit held-out regions and known components."""
import numpy as np
from scipy.signal import butter, sosfilt, chirp


def generate(spec):
    rng = np.random.default_rng(int(spec["seed"]))
    sr, n = int(spec["sample_rate"]), int(spec["num_samples"])
    t = np.arange(n, dtype=np.float64) / sr
    regime = spec.get("regime", "iid")
    family = spec.get("family", "mixture")
    if regime == "families":
        family = ["chirp", "am_fm", "damped"][int(spec["seed"]) % 3]
    if family != "mixture":
        f0 = rng.uniform(100, min(1200, sr * .15))
        if family == "chirp":
            y = .25 * chirp(t, f0=f0, f1=min(sr * .4, f0 * 3), t1=max(t[-1], 1 / sr), method="quadratic")
        elif family == "am_fm":
            y = .25 * (1 + .5 * np.sin(2 * np.pi * 7 * t)) * np.sin(2 * np.pi * f0 * t + 4 * np.sin(2 * np.pi * 3 * t))
        elif family == "damped":
            y = .4 * np.exp(-rng.uniform(3, 12) * t) * np.sin(2 * np.pi * f0 * t)
        else:
            raise ValueError(f"Unknown synthetic family: {family}")
        return y.astype(np.float32), {family: y.astype(np.float32)}, {"family": family, "frequency": f0}
    nyquist = sr / 2
    if regime == "extrapolation":
        f0 = rng.uniform(420, min(700, sr * .2))
        cutoff = rng.uniform(.70, .88) * nyquist
    elif regime == "combinations":
        f0 = rng.uniform(220, 360)
        cutoff = rng.uniform(.375, .60) * nyquist
    else:
        # The high-frequency / high-cutoff quadrant is withheld from training.
        while True:
            f0 = rng.uniform(70, min(360, sr * .2))
            cutoff = rng.uniform(.08, .60) * nyquist
            if not (f0 > 220 and cutoff > .375 * nyquist):
                break
    phase = rng.uniform(-np.pi, np.pi)
    modulation = rng.uniform(2, 5) if regime != "extrapolation" else rng.uniform(8, 12)
    envelope = .65 + .25 * np.sin(2 * np.pi * modulation * t + phase)
    periodic = np.zeros(n)
    harmonics = min(24, int((nyquist - 1) / f0))
    for h in range(1, harmonics + 1):
        periodic += np.sin(2 * np.pi * h * f0 * t + phase * h) / h
    periodic *= rng.uniform(.12, .35) * envelope / max(1, sum(1 / h for h in range(1, harmonics + 1)))
    noise = rng.normal(0, rng.uniform(.005, .05), n) * envelope
    transient = np.zeros(n)
    for _ in range(int(rng.integers(1, 4))):
        start = int(rng.integers(0, max(1, n - 1)))
        length = min(n - start, max(8, int(sr * rng.uniform(.003, .02))))
        transient[start:start + length] += rng.normal(0, .12, length) * np.exp(-np.linspace(0, 5, length))
    interacting = bool(rng.integers(0, 2))
    sos1 = butter(3, cutoff, fs=sr, output="sos")
    sos2 = butter(3, min(nyquist * .95, cutoff * 1.4), fs=sr, output="sos")
    blend = np.linspace(0, 1, n) if interacting else np.zeros(n)
    parts = {}
    for name, values in {"harmonic": periodic, "noise": noise, "transient": transient}.items():
        parts[name] = ((1 - blend) * sosfilt(sos1, values) + blend * sosfilt(sos2, values)).astype(np.float32)
    signal = sum(parts.values())
    factors = {"family": "mixture", "f0": float(f0), "cutoff": float(cutoff), "phase": float(phase), "modulation": float(modulation), "interacting": interacting}
    return signal.astype(np.float32), parts, factors


def records(split, count, sample_rate=16000, seconds=1.0):
    offsets = {"train": 0, "validation": 100000000, "iid": 200000000, "combinations": 300000000, "extrapolation": 400000000, "families": 500000000}
    if count >= 100000000:
        raise ValueError("Synthetic count exceeds reserved seed range")
    for i in range(count):
        yield {"id": f"synthetic:{split}:{i:07d}", "corpus": "synthetic", "split": split, "role": "train" if split == "train" else "validation" if split == "validation" else "test", "speaker": None, "sample_rate": sample_rate, "num_samples": round(sample_rate * seconds), "seed": offsets[split] + i, "regime": "iid" if split in {"train", "validation"} else split, "family": "mixture"}
