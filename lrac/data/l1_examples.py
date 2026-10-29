"""Deterministic pilot cohorts selected before inspecting sparse-coding results."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from lrac.data.preprocessing import load_audio
from lrac.linear import CosineDictionary
from lrac.utils import file_hash


def synthetic_examples(samples, sample_rate, seed):
    """Return impulses, an exact cosine atom, mixed structure, noise and silence."""
    rng = np.random.default_rng(seed)
    t = np.arange(samples)/sample_rate
    impulse = np.zeros(samples)
    impulse[samples//3] = .8
    coefficients = np.zeros(samples)
    coefficients[min(samples-1, max(1, round(440*2*samples/sample_rate)))] = .15*np.sqrt(samples)
    cosine = CosineDictionary(samples, "dct").synthesis(coefficients)
    mixture = .15*np.sin(2*np.pi*130*t) + .07*np.sin(2*np.pi*390*t)
    mixture[samples//3] += .5
    signals = dict(silence=np.zeros(samples), impulse=impulse, cosine=cosine,
                   mixture=mixture, noise=.05*rng.standard_normal(samples))
    return [dict(id=name, speaker=name, role="synthetic", waveform=x,
                 sample_rate=sample_rate, seed=seed) for name, x in signals.items()]


def speech_examples(manifest, exclusions, *, samples, sample_rate, speakers=40,
                    clips_per_speaker=3, seed=0):
    """Select centered crops from distinct unused dev-clean utterances.

    No energy, pitch, model response or reconstruction score influences selection.
    The shuffled speaker split is fixed before fitting/coding any recording.
    """
    manifest, exclusions = Path(manifest), Path(exclusions)
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    with exclusions.open(newline="", encoding="utf-8") as stream:
        excluded = {r["id"] for r in csv.DictReader(stream)}
    if not excluded or not excluded <= {r["id"] for r in rows}:
        raise ValueError("Checkpoint-selection exclusions must belong to this manifest")
    if speakers < 2 or speakers % 2 or clips_per_speaker < 1:
        raise ValueError("Use an even positive speaker count and positive clips per speaker")
    grouped = {}
    for row in rows:
        if row.get("split") != "dev-clean" or row.get("role") != "validation":
            raise ValueError("The L1 speech pilot only accepts development recordings")
        if row["sample_rate"] != sample_rate:
            raise ValueError("Pilot source sample rates must match; no hidden resampling")
        extent = int(row.get("stop", row["num_samples"])) - int(row.get("start", 0))
        if row["id"] not in excluded and extent >= samples:
            grouped.setdefault(str(row["speaker"]), []).append(row)
    ordered = sorted(grouped, key=int)
    if len(ordered) < speakers:
        raise ValueError("Not enough eligible development speakers")
    selected = list(np.random.default_rng(seed).permutation(ordered)[:speakers])
    cohort = []
    for index, speaker in enumerate(selected):
        candidates = sorted(grouped[speaker], key=lambda r: r["id"])
        if len(candidates) < clips_per_speaker:
            raise ValueError(f"Not enough unused utterances for speaker {speaker}")
        for source in candidates[:clips_per_speaker]:
            row = dict(source)
            start, stop = int(row.get("start", 0)), int(row.get("stop", row["num_samples"]))
            row["start"] = start + (stop-start-samples)//2
            row["stop"] = row["start"] + samples
            row["role"] = "calibration" if index < speakers//2 else "measurement"
            row["source_sha256"] = file_hash(row["path"])
            cohort.append(row)
    if len({r["id"] for r in cohort}) != len(cohort):
        raise ValueError("Repeated source recording in pilot")
    return cohort, dict(manifest_sha256=file_hash(manifest), exclusion_csv_sha256=file_hash(exclusions),
                        excluded_ids=sorted(excluded), source_speakers=ordered,
                        calibration_speakers=selected[:speakers//2], measurement_speakers=selected[speakers//2:])


def read_example(row, sample_rate):
    """Read the exact selected extent; retain original amplitude and DC offset."""
    x = np.asarray(row["waveform"], dtype=np.float64) if "waveform" in row else load_audio(row, sample_rate).astype(np.float64)
    digest = hashlib.sha256(np.asarray(x, dtype="<f8").tobytes()).hexdigest()
    return x, digest
