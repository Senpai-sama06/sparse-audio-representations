"""Standalone adaptive STFT dictionary model and LibriSpeech training script.

Copy this file to another machine. No project modules are required.
Python 3.11+; install: pip install torch numpy soundfile kagglehub
Install a PyTorch build that matches the target machine's CUDA installation.

Commands (no command prints help and does not acquire data):
  python adaptive_stft_dictionary.py --help
  python adaptive_stft_dictionary.py self-check
  python adaptive_stft_dictionary.py prepare --download --output runs/stft_adaptive
  python adaptive_stft_dictionary.py train --output runs/stft_adaptive
  python adaptive_stft_dictionary.py train --download --output runs/new_stft_run
  python adaptive_stft_dictionary.py train --output runs/stft_adaptive --resume
  python adaptive_stft_dictionary.py evaluate --output runs/stft_adaptive --split test-clean

Set ordinary Python defaults in Settings below, or override them on the CLI.
Downloads require --download. Model import, --help and self-check are offline.
prepare indexes data but never trains. train never evaluates test-clean.

Default source: pypiahmad/librispeech-asr-corpus/versions/1 on Kaggle.
This mirror contains train-clean-360, dev-clean and test-clean. Credentials, if
required, use KaggleHub's normal environment or login setup; none are embedded.
An existing extracted corpus can instead be supplied with --data-root PATH.
Only complete fixed-length excerpts enter a batch. Short utterances are listed
as exclusions, so sample/frame masks are all valid; STFT boundary padding still
uses the declared constant-padding rule. Variable-length batching is not used.

Architecture: normalized waveform -> complex STFT -> frequency-aware CNN ->
query-conditioned atom generator -> complex coefficients -> shared alternating
updates -> projected dictionary product -> inverse STFT -> restored gain.
One dictionary serves each full excerpt. There is no waveform decoder bypass.
This first experiment permits dense coefficients. It does not claim compression.

Variants: refined = generated atoms plus D/A updates; generated = generated atoms
plus A-only updates; fixed = shared trainable atoms plus A-only updates. All use
the same configured number of coefficient steps. Refined adds dictionary work;
wall time and parameter counts must also be considered in comparisons.

Outputs: settings, speaker audit, source and data identity, progress.json,
history.jsonl, validation records, initial/best/last checkpoints, stage snapshots,
CSV tables and a plain self-contained HTML report with embedded listening audio.
Validation uses fixed excerpts, not a full-corpus perceptual-quality benchmark.
Explicit evaluate is separate and does not modify the training checkpoint.

Inspiration (our implementation is a new complex spectral model):
  Zheng et al., Deep Convolutional Dictionary Learning for Image Denoising, CVPR 2021
  Li & Liu, seismic DCDL, 2024; doi:10.1016/j.rinam.2024.100516
  Szatkowski et al., HyperSound, ECML PKDD 2023; arxiv.org/abs/2302.04959
Design notes: documentation/adaptive_stft_dictionary_design.html (optional).
"""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import html
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from dataclasses import asdict, dataclass, fields
from typing import Any

import numpy as np
import soundfile as sf
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class Settings:
    """Edit these Python values to define a run. No YAML is required."""

    dataset: str = "pypiahmad/librispeech-asr-corpus/versions/1"
    train_split: str = "train-clean-360"
    validation_split: str = "dev-clean"
    test_split: str = "test-clean"
    sample_rate: int = 16000
    seconds: float = 1.0
    n_fft: int = 512
    hop: int = 128
    atoms: int = 32
    width: int = 128
    heads: int = 4
    stages: int = 4
    initial_steps: int = 4
    variant: str = "refined"
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    steps: int = 20000
    batch_size: int = 8
    workers: int = 0
    validate_every: int = 500
    checkpoint_every: int = 250
    log_every: int = 20
    validation_examples: int = 64
    control_examples: int = 4
    inspection_examples: int = 3
    seed: int = 42
    complex_weight: float = 1.0
    waveform_weight: float = 1.0
    spectral_weight: float = 1.0
    intermediate_weight: float = 0.5
    gain_floor: float = 1e-5
    clip_grad: float = 1.0

    @property
    def samples(self) -> int:
        return round(self.seconds * self.sample_rate)

    def validate(self) -> None:
        positive = ("sample_rate", "seconds", "n_fft", "hop", "atoms", "width",
                    "heads", "steps", "batch_size", "validate_every", "log_every",
                    "checkpoint_every", "validation_examples", "gain_floor", "clip_grad")
        for key in positive:
            if not math.isfinite(float(getattr(self, key))) or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be positive and finite")
        if self.n_fft % 2 or self.hop > self.n_fft // 2:
            raise ValueError("Use an even FFT length and a hop no greater than half its length")
        if self.width % self.heads or self.width < 8:
            raise ValueError("width must be at least 8 and divisible by heads")
        if self.samples < self.n_fft:
            raise ValueError("The excerpt must be at least one FFT window long")
        for key in ("stages", "initial_steps", "workers", "control_examples", "inspection_examples",
                    "complex_weight", "waveform_weight", "spectral_weight", "intermediate_weight",
                    "weight_decay"):
            if not math.isfinite(float(getattr(self, key))) or getattr(self, key) < 0:
                raise ValueError(f"{key} must be nonnegative and finite")
        if self.learning_rate <= 0 or not math.isfinite(self.learning_rate):
            raise ValueError("learning_rate must be positive and finite")
        if self.complex_weight + self.waveform_weight + self.spectral_weight == 0:
            raise ValueError("At least one reconstruction loss must be active")
        if self.variant not in {"fixed", "generated", "refined"}:
            raise ValueError("Unknown model variant")
        if not self.train_split.startswith("train-"):
            raise ValueError("Training must use an official training partition")
        if not self.validation_split.startswith("dev-") or not self.test_split.startswith("test-"):
            raise ValueError("Use official dev and test partitions for their corresponding roles")


def as_channels(x: Tensor) -> Tensor:
    """Pack B x F x T complex values into B x 2 x F x T real values."""
    return torch.stack((x.real, x.imag), dim=1)


def as_complex(x: Tensor) -> Tensor:
    return torch.complex(x[:, 0], x[:, 1])


def project_endpoints(x: Tensor) -> Tensor:
    """Pi is a real-linear, self-adjoint projection for the even FFT used here."""
    keep = x.real.new_ones(x.shape[-2])
    keep[0] = keep[-1] = 0
    return torch.complex(x.real, x.imag * keep[:, None])


def normalize_pair(d: Tensor, a: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
    """Normalize columns and compensate coefficient rows, preserving D @ A."""
    norms = d.abs().square().sum(dim=1, keepdim=True).sqrt().clamp_min(1e-8)
    return d / norms, None if a is None else a * norms.transpose(1, 2)


class AudioTransform(nn.Module):
    """One reversible gain and an explicitly matched STFT/iSTFT pair."""

    def __init__(self, cfg: Settings):
        super().__init__()
        self.cfg = cfg
        self.register_buffer("window", torch.hann_window(cfg.n_fft, periodic=True))

    def stft(self, x: Tensor, n_fft: int | None = None) -> Tensor:
        n = n_fft or self.cfg.n_fft
        window = self.window if n == self.cfg.n_fft else torch.hann_window(n, device=x.device, dtype=x.dtype)
        return torch.stft(x, n, hop_length=self.cfg.hop if n == self.cfg.n_fft else n // 4,
                          win_length=n, window=window, center=True, pad_mode="constant",
                          normalized=False, onesided=True, return_complex=True)

    def analyze(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if x.ndim != 2 or not torch.isfinite(x).all():
            raise ValueError("Expected finite audio shaped (batch, samples)")
        gain = x.square().mean(-1, keepdim=True).sqrt().clamp_min(self.cfg.gain_floor)
        normalized = x / gain
        return self.stft(normalized), normalized, gain

    def inverse(self, s: Tensor, samples: int) -> Tensor:
        return torch.istft(project_endpoints(s), self.cfg.n_fft, hop_length=self.cfg.hop,
                           win_length=self.cfg.n_fft, window=self.window, center=True,
                           normalized=False, onesided=True, length=samples)


class ResidualBlock(nn.Module):
    """Convolutions preserve the time grid; normalization is per example."""

    def __init__(self, channels: int):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1),
                                    nn.GroupNorm(8, channels), nn.SiLU(),
                                    nn.Conv2d(channels, channels, 3, padding=1),
                                    nn.GroupNorm(8, channels))

    def forward(self, x: Tensor) -> Tensor:
        return F.silu(x + self.layers(x))


class RecordingEncoder(nn.Module):
    """Observe both complex channels and absolute frequency coordinates."""

    def __init__(self, cfg: Settings):
        super().__init__()
        layers: list[nn.Module] = []
        previous = 3
        for i, channels in enumerate((32, 64, 128)):
            layers.extend((nn.Conv2d(previous, channels, 3, stride=(1 if i == 0 else 2, 1), padding=1),
                           nn.GroupNorm(8, channels), nn.SiLU(), ResidualBlock(channels)))
            previous = channels
        self.features = nn.Sequential(*layers)
        self.tokens = nn.Sequential(nn.Linear(128 * 8, cfg.width), nn.LayerNorm(cfg.width))

    def forward(self, s: Tensor) -> Tensor:
        ri = as_channels(s)
        freq = torch.linspace(0, 1, s.shape[1], device=s.device, dtype=s.real.dtype)
        coords = freq[None, None, :, None].expand(s.shape[0], 1, -1, s.shape[2])
        x = self.features(torch.cat((ri, coords), dim=1))
        x = F.adaptive_avg_pool2d(x, (8, x.shape[-1]))
        return self.tokens(x.flatten(1, 2).transpose(1, 2))


class AtomGenerator(nn.Module):
    """Queries are shared; the resulting atoms depend on the recording."""

    def __init__(self, cfg: Settings):
        super().__init__()
        self.bins = cfg.n_fft // 2 + 1
        self.queries = nn.Parameter(torch.randn(cfg.atoms, cfg.width) / math.sqrt(cfg.width))
        self.attention = nn.MultiheadAttention(cfg.width, cfg.heads, dropout=0, batch_first=True)
        self.norm = nn.LayerNorm(cfg.width)
        self.ffn = nn.Sequential(nn.Linear(cfg.width, 2 * cfg.width), nn.SiLU(),
                                 nn.Linear(2 * cfg.width, cfg.width))
        self.output = nn.Linear(cfg.width, 2 * self.bins)

    def forward(self, h: Tensor) -> tuple[Tensor, Tensor]:
        q = self.queries[None].expand(h.shape[0], -1, -1)
        attended, weights = self.attention(q, h, h, need_weights=True)
        q = self.norm(q + attended)
        q = q + self.ffn(q)
        ri = self.output(q).reshape(h.shape[0], -1, 2, self.bins).permute(0, 2, 3, 1)
        d, _ = normalize_pair(as_complex(ri))
        return d, weights


class JointRefinement(nn.Module):
    """A shared unrolled stage with exact projected-fidelity data gradients.

    For Lc = ||Pi(DA)-S||^2 / energy, grad_A = 2 D^H Pi(R)/energy.
    grad_D = 2 Pi(R) A^H/energy. Pi is self-adjoint over real/imag values.
    Frobenius norms bound the operator norms used for conservative step sizes.
    The loss denominator cancels in eta * grad. This implementation keeps that
    cancellation explicit instead of taking tiny steps on a normalized loss.
    All operations remain differentiable through the unrolled stages.
    """

    def __init__(self, cfg: Settings):
        super().__init__()
        self.cfg = cfg
        self.context_a = nn.Linear(cfg.width, 16)
        self.context_d = nn.Linear(cfg.width, 8)
        self.correct_a = nn.Sequential(nn.Conv1d(2 * cfg.atoms + 16, 64, 3, padding=1),
                                       nn.SiLU(), nn.Conv1d(64, 2 * cfg.atoms, 3, padding=1))
        self.correct_d = nn.Sequential(nn.Conv1d(10, 32, 3, padding=1), nn.SiLU(),
                                       nn.Conv1d(32, 2, 3, padding=1))
        for network in (self.correct_a, self.correct_d):
            nn.init.zeros_(network[-1].weight)
            nn.init.zeros_(network[-1].bias)
        self.step_a = nn.Parameter(torch.tensor(0.0))
        self.step_d = nn.Parameter(torch.tensor(0.0))

    @staticmethod
    def a_step(d: Tensor, a: Tensor, s: Tensor, fraction: Tensor | float = 0.9) -> Tensor:
        residual = project_endpoints(d @ a) - s
        bound = d.abs().square().sum((1, 2), keepdim=True).clamp_min(1e-8)
        return a - fraction * (d.mH @ project_endpoints(residual)) / bound

    def forward(self, d: Tensor, a: Tensor, s: Tensor, h: Tensor,
                update_dictionary: bool) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        def error(di: Tensor, ai: Tensor) -> Tensor:
            return (project_endpoints(di @ ai) - s).abs().square().sum((1, 2)) / s.abs().square().sum((1, 2)).clamp_min(1e-8)

        stats = {"before": error(d, a)}
        a = self.a_step(d, a, s, self.step_a.sigmoid())
        stats["after_a_data"] = error(d, a)
        scale_a = a.abs().square().mean((1, 2), keepdim=True).sqrt().detach().clamp_min(1e-4)
        ari = as_channels(a / scale_a).flatten(1, 2)
        ca = self.context_a(h).transpose(1, 2)
        delta_a = self.correct_a(torch.cat((ari, ca), dim=1)).tanh()
        a = a + 0.05 * scale_a * as_complex(delta_a.reshape(a.shape[0], 2, a.shape[1], a.shape[2]))
        stats["after_a_prior"] = error(d, a)
        if update_dictionary:
            residual = project_endpoints(d @ a) - s
            bound = a.abs().square().sum((1, 2), keepdim=True).clamp_min(1e-8)
            d = d - self.step_d.sigmoid() * (project_endpoints(residual) @ a.mH) / bound
            stats["after_d_data"] = error(d, a)
            batch, bins, atoms = d.shape
            dri = as_channels(d).permute(0, 3, 1, 2).reshape(batch * atoms, 2, bins)
            cd = self.context_d(h.mean(1))[:, None, :, None].expand(-1, atoms, -1, bins)
            delta_d = self.correct_d(torch.cat((dri, cd.reshape(batch * atoms, 8, bins)), dim=1)).tanh()
            delta_d = delta_d.reshape(batch, atoms, 2, bins).permute(0, 2, 3, 1)
            d = d + (0.05 / math.sqrt(bins)) * as_complex(delta_d)
            stats["after_d_prior"] = error(d, a)
        d, a = normalize_pair(d, a)
        stats["after_normalization"] = error(d, a)
        return d, a, stats


@dataclass
class ModelOutput:
    spectrum: Tensor
    normalized_audio: Tensor
    gain: Tensor
    dictionaries: list[Tensor]
    coefficients: list[Tensor]
    spectra: list[Tensor]
    waveforms: list[Tensor]
    attention: Tensor | None
    updates: list[dict[str, Tensor]]

    @property
    def audio(self) -> Tensor:
        return self.waveforms[-1] * self.gain


class AdaptiveSTFTDictionary(nn.Module):
    """The full model. All output audio passes through Pi(D @ A)."""

    def __init__(self, cfg: Settings):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.transform = AudioTransform(cfg)
        if cfg.variant == "fixed":
            self.fixed_atoms = nn.Parameter(torch.randn(1, 2, cfg.n_fft // 2 + 1, cfg.atoms))
        else:
            self.encoder = RecordingEncoder(cfg)
            self.generator = AtomGenerator(cfg)
        self.refinement = JointRefinement(cfg) if cfg.stages else None

    def forward(self, x: Tensor) -> ModelOutput:
        if x.shape[-1] != self.cfg.samples:
            raise ValueError(f"This run expects {self.cfg.samples} samples per excerpt")
        s, normalized, gain = self.transform.analyze(x)
        if self.cfg.variant == "fixed":
            d, _ = normalize_pair(as_complex(self.fixed_atoms).expand(x.shape[0], -1, -1))
            h = x.new_zeros(x.shape[0], s.shape[-1], self.cfg.width)
            attention = None
        else:
            h = self.encoder(s)
            d, attention = self.generator(h)
        bound = d.abs().square().sum((1, 2), keepdim=True).clamp_min(1e-8)
        a = d.mH @ s / bound
        for _ in range(self.cfg.initial_steps):
            a = JointRefinement.a_step(d, a, s)
        dictionaries, coefficients, spectra, waves, updates = [], [], [], [], []

        def retain() -> None:
            shat = project_endpoints(d @ a)
            dictionaries.append(d)
            coefficients.append(a)
            spectra.append(shat)
            waves.append(self.transform.inverse(shat, x.shape[-1]))

        retain()
        for _ in range(self.cfg.stages):
            d, a, stats = self.refinement(d, a, s, h, self.cfg.variant == "refined")
            updates.append(stats)
            retain()
        return ModelOutput(s, normalized, gain, dictionaries, coefficients, spectra, waves, attention, updates)


def reconstruction_terms(transform: AudioTransform, target: Tensor, s: Tensor,
                         estimate: Tensor, shat: Tensor) -> dict[str, Tensor]:
    """Normalized per-example losses. Silence uses explicit energy floors."""
    complex_loss = (shat - s).abs().square().sum((1, 2)) / s.abs().square().sum((1, 2)).clamp_min(1e-8)
    wave_loss = (target - estimate).abs().mean(-1) / target.abs().mean(-1).clamp_min(1e-4)
    spectral = target.new_zeros(target.shape[0])
    for n in (128, 256, 512, 1024):
        ref = transform.stft(target, n).abs()
        pred = transform.stft(estimate, n).abs()
        convergence = (ref - pred).flatten(1).norm(dim=1) / ref.flatten(1).norm(dim=1).clamp_min(1e-4)
        log_error = ((ref + 1e-4).log() - (pred + 1e-4).log()).abs().mean((1, 2))
        spectral = spectral + (convergence + log_error) / 4
    return {"complex": complex_loss, "waveform_l1": wave_loss, "spectral": spectral}


def objective(model: AdaptiveSTFTDictionary, result: ModelOutput) -> tuple[Tensor, dict[str, float]]:
    cfg = model.cfg
    score = result.normalized_audio.new_zeros(())
    summary: dict[str, float] = {}
    count = len(result.spectra)
    for stage, (shat, wave) in enumerate(zip(result.spectra, result.waveforms)):
        terms = reconstruction_terms(model.transform, result.normalized_audio, result.spectrum, wave, shat)
        value = (cfg.complex_weight * terms["complex"] + cfg.waveform_weight * terms["waveform_l1"]
                 + cfg.spectral_weight * terms["spectral"]).mean()
        weight = 1.0 if stage == count - 1 else cfg.intermediate_weight / max(count - 1, 1)
        score = score + weight * value
        for key, term in terms.items():
            summary[f"stage_{stage}_{key}"] = float(term.mean().detach())
        summary[f"stage_{stage}_score"] = float(value.detach())
    summary["objective"] = float(score.detach())
    return score, summary


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def identity(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    replace_file(temp, path)


def replace_file(temp: Path, target: Path) -> None:
    """Retry short Windows file locks from readers of a live run."""
    for attempt in range(6):
        try:
            os.replace(temp, target)
            return
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(0.05 * (attempt + 1))


def append_json(path: Path, value: Any) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, allow_nan=False) + "\n")


def seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def environment() -> dict[str, Any]:
    versions = {}
    for name in ("torch", "numpy", "soundfile", "kagglehub"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"python": sys.version, "packages": versions, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


def stable_int(*parts: Any) -> int:
    return int.from_bytes(hashlib.sha256(":".join(map(str, parts)).encode()).digest()[:8], "little")


def data_settings(cfg: Settings) -> dict[str, Any]:
    keys = ("dataset", "train_split", "validation_split", "test_split", "sample_rate",
            "seconds", "seed", "validation_examples")
    return {key: getattr(cfg, key) for key in keys}


def select_excerpts(records: list[dict], count: int, cfg: Settings) -> list[dict]:
    """Round-robin speakers; fixed hashed utterance ordering and crop offsets."""
    speakers: dict[str, list[dict]] = {}
    for record in records:
        speakers.setdefault(record["speaker"], []).append(record)
    order = sorted(speakers, key=lambda s: stable_int(cfg.seed, "speaker", s))
    for speaker in order:
        speakers[speaker].sort(key=lambda row: stable_int(cfg.seed, row["relative"]))
    chosen = []
    depth = 0
    while len(chosen) < min(count, len(records)):
        for speaker in order:
            if depth < len(speakers[speaker]) and len(chosen) < count:
                row = dict(speakers[speaker][depth])
                row["offset"] = stable_int(cfg.seed, "offset", row["relative"]) % (row["frames"] - cfg.samples + 1)
                chosen.append(row)
        depth += 1
    return chosen


def prepare_data(args: argparse.Namespace, cfg: Settings, out: Path) -> dict:
    """Resolve KaggleHub only on explicit request; then index official splits.

    The source may contain nested split/LibriSpeech/split folders. Split names
    must occur as exact path components. Files shorter than the fixed excerpt
    are excluded and listed. No energy or reconstruction-quality filtering is
    performed. A file with an invalid header or naming scheme stops preparation.
    The roster fingerprint covers metadata, not every source audio byte.
    """
    manifest_path = out / "data_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        actual = identity({key: value for key, value in manifest.items() if key not in {"root", "fingerprint"}})
        if actual != manifest["fingerprint"]:
            raise ValueError("The saved data manifest changed or is incomplete")
        if manifest["data_settings"] != data_settings(cfg):
            raise ValueError("Existing data selection differs; choose a fresh output directory")
        root = Path(args.data_root).resolve() if args.data_root else Path(manifest["root"])
        if not root.is_dir():
            raise FileNotFoundError(f"Data root is missing: {root}. Use --data-root to relocate it")
        return {**manifest, "root": str(root)}
    if args.data_root:
        root = Path(args.data_root).expanduser().resolve()
    elif args.download:
        try:
            import kagglehub
        except ImportError as error:
            raise ImportError("Install kagglehub or supply --data-root") from error
        print(f"KaggleHub source: {cfg.dataset}", flush=True)
        root = Path(kagglehub.dataset_download(cfg.dataset)).resolve()
    else:
        raise ValueError("Supply --data-root or explicitly request KaggleHub with --download")
    if not root.is_dir():
        raise FileNotFoundError(root)
    out.mkdir(parents=True, exist_ok=True)
    partitions = (cfg.train_split, cfg.validation_split, cfg.test_split)
    records: dict[str, list[dict]] = {part: [] for part in partitions}
    excluded = []
    utterances: dict[str, str] = {}
    seen = 0
    for path in sorted(root.rglob("*.flac")):
        relative = path.relative_to(root)
        labels = set((root.name, *relative.parts[:-1])).intersection(partitions)
        if not labels:
            continue
        if len(labels) != 1:
            raise ValueError(f"Ambiguous split in path: {relative}")
        split = labels.pop()
        parts = path.stem.split("-")
        if len(parts) != 3 or not all(piece.isdigit() for piece in parts):
            raise ValueError(f"Expected LibriSpeech speaker-chapter-utterance naming: {path}")
        if path.stem in utterances:
            raise ValueError(f"Duplicate utterance {path.stem}: {relative} and {utterances[path.stem]}")
        utterances[path.stem] = relative.as_posix()
        info = sf.info(path)
        if info.samplerate != cfg.sample_rate or info.channels != 1:
            raise ValueError(f"Expected mono {cfg.sample_rate} Hz LibriSpeech, found {info}: {path}")
        stat = path.stat()
        record = {"relative": relative.as_posix(), "speaker": parts[0], "utterance": path.stem,
                  "split": split, "frames": info.frames, "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        if info.frames < cfg.samples:
            excluded.append({**record, "reason": "shorter_than_fixed_excerpt"})
        else:
            records[split].append(record)
        seen += 1
        if seen % 1000 == 0:
            atomic_json(out / "progress.json", {"status": "preparing", "files_indexed": seen, "heartbeat": time.time()})
            print(f"Indexed {seen} audio headers", flush=True)
    for split in partitions:
        if not records[split]:
            raise ValueError(f"No eligible extracted FLAC files for {split} under {root}. No fallback split is used")
    sets = {split: {row["speaker"] for row in rows} for split, rows in records.items()}
    for i, split in enumerate(partitions):
        for other in partitions[i + 1:]:
            if sets[split] & sets[other]:
                raise ValueError(f"Speaker leakage between {split} and {other}")
    manifest = {"format": 1, "root": str(root), "data_settings": data_settings(cfg),
                "records": records, "excluded": excluded,
                "validation": select_excerpts(records[cfg.validation_split], cfg.validation_examples, cfg),
                "fingerprint_scope": "relative paths, audio headers, file sizes and modification times; not full audio hashes"}
    manifest["fingerprint"] = identity({key: value for key, value in manifest.items() if key != "root"})
    atomic_json(manifest_path, manifest)
    atomic_json(out / "speaker_audit.json", {"disjoint": True, "partitions": {
        split: {"utterances": len(records[split]), "speakers": sorted(sets[split])} for split in partitions}})
    atomic_json(out / "progress.json", {"status": "prepared", "files_indexed": seen,
                                        "training_started": False, "heartbeat": time.time()})
    print(f"Prepared {out / 'data_manifest.json'}; no training in prepare mode", flush=True)
    return manifest


def read_excerpt(root: Path, row: dict, offset: int, cfg: Settings) -> Tensor:
    path = root / row["relative"]
    stat = path.stat()
    if stat.st_size != row["bytes"] or stat.st_mtime_ns != row["mtime_ns"]:
        raise ValueError(f"Audio metadata changed after preparation: {path}")
    audio, rate = sf.read(path, start=offset, frames=cfg.samples, dtype="float32", always_2d=True)
    if rate != cfg.sample_rate or audio.shape != (cfg.samples, 1) or not np.isfinite(audio).all():
        raise ValueError(f"Invalid excerpt at offset {offset}: {path}")
    return torch.from_numpy(audio[:, 0].copy())


class SpeechExcerpts(Dataset):
    """Stateless crop draws make resume independent of worker prefetch order."""

    def __init__(self, manifest: dict, cfg: Settings):
        self.root, self.cfg = Path(manifest["root"]), cfg
        self.by_speaker: dict[str, list[dict]] = {}
        for row in manifest["records"][cfg.train_split]:
            self.by_speaker.setdefault(row["speaker"], []).append(row)
        self.speakers = sorted(self.by_speaker)

    def __len__(self) -> int:
        return self.cfg.steps * self.cfg.batch_size

    def __getitem__(self, index: int) -> Tensor:
        rng = random.Random(stable_int(self.cfg.seed, "training_draw", index))
        speaker = self.speakers[rng.randrange(len(self.speakers))]
        rows = self.by_speaker[speaker]
        row = rows[rng.randrange(len(rows))]
        offset = rng.randrange(row["frames"] - self.cfg.samples + 1)
        return read_excerpt(self.root, row, offset, self.cfg)


def waveform_metrics(reference: Tensor, estimate: Tensor, edge: int) -> dict[str, float | None]:
    """No time alignment or gain fit. SI-SDR has its normal scale-invariant definition."""
    x, y = reference.double().flatten(), estimate.double().flatten()
    error = (x - y).square()
    energy = float(x.square().sum())
    centered = x - x.mean()
    y_centered = y - y.mean()
    center_energy = centered.square().sum()
    if float(center_energy) > 1e-12:
        desired = centered * (centered @ y_centered) / center_energy
        desired_energy = float(desired.square().sum())
        si_sdr = 10 * math.log10(max(desired_energy, 1e-20) / max(float((y_centered - desired).square().sum()), 1e-20))
    else:
        si_sdr = None
    border = min(edge, max(1, len(x) // 4))
    edges = torch.cat((error[:border], error[-border:]))
    edge_energy = float(torch.cat((x[:border], x[-border:])).square().sum())
    interior_energy = float(x[border:-border].square().sum())
    return {"mse": float(error.mean()), "nmse": float(error.sum()) / energy if energy > 1e-12 else None,
            "si_sdr_db": si_sdr, "edge_nmse": float(edges.sum()) / edge_energy if edge_energy > 1e-12 else None,
            "interior_nmse": float(error[border:-border].sum()) / interior_energy if interior_energy > 1e-12 else None,
            "silence": energy <= 1e-12}


def fixed_dictionary_solve(d: Tensor, s: Tensor) -> Tensor:
    """Strong projected-objective control, solved on CPU in float64.

    Realify D @ A, remove the unobservable imaginary endpoint rows, and solve
    the resulting real least-squares problem by rank-revealing SVD. This solves
    the same Pi objective, unlike a naive complex solve followed by projection.
    """
    dc, sc = d[0].detach().cpu().to(torch.complex128), s[0].detach().cpu().to(torch.complex128)
    top = torch.cat((dc.real, -dc.imag), dim=1)
    bottom = torch.cat((dc.imag, dc.real), dim=1)
    bottom = bottom.clone()
    bottom[0] = bottom[-1] = 0
    matrix = torch.cat((top, bottom), dim=0)
    target = torch.cat((sc.real, sc.imag), dim=0)
    solution = torch.linalg.lstsq(matrix, target, driver="gelsd").solution
    atoms = dc.shape[1]
    a = torch.complex(solution[:atoms], solution[atoms:])[None].to(s.device, s.dtype)
    return project_endpoints(d @ a)


def aggregate(rows: list[dict]) -> dict:
    result = {}
    for stage in sorted({row["stage"] for row in rows}):
        subset = [row for row in rows if row["stage"] == stage]
        metrics = {}
        for key in subset[0]:
            values = [float(row[key]) for row in subset if isinstance(row.get(key), (float, int))
                      and not isinstance(row.get(key), bool) and math.isfinite(row[key])]
            if values:
                metrics[key] = {"mean": sum(values) / len(values), "count": len(values)}
        result[stage] = {"examples": len(subset), "metrics": metrics}
    return result


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def audio_uri(x: Tensor, rate: int) -> str:
    stream = io.BytesIO()
    sf.write(stream, x.detach().cpu().numpy().reshape(-1), rate, format="WAV", subtype="FLOAT")
    return "data:audio/wav;base64," + base64.b64encode(stream.getvalue()).decode("ascii")


def write_report(folder: Path, summary: dict, examples: list[dict], cfg: Settings, role: str, step: int) -> None:
    """Plain, portable HTML. All listening audio remains inside this one file."""
    esc = html.escape
    parts = ["<!doctype html><html lang='en'><meta charset='utf-8'>",
             "<meta name='viewport' content='width=device-width,initial-scale=1'>",
             "<title>Adaptive STFT dictionary results</title><body>",
             "<h1>Adaptive STFT dictionary results</h1>",
             f"<p>Role: {esc(role)}. Step: {step}. Variant: {esc(cfg.variant)}. Atoms: {cfg.atoms}.</p>",
             "<p>These results use fixed excerpts. They do not establish full-utterance perceptual quality or compression.</p>",
             "<p>Control rows can use fewer examples. Their counts are explicit. Raw SVD error is a complex-matrix reference; it is not a bound on projected synthesis or waveform loss.</p>",
             "<table border='1' cellpadding='6'><tr><th>Stage</th><th>Examples</th><th>Metric</th><th>Mean</th><th>Valid count</th></tr>"]
    for stage, item in summary.items():
        for metric, value in item["metrics"].items():
            parts.append(f"<tr><td>{esc(stage)}</td><td>{item['examples']}</td><td>{esc(metric)}</td>"
                         f"<td>{value['mean']:.7g}</td><td>{value['count']}</td></tr>")
    parts.append("</table><h2>Listening examples</h2><p>Audio uses the original input gain. No output alignment or automatic gain fit is applied.</p>")
    for example in examples:
        parts.append(f"<h3>{esc(example['utterance'])}; offset {example['offset']} samples</h3>")
        for label, wave in example["audio"].items():
            parts.append(f"<p>{esc(label)}</p><audio controls preload='none' src='{audio_uri(wave, cfg.sample_rate)}'></audio>")
    parts.append("<h2>Settings</h2><pre>" + esc(json.dumps(asdict(cfg), indent=2)) + "</pre></body></html>")
    temporary = folder / "report.html.tmp"
    temporary.write_text("\n".join(parts), encoding="utf-8")
    replace_file(temporary, folder / "report.html")


@torch.no_grad()
def evaluate(model: AdaptiveSTFTDictionary, manifest: dict, rows: list[dict],
             folder: Path, role: str, step: int, device: torch.device) -> tuple[float, dict]:
    """Inspect frozen model weights; analytic state updates also work under no_grad."""
    cfg = model.cfg
    model.eval()
    folder.mkdir(parents=True, exist_ok=True)
    records, examples, source_rows = [], [], []
    previous_d = None
    for index, row in enumerate(rows):
        x = read_excerpt(Path(manifest["root"]), row, row["offset"], cfg)[None].to(device)
        result = model(x)
        _, terms = objective(model, result)
        s = result.spectrum
        for stage, (d, a, shat, wave) in enumerate(zip(result.dictionaries, result.coefficients,
                                                     result.spectra, result.waveforms)):
            rec = {"utterance": row["utterance"], "speaker": row["speaker"], "offset": row["offset"],
                   "stage": f"model_{stage}", **waveform_metrics(x[0], wave[0] * result.gain[0], cfg.n_fft // 2),
                   "complex_nmse": float((shat - s).abs().square().sum() / s.abs().square().sum().clamp_min(1e-8)),
                   "score": terms[f"stage_{stage}_score"]}
            consistency = model.transform.stft(wave) - shat
            rec["consistency_nmse"] = float(consistency.abs().square().sum() / shat.abs().square().sum().clamp_min(1e-8))
            gram = (d.mH @ d).abs()
            off_diagonal = gram * (1 - torch.eye(cfg.atoms, device=device)[None])
            rec["max_atom_coherence"] = float(off_diagonal.max())
            rec["max_atom_norm_error"] = float((d.abs().square().sum(1).sqrt() - 1).abs().max())
            rec["coefficient_rms"] = float(a.abs().square().mean().sqrt())
            if stage:
                previous = result.dictionaries[stage - 1]
                rec["dictionary_relative_update"] = float((d - previous).abs().norm() / previous.abs().norm().clamp_min(1e-8))
            if stage:
                rec.update({key: float(value.mean()) for key, value in result.updates[stage - 1].items()})
            records.append(rec)
        if index < cfg.control_examples:
            for label, dictionary in (("same_dictionary_optimal_A", result.dictionaries[-1]),
                                      ("previous_recording_dictionary_optimal_A", previous_d)):
                if dictionary is None:
                    continue
                shat = fixed_dictionary_solve(dictionary, s)
                wave = model.transform.inverse(shat, cfg.samples) * result.gain
                records.append({"utterance": row["utterance"], "stage": label,
                                **waveform_metrics(x[0], wave[0], cfg.n_fft // 2),
                                "complex_nmse": float((shat - s).abs().square().sum() / s.abs().square().sum().clamp_min(1e-8))})
            u, singular, vh = torch.linalg.svd(s[0].cpu(), full_matrices=False)
            k = min(cfg.atoms, len(singular))
            raw = ((u[:, :k] * singular[:k]) @ vh[:k])[None].to(device)
            projected = project_endpoints(raw)
            wave = model.transform.inverse(projected, cfg.samples) * result.gain
            records.append({"utterance": row["utterance"], "stage": "svd_projected",
                            **waveform_metrics(x[0], wave[0], cfg.n_fft // 2),
                            "raw_complex_nmse": float((raw - s).abs().square().sum() / s.abs().square().sum().clamp_min(1e-8)),
                            "projected_complex_nmse": float((projected - s).abs().square().sum() / s.abs().square().sum().clamp_min(1e-8))})
        previous_d = result.dictionaries[-1].detach()
        source_rows.append({**row, "sha256": file_hash(Path(manifest["root"]) / row["relative"])})
        if index < cfg.inspection_examples:
            audio = {"original": x[0].cpu()}
            for stage, wave in enumerate(result.waveforms):
                audio[f"model stage {stage}"] = (wave[0] * result.gain[0]).cpu()
            examples.append({"utterance": row["utterance"], "offset": row["offset"], "audio": audio})
            arrays = {"reference": x[0].cpu().numpy(), "gain": result.gain.cpu().numpy(),
                      "spectrum": s.cpu().numpy(), "sample_mask": np.ones(cfg.samples, dtype=bool),
                      "frame_mask": np.ones(s.shape[-1], dtype=bool)}
            if result.attention is not None:
                arrays["attention"] = result.attention.cpu().numpy()
            for stage, (d, a) in enumerate(zip(result.dictionaries, result.coefficients)):
                arrays[f"dictionary_{stage}"] = d.cpu().numpy()
                arrays[f"coefficients_{stage}"] = a.cpu().numpy()
            np.savez_compressed(folder / f"inspection_{index:03d}.npz", **arrays)
        print(f"{role}: {index + 1}/{len(rows)} excerpts", flush=True)
    summary = aggregate(records)
    score = summary[f"model_{cfg.stages}"]["metrics"]["score"]["mean"]
    if not math.isfinite(score):
        raise FloatingPointError("Nonfinite validation score")
    write_csv(folder / "metrics.csv", records)
    atomic_json(folder / "summary.json", summary)
    atomic_json(folder / "sources.json", {"role": role, "examples": source_rows,
                                         "checkpoint_step": step, "source_sha256": file_hash(Path(__file__))})
    write_report(folder, summary, examples, cfg, role, step)
    return score, summary


class RunLock:
    """An OS lock prevents concurrent writers and releases after a process exit."""

    def __init__(self, out: Path):
        self.path = out / "run.lock"

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("a+b")
        self.stream.seek(0, 2)
        if self.stream.tell() == 0:
            self.stream.write(b"0")
            self.stream.flush()
        self.stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            self.stream.close()
            raise RuntimeError(f"Another process owns {self.path}") from error
        return self

    def __exit__(self, *unused):
        self.stream.close()


def choose_device(name: str) -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if name == "auto" else torch.device(name)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("This script supports CPU and CUDA complex operations")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def save_checkpoint(path: Path, model: AdaptiveSTFTDictionary, optimizer: torch.optim.Optimizer,
                    step: int, best: float, manifest: dict, elapsed: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {"format": 1, "settings": asdict(model.cfg), "model": model.state_dict(),
             "optimizer": optimizer.state_dict(), "step": step, "best_validation": best,
             "data_fingerprint": manifest["fingerprint"], "source_sha256": file_hash(Path(__file__)),
             "elapsed_active_seconds": elapsed, "torch_rng": torch.get_rng_state(),
             "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
             "python_rng": random.getstate()}
    temporary = path.with_suffix(".pt.tmp")
    torch.save(state, temporary)
    replace_file(temporary, path)


def train(args: argparse.Namespace, cfg: Settings, out: Path, device: torch.device) -> None:
    """Only this explicit command can perform optimizer updates."""
    cfg.validate()
    out.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out / "checkpoints" / "last.pt"
    if checkpoint_path.exists() and not args.resume:
        raise ValueError("A checkpoint exists. Use --resume or a fresh output directory")
    if args.resume and not checkpoint_path.exists():
        raise FileNotFoundError(f"No checkpoint to resume: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True) if args.resume else None
    if state:
        for key, value in state["settings"].items():
            if key not in {"steps", "workers"} and getattr(cfg, key) != value:
                raise ValueError(f"Resume setting differs: {key}")
        if cfg.steps < state["step"]:
            raise ValueError("Total steps precede the saved checkpoint")
        if state["source_sha256"] != file_hash(Path(__file__)):
            raise ValueError("The script changed since this checkpoint; use a fresh run")
    manifest = prepare_data(args, cfg, out)
    if state and state["data_fingerprint"] != manifest["fingerprint"]:
        raise ValueError("Resume data fingerprint differs")
    seed_all(cfg.seed)
    model = AdaptiveSTFTDictionary(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    start, best, prior_elapsed = 0, float("inf"), 0.0
    if state:
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start, best = state["step"], state["best_validation"]
        prior_elapsed = state["elapsed_active_seconds"]
        torch.set_rng_state(state["torch_rng"].cpu())
        random.setstate(state["python_rng"])
        if device.type == "cuda" and state["cuda_rng"]:
            if len(state["cuda_rng"]) != torch.cuda.device_count():
                raise ValueError("Resume GPU count differs; use the original visible-device configuration")
            torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda_rng"]])
        append_json(out / "resumes.jsonl", {"step": start, "time": time.time(), "settings": asdict(cfg)})
    elif (out / "history.jsonl").exists():
        raise ValueError("History exists without a checkpoint; choose a fresh output directory")
    atomic_json(out / "settings.json", asdict(cfg))
    atomic_json(out / "environment.json", environment())
    atomic_json(out / "model_information.json", {
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "device": str(device), "source_sha256": file_hash(Path(__file__)),
        "description": "One generated dictionary per excerpt; dense complex coefficients; no waveform bypass",
        "stage_count": cfg.stages, "variant": cfg.variant,
        "resume_scope": "model/optimizer/RNG restored; crop draws depend only on absolute draw index; cross-device bitwise identity not promised"})
    began = time.monotonic()
    last_saved = start

    def elapsed() -> float:
        return prior_elapsed + time.monotonic() - began

    def progress(status: str, step: int, **extra: Any) -> None:
        completed_here = max(0, step - start)
        eta = ((time.monotonic() - began) / completed_here * (cfg.steps - step)) if completed_here else None
        atomic_json(out / "progress.json", {"status": status, "step": step, "total_steps": cfg.steps,
            "elapsed_active_seconds": elapsed(), "eta_seconds": eta,
            "eta_note": "Provisional; validation and data preparation have different costs",
            "best_validation": best if math.isfinite(best) else None,
            "last_saved_step": last_saved, "heartbeat": time.time(), **extra})

    def validate(step: int) -> float:
        progress("validating", step)
        score, summary = evaluate(model, manifest, manifest["validation"],
                                  out / "validation" / f"step_{step:07d}", "validation", step, device)
        append_json(out / "validation.jsonl", {"step": step, "score": score, "summary": summary})
        return score

    current = start
    try:
        if not state:
            best = validate(0)
            for name in ("initial.pt", "best.pt", "last.pt"):
                save_checkpoint(out / "checkpoints" / name, model, optimizer, 0, best, manifest, elapsed())
        dataset = SpeechExcerpts(manifest, cfg)
        generator = torch.Generator().manual_seed(cfg.seed)
        loader = DataLoader(dataset, batch_size=cfg.batch_size,
                            sampler=range(start * cfg.batch_size, cfg.steps * cfg.batch_size),
                            num_workers=cfg.workers, pin_memory=device.type == "cuda", generator=generator)
        for current, batch in enumerate(loader, start=start + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            result = model(batch.to(device, non_blocking=True))
            loss, terms = objective(model, result)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at step {current}")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad, error_if_nonfinite=True)
            optimizer.step()
            if current == start + 1 or current % cfg.log_every == 0:
                record = {"step": current, "seconds": elapsed(), "gradient_norm": float(gradient_norm), **terms}
                append_json(out / "history.jsonl", record)
                progress("training", current, training=record)
                print(f"step={current}/{cfg.steps} loss={float(loss.detach()):.6g}", flush=True)
            # Release the training graph before validation and checkpoint work.
            del result, loss
            if current % cfg.validate_every == 0 or current == cfg.steps:
                score = validate(current)
                if score < best:
                    best = score
                    save_checkpoint(out / "checkpoints" / "best.pt", model, optimizer, current, best, manifest, elapsed())
            if current % cfg.checkpoint_every == 0 or current % cfg.validate_every == 0 or current == cfg.steps:
                save_checkpoint(checkpoint_path, model, optimizer, current, best, manifest, elapsed())
                last_saved = current
        progress("complete", current)
        atomic_json(out / "completion.json", {"step": current, "best_validation": best,
                    "elapsed_active_seconds": elapsed(), "test_evaluated": False})
    except (Exception, KeyboardInterrupt) as error:
        progress("interrupted" if isinstance(error, KeyboardInterrupt) else "failed", current,
                 error=str(error), resume_from=str(checkpoint_path))
        raise


def self_check(device: torch.device) -> dict:
    """Offline numerical checks. Never construct an optimizer or acquire data."""
    seed_all(7)
    torch.set_num_threads(1)
    cfg = Settings(seconds=0.032, n_fft=64, hop=16, atoms=4, width=16, heads=4, stages=2,
                   initial_steps=2, validation_examples=2)
    model = AdaptiveSTFTDictionary(cfg).to(device)
    t = torch.arange(cfg.samples, device=device) / cfg.sample_rate
    x = torch.stack((0.2 * torch.sin(2 * torch.pi * 220 * t),
                     0.15 * torch.cos(2 * torch.pi * 733 * t)))
    s, normalized, gain = model.transform.analyze(x)
    roundtrip = model.transform.inverse(s, cfg.samples) * gain
    roundtrip_error = float((roundtrip - x).square().mean() / x.square().mean())
    assert roundtrip_error < 1e-10, roundtrip_error
    result = model(x)
    assert result.audio.shape == x.shape and torch.isfinite(result.audio).all()
    assert len(result.dictionaries) == cfg.stages + 1
    assert result.spectra[-1][:, (0, -1)].imag.abs().max() == 0
    assert float((result.dictionaries[0][0] - result.dictionaries[0][1]).abs().max().detach()) > 1e-5
    loss, _ = objective(model, result)
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    for module in (model.encoder, model.generator, model.refinement):
        assert any(p.grad is not None and bool(torch.count_nonzero(p.grad)) for p in module.parameters())
    with torch.no_grad():
        model.eval()
        batch_result = model(x).audio
        single_result = torch.cat([model(item[None]).audio for item in x])
        torch.testing.assert_close(batch_result, single_result, rtol=5e-4, atol=2e-5)
        zero = model(torch.zeros_like(x))
        assert torch.isfinite(zero.audio).all()
        assert float(zero.audio.abs().max()) == 0
        d = torch.randn(1, 5, 3, dtype=torch.complex128)
        a = torch.randn(1, 3, 7, dtype=torch.complex128)
        dn, an = normalize_pair(d, a)
        torch.testing.assert_close(d @ a, dn @ an)
        probe = project_endpoints(torch.randn(1, 5, 7, dtype=torch.complex128))
        before = (project_endpoints(d @ a) - probe).abs().square().sum()
        after_spectrum = fixed_dictionary_solve(d, probe)
        assert (after_spectrum - probe).abs().square().sum() <= before + 1e-9
        for variant in ("fixed", "generated"):
            settings = Settings(**{**asdict(cfg), "variant": variant})
            control = AdaptiveSTFTDictionary(settings).to(device)
            assert torch.isfinite(control(x).audio).all()
    # Verify both projected complex data gradients against PyTorch autograd.
    d = torch.randn(1, 5, 3, dtype=torch.complex128, requires_grad=True)
    a = torch.randn(1, 3, 7, dtype=torch.complex128, requires_grad=True)
    target = project_endpoints(torch.randn(1, 5, 7, dtype=torch.complex128))
    residual = project_endpoints(d @ a) - target
    energy = target.abs().square().sum()
    gd, ga = torch.autograd.grad(residual.abs().square().sum() / energy, (d, a))
    torch.testing.assert_close(gd, 2 * project_endpoints(residual) @ a.mH / energy)
    torch.testing.assert_close(ga, 2 * d.mH @ project_endpoints(residual) / energy)
    return {"passed": True, "device": str(device), "stft_roundtrip_nmse": roundtrip_error,
            "checks": ["gain inversion", "projected complex gradients", "finite end-to-end backward",
                       "input-dependent atoms", "normalization preserves product", "sample isolation",
                       "silence", "fixed-D numerical solve", "all variants", "no_grad inference"],
            "optimizer_steps": 0, "dataset_downloads": 0}


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = p.add_subparsers(dest="command")
    check = commands.add_parser("self-check", help="Offline synthetic numerical checks; no weight updates")
    check.add_argument("--device", default="cpu")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--output", type=Path, default=Path("runs/adaptive_stft_dictionary"))
    common.add_argument("--data-root", help="Existing extracted corpus; no download")
    common.add_argument("--download", action="store_true", help="Explicitly allow KaggleHub acquisition")
    common.add_argument("--device", default="auto")
    common.add_argument("--threads", type=int, default=1)
    defaults = Settings()
    for field in fields(defaults):
        common.add_argument("--" + field.name.replace("_", "-"), type=type(getattr(defaults, field.name)),
                            default=argparse.SUPPRESS, help=f"Python default: {getattr(defaults, field.name)}")
    commands.add_parser("prepare", parents=[common], help="Acquire/index splits; do not train")
    training = commands.add_parser("train", parents=[common], help="Explicitly start training")
    training.add_argument("--resume", action="store_true")
    evaluation = commands.add_parser("evaluate", parents=[common], help="Evaluate a saved checkpoint; no optimizer")
    evaluation.add_argument("--checkpoint", default="best.pt")
    evaluation.add_argument("--split", help="Default: saved test split; dev split is also allowed")
    evaluation.add_argument("--examples", type=int, help="Default: saved validation-example count")
    return p


def main(argv: list[str] | None = None) -> None:
    p = parser()
    args = p.parse_args(argv)
    if not args.command:
        p.print_help()
        return
    if args.command == "self-check":
        print(json.dumps(self_check(choose_device(args.device)), indent=2))
        return
    if args.threads < 1:
        p.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    out = args.output.expanduser().resolve()
    if args.command == "evaluate" or (args.command == "train" and args.resume):
        saved = json.loads((out / "settings.json").read_text(encoding="utf-8"))
    else:
        saved = asdict(Settings())
    overrides = {field.name: getattr(args, field.name) for field in fields(Settings()) if hasattr(args, field.name)}
    if args.command == "evaluate" and overrides:
        raise ValueError("Evaluation uses the checkpoint settings. Use --examples for evaluation coverage")
    cfg = Settings(**{**saved, **overrides})
    cfg.validate()
    device = choose_device(args.device)
    with RunLock(out):
        if args.command == "prepare":
            prepare_data(args, cfg, out)
        elif args.command == "train":
            train(args, cfg, out, device)
        else:
            manifest = prepare_data(args, cfg, out)
            path = Path(args.checkpoint)
            if not path.is_absolute():
                path = out / "checkpoints" / path
            state = torch.load(path, map_location="cpu", weights_only=True)
            compatible = all(value == getattr(cfg, key) for key, value in state["settings"].items()
                             if key not in {"steps", "workers"})
            if not compatible or state["data_fingerprint"] != manifest["fingerprint"]:
                raise ValueError("Checkpoint settings or data differ")
            if state["source_sha256"] != file_hash(Path(__file__)):
                raise ValueError("Checkpoint script identity differs")
            split = args.split or cfg.test_split
            if split not in {cfg.test_split, cfg.validation_split}:
                raise ValueError("Explicit evaluation accepts only the saved dev or test split")
            count = cfg.validation_examples if args.examples is None else args.examples
            if count <= 0:
                raise ValueError("--examples must be positive")
            rows = select_excerpts(manifest["records"][split], count, cfg)
            folder = out / "evaluation" / f"{split}_{path.stem}_step_{state['step']:07d}_n{len(rows)}"
            if folder.exists():
                raise FileExistsError(f"Evaluation folder already exists: {folder}")
            model = AdaptiveSTFTDictionary(cfg).to(device)
            model.load_state_dict(state["model"])
            score, _ = evaluate(model, manifest, rows, folder, split, state["step"], device)
            atomic_json(folder / "checkpoint.json", {"path": str(path), "sha256": file_hash(path), "score": score})
            print(f"Report: {folder / 'report.html'}", flush=True)


if __name__ == "__main__":
    main()
