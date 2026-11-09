"""Local LibriSpeech discovery, immutable manifests, and epoch-stable crops."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import random

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset
from .acquire import split_folder
from .runtime import atomic_json, file_lock

SAMPLE_RATE = 16000


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    atomic_json(path, value)


def scan_split(root, split):
    folder = split_folder(root, split)
    if folder is None:
        raise FileNotFoundError(f"Missing LibriSpeech split {split} under {root}")
    paths = sorted(folder.rglob("*.flac"))
    if not paths:
        raise ValueError(f"No FLAC recordings in {folder}")
    rows = []
    for i, path in enumerate(paths):
        info = sf.info(path)
        fields = path.stem.split("-")
        if len(fields) != 3 or not all(field.isdigit() for field in fields):
            raise ValueError(f"Expected LibriSpeech speaker-chapter-utterance name: {path}")
        if info.samplerate != SAMPLE_RATE or info.channels != 1 or info.frames <= 0:
            raise ValueError(f"Expected nonempty mono 16-kHz LibriSpeech audio: {path}")
        stat = path.stat()
        rows.append(dict(id=path.stem, speaker=fields[0], split=split,
                         path=str(path.resolve()), frames=info.frames,
                         sample_rate=info.samplerate, size=stat.st_size,
                         mtime_ns=stat.st_mtime_ns, sha256=digest(path)))
        if (i + 1) % 1000 == 0:
            print(f"Indexed and hashed {split}: {i + 1}/{len(paths)}", flush=True)
    return rows


def balanced_subset(rows, hours, seed):
    if hours is None:
        return rows
    if hours <= 0:
        raise ValueError("train_hours must be positive")
    groups = {}
    for row in rows:
        groups.setdefault(row["speaker"], []).append(row)
    rng = random.Random(seed)
    speakers = sorted(groups)
    rng.shuffle(speakers)
    for values in groups.values():
        rng.shuffle(values)
    result, frames, index = [], 0, 0
    while frames < hours * 3600 * SAMPLE_RATE:
        changed = False
        for speaker in speakers:
            if index < len(groups[speaker]):
                row = groups[speaker][index]
                result.append(row)
                frames += row["frames"]
                changed = True
                if frames >= hours * 3600 * SAMPLE_RATE:
                    break
        if not changed:
            break
        index += 1
    return result


def prepare(root, output, train_split="train-clean-100", dev_split="dev-clean",
            test_split="test-clean", train_hours=None, seed=0):
    output = Path(output).resolve()
    with file_lock(output.parent / (output.name + ".lock")):
        return _prepare(root, output, train_split, dev_split, test_split, train_hours, seed)


def _prepare(root, output, train_split, dev_split, test_split, train_hours, seed):
    if train_hours is not None and (not math.isfinite(train_hours) or train_hours <= 0):
        raise ValueError("train_hours must be finite and positive")
    if not 0 <= seed < 2**32:
        raise ValueError("seed must be in [0, 2**32)")
    root, output = Path(root).resolve(), Path(output).resolve()
    if (root / "LibriSpeech").is_dir():
        root = root / "LibriSpeech"
    specification = dict(root=str(root), train_split=train_split, dev_split=dev_split,
                         test_split=test_split, train_hours=train_hours, seed=seed)
    if train_split == dev_split or test_split in {train_split, dev_split}:
        raise ValueError("Training, development, and test split names must differ")
    metadata = output / "manifest_info.json"
    if metadata.exists():
        existing = json.loads(metadata.read_text(encoding="utf-8"))
        if existing["specification"] != specification:
            raise ValueError(f"Manifest settings changed; use a new directory: {output}")
        for name, sha in existing["manifest_hashes"].items():
            if digest(output / name) != sha:
                raise ValueError(f"Manifest changed: {output / name}")
        return output
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Refusing to replace an incomplete/nonempty manifest directory: {output}")
    train = balanced_subset(scan_split(root, train_split), train_hours, seed)
    dev = scan_split(root, dev_split)
    speakers = sorted({row["speaker"] for row in dev},
                      key=lambda value: hashlib.sha256(f"{seed}:{value}".encode()).hexdigest())
    if len(speakers) < 2:
        raise ValueError("Development split needs at least two speakers for validation/calibration")
    calibration_speakers = set(speakers[:len(speakers) // 2])
    partitions = dict(train=train,
                      validation=[r for r in dev if r["speaker"] not in calibration_speakers],
                      calibration=[r for r in dev if r["speaker"] in calibration_speakers])
    if test_split and split_folder(root, test_split) is not None:
        partitions["test"] = scan_split(root, test_split)
    audit_partitions(partitions)
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in partitions.items():
        (output / f"{name}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    write_json(metadata, dict(specification=specification,
                             manifest_hashes={f"{name}.jsonl": digest(output / f"{name}.jsonl") for name in partitions},
                             counts={name: dict(utterances=len(rows),
                                                speakers=len({r["speaker"] for r in rows}),
                                                hours=sum(r["frames"] for r in rows) / SAMPLE_RATE / 3600)
                                     for name, rows in partitions.items()},
                             test_exposure="Historical exposure must be audited; no fresh-test claim."))
    return output


def audit_partitions(partitions):
    names = list(partitions)
    for name, rows in partitions.items():
        if not rows or len({r["id"] for r in rows}) != len(rows):
            raise ValueError(f"Empty partition or duplicate recording IDs: {name}")
    for i, name in enumerate(names):
        for other in names[i + 1:]:
            a, b = partitions[name], partitions[other]
            if {r["speaker"] for r in a} & {r["speaker"] for r in b}:
                raise ValueError(f"Speaker leakage: {name}/{other}")
            if {r["sha256"] for r in a} & {r["sha256"] for r in b}:
                raise ValueError(f"Duplicate audio across {name}/{other}")


def read_manifests(folder, verify_audio=False):
    folder = Path(folder)
    info = json.loads((folder / "manifest_info.json").read_text(encoding="utf-8"))
    partitions = {}
    for filename, expected in info["manifest_hashes"].items():
        path = folder / filename
        if digest(path) != expected:
            raise ValueError(f"Manifest checksum mismatch: {path}")
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        partitions[path.stem] = rows
        # Test audio is never loaded or verified by the training path.
        if path.stem == "test":
            continue
        for row in rows:
            audio = Path(row["path"])
            stat = audio.stat()
            if stat.st_size != row["size"] or stat.st_mtime_ns != row["mtime_ns"]:
                raise ValueError(f"Audio changed since manifest preparation: {audio}")
            if verify_audio and digest(audio) != row["sha256"]:
                raise ValueError(f"Audio checksum mismatch: {audio}")
    audit_partitions(partitions)
    if not {"train", "validation", "calibration"}.issubset(partitions):
        raise ValueError("Need train, validation, and calibration manifests")
    return partitions, digest(folder / "manifest_info.json")


def speaker_limit(rows, limit):
    if not limit:
        return rows
    groups = {}
    for row in rows:
        groups.setdefault(row["speaker"], []).append(row)
    ordered = [groups[key] for key in sorted(groups)]
    result = []
    for index in range(max(map(len, ordered))):
        for group in ordered:
            if index < len(group):
                result.append(group[index])
                if len(result) == limit:
                    return result
    return result


class SpeechDataset(Dataset):
    def __init__(self, rows, crop_samples=0, random_crop=False, seed=0):
        self.rows = rows
        self.crop_samples = crop_samples
        self.random_crop = random_crop
        self.seed, self.epoch = seed, 0

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        count = row["frames"]
        start, frames = 0, -1
        if self.crop_samples:
            maximum = max(0, count - self.crop_samples)
            rng = random.Random(f"{self.seed}:{self.epoch}:{row['id']}")
            start = rng.randint(0, maximum) if self.random_crop else maximum // 2
            frames = min(count, self.crop_samples)
        audio, rate = sf.read(row["path"], start=start, frames=frames,
                              dtype="float32", always_2d=True)
        if rate != SAMPLE_RATE or audio.shape[1] != 1 or not len(audio) or not np.isfinite(audio).all():
            raise ValueError(f"Invalid mono 16-kHz audio: {row['path']}")
        return torch.from_numpy(audio.T.copy()), row["id"]


def collate(batch):
    lengths = torch.tensor([audio.shape[-1] for audio, _ in batch])
    size = int(lengths.max())
    audio = torch.stack([torch.nn.functional.pad(x, (0, size - x.shape[-1])) for x, _ in batch])
    return audio, lengths, [item[1] for item in batch]
