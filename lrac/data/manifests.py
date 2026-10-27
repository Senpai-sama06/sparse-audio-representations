from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json
import random
import soundfile as sf
from lrac.data.synthetic import records


def read_manifest(path):
    with open(path, encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_manifest(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    temporary.replace(path)


def configured_manifests(cfg):
    result = {"train": cfg.manifest(cfg.data.train_manifest), "validation": cfg.manifest(cfg.data.val_manifest)}
    result.update({f"test:{key}": cfg.manifest(value) for key, value in cfg.data.test_manifests.items()})
    return result


def audio_record(path, corpus, split, role, speaker, utterance, chapter=None):
    info = sf.info(str(path))
    if info.channels != 1:
        raise ValueError(f"Expected mono audio: {path}")
    return {"id": f"{corpus}:{utterance}", "path": str(Path(path).resolve()), "corpus": corpus, "split": split, "role": role, "speaker": str(speaker), "chapter": chapter, "utterance": utterance, "sample_rate": info.samplerate, "num_samples": info.frames, "start": 0, "stop": info.frames}


def balanced_subset(rows, hours, seed=0):
    groups = {}
    for row in rows:
        groups.setdefault(row["speaker"], []).append(row)
    if not groups:
        raise ValueError("Cannot subset an empty corpus")
    budget = hours * 3600 / len(groups)
    selected = []
    for speaker in sorted(groups):
        shuffled = sorted(groups[speaker], key=lambda x: x["id"])
        random.Random(f"{seed}:{speaker}").shuffle(shuffled)
        remaining = budget
        for row in shuffled:
            if remaining <= 0:
                break
            take = min(row["num_samples"], round(remaining * row["sample_rate"]))
            if take <= 0:
                break
            item = dict(row, stop=take)
            selected.append(item)
            remaining -= take / row["sample_rate"]
    return selected


def librispeech_split_root(root, split):
    """Resolve official extraction or Kaggle's per-archive nested layout."""
    root = Path(root)
    for candidate in [root / "LibriSpeech" / split, root / split / "LibriSpeech" / split, root / split]:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"Missing LibriSpeech partition {split} under {root}")


def prepare(cfg, download=False):
    from lrac.data.downloads import acquire
    dest = cfg.path(cfg.data.manifest_dir)
    if cfg.data.source == "synthetic":
        for split, count, filename in [("train", cfg.data.train_count, "train.jsonl"), ("validation", cfg.data.validation_count, "validation.jsonl"), *[(s, cfg.data.test_count, f"test_{s}.jsonl") for s in ["iid", "combinations", "extrapolation", "families"]]]:
            write_manifest(dest / filename, records(split, count, cfg.data.sample_rate, cfg.data.synthetic_seconds))
        return dest
    root = cfg.path(cfg.data.root)
    if download:
        partitions = [cfg.data.librispeech_train_split, "dev-clean", *[f"test-{key}" for key in cfg.data.test_manifests]] if cfg.data.source == "librispeech" else None
        acquire(cfg.data.source, root, librispeech_splits=partitions)
    if cfg.data.source == "librispeech":
        unknown = set(cfg.data.test_manifests) - {"clean", "other"}
        if unknown:
            raise ValueError(f"Unknown LibriSpeech test partitions: {sorted(unknown)}")
        partitions = [(cfg.data.librispeech_train_split, "train", "train.jsonl"), ("dev-clean", "validation", cfg.data.val_manifest)]
        partitions.extend((f"test-{key}", "test", name) for key, name in cfg.data.test_manifests.items())
        for split, role, name in partitions:
            folder = librispeech_split_root(root, split)
            paths = sorted(folder.rglob("*.flac"))
            if not paths:
                raise FileNotFoundError(f"No FLAC recordings in {folder}")
            rows = []
            def record(path):
                speaker, chapter, _ = path.stem.split("-")
                return audio_record(path, "librispeech", split, role, speaker, path.stem, chapter)
            # Header reads are independent file I/O. Preserve sorted output order
            # while avoiding a long serial index of the cached 360h partition.
            with ThreadPoolExecutor(max_workers=8) as executor:
                for count, row in enumerate(executor.map(record, paths), 1):
                    rows.append(row)
                    if count % 10000 == 0 or count == len(paths):
                        print(f"Indexed {split}: {count}/{len(paths)} recordings", flush=True)
            write_manifest(dest / name, rows)
            if role == "train":
                for hours in cfg.data.subsets_hours:
                    write_manifest(dest / f"train_{hours:g}h.jsonl", balanced_subset(rows, hours))
    else:
        base = root / "VCTK-Corpus-0.92" if (root / "VCTK-Corpus-0.92").exists() else root
        groups = {}
        for path in sorted(base.rglob("*_mic2.flac")):
            speaker = path.stem.split("_")[0]
            groups.setdefault(speaker, []).append(path)
        if not groups:
            raise FileNotFoundError(f"No VCTK mic2 recordings under {base}")
        if cfg.data.vctk_mode == "external":
            rows = []
            for speaker, paths in sorted(groups.items()):
                random.Random(f"vctk:{speaker}").shuffle(paths)
                for path in paths[:cfg.data.vctk_per_speaker]:
                    rows.append(audio_record(path, "vctk", "external", "test", speaker, path.stem))
            write_manifest(dest / "external.jsonl", rows)
        else:
            speakers = sorted(groups)
            random.Random(0).shuffle(speakers)
            if len(speakers) < 10:
                raise ValueError("Full-band speaker-split study requires at least ten speakers")
            n_train, n_val = int(.8 * len(speakers)), max(1, int(.1 * len(speakers)))
            partitions = [("train", speakers[:n_train], "train.jsonl"), ("validation", speakers[n_train:n_train + n_val], "validation.jsonl"), ("test", speakers[n_train + n_val:], "test.jsonl")]
            for role, ids, filename in partitions:
                rows = [audio_record(path, "vctk", role, role, speaker, path.stem) for speaker in ids for path in sorted(groups[speaker])]
                write_manifest(dest / filename, rows)
    return dest
