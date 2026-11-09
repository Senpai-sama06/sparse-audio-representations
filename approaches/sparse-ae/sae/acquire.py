"""Cache-first LibriSpeech resolution and serialized KaggleHub acquisition."""
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

from .runtime import atomic_json, file_lock, read_json

DEFAULT_DATASET = "pypiahmad/librispeech-asr-corpus/versions/1"


def cache_directory(value=None):
    return Path(value or os.environ.get("SPARSE_AE_CACHE", Path.home() / ".cache" / "sparse-ae")).expanduser().resolve()


def split_folder(root, split):
    """Accept ordinary and nested Kaggle layouts without traversing all speaker folders."""
    root = Path(root)
    mapping = read_json(root / "split_locations.json", {})
    if split in mapping:
        candidate = Path(mapping[split])
        if candidate.is_dir() and next(candidate.rglob("*.flac"), None) is not None:
            return candidate
    frontier = [root]
    for _ in range(6):
        following = []
        for folder in frontier:
            if not folder.is_dir():
                continue
            if folder.name == split and next(folder.rglob("*.flac"), None) is not None:
                return folder
            try:
                children = sorted(p for p in folder.iterdir() if p.is_dir())
            except PermissionError:
                continue
            for child in children:
                if child.name == split and next(child.rglob("*.flac"), None) is not None:
                    return child
                # Other split folders contain many thousands of chapter directories.
                if child.name.startswith(("train-", "dev-", "test-")) and child.name != split:
                    continue
                if not child.name.isdigit() and child.name not in {"extracted", ".git", "__pycache__"}:
                    following.append(child)
        frontier = following
    return None


def has_splits(root, splits):
    return all(split_folder(root, split) is not None for split in splits)


def cached_sources(dataset):
    kaggle_cache = Path(os.environ.get("KAGGLEHUB_CACHE", Path.home() / ".cache" / "kagglehub")).expanduser()
    parts = dataset.split("/")
    preferred = kaggle_cache / "datasets" / parts[0] / parts[1]
    candidates = []
    if len(parts) == 4 and parts[2] == "versions":
        candidates.append(preferred / "versions" / parts[3])
    # Reuse other already-cached LibriSpeech mirrors instead of downloading again.
    folders = [preferred]
    base = kaggle_cache / "datasets"
    if base.is_dir():
        folders += sorted(p for p in base.glob("*/*") if p.is_dir() and "libri" in p.name.lower())
    for folder in folders:
        versions = folder / "versions"
        if versions.is_dir():
            candidates += sorted((p for p in versions.iterdir() if p.is_dir()),
                                 key=lambda p: (p.name.isdigit(), int(p.name) if p.name.isdigit() else 0), reverse=True)
        else:
            candidates.append(folder)
    return list(dict.fromkeys(p.resolve() for p in candidates if p.is_dir()))


def safe_extract(archive, output):
    """Only ordinary files/directories, inside our own extraction directory."""
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:*") as stream:
        for member in stream:
            if member.name.startswith(("/", "\\")) or "\\" in member.name:
                raise ValueError(f"Unsafe archive path: {member.name}")
            target = (output / member.name).resolve()
            if not target.is_relative_to(output):
                raise ValueError(f"Archive escapes extraction directory: {member.name}")
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                target.parent.mkdir(parents=True, exist_ok=True)
                source = stream.extractfile(member)
                if source is None:
                    raise ValueError(f"Unreadable archive member: {member.name}")
                temporary = target.with_name(target.name + ".extracting")
                with source, open(temporary, "wb") as destination:
                    shutil.copyfileobj(source, destination, length=1024 * 1024)
                temporary.replace(target)
            else:
                raise ValueError(f"Archive links/special files are unsupported: {member.name}")


def materialize(source, cache, required, optional):
    from .data import digest
    source = Path(source).resolve()
    if has_splits(source, required):
        return source
    key = hashlib.sha256(str(source).encode()).hexdigest()[:20]
    view = cache / "extracted" / key
    mapping = {}
    # Archive mirrors may keep one tar.gz for each official split.
    for split in list(dict.fromkeys([*required, *optional])):
        folder = split_folder(source, split)
        if folder is None:
            archives = []
            for suffix in (".tar.gz", ".tgz", ".tar"):
                archives.extend(source.rglob(split + suffix))
            if archives:
                archive = sorted(archives)[0]
                stat = archive.stat()
                identity = dict(path=str(archive), size=stat.st_size, mtime_ns=stat.st_mtime_ns)
                destination = view / "extracted" / split
                marker = destination / "archive_ready.json"
                if read_json(marker, {}).get("archive") != identity:
                    print(f"Extracting cached LibriSpeech archive: {archive}", flush=True)
                    safe_extract(archive, destination)
                    atomic_json(marker, dict(archive=identity, sha256=digest(archive)))
                folder = split_folder(destination, split)
        if folder is None:
            if split in required:
                return None
        else:
            mapping[split] = str(folder.resolve())
    atomic_json(view / "split_locations.json", mapping)
    return view


def acquire_root(root, dataset, cache_dir, train_split, dev_split, test_split, offline=False):
    cache = cache_directory(cache_dir)
    required, optional = (train_split, dev_split), ([test_split] if test_split else [])
    pointer_key = hashlib.sha256(json.dumps([dataset, *required, *optional]).encode()).hexdigest()[:20]
    pointer = cache / "resolved" / (pointer_key + ".json")
    with file_lock(cache / "dataset_setup.lock"):
        if root or os.environ.get("LIBRISPEECH_ROOT"):
            source = Path(root or os.environ["LIBRISPEECH_ROOT"]).expanduser().resolve()
            if not source.is_dir():
                raise FileNotFoundError(f"LibriSpeech root does not exist: {source}")
            result = materialize(source, cache, required, optional)
            if result is None:
                raise ValueError(f"Need {train_split} and {dev_split} FLAC audio or tar archives under {source}")
            return result, dict(source="explicit_root", path=str(source), resolved=str(result))
        resolved = read_json(pointer, {})
        if resolved.get("root") and has_splits(Path(resolved["root"]), required):
            return Path(resolved["root"]), dict(source="saved_cache", **resolved)
        local = [Path.cwd() / "data" / "LibriSpeech", Path.cwd() / "LibriSpeech",
                 Path.cwd() / "data", Path("/data/LibriSpeech")]
        for source in [*local, *cached_sources(dataset)]:
            if not source.is_dir():
                continue
            result = materialize(source, cache, required, optional)
            if result is not None:
                record = dict(root=str(result), original_source=str(source), requested_dataset=dataset)
                atomic_json(pointer, record)
                print(f"Using existing LibriSpeech: {result}", flush=True)
                return result, dict(source="existing_cache", **record)
        if offline:
            raise FileNotFoundError("LibriSpeech not found locally. Supply --root or set LIBRISPEECH_ROOT/KAGGLEHUB_CACHE; offline mode forbids downloading.")
        try:
            kagglehub = importlib.import_module("kagglehub")
        except ImportError:
            print("Installing missing KaggleHub package in this Python environment...", flush=True)
            subprocess.run([sys.executable, "-m", "pip", "install", "kagglehub>=1.0.3,<2"], check=True)
            importlib.invalidate_caches()
            kagglehub = importlib.import_module("kagglehub")
        print(f"Obtaining LibriSpeech through KaggleHub: {dataset}", flush=True)
        source = Path(kagglehub.dataset_download(dataset)).resolve()
        result = materialize(source, cache, required, optional)
        if result is None:
            raise ValueError(f"Kaggle dataset {dataset} does not contain required {train_split}/{dev_split} FLAC files or tar archives. Supply another --dataset or --root.")
        record = dict(root=str(result), original_source=str(source), requested_dataset=dataset)
        atomic_json(pointer, record)
        return result, dict(source="kagglehub", **record)


def ensure_manifests(config, progress=None):
    from .data import prepare
    if config.get("manifest_dir"):
        return Path(config["manifest_dir"]).resolve(), dict(source="prepared_manifests")
    if progress:
        progress.update(phase="resolving_data")
    root, record = acquire_root(config.get("root"), config.get("dataset", DEFAULT_DATASET),
                                config.get("cache_dir"), config["train_split"], config["dev_split"],
                                config["test_split"], config.get("offline", False))
    selection = dict(root=str(root), train_split=config["train_split"], dev_split=config["dev_split"],
                     test_split=config["test_split"], train_hours=config["train_hours"], seed=config["seed"])
    key = hashlib.sha256(json.dumps(selection, sort_keys=True).encode()).hexdigest()[:20]
    manifests = cache_directory(config.get("cache_dir")) / "manifests" / key
    if progress:
        progress.update(phase="preparing_data", data_root=str(root), manifest_dir=str(manifests))
    prepare(root, manifests, config["train_split"], config["dev_split"], config["test_split"],
            config["train_hours"], config["seed"])
    record.update(manifest_dir=str(manifests))
    return manifests, record
