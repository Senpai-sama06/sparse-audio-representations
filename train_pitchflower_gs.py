"""Copy this file to another machine and run: python train_pitchflower_gs.py

Requires Python 3.11 or 3.12, internet access, and an NVIDIA CUDA GPU/driver.
No repository checkout, Git, PowerShell, or YAML file is required. The first run
creates an isolated environment, downloads pinned training code, downloads
LibriSpeech train-clean-360/dev-clean, prepares WORLD data, and trains GS only.
Allow several hundred GB for audio, derived caches, downloads, and checkpoints.

To use existing audio: python train_pitchflower_gs.py --corpus-root E:/LibriSpeech
To choose storage: python train_pitchflower_gs.py --workspace E:/pitchflower_gs
Rerun the same command to resume. A new recipe needs a new workspace. Defaults
match the hardware profile: batch 64, 125,000 updates or 72 active training hours.
This trains the GS codec; it does not train RVQ, the predictor, or scalar PTQ.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import urllib.request
import venv
import zipfile


PROJECT_REVISION = "11b5d4a029664bce8e7c945ff56e8e85c8ca7745"
SOURCES = {
    "project": ("Senpai-sama06/sparse-audio-representations", PROJECT_REVISION),
    "PitchFlower": ("diegotg2000/PitchFlower", "4cef72ac9630cab632f13bd1d26b22cdae81cf17"),
    "gs-codec": ("ronaluf/gs-codec", "c5261bbc3f90838dc21b3b134ceadd15d702f007"),
}
PACKAGES = [
    "numpy>=2,<3", "scipy>=1.13", "soundfile>=0.13", "librosa==0.11.0",
    "scikit-learn>=1.7", "PyYAML>=6", "matplotlib>=3.9", "pandas==2.3.3",
    "psutil>=7", "pytorch-lightning==2.5.5", "einops==0.8.1",
    "rotary-embedding-torch==0.8.9", "torchdyn==1.0.6", "vocos==0.1.0",
    "pyworld==0.3.5", "tqdm>=4.67", "tensorboard>=2.18",
]


def parser():
    command = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    command.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parent / "pitchflower_gs_work")
    command.add_argument("--corpus-root", type=Path, help="Existing LibriSpeech parent directory; missing partitions are downloaded here")
    command.add_argument("--cache-dir", type=Path, help="Existing compatible seed-0 WORLD/mapped cache, or a new cache location")
    command.add_argument("--steps", type=int, default=125000)
    command.add_argument("--hours", type=float, default=72.0, help="Cumulative active GS training time cap")
    command.add_argument("--loader-workers", type=int, default=4)
    command.add_argument("--prepare-workers", type=int, default=2)
    command.add_argument("--prepare-only", action="store_true", help="Prepare data and exit without training")
    command.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return command


def sha(path, algorithm="sha256"):
    result = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def download(url, destination, checksum=None, algorithm="sha256"):
    """Resume partial HTTP downloads, then check the published digest if supplied."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if checksum is None or sha(destination, algorithm) == checksum:
            return destination
        raise RuntimeError(f"Checksum mismatch in {destination}; move that file aside and retry")
    partial = destination.with_suffix(destination.suffix + ".partial")
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": "PitchFlower-GS-training/1.0", "Accept-Encoding": "identity"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    print(f"Downloading {url}", flush=True)
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120) as response:
        resumed = offset > 0 and response.status == 206
        if resumed and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
            raise RuntimeError("Server returned an unexpected download range")
        offset = offset if resumed else 0
        count = offset
        total = int(response.headers.get("Content-Length", 0)) + offset
        last = time.monotonic()
        with partial.open("ab" if resumed else "wb") as output:
            while True:
                block = response.read(4 * 1024 * 1024)
                if not block:
                    break
                output.write(block)
                count += len(block)
                if time.monotonic() - last >= 5:
                    print(f"  {count / 1e9:.2f} / {total / 1e9:.2f} GB", flush=True)
                    last = time.monotonic()
    if total and count != total:
        raise RuntimeError(f"Incomplete download: {partial}; rerun to resume")
    if checksum and sha(partial, algorithm) != checksum:
        raise RuntimeError(f"Checksum mismatch: {partial}; move it aside before retrying")
    partial.replace(destination)
    return destination


def source_archive(key, destination, downloads):
    """Install immutable source into a new directory, retaining upstream licenses."""
    repo, revision = SOURCES[key]
    marker = destination / ".standalone-source.json"
    expected = {"repository": repo, "revision": revision}
    if marker.exists():
        if json.loads(marker.read_text()) != expected:
            raise RuntimeError(f"Source version mismatch at {destination}")
        return
    if destination.exists():
        raise RuntimeError(f"Unrecognized source directory {destination}; choose another workspace")
    archive = download(f"https://codeload.github.com/{repo}/zip/{revision}", downloads / f"{key}-{revision}.zip")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
        staging = Path(temporary).resolve()
        with zipfile.ZipFile(archive) as source:
            for member in source.infolist():
                target = (staging / member.filename).resolve()
                if not target.is_relative_to(staging) or (member.external_attr >> 16) & 0o170000 == 0o120000:
                    raise RuntimeError("Unsafe source archive entry")
            source.extractall(staging)
        children = list(staging.iterdir())
        if len(children) != 1 or not children[0].is_dir():
            raise RuntimeError("Unexpected source archive layout")
        children[0].rename(destination)
    save_json(marker, expected)


def bootstrap(args):
    """Install into a dedicated environment and execute this file there."""
    folder = args.workspace / ".venv"
    python = folder / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    stamp = folder / "gs-dependencies.json"
    recipe = {"torch": "2.8.0+cu128", "packages": PACKAGES}
    if not python.exists():
        venv.EnvBuilder(with_pip=True).create(folder)
    if not stamp.exists() or json.loads(stamp.read_text()) != recipe:
        subprocess.run([str(python), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"], check=True)
        subprocess.run([str(python), "-m", "pip", "install", "torch==2.8.0", "torchaudio==2.8.0",
                        "--index-url", "https://download.pytorch.org/whl/cu128"], check=True)
        subprocess.run([str(python), "-m", "pip", "install", *PACKAGES], check=True)
        save_json(stamp, recipe)
    child = subprocess.Popen([str(python), "-u", str(Path(__file__).resolve()), *sys.argv[1:], "--worker"])
    # Ctrl+C also reaches the worker through the console. Give Lightning time
    # to finish its checkpoint instead of killing it from the bootstrap parent.
    while True:
        try:
            return child.wait()
        except KeyboardInterrupt:
            print("Waiting for the training worker to save and stop...", flush=True)


def data_manifests(root, destination, downloads):
    """Index the full official train/dev partitions using destination-local paths."""
    from lrac.data.downloads import safe_extract
    from lrac.data.manifests import audio_record, librispeech_split_root, write_manifest
    base = "https://www.openslr.org/resources/12/"
    checks = None
    for split, role, filename, expected in [
        ("train-clean-360", "train", "train.jsonl", 104014),
        ("dev-clean", "validation", "validation.jsonl", 2703),
    ]:
        try:
            folder = librispeech_split_root(root, split)
            complete = len(list(folder.rglob("*.flac"))) == expected
        except FileNotFoundError:
            complete = False
        if not complete:
            if checks is None:
                with urllib.request.urlopen(base + "md5sum.txt", timeout=60) as response:
                    checks = {line.split()[-1].lstrip("*"): line.split()[0]
                              for line in response.read().decode().splitlines() if len(line.split()) == 2}
            name = split + ".tar.gz"
            archive = download(base + name, downloads / name, checks[name], "md5")
            print(f"Extracting {name} ...", flush=True)
            safe_extract(archive, root)
            folder = librispeech_split_root(root, split)
        paths = sorted(folder.rglob("*.flac"))
        if len(paths) != expected:
            raise RuntimeError(f"Expected {expected} recordings in {folder}, found {len(paths)}")
        def record(path):
            speaker, chapter, _ = path.stem.split("-")
            return audio_record(path, "librispeech", split, role, speaker, path.stem, chapter)
        rows = []
        with ThreadPoolExecutor(max_workers=8) as executor:
            for count, row in enumerate(executor.map(record, paths), 1):
                rows.append(row)
                if count % 10000 == 0 or count == len(paths):
                    print(f"Indexed {split}: {count}/{len(paths)}", flush=True)
        write_manifest(destination / filename, rows)


def worker(args):
    # Set these before importing numpy, torch, WORLD, or Lightning.
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ["HF_HOME"] = str(args.workspace / "downloads/huggingface")
    downloads = args.workspace / "downloads"
    project = args.workspace / "sources" / f"lrac-{PROJECT_REVISION}"
    source_archive("project", project, downloads)
    for name in ("PitchFlower", "gs-codec"):
        source_archive(name, project / "runs/vendor" / name, downloads)
    sys.path.insert(0, str(project))
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is unavailable. Install/update the NVIDIA driver before retrying.")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("This training recipe requires a GPU that supports BF16.")
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    from lrac.pitchflower import audit_data, prepare, fit
    from lrac.data.pitchflower import cache_lock
    from lrac.utils import source_fingerprint
    from vocos import Vocos
    # Check/download final waveform-export assets before starting lengthy work.
    model = Vocos.from_pretrained("charactr/vocos-mel-24khz").eval()
    del model
    output = args.workspace / "runs/pitchflower_gs"
    output.mkdir(parents=True, exist_ok=True)
    manifests = args.workspace / "data/manifests"
    settings = dict(
        run_dir=str(output), train_manifest=str(manifests / "train.jsonl"),
        validation_manifest=str(manifests / "validation.jsonl"),
        cache_dir=str(args.cache_dir or args.workspace / "data/processed/pitchflower_24k_seed0"),
        expected_train_files=104014, expected_train_speakers=921, seed=0,
        samples=36096, micro_batch=64, accumulation=1, steps=args.steps,
        predictor_steps=100000, predictor_accumulation=1,
        learning_rate=0.0001414213562373095, loader_workers=args.loader_workers,
        prepare_workers=args.prepare_workers, validation_examples=120,
        validation_interval=500, checkpoint_interval=250,
        gs_density=80/225, gs_channels=32, gs_iterations=100,
        rvq_commitment=0.25, gs_commitment=20.0,
        stage_hours={"rvq": 72, "gs": args.hours, "predictor": 12},
        codec_warmup_steps=1000, mapped_cache=True, cache_metadata=True,
    )
    with cache_lock(output / "campaign.lock"):
        print(f"GS outputs: {output}", flush=True)
        print("Checking/downloading train-clean-360 and dev-clean; substantial disk space is required.", flush=True)
        data_manifests(args.corpus_root or args.workspace / "data/corpus", manifests, downloads)
        identity = {"sources": SOURCES, "recipe": source_fingerprint(project),
                    "corpus_root": str(args.corpus_root or args.workspace / "data/corpus")}
        # JSON round-trip normalizes tuples in SOURCES before comparing resumes.
        identity = json.loads(json.dumps(identity))
        for name, value in (("settings.json", settings), ("standalone_provenance.json", identity)):
            path = output / name
            if path.exists() and json.loads(path.read_text()) != value:
                raise RuntimeError(f"{name} changed; select a new --workspace for a different recipe")
            save_json(path, value)
        try:
            train, validation, routine, _ = audit_data(settings, output)
            prepare(train + validation, settings, output)
            if not args.prepare_only:
                fit("gs", train, routine, settings, output)
                save_json(output / "progress.json", {"stage": "gs", "status": "complete"})
        except BaseException as error:
            path = output / "progress.json"
            progress = json.loads(path.read_text()) if path.exists() else {}
            progress.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                            error=f"{type(error).__name__}: {error}")
            save_json(path, progress)
            raise


def main():
    args = parser().parse_args()
    if sys.version_info[:2] not in ((3, 11), (3, 12)):
        raise SystemExit("Use Python 3.11 or 3.12 for the pinned PyTorch/WORLD environment.")
    if args.steps < 1 or args.hours <= 0 or args.loader_workers < 0 or args.prepare_workers < 1:
        raise SystemExit("Step/time/worker values are invalid")
    args.workspace = args.workspace.expanduser().resolve()
    if args.corpus_root:
        args.corpus_root = args.corpus_root.expanduser().resolve()
    if args.cache_dir:
        args.cache_dir = args.cache_dir.expanduser().resolve()
    args.workspace.mkdir(parents=True, exist_ok=True)
    if args.worker:
        worker(args)
        return 0
    print(f"Workspace: {args.workspace}\nPreparing a dedicated Python environment...", flush=True)
    return bootstrap(args)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
