"""Acquisition, parallel process lifecycle, detachment, and reconnectable status."""
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time
from types import SimpleNamespace

import pytest

from sae import acquire
from sae.campaign import ENTRY_DIR, print_status, snapshot
from sae.data import prepare, read_manifests
from sae.runtime import atomic_json, file_lock, process_alive, process_identity, read_json


def isolated_cache(monkeypatch, tmp_path):
    working = tmp_path / "working"
    working.mkdir()
    monkeypatch.chdir(working)
    monkeypatch.delenv("LIBRISPEECH_ROOT", raising=False)
    monkeypatch.setenv("KAGGLEHUB_CACHE", str(tmp_path / "kagglehub"))


def test_existing_nested_kaggle_cache_needs_no_network(corpus, tmp_path, monkeypatch):
    isolated_cache(monkeypatch, tmp_path)
    cached = tmp_path / "kagglehub" / "datasets" / "someone" / "librispeech-mirror" / "versions" / "3"
    shutil.copytree(corpus, cached / "nested" / "LibriSpeech")
    def no_import(name):
        raise AssertionError("Existing data must not need importing KaggleHub")
    monkeypatch.setattr(acquire.importlib, "import_module", no_import)
    root, record = acquire.acquire_root(None, acquire.DEFAULT_DATASET, tmp_path / "setup", "train-clean-100", "dev-clean", "test-clean", True)
    assert record["source"] == "existing_cache"
    assert acquire.has_splits(root, ["train-clean-100", "dev-clean"])


def test_missing_kagglehub_installed_once_and_resolved_cache_reused(corpus, tmp_path, monkeypatch):
    isolated_cache(monkeypatch, tmp_path)
    calls = dict(imports=0, installs=0, downloads=0)
    def download(handle):
        calls["downloads"] += 1
        assert handle == acquire.DEFAULT_DATASET
        return str(corpus)
    def importer(name):
        assert name == "kagglehub"
        calls["imports"] += 1
        if calls["imports"] == 1:
            raise ImportError("test missing package")
        return SimpleNamespace(dataset_download=download)
    def install(command, check):
        calls["installs"] += 1
        assert command[:4] == [sys.executable, "-m", "pip", "install"] and check
    monkeypatch.setattr(acquire.importlib, "import_module", importer)
    monkeypatch.setattr(acquire.subprocess, "run", install)
    for _ in range(2):
        root, _ = acquire.acquire_root(None, acquire.DEFAULT_DATASET, tmp_path / "setup", "train-clean-100", "dev-clean", "test-clean")
        assert root == corpus
    assert calls == dict(imports=2, installs=1, downloads=1)


def test_offline_mode_does_not_install_or_download(tmp_path, monkeypatch):
    isolated_cache(monkeypatch, tmp_path)
    with pytest.raises(FileNotFoundError, match="offline"):
        acquire.acquire_root(None, acquire.DEFAULT_DATASET, tmp_path / "setup", "train-clean-100", "dev-clean", "test-clean", True)


def test_cached_tar_archives_extract_once_and_prepare_manifests(corpus, tmp_path):
    source = tmp_path / "archives"
    source.mkdir()
    for split in ("train-clean-100", "dev-clean", "test-clean"):
        with tarfile.open(source / (split + ".tar.gz"), "w:gz") as stream:
            stream.add(corpus / split, arcname="LibriSpeech/" + split)
    root, _ = acquire.acquire_root(source, acquire.DEFAULT_DATASET, tmp_path / "setup", "train-clean-100", "dev-clean", "test-clean", True)
    manifests = tmp_path / "extracted_manifests"
    prepare(root, manifests)
    partitions, _ = read_manifests(manifests, verify_audio=True)
    assert len(partitions["train"]) == 4 and len(partitions["test"]) == 4
    ready = next((tmp_path / "setup").rglob("archive_ready.json"))
    timestamp = ready.stat().st_mtime_ns
    acquire.acquire_root(source, acquire.DEFAULT_DATASET, tmp_path / "setup", "train-clean-100", "dev-clean", "test-clean", True)
    assert ready.stat().st_mtime_ns == timestamp


def test_archive_traversal_is_rejected(tmp_path):
    archive = tmp_path / "unsafe.tar"
    with tarfile.open(archive, "w") as stream:
        member = tarfile.TarInfo("../outside.txt")
        member.size = 3
        stream.addfile(member, io.BytesIO(b"bad"))
    with pytest.raises(ValueError, match="escapes"):
        acquire.safe_extract(archive, tmp_path / "destination")
    assert not (tmp_path / "outside.txt").exists()


def test_process_birth_tokens_and_automatic_lock_release(tmp_path):
    identity = process_identity()
    assert process_alive(identity)
    assert not process_alive(dict(identity, birth="recycled PID"))
    path = tmp_path / "setup.lock"
    with file_lock(path):
        with pytest.raises(TimeoutError):
            with file_lock(path, timeout=0):
                pass
    with file_lock(path, timeout=0):
        pass


def trainer_extras():
    return ["--channels", "4", "4", "4", "4", "--latent-width", "8", "--k", "2",
            "--crop-seconds", ".064", "--spectral-weight", "0", "--train-limit", "2",
            "--validation-limit", "2", "--calibration-limit", "2", "--log-every", "0"]


def wait_campaign(output, timeout=90):
    deadline = time.monotonic() + timeout
    peak = 0
    while time.monotonic() < deadline:
        state = snapshot(output)
        peak = max(peak, sum(job["alive"] for job in state["jobs"]))
        if state["state"] in {"completed", "completed_with_failures", "failed", "stopped"} and not state["supervisor_alive"]:
            return state, peak
        time.sleep(.1)
    atomic_json(output / "stop.request", dict(reason="test cleanup"))
    raise AssertionError(f"Campaign timed out: {snapshot(output)}")


def test_detached_parallel_campaign_survives_launcher_exit_and_reports_status(manifests, tmp_path):
    output = tmp_path / "campaign"
    command = [sys.executable, str(ENTRY_DIR / "run_all.py"), "start", "--manifest-dir", str(manifests),
               "--output", str(output), "--methods", "l1", "topk", "batchtopk", "--parallel", "2",
               "--launch-delay", "0", "--epochs", "1", "--patience", "0", "--batch-size", "2",
               "--device", "cpu", "--threads", "1", "--", *trainer_extras()]
    launched = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert launched.returncode == 0, launched.stderr
    state, peak = wait_campaign(output)
    assert peak == 2
    assert state["state"] == "completed", (state, (output / "supervisor.log").read_text())
    assert all(job["state"] == "completed" and job["completed_epochs"] == 1 for job in state["jobs"])
    assert (output / "supervisor.log").is_file()
    for job in state["jobs"]:
        assert Path(job["log"]).is_file()
        assert (Path(job["output"]) / "best_weights.pt").is_file()
    # A new process reads the persisted campaign after the launching process has exited.
    viewed = subprocess.run([sys.executable, str(ENTRY_DIR / "run_all.py"), "status", "--output", str(output), "--json"],
                            capture_output=True, text=True, timeout=15)
    assert json.loads(viewed.stdout)["state"] == "completed"
    resumed = subprocess.run([sys.executable, str(ENTRY_DIR / "run_all.py"), "resume", "--output", str(output)],
                             capture_output=True, text=True, timeout=15)
    assert resumed.returncode == 0, resumed.stderr
    state, _ = wait_campaign(output)
    assert all(len(job["attempts"]) == 1 for job in read_json(output / "campaign.json")["jobs"].values())


def test_failed_parallel_job_is_visible_with_its_log(manifests, tmp_path):
    output = tmp_path / "failed_campaign"
    command = [sys.executable, str(ENTRY_DIR / "run_all.py"), "start", "--foreground", "--manifest-dir", str(manifests),
               "--output", str(output), "--methods", "topk", "--parallel", "1", "--epochs", "1",
               "--device", "cpu", "--", "--latent-width", "8", "--k", "99"]
    finished = subprocess.run(command, capture_output=True, text=True, timeout=40)
    assert finished.returncode == 1
    state = snapshot(output)
    assert state["state"] == "completed_with_failures"
    assert state["jobs"][0]["state"] == "failed"
    assert "k <= latent_width" in Path(state["jobs"][0]["log"]).read_text()


def test_status_before_first_gpu_snapshot_does_not_require_torch(tmp_path, capsys):
    from sae.campaign import prepare_start
    output = tmp_path / "queued"
    prepare_start(dict(output=str(output), methods=["l1"]))
    print_status(snapshot(output))
    assert "queued" in capsys.readouterr().out
