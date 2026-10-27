from dataclasses import replace
import numpy as np
import pytest
import soundfile as sf
from lrac.config import Config, DataConfig
from lrac.data.audit import audit, AuditError
from lrac.data.manifests import write_manifest, audio_record, balanced_subset


def fixture_config(tmp_path, overlap=False, duplicate=False):
    cfg = Config(project_root=str(tmp_path), data=DataConfig(source="librispeech", manifest_dir="manifests", test_manifests={"clean": "test.jsonl"}))
    rng = np.random.default_rng(0)
    first = rng.normal(0, .1, 1600).astype(np.float32)
    for i, (name, split, role) in enumerate([("train.jsonl", "train-clean-100", "train"), ("validation.jsonl", "dev-clean", "validation"), ("test.jsonl", "test-clean", "test")]):
        path = tmp_path / f"{role}.wav"
        sf.write(path, first if duplicate else rng.normal(0, .1, 1600), 16000)
        speaker = "same" if overlap else str(i)
        row = audio_record(path, "librispeech", split, role, speaker, f"{speaker}-{role}")
        write_manifest(tmp_path / "manifests" / name, [row])
    return cfg


def test_valid_speaker_disjoint_audit(tmp_path):
    result = audit(fixture_config(tmp_path))
    assert result["passed"]
    assert result["manifests"]["train"]["speakers"] == 1


def test_speaker_overlap_fails(tmp_path):
    with pytest.raises(AuditError, match="speaker overlap"):
        audit(fixture_config(tmp_path, overlap=True))


def test_duplicate_audio_fails_even_with_different_names(tmp_path):
    with pytest.raises(AuditError, match="decoded audio overlap"):
        audit(fixture_config(tmp_path, duplicate=True))


def test_interval_bounds_fail(tmp_path):
    cfg = fixture_config(tmp_path)
    from lrac.data.manifests import read_manifest
    rows = read_manifest(cfg.manifest("train.jsonl"))
    rows[0]["stop"] = 9999
    write_manifest(cfg.manifest("train.jsonl"), rows)
    with pytest.raises(AuditError, match="Invalid source interval"):
        audit(cfg)


def test_balanced_subsets_are_nested_and_cover_speakers():
    rows = [{"speaker": str(s), "id": f"{s}-{i}", "num_samples": 160000, "sample_rate": 16000, "start": 0, "stop": 160000} for s in range(5) for i in range(20)]
    short = balanced_subset(rows, .01)
    long = balanced_subset(rows, .02)
    lookup = {row["id"]: row for row in long}
    assert {r["speaker"] for r in short} == {str(i) for i in range(5)}
    assert all(row["id"] in lookup and row["stop"] <= lookup[row["id"]]["stop"] for row in short)


def test_kaggle_nested_librispeech_360_preparation_and_audit(tmp_path):
    from lrac.data.manifests import prepare, read_manifest
    rng = np.random.default_rng(5)
    for split, speakers in [("train-clean-360",[1,2]),("dev-clean",[3]),("test-clean",[4])]:
        for speaker in speakers:
            folder = tmp_path/"cache"/split/"LibriSpeech"/split/str(speaker)/"10"
            folder.mkdir(parents=True)
            sf.write(folder/f"{speaker}-10-0000.flac",rng.normal(0,.1,16000),16000)
    cfg = Config(project_root=str(tmp_path),data=DataConfig(source="librispeech",root="cache",manifest_dir="manifests",librispeech_train_split="train-clean-360",subsets_hours=[.0005],test_manifests={"clean":"test_clean.jsonl"},expected_speakers={"train":2,"validation":1,"test:clean":1}))
    prepare(cfg)
    assert audit(cfg)["passed"]
    assert {r["split"] for r in read_manifest(cfg.manifest("train.jsonl"))} == {"train-clean-360"}
    assert len(read_manifest(cfg.manifest("train_0.0005h.jsonl"))) == 2
