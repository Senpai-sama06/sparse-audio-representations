"""Protect existing results and run bookkeeping while experiments are reorganized."""
import csv
import json
from pathlib import Path

import pytest

from lrac.config import StudyConfig, load_study_config
from lrac.data.sparse_examples import cue_examples, search_examples
from lrac.study import run_study
from lrac.utils import file_hash


ROOT = Path(__file__).resolve().parents[1]


def test_moved_datasets_match_all_published_sequences():
    search = json.loads((ROOT / "documentation/sparse_search_results.json").read_text(encoding="utf-8"))
    cues = json.loads((ROOT / "documentation/sparse_cues_results.json").read_text(encoding="utf-8"))
    assert {name: [str(v) for v in x] for name, x in search_examples().items()} == {
        name: row["sequence"] for name, row in search["examples"].items()
    }
    assert {row["name"]: [str(v) for v in row["x"]] for row in cue_examples()} == {
        name: row["sequence"] for name, row in cues["witnesses"].items()
    }


def test_search_study_reproduces_results_and_records_artifacts(tmp_path, monkeypatch):
    # Exercise the actual public command outside the project working directory.
    from lrac.cli import main

    monkeypatch.chdir(tmp_path)
    destination = tmp_path / "search"
    main(["study", "--config", str(ROOT / "configs/study_sparse_search.yaml"), "--output", str(destination)])
    for filename in ("sparse_search_results.csv", "sparse_search_curves.csv"):
        with (destination / filename).open(newline="", encoding="utf-8") as actual, (ROOT / "documentation" / filename).open(newline="", encoding="utf-8") as expected:
            assert list(csv.DictReader(actual)) == list(csv.DictReader(expected))
    config = json.loads((destination / "config.json").read_text())
    assert config["parameters"] == {"seed": 0}
    assert json.loads((destination / "progress.json").read_text())["status"] == "complete"
    completion = json.loads((destination / "completion.json").read_text())
    for filename, record in completion["artifacts"].items():
        assert record["sha256"] == file_hash(destination / filename)
    assert "lrac/study_sparse_search.py" in json.loads((destination / "source.json").read_text())["files"]


def test_study_rejects_unknown_parameters_and_preserves_existing_directories(tmp_path):
    cfg = StudyConfig("search", "sparse_search", parameters={"sead": 0}, project_root=str(tmp_path))
    with pytest.raises(ValueError, match="Invalid sparse_search parameters"):
        run_study(cfg, "new")
    assert not (tmp_path / "new").exists()
    cfg.parameters = {}
    existing = tmp_path / "existing"
    existing.mkdir()
    sentinel = existing / "config.json"
    sentinel.write_text("preserve this")
    with pytest.raises(FileExistsError):
        run_study(cfg, "existing")
    assert sentinel.read_text() == "preserve this"
    cfg.kind = "unknown"
    with pytest.raises(ValueError, match="Unknown study kind"):
        run_study(cfg, "new")
    assert not (tmp_path / "new").exists()


def test_failed_study_records_failure_and_resolved_defaults(tmp_path):
    cfg = StudyConfig("cues", "sparse_cues", parameters={"beam_width": 0}, project_root=str(tmp_path))
    with pytest.raises(ValueError, match="beam_width must be a positive integer"):
        run_study(cfg, "failed")
    destination = tmp_path / "failed"
    assert json.loads((destination / "config.json").read_text())["parameters"] == {"seed": 20261029, "beam_width": 0}
    assert json.loads((destination / "progress.json").read_text())["status"] == "failed"
    assert not (destination / "completion.json").exists()


def test_study_yaml_reuses_inheritance_and_project_paths(tmp_path):
    (tmp_path / "pyproject.toml").touch()
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "base.yaml").write_text("name: search\nkind: sparse_search\nparameters:\n  seed: 1\n")
    path = configs / "child.yaml"
    path.write_text("extends: base.yaml\nparameters:\n  seed: 2\n")
    cfg = load_study_config(path)
    assert cfg.parameters == {"seed": 2}
    assert cfg.path(cfg.runs_dir) == tmp_path / "runs"
    assert cfg.path("~/datasets") == Path.home() / "datasets"
    for content in (
        "name: ../bad\nkind: sparse_search\n",
        "name: search\nkind: sparse_search\nparameters: []\n",
        "name: search\nkind: sparse_search\nparamters: {}\n",
    ):
        path.write_text(content)
        with pytest.raises(ValueError):
            load_study_config(path)
