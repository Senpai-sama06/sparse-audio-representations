import json
from dataclasses import replace
from pathlib import Path
import torch
import pytest
from lrac.config import Config, DataConfig, ModelConfig, TrainConfig, EvaluationConfig, load_config, load_study_config
from lrac.data.manifests import prepare
from lrac.train import train, load_run
from lrac.evaluate import evaluate
from lrac.report import report


def test_saved_pipeline_and_evaluation(tmp_path):
    cfg = Config(project_root=str(tmp_path), data=DataConfig(manifest_dir="manifests", train_count=6, validation_count=2, test_count=2, synthetic_seconds=.032, crop_seconds=.032), model=ModelConfig(kind="orthogonal", frame_length=32, hop=8), train=TrainConfig(steps=2, batch_size=2, val_examples=1, val_interval=1, log_interval=1, device="cpu"), evaluation=EvaluationConfig(audio_examples=1, bootstrap_samples=10))
    prepare(cfg)
    run = train(cfg)
    _, model, state = load_run(run, "cpu")
    x = torch.randn(1, 1, 517)
    rep = model.encode(x)
    output = model.decode(rep)
    x.zero_()
    assert torch.equal(output, model.decode(rep))
    paths = evaluate(run, split="iid", limit=2, device="cpu", budgets=[0, 4])
    assert len(paths) == 2
    summary = json.loads((paths[0] / "summary.json").read_text())
    assert summary["limited"] and summary["examples"] == 2
    report([run], tmp_path / "report")
    assert (tmp_path / "report/index.html").exists()


def test_repository_configurations_are_valid():
    root = Path(__file__).resolve().parents[1]
    for path in (root / "configs").rglob("*.yaml"):
        if path.name.startswith("study_"):
            load_study_config(path)
        elif not path.name.startswith("sweep_"):
            load_config(path)


def test_config_paths_expand_home_without_changing_project_relative_paths(tmp_path):
    cfg = Config(project_root=str(tmp_path))
    assert cfg.path("~/datasets") == Path.home() / "datasets"
    assert cfg.path("data/raw") == tmp_path / "data/raw"


def test_resume_and_reject_changed_dataset_without_overwriting_run(tmp_path):
    cfg = Config(project_root=str(tmp_path), data=DataConfig(manifest_dir="manifests", train_count=4, validation_count=1, test_count=1, synthetic_seconds=.032, crop_seconds=.032), model=ModelConfig(kind="orthogonal", frame_length=32, hop=8), train=TrainConfig(steps=1, batch_size=2, val_examples=1, val_interval=1, device="cpu"))
    prepare(cfg)
    run = train(cfg)
    cfg.train.steps = 2
    train(cfg, resume=run / "checkpoints/last.pt")
    _, _, state = load_run(run, "cpu", "last.pt")
    assert state["step"] == 2
    previous = (run / "config.json").read_text()
    cfg.data.crop_seconds = .064
    with pytest.raises(ValueError, match="dataset configuration"):
        train(cfg, resume=run / "checkpoints/last.pt")
    assert (run / "config.json").read_text() == previous


def test_simple_experiment_writes_report_and_leaves_final_tests_alone(tmp_path):
    from lrac.experiment import run_experiment

    cfg = Config(
        project_root=str(tmp_path),
        data=DataConfig(
            manifest_dir="manifests",
            train_count=4,
            validation_count=2,
            test_count=2,
            synthetic_seconds=.032,
            crop_seconds=.032,
        ),
        model=ModelConfig(kind="dct", frame_length=32, hop=8, threshold=0),
        train=TrainConfig(name="example", steps=0, val_examples=1, device="cpu"),
        evaluation=EvaluationConfig(audio_examples=1, bootstrap_samples=10, analysis_examples=1),
    )
    run = run_experiment(cfg)
    assert (run / "report/index.html").exists()
    assert (run / "analysis/validation/diagnostics.json").exists()
    assert {p.name for p in (run / "evaluation").iterdir() if p.is_dir()} == {"validation"}
    score = next((run / "evaluation/validation").rglob("summary.json"))
    assert json.loads(score.read_text())["examples"] == 1
