import itertools
import json
import pytest
import torch
from lrac.config import Config, DataConfig, ModelConfig, TrainConfig, EvaluationConfig
from lrac.models import build_model
from lrac.models.block_orthogonal import select_coefficients
from lrac.data.manifests import prepare
from lrac.train import train, load_run
from lrac.evaluate import evaluate


def model(n=8, **options):
    return build_model(ModelConfig(kind="block_orthogonal", frame_length=n, hop=n, groups=2, **options), 16000)


@pytest.mark.parametrize("length", [1, 5, 8, 19, 32])
def test_exact_whole_signal_roundtrip_including_partial_blocks(length):
    torch.manual_seed(1)
    m = model(initialization="random", keep_fraction=1)
    x = torch.randn(2, 1, length, dtype=torch.float64)
    y, rep = m(x)
    assert (y - x).norm() / x.norm() < 1e-12
    assert int(rep.coefficient_mask.sum()) == x.numel()
    assert rep.timing["tail_lengths"] == [length % 8] * 2


def test_global_budget_can_spend_every_coefficient_in_one_block():
    m = model(4, initialization="identity", sample_budget=1)
    x = torch.tensor([[[1., 0., 0., 0., 10., 9., 0., 0.]]])
    y, rep = m(x)
    assert int(rep.selected_mask.sum()) == 1
    assert y[0, 0, 4] == 10
    assert torch.count_nonzero(y[..., :4]) == 0


def test_padding_cannot_consume_budget_or_affect_short_recording():
    m = model(keep_fraction=1)
    x = torch.randn(2, 1, 19, dtype=torch.float64)
    x[1, 0, 5:] = 1000
    y, rep = m(x, torch.tensor([19, 5]))
    assert torch.allclose(y[1, 0, :5], x[1, 0, :5], atol=1e-12)
    assert torch.count_nonzero(y[1, 0, 5:]) == 0
    assert rep.coefficient_mask[1].sum() == 5


@pytest.mark.parametrize("target", [0., .01, .2, 1.])
def test_minimum_support_matches_exhaustive_search(target):
    torch.manual_seed(2)
    m = model(4, initialization="random", keep_fraction=1)
    x = torch.randn(1, 1, 6, dtype=torch.float64)
    full = m.encode(x)
    values = full.concatenate()
    valid_indices = full.coefficient_mask.reshape(-1).nonzero().flatten().tolist()
    best_errors = []
    for count in range(7):
        errors = []
        for support in itertools.combinations(valid_indices, count):
            candidate = torch.zeros_like(values)
            candidate.reshape(-1)[list(support)] = values.reshape(-1)[list(support)]
            y = m.decode(full.with_values(candidate))
            errors.append(float((x - y).square().sum() / x.square().sum()))
        best_errors.append(min(errors))
    m.config.target_nmse = target
    y, rep = m(x)
    count = int(rep.selected_mask.sum())
    # A zero tolerance means all nonzero transform coefficients; floating inverse
    # errors remain at machine precision rather than literally zero.
    oracle = 6 if target == 0 else next(s for s, error in enumerate(best_errors) if error <= target + 1e-14)
    assert count == oracle
    assert abs(float((x - y).square().sum() / x.square().sum()) - rep.timing["sparse_coding"][0]["predicted_nmse"]) < 1e-12


def test_zero_signal_and_zero_budget_fraction():
    m = model(target_nmse=.01)
    _, rep = m(torch.zeros(1, 1, 9))
    assert int(rep.selected_mask.sum()) == 0
    m.config.target_nmse = None
    m.config.keep_fraction = 0
    assert torch.count_nonzero(m(torch.randn(1, 1, 9))[0]) == 0


def test_procrustes_update_cannot_increase_same_batch_objective():
    torch.manual_seed(3)
    m = model(16, initialization="random", keep_fraction=.25)
    x = torch.randn(3, 1, 37, dtype=torch.float64)
    lengths = torch.tensor([37, 32, 9])
    for _ in range(5):
        stats = m.update_dictionary(x, lengths)
        assert stats["mse_fixed_codes"] <= stats["mse_before"] + 1e-12
        assert stats["mse_after"] <= stats["mse_fixed_codes"] + 1e-12
        assert stats["orthogonality_error"] < 1e-12


def test_closed_form_training_checkpoint_and_tolerance_evaluation(tmp_path):
    cfg = Config(
        project_root=str(tmp_path),
        data=DataConfig(manifest_dir="manifests", train_count=8, validation_count=2, test_count=2, synthetic_seconds=.032, crop_seconds=.032),
        model=ModelConfig(kind="block_orthogonal", frame_length=16, hop=16, initialization="random", groups=2),
        train=TrainConfig(name="block", steps=3, batch_size=4, val_examples=2, val_interval=1, log_interval=1, device="cpu"),
        evaluation=EvaluationConfig(audio_examples=1, bootstrap_samples=10),
    )
    prepare(cfg)
    run = train(cfg)
    assert (run / "checkpoints/initial.pt").exists()
    _, m, state = load_run(run, "cpu", "last.pt")
    assert state["step"] == 3
    assert not list(m.parameters())
    stats = json.loads((run / "training_statistics.json").read_text())
    assert stats["training_method"] == "orthogonal_procrustes"
    paths = evaluate(run, "validation", device="cpu", nmse_targets=[.1, .01])
    for path in paths:
        report = json.loads((path / "summary.json").read_text())
        assert report["minimum_support_verified_examples"] == 2
        assert report["metrics"]["nmse"]["mean"] <= report["target_nmse"] + 1e-10
    initial_paths = evaluate(run, "validation", device="cpu", checkpoint="initial.pt")
    assert "initial" in initial_paths[0].parts
