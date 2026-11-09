"""Behavioral tests for stopping, native SAE gradients, data isolation, and resume."""
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from sae.checkpoint import Stopper
from sae.cli import main
from sae.data import SpeechDataset, prepare, read_manifests
from sae.models import AudioSAE, Jump, METHODS, ModelConfig, Step, WaveformDecoder
from sae.objectives import objective


@pytest.fixture(autouse=True)
def threads():
    torch.set_num_threads(1)


def args(manifests, output, epochs=1):
    return ["--manifest-dir", str(manifests), "--output", str(output), "--epochs", str(epochs),
            "--channels", "4", "4", "4", "4", "--latent-width", "8", "--k", "2",
            "--batch-size", "2", "--crop-seconds", ".064", "--spectral-weight", "0",
            "--device", "cpu", "--threads", "1", "--log-every", "0", "--patience", "0"]


def test_stopping_tracks_small_best_improvements_separately():
    stopper = Stopper(patience=2, min_delta=.1)
    assert stopper.update(1, 1) == (True, False)
    assert stopper.update(.96, 2) == (True, False)
    assert stopper.update(.95, 3) == (True, True)
    assert stopper.best_epoch == 3 and stopper.anchor == 1
    with pytest.raises(FloatingPointError):
        stopper.update(float("nan"), 4)


def test_stopping_accumulates_improvement_and_respects_min_epochs():
    stopper = Stopper(patience=1, min_delta=.1, min_epochs=3)
    assert not stopper.update(1, 1)[1]
    assert not stopper.update(.96, 2)[1]
    assert not stopper.update(.85, 3)[1]
    assert stopper.update(.85, 4)[1]
    disabled = Stopper(patience=0)
    assert all(not disabled.update(1, epoch)[1] for epoch in range(1, 20))


def test_jumprelu_uses_threshold_kernel_and_l0_gradients():
    x = torch.tensor([[.95, 1.05]], requires_grad=True)
    threshold = torch.tensor([1., 1.], requires_grad=True)
    Jump.apply(x, threshold, .2).sum().backward()
    assert torch.equal(x.grad, torch.tensor([[0., 1.]]))
    assert torch.allclose(threshold.grad, torch.tensor([-5., -5.]))
    x.grad = None
    threshold.grad = None
    Step.apply(x, threshold, .2).sum().backward()
    assert torch.count_nonzero(x.grad) == 0
    assert torch.allclose(threshold.grad, torch.tensor([-5., -5.]))


def test_gated_auxiliary_trains_gate_without_decoder_gradients():
    torch.manual_seed(4)
    model = AudioSAE(ModelConfig(method="gated", channels=(4, 4, 4, 4), latent_width=8, k=2))
    x = torch.randn(2, 1, 1024) * .1
    result = model(x, torch.tensor([1024, 1024]))
    model.detached_decode(result["gate_code"], 1024).square().mean().backward()
    assert model.project.weight.grad is not None and model.project.weight.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in model.decoder.parameters())
    assert model.magnitude_bias.grad is None


def test_batchtopk_budget_and_batch_independent_inference():
    model = AudioSAE(ModelConfig(method="batchtopk", channels=(4, 4, 4, 4), latent_width=8, k=2))
    values = torch.arange(1., 33.).reshape(2, 2, 8)
    valid = torch.tensor([[True, False], [True, True]])
    selected = model.batch_selection(values, valid)
    assert torch.count_nonzero(selected) == 6
    assert torch.count_nonzero(selected[~valid]) == 0
    model.eval()
    x = torch.randn(2, 1, 1024)
    with pytest.raises(RuntimeError, match="calibration"):
        model(x, torch.tensor([1024, 1024]))
    model.inference_threshold.fill_(.5)
    model.threshold_calibrated.fill_(True)
    with torch.no_grad():
        together = model(x, torch.tensor([1024, 1024]))["z"]
        separate = torch.cat([model(item[None], torch.tensor([1024]))["z"] for item in x])
    assert torch.allclose(together, separate, atol=1e-6)


def test_manifest_speaker_isolation_and_epoch_stable_crops(manifests):
    partitions, _ = read_manifests(manifests, verify_audio=True)
    assert {r["speaker"] for r in partitions["validation"]}.isdisjoint(
        {r["speaker"] for r in partitions["calibration"]})
    dataset = SpeechDataset(partitions["train"], 256, random_crop=True, seed=4)
    dataset.epoch = 2
    a = dataset[0][0]
    assert torch.equal(a, dataset[0][0])
    dataset.epoch = 3
    assert not torch.equal(a, dataset[0][0])
    audio = Path(partitions["train"][0]["path"])
    audio.write_bytes(audio.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="changed"):
        read_manifests(manifests)


@pytest.mark.parametrize("method", METHODS)
def test_each_trainer_saves_best_resume_and_independent_decoder(method, manifests, tmp_path):
    output = tmp_path / method
    summary = main(method, args(manifests, output))
    assert summary["completed_epochs"] == 1 and summary["stop_reason"] == "max_epochs"
    assert summary["test_used"] is False
    weights = torch.load(output / "best_weights.pt", weights_only=True)
    model = AudioSAE(ModelConfig(**weights["model_config"]))
    model.load_state_dict(weights["model_state"])
    decoder_payload = torch.load(output / "decoder_best.pt", weights_only=True)
    decoder = WaveformDecoder(decoder_payload["model_config"]["channels"], 8)
    decoder.load_state_dict(decoder_payload["decoder_state"])
    code = torch.rand(1, 4, 8)
    with torch.no_grad():
        assert torch.equal(model.decoder(code), decoder(code))
    assert (output / "last.pt").is_file()
    assert (output / "history.csv").is_file()
    assert json.loads((output / "best_validation.json").read_text())["epoch"] == 1
    if method == "batchtopk":
        assert bool(model.threshold_calibrated)


def test_resume_matches_uninterrupted_training(manifests, tmp_path):
    complete, resumed = tmp_path / "complete", tmp_path / "resumed"
    main("topk", args(manifests, complete, 2))
    main("topk", args(manifests, resumed, 1))
    main("topk", ["--resume", str(resumed / "last.pt"), "--epochs", "2"])
    a = torch.load(complete / "last.pt", weights_only=False)
    b = torch.load(resumed / "last.pt", weights_only=False)
    assert a["epoch"] == b["epoch"] == 2
    assert a["stopper"] == b["stopper"]
    assert all(torch.equal(value, b["model_state"][key]) for key, value in a["model_state"].items())
    assert [row["epoch"] for row in b["history"]] == [1, 2]
    with pytest.raises(SystemExit):
        main("topk", ["--resume", str(resumed / "last.pt"), "--k", "3"])


def test_real_training_stops_on_validation_plateau(manifests, tmp_path):
    output = tmp_path / "stopped"
    options = args(manifests, output, 10) + ["--patience", "1", "--min-delta", "1000000"]
    summary = main("l1", options)
    assert summary["stop_reason"] == "early_stopping"
    assert summary["completed_epochs"] == 2
    saved = torch.load(output / "last.pt", weights_only=False)
    assert saved["stopper"]["bad_epochs"] == 1
    unchanged = main("l1", ["--resume", str(output / "last.pt")])
    assert unchanged["completed_epochs"] == 2


def test_stft_objective_and_dead_latent_auxiliary_have_finite_gradients():
    config = dict(mse_scale=.1, spectral_weight=.1, kl_target=.05,
                  sparsity_weight=0, gated_aux_weight=1, aux_weight=.03)
    model = AudioSAE(ModelConfig(method="topk", channels=(4, 4, 4, 4), latent_width=8, k=2))
    model.inactive_steps.fill_(1000)
    x, lengths = torch.randn(2, 1, 768) * .1, torch.tensor([768, 517])
    result = model(x, lengths)
    losses = objective(model, result, x, lengths, config)
    assert result["aux_code"] is not None and losses["auxiliary"] > 0
    losses["loss"].backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_export_evaluation_uses_full_audio_and_records_no_bitrate(manifests, tmp_path):
    from evaluate import main as evaluate
    output = tmp_path / "trained"
    main("dense", args(manifests, output))
    evaluation = tmp_path / "evaluation"
    result = evaluate(["--checkpoint", str(output / "best_weights.pt"), "--manifest", str(manifests / "test.jsonl"),
                       "--output", str(evaluation), "--device", "cpu", "--threads", "1", "--save-audio", "1"])
    assert result["cases"] == 4 and result["bitrate"] is None
    cases = json.loads((evaluation / "cases.json").read_text())
    assert {row["samples"] for row in cases} == {1024, 1041}
    assert len(list(evaluation.glob("*.wav"))) == 2


def test_optional_speech_metrics_preserve_invalid_cases():
    pytest.importorskip("pesq")
    pytest.importorskip("pystoi")
    from sae.metrics import audio_metrics
    rng = np.random.default_rng(5)
    t = np.arange(32000) / 16000
    x = rng.normal(0, .03, len(t)) * (.6 + .4 * np.sin(2 * np.pi * 4 * t))
    reference = audio_metrics(x, x, speech=True)
    assert reference["pesq_wb"] > 4 and reference["stoi"] > .99
    silent = audio_metrics(np.zeros(16000), np.zeros(16000), speech=True)
    assert silent["stoi"] is None and silent["statuses"]["stoi"] == "silent_reference"
    short = audio_metrics(x[:100], x[:100], speech=True)
    assert short["pesq_wb"] is None and short["stoi"] is None


def test_periodic_resume_restores_its_best_snapshot(manifests, tmp_path):
    output = tmp_path / "periodic"
    main("topk", args(manifests, output, 2) + ["--save-every", "1"])
    first = torch.load(output / "epoch_0001.pt", weights_only=False)
    main("topk", ["--resume", str(output / "epoch_0001.pt"), "--epochs", "1"])
    best = torch.load(output / "best_weights.pt", weights_only=True)
    last = torch.load(output / "last.pt", weights_only=False)
    assert best["epoch"] == first["best_payload"]["epoch"] == 1
    assert last["epoch"] == 1
    assert all(torch.equal(value, best["model_state"][key]) for key, value in first["best_payload"]["model_state"].items())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA runtime not installed")
def test_unindexed_cuda_device_accepts_allocator_cap_and_checkpoint_resume(manifests, tmp_path):
    output = tmp_path / "cuda_cap"
    try:
        summary = main("topk", args(manifests, output) + ["--device", "cuda", "--gpu-memory-fraction", ".2"])
        assert summary["completed_epochs"] == 1
        resumed = main("topk", ["--resume", str(output / "last.pt"), "--epochs", "2"])
        assert resumed["completed_epochs"] == 2
    finally:
        torch.cuda.set_per_process_memory_fraction(1.0)
