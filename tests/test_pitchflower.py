"""Synthetic component and disposable Trainer checks; no production training."""
import hashlib
import copy
import json
from types import MethodType, SimpleNamespace

import numpy as np
import pytest
import torch

from lrac.data.pitchflower import PitchDataset
from lrac.models.pitchflower import GaussianBottleneck, GaussianPredictor, PitchFlowerModel, render_gaussians


@pytest.mark.parametrize("device,channels", [("cpu", 32), ("cuda", 256), ("cuda", 512)])
def test_sdpa_matches_reference_values_gradients_and_checkpoint_keys(device, channels):
    """Check the actual upstream block, including RoPE and both CUDA head widths."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    from lrac.models.pitchflower import reference_path, sdpa_attention
    reference_path()
    from pitchflower.transformer import AttnBlock
    original = AttnBlock(channels).to(device)
    optimized = copy.deepcopy(original)
    optimized.forward = MethodType(sdpa_attention, optimized)
    x = torch.randn(2, channels, 17, device=device, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
        a, b = original(x), optimized(y)
    tolerance = dict(rtol=.025, atol=.015) if device == "cuda" else dict(rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(a, b, **tolerance)
    a.square().mean().backward()
    b.square().mean().backward()
    torch.testing.assert_close(x.grad, y.grad, **tolerance)
    assert original.state_dict().keys() == optimized.state_dict().keys()
    for p, q in zip(original.parameters(), optimized.parameters()):
        torch.testing.assert_close(p.grad, q.grad, **tolerance)


def test_larger_batch_preserves_sequential_rvq_updates_and_commitment():
    model = PitchFlowerModel("rvq", hidden=32, flow_channels=32, ae_layers=1, flow_layers=1)
    # Bypass one-time k-means initialization; compare steady-state EMA updates
    # from identical populated codebooks, as in the resumed production run.
    for name, buffer in model.autoencoder.quantizer.named_buffers():
        if name.endswith("inited"):
            buffer.fill_(True)
        elif name.endswith("embed") or name.endswith("embed_avg"):
            buffer.normal_()
        elif name.endswith("cluster_size"):
            buffer.fill_(10.)
    baseline = copy.deepcopy(model)
    latent = torch.randn(8, 256, 8, requires_grad=True)
    reference_latent = latent.detach().clone().requires_grad_()
    value, penalty = model.condition(latent)
    parts = [baseline.condition(part) for part in reference_latent.split(4)]
    torch.testing.assert_close(value, torch.cat([part[0] for part in parts]), rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(penalty, sum(part[1] for part in parts)/2)
    penalty.backward()
    (sum(part[1] for part in parts)/2).backward()
    torch.testing.assert_close(latent.grad, reference_latent.grad)
    for name, buffer in model.autoencoder.quantizer.named_buffers():
        torch.testing.assert_close(buffer, dict(baseline.autoencoder.quantizer.named_buffers())[name])


def test_execution_migration_preserves_history_and_rejects_objective_changes(tmp_path, monkeypatch):
    import lrac.pitchflower as recipe
    old = dict(micro_batch=4, accumulation=8, predictor_accumulation=4, loader_workers=2,
               steps=1000, gs_iterations=100)
    (tmp_path/"settings.json").write_text(json.dumps(old))
    (tmp_path/"provenance.json").write_text(json.dumps(dict(references={}, recipe_sha256={})))
    folder = tmp_path/"rvq/checkpoints"
    folder.mkdir(parents=True)
    torch.save(dict(global_step=500, marker="unchanged"), folder/"last.ckpt")
    monkeypatch.setattr(recipe, "source_fingerprint", lambda _: {})
    monkeypatch.setattr(recipe, "environment", lambda: {})
    with pytest.raises(ValueError, match="execution-only"):
        recipe.migrate_execution(dict(old, gs_iterations=50), tmp_path)
    with pytest.raises(ValueError, match="batch must remain 32"):
        recipe.migrate_execution(dict(old, micro_batch=8), tmp_path)
    new = dict(old, micro_batch=32, accumulation=1, predictor_accumulation=1)
    assert recipe.migrate_execution(new, tmp_path) == 500
    assert json.loads((tmp_path/"settings.json").read_text()) == new
    history = json.loads(next((tmp_path/"execution_history").glob("*.json")).read_text())
    assert history["old_settings"] == old and history["resume_step"] == 500
    assert torch.load(folder/"last.ckpt", weights_only=False)["marker"] == "unchanged"


def test_interruption_callback_saves_optimizer_checkpoint(tmp_path):
    from lrac.pitchflower import LiveProgress
    saved = []
    trainer = SimpleNamespace(optimizers=[object()], global_step=994, save_checkpoint=saved.append)
    callback = LiveProgress(tmp_path, "rvq", 1000000)
    callback.on_exception(trainer, None, SystemExit(1))
    assert saved == [tmp_path/"checkpoints/last.ckpt"]
    assert json.loads((tmp_path/"progress.json").read_text())["step"] == 994


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA fused Adam required")
def test_fused_optimizer_state_can_resume_with_clipping_compatible_adam(monkeypatch):
    """A fused checkpoint can return to ordinary Adam without losing moments."""
    import lrac.pitchflower as recipe
    monkeypatch.setattr(recipe, "PitchFlowerModel", lambda *args, **kwargs: torch.nn.Linear(2, 2))
    settings = dict(gs_density=.35, gs_iterations=100, gs_channels=32, samples=36096)
    module = recipe.TrainingModule(settings, "rvq").cuda()
    optimizer = torch.optim.AdamW(module.net.parameters(), fused=True)
    module.net(torch.ones(2, 2, device="cuda")).square().mean().backward()
    optimizer.step()
    saved = [state["exp_avg"].clone() for state in optimizer.state.values()]
    assert all(state["step"].device.type == "cuda" for state in optimizer.state.values())
    module._trainer = SimpleNamespace(optimizers=[optimizer])
    module.on_train_start()
    for before, state in zip(saved, optimizer.state.values()):
        torch.testing.assert_close(before, state["exp_avg"])
        assert state["step"].device.type == "cpu" and float(state["step"]) == 1
    optimizer.zero_grad(set_to_none=True)
    module.net(torch.ones(2, 2, device="cuda")).square().mean().backward()
    optimizer.step()
    assert all(float(state["step"]) == 2 for state in optimizer.state.values())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA BF16 integration check")
@pytest.mark.parametrize("stage", ["rvq", "gs", "predictor"])
def test_lightning_bf16_clipping_and_checkpoint_resume(tmp_path, monkeypatch, stage):
    """Exercise the actual AMP/optimizer/clip/restore path missed by raw benchmarks."""
    import lrac.pitchflower as recipe
    monkeypatch.setattr(recipe, "PitchFlowerModel", lambda *args, **kwargs: torch.nn.Linear(2, 2))
    monkeypatch.setattr(recipe, "GaussianPredictor", lambda *args, **kwargs: torch.nn.Linear(2, 2))
    settings = dict(micro_batch=4, gs_density=.35, gs_iterations=100, gs_channels=32,
                    samples=36096, learning_rate=1e-4, predictor_steps=100000, seed=0)
    norms = []

    class SyntheticModule(recipe.TrainingModule):
        def _loss(self, batch, training):
            network = self.predictor if self.predictor_stage else self.net
            loss = (network(batch[0]).float()-batch[1]).square().mean()
            return loss, {"synthetic_mse": loss.detach()}

        def configure_gradient_clipping(self, optimizer, gradient_clip_val, gradient_clip_algorithm):
            super().configure_gradient_clipping(optimizer, gradient_clip_val, gradient_clip_algorithm)
            gradients = [p.grad for group in optimizer.param_groups for p in group["params"] if p.grad is not None]
            norms.append(float(torch.stack([grad.norm() for grad in gradients]).norm()))

    data = torch.utils.data.TensorDataset(torch.full((16, 2), 10.), torch.full((16, 2), 30.))
    loader = torch.utils.data.DataLoader(data, batch_size=4)
    def trainer(steps):
        return recipe.pl.Trainer(accelerator="gpu", devices=1, precision="bf16-mixed",
            max_steps=steps, max_epochs=-1, gradient_clip_val=1., num_sanity_val_steps=0,
            logger=False, enable_checkpointing=False, enable_progress_bar=False,
            enable_model_summary=False, default_root_dir=tmp_path)
    first = SyntheticModule(settings, "gs" if stage == "predictor" else stage, stage == "predictor")
    initial = trainer(2)
    initial.fit(first, loader)
    checkpoint = tmp_path/"synthetic.ckpt"
    initial.save_checkpoint(checkpoint)
    second = SyntheticModule(settings, "gs" if stage == "predictor" else stage, stage == "predictor")
    resumed = trainer(4)
    resumed.fit(second, loader, ckpt_path=str(checkpoint))
    assert resumed.global_step == 4 and len(norms) == 4
    assert all(norm <= 1.00001 for norm in norms)
    assert all(not group["fused"] for group in resumed.optimizers[0].param_groups)
    assert all(float(state["step"]) == 4 for state in resumed.optimizers[0].state.values() if "step" in state)


def test_resume_converts_microbatch_counts_without_changing_optimizer_steps(monkeypatch):
    import lrac.pitchflower as recipe
    monkeypatch.setattr(recipe, "PitchFlowerModel", lambda *args, **kwargs: torch.nn.Linear(2, 2))
    settings = dict(micro_batch=32, gs_density=.35, gs_iterations=100, gs_channels=32, samples=36096)
    module = recipe.TrainingModule(settings, "rvq")
    counters = dict(ready=4000, started=4000, processed=4000, completed=3999)
    checkpoint = dict(global_step=500, hyper_parameters=dict(settings=dict(settings, micro_batch=4)),
        loops={"fit_loop": {"epoch_loop.batch_progress": dict(current=dict(counters), total=dict(counters)),
                            "epoch_loop.state_dict": {"_batches_that_stepped": 499}}})
    module.on_load_checkpoint(checkpoint)
    for scope in ("current", "total"):
        assert checkpoint["loops"]["fit_loop"]["epoch_loop.batch_progress"][scope] == dict(
            ready=500, started=500, processed=500, completed=499)
    assert checkpoint["global_step"] == 500
    assert checkpoint["loops"]["fit_loop"]["epoch_loop.state_dict"]["_batches_that_stepped"] == 499


def test_gaussian_renderer_and_straight_through_gradient():
    weights = torch.tensor([[[2., -1.]]])
    scales = torch.ones(1, 2)
    rendered = render_gaussians(weights, scales, 3)
    grid = torch.arange(3).float()
    expected = 2*torch.exp(-.5*grid.square())-torch.exp(-.5*(grid-2).square())
    torch.testing.assert_close(rendered[0, 0], expected)
    latent = torch.randn(2, 3, 24, requires_grad=True)
    quantizer = GaussianBottleneck(iterations=3)
    fitted, penalty = quantizer(latent)
    assert penalty.isfinite() and fitted.shape == latent.shape
    assert quantizer.last_parameters["scales"].shape[0] == 2
    fitted.sum().backward()
    torch.testing.assert_close(latent.grad, torch.ones_like(latent))
    # Validation disables outer gradients, but the inner fit must remain usable.
    with torch.no_grad():
        other, _ = quantizer(latent.detach())
    assert torch.isfinite(other).all()


@pytest.mark.parametrize("bottleneck", ["rvq", "gs"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_reference_backbone_shapes_and_encoder_flow_gradients(bottleneck, device):
    pytest.importorskip("rotary_embedding_torch")
    pytest.importorskip("torchdyn")
    from pathlib import Path
    if not (Path(__file__).resolve().parents[1]/"runs/vendor/PitchFlower/src").exists():
        pytest.skip("Run tools/setup_pitchflower.py to check the pinned reference")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA optional for synthetic component checks")
    # Reduced hidden widths/layers keep this an isolated mathematical check, not
    # a miniature speech-training run. Main configurations retain paper widths.
    model = PitchFlowerModel(bottleneck, hidden=32, flow_channels=32,
                            ae_layers=1, flow_layers=1, gs_iterations=2).to(device)
    waveform = torch.randn(1, 1792, device=device)*.05  # Eight mel frames -> four latent frames.
    pitch = torch.full((1, 8), 7., device=device)
    with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
        loss, terms = model(waveform, waveform, pitch, drop_probability=0)
    assert loss.isfinite() and all(v.isfinite() for v in terms.values())
    loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.autoencoder.encoder.parameters())
    assert any(p.grad is not None for p in model.flow.parameters())
    assert model.autoencoder.quantizer_in.out_channels == (256 if bottleneck == "rvq" else 32)


def test_cached_crop_pitch_alignment_without_world_or_training(tmp_path):
    row = {"id": "test-source"}
    identity = hashlib.sha256(b"test-source:0:pitchflower24k-v1").hexdigest()
    original = np.linspace(-1, 1, 4096, dtype=np.float32)
    pitch = np.arange(17, dtype=np.float32)
    np.savez_compressed(tmp_path/f"{identity}.npz", original=original,
                        perturbed=original*.5, pitch=pitch)
    dataset = PitchDataset([row], tmp_path, samples=1792, training=False)
    a, b, f0 = dataset[0]
    assert a.shape == b.shape == (1792,) and f0.shape == (8,)
    torch.testing.assert_close(f0, torch.arange(8).float())
    torch.testing.assert_close(a, b)


def test_official_predictor_renders_variable_primitive_counts():
    from pathlib import Path
    if not (Path(__file__).resolve().parents[1]/"runs/vendor/gs-codec").exists():
        pytest.skip("Run tools/setup_pitchflower.py for the official predictor")
    model = GaussianPredictor(channels=3, frames=24, count=8)
    latent = torch.randn(2, 3, 24)
    for count in (5, 8):
        weights, scales = model(latent, count)
        assert weights.shape == (2, 3, count) and scales.shape == (2, count)
        assert (scales > 0).all()
        reconstruction = render_gaussians(weights, scales, 24)
        assert torch.isfinite(reconstruction).all()
    reconstruction.square().mean().backward()
    assert any(p.grad is not None for p in model.parameters())


def test_training_configuration_constructs_without_fit(tmp_path, monkeypatch):
    """Validate the installed Trainer API without starting any optimizer/data loop."""
    from pathlib import Path
    from lrac.config import load_study_config
    import lrac.pitchflower as recipe
    from pytorch_lightning.callbacks import ModelCheckpoint
    root = Path(__file__).resolve().parents[1]
    settings = load_study_config(root/"configs/study_pitchflower_librispeech.yaml").parameters
    monkeypatch.setattr(recipe, "PitchFlowerModel", lambda *args, **kwargs: torch.nn.Linear(2, 2))
    module = recipe.TrainingModule(settings, "rvq")
    assert module.configure_optimizers().param_groups[0]["lr"] == 1e-4
    best = ModelCheckpoint(dirpath=tmp_path, monitor="val/loss", save_top_k=1,
                           save_last=False, save_on_train_epoch_end=False)
    latest = ModelCheckpoint(dirpath=tmp_path, save_top_k=0, save_last=True,
        every_n_train_steps=500, save_on_train_epoch_end=False)
    trainer = recipe.pl.Trainer(accelerator="cpu", devices=1, logger=False,
        enable_progress_bar=False, callbacks=[best, latest], max_steps=1, max_epochs=-1,
        accumulate_grad_batches=8, gradient_clip_val=1., val_check_interval=8000,
        check_val_every_n_epoch=None, num_sanity_val_steps=0, inference_mode=False,
        default_root_dir=tmp_path)
    assert trainer.accumulate_grad_batches == 8
    assert not (tmp_path/"last.ckpt").exists()  # Constructor only: no fit/save.
