"""Component checks only: synthetic tensors; no Trainer.fit or training run."""
import hashlib

import numpy as np
import pytest
import torch

from lrac.data.pitchflower import PitchDataset
from lrac.models.pitchflower import GaussianBottleneck, GaussianPredictor, PitchFlowerModel, render_gaussians


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
