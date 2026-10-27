import pytest
import torch
from lrac.config import ModelConfig
from lrac.models import build_model
from lrac.models.transforms import dct_basis, frame, unframe


@pytest.mark.parametrize("length", [1, 17, 512, 777, 2048])
def test_windowed_roundtrip_including_boundaries(length):
    torch.manual_seed(0)
    x = torch.randn(2, 1, length, dtype=torch.float64)
    window = torch.hann_window(64, dtype=torch.float64).sqrt()
    values, timing = frame(x, window, 16)
    basis = dct_basis(64, dtype=torch.float64)
    restored = unframe((values @ basis) @ basis.T, window, timing, length)
    assert torch.linalg.vector_norm(restored - x) / x.norm() < 1e-12


@pytest.mark.parametrize("kind", ["dct", "random", "orthogonal", "grouped"])
def test_model_roundtrip_and_dictionary_geometry(kind):
    torch.manual_seed(1)
    model = build_model(ModelConfig(kind=kind, frame_length=64, hop=16, threshold=0), 16000)
    x = torch.randn(2, 1, 777)
    restored, rep = model(x)
    assert (restored - x).norm() / x.norm() < 1e-4
    assert rep.original_samples == 777
    assert torch.allclose(model.matrix().T @ model.matrix(), torch.eye(64), atol=1e-5)


def test_code_budget_changes_representation_not_decoder_access():
    model = build_model(ModelConfig(kind="dct", frame_length=64, hop=16, threshold=0, code_budget=8), 16000)
    x = torch.randn(1, 1, 1000)
    rep = model.encode(x)
    expected = model.decode(rep)
    x.zero_()
    assert torch.equal(expected, model.decode(rep))
    assert (rep.concatenate().abs() > 0).sum(-1).max() <= 8


def test_pca_roundtrip_includes_mean():
    model = build_model(ModelConfig(kind="pca", frame_length=16, hop=4, groups=4, threshold=0), 16000)
    x = torch.randn(2, 1, 500) + 2
    model.fit([(x, torch.tensor([500, 500]), [])], torch.device("cpu"), 200)
    restored, _ = model(x)
    assert (restored - x).norm() / x.norm() < 1e-5
    assert model.frame_mean.abs().sum() > 0
