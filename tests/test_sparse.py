import torch
from lrac.sparse import fista, soft_threshold
from lrac.config import ModelConfig
from lrac.models import build_model


def test_sparse_inference_matches_known_solution():
    torch.manual_seed(2)
    y = torch.randn(2, 5, 12)
    actual = fista(y, lambda v: v, lambda v: v, y.shape, .2, iterations=100, tolerance=1e-8)
    assert torch.allclose(actual, soft_threshold(y, .2), atol=1e-5)


def test_overcomplete_model_learns_unit_norm_atoms():
    model = build_model(ModelConfig(kind="overcomplete", frame_length=16, hop=4, atoms=32, inference_steps=40), 16000)
    x = torch.randn(1, 1, 128)
    y, rep = model(x)
    loss = (y - x).square().mean()
    loss.backward()
    assert model.dictionary.weight.grad is not None
    assert torch.allclose(model.matrix().norm(dim=0), torch.ones(32), atol=1e-6)
    assert rep.concatenate().shape[-1] == 32


def test_convolutional_adjoint_and_learning():
    import torch.nn.functional as F
    model = build_model(ModelConfig(kind="convolutional", conv_kernels=4, conv_length=32, conv_hop=8, groups=2, inference_steps=25), 16000)
    x, codes = torch.randn(1, 1, 128), torch.randn(1, 4, 16)
    analyzed = F.conv1d(x, model.kernels, stride=8, padding=model.padding)
    synthesized = F.conv_transpose1d(codes, model.kernels, stride=8, padding=model.padding)
    assert torch.allclose((analyzed * codes).sum(), (x * synthesized).sum(), atol=1e-5)
    y, _ = model(x)
    assert y.shape == x.shape
    (y - x).square().mean().backward()
    assert torch.isfinite(model.kernels.grad).all()
