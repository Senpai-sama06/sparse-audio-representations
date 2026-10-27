import torch
from lrac.losses import functional_interaction, exact_interaction
from lrac.config import ModelConfig
from lrac.models import build_model


def test_orthogonal_and_redundant_functional_effects():
    z = torch.tensor([.3, -.2], dtype=torch.float64)
    orthogonal = lambda v: v
    redundant = lambda v: torch.stack([v.sum(), v.sum()])
    assert float(exact_interaction(orthogonal, z, 2)) == 0
    assert torch.allclose(exact_interaction(redundant, z, 2), torch.tensor(1., dtype=torch.float64))
    assert torch.allclose(functional_interaction(redundant, z, 2, probes=64), exact_interaction(redundant, z, 2), atol=1e-8)


def test_randomized_estimate_agrees_with_exact_and_has_gradient():
    torch.manual_seed(4)
    weight = torch.randn(5, 4, dtype=torch.float64, requires_grad=True)
    z = torch.randn(4, dtype=torch.float64)
    synthesis = lambda value: weight @ value
    estimate = functional_interaction(synthesis, z, 2, probes=256)
    exact = exact_interaction(synthesis, z, 2)
    assert abs(float((estimate - exact).detach())) < .07
    estimate.backward()
    assert torch.isfinite(weight.grad).all()


def test_jvp_matches_finite_difference():
    z = torch.tensor([.2, .4], dtype=torch.float64)
    direction = torch.tensor([.5, -.3], dtype=torch.float64)
    synthesis = lambda v: torch.stack([v[0].sin(), v[0] * v[1], v[1].square()])
    _, response = torch.func.jvp(synthesis, (z,), (direction,))
    eps = 1e-6
    numerical = (synthesis(z + eps * direction) - synthesis(z - eps * direction)) / (2 * eps)
    assert torch.allclose(response, numerical, atol=1e-8)


def test_nonlinear_model_lengths_and_functional_backprop():
    cfg = ModelConfig(kind="autoencoder", channels=[4, 4, 8, 8], latent_width=8, groups=2)
    model = build_model(cfg, 16000)
    x = torch.randn(1, 1, 513)
    output, representation = model(x)
    assert output.shape == x.shape
    z = representation.concatenate()
    loss = functional_interaction(lambda v: model.decode(representation.with_values(v)), z, 2, probes=2)
    (loss + (output - x).square().mean()).backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
