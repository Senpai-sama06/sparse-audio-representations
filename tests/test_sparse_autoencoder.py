import json
import pytest
import torch
from lrac.config import Config, DataConfig, ModelConfig, TrainConfig, EvaluationConfig, from_dict
from lrac.models import build_model
from lrac.models.base import Representation
from lrac.data.manifests import prepare
from lrac.train import train, load_run
from lrac.evaluate import evaluate


def model():
    return build_model(ModelConfig(kind="sparse_autoencoder", channels=[2, 2, 4, 4], latent_width=8, groups=1, keep_fraction=0.25), 16000)


def test_global_selection_and_padding_exclusion():
    net = model()
    values = torch.tensor([[[1., 2., 3., 4.], [5., 6., 7., 8.], [100., 100., 100., 100.]]], requires_grad=True)
    rep = Representation({"group_0": values}, torch.tensor([300]), 16000, 300, {"hop": 256}, coefficient_mask=torch.tensor([[[True]*4, [True]*4, [False]*4]]))
    code = net.sparsify(rep, sample_budget=3)
    assert code.concatenate().nonzero().tolist() == [[0, 1, 1], [0, 1, 2], [0, 1, 3]]
    assert torch.count_nonzero(net.sparsify(rep, sample_budget=0).concatenate()) == 0
    assert torch.count_nonzero(net.sparsify(rep, keep_fraction=1.).concatenate()) == 8
    code.concatenate().sum().backward()
    assert int(values.grad.count_nonzero()) == 3


def test_boundary_lengths_and_batch_padding_do_not_change_codes():
    torch.manual_seed(1)
    net = model().eval()
    long = torch.randn(1, 1, 513)
    for length in [1, 255, 256, 257]:
        short = torch.randn(1, 1, length)
        alone = net.encode(short)
        mixed = torch.cat([torch.nn.functional.pad(short, (0, 513-length), value=123), long])
        together = net.encode(mixed, torch.tensor([length, 513]))
        steps = (length + 255)//256
        assert torch.allclose(alone.concatenate()[0], together.concatenate()[0, :steps], atol=1e-7)
        assert not together.selected_mask[0, steps:].any()
        single_output = net.decode(alone)
        batch_output = net.decode(together)
        assert single_output.shape == short.shape
        assert torch.allclose(single_output[0], batch_output[0, :, :length], atol=1e-7)
        assert not batch_output[0, :, length:].any()


def test_reconstruction_uses_only_code_and_gradients_reach_both_networks():
    net = model()
    x = torch.randn(2, 1, 512)
    estimate, code = net(x)
    estimate.square().mean().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in net.encoder.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in net.decoder.parameters())
    assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)
    before = net.decode(code).detach()
    x.zero_()
    assert torch.equal(before, net.decode(code))


def test_affine_control_preserves_parameter_budget_and_has_affine_dense_mapping():
    from dataclasses import replace
    torch.manual_seed(2)
    config = ModelConfig(kind="sparse_autoencoder",channels=[2,2,4,4],latent_width=8,groups=1)
    nonlinear = build_model(config,16000).eval()
    torch.manual_seed(2)
    affine = build_model(replace(config,activation="identity"),16000).eval()
    assert not any(isinstance(layer,torch.nn.LeakyReLU) for layer in affine.modules())
    assert sum(p.numel() for p in nonlinear.parameters()) == sum(p.numel() for p in affine.parameters())
    for key,value in nonlinear.state_dict().items():
        assert torch.equal(value,affine.state_dict()[key])
    x,y=torch.randn(1,1,512),torch.randn(1,1,512)
    def encode(value):return affine.encode_dense(value).concatenate()
    assert torch.allclose(encode(x+y),encode(x)+encode(y)-encode(torch.zeros_like(x)),atol=1e-6)
    code=affine.encode_dense(x)
    z,w=torch.randn_like(code.concatenate()),torch.randn_like(code.concatenate())
    def decode(value):return affine.decode(code.with_values(value))
    assert torch.allclose(decode(z+w),decode(z)+decode(w)-decode(torch.zeros_like(z)),atol=1e-6)


def test_checkpoint_training_and_fraction_evaluation(tmp_path):
    cfg = Config(project_root=str(tmp_path),
        data=DataConfig(manifest_dir="manifests",train_count=4,validation_count=2,test_count=2,synthetic_seconds=.032,crop_seconds=.032),
        model=ModelConfig(kind="sparse_autoencoder",channels=[2,2,4,4],latent_width=8,groups=1),
        train=TrainConfig(steps=2,batch_size=2,val_examples=1,val_interval=1,log_interval=1,device="cpu",spectral_weight=0.,keep_fractions=[.25,1.]),
        evaluation=EvaluationConfig(audio_examples=0,bootstrap_samples=2))
    prepare(cfg)
    run = train(cfg)
    _, net, state = load_run(run,"cpu")
    assert state["step"] in [0,1,2]
    assert json.loads((run/"progress.json").read_text())["status"] == "complete"
    assert len(json.loads((run/"validation.jsonl").read_text().splitlines()[-1])["keep_fraction_losses"]) == 2
    paths = evaluate(run,split="validation",limit=1,device="cpu",keep_fractions=[0.,.25,1.])
    zero = json.loads((paths[0]/"summary.json").read_text())
    assert zero["metrics"]["active_coefficients"]["mean"] == 0
    assert zero["budget_scope"] == "whole recording"
    search = json.loads((paths[-1].parent/"support_search.json").read_text())
    assert len(search["targets"]) == 3
    assert all(row["met"] + row["unmet"] + row["silent"] == 1 for row in search["targets"])
    assert json.loads((paths[1]/"summary.json").read_text())["metrics"]["active_coefficients"]["mean"] == 4
    before = [(path/"metrics.csv").read_bytes() for path in paths]
    reused = evaluate(run,split="validation",limit=1,device="cpu",keep_fractions=[0.,.25,1.],resume=True)
    assert before == [(path/"metrics.csv").read_bytes() for path in reused]
    assert json.loads((paths[-1].parent/"support_search.json").read_text()) == search
    assert json.loads((paths[-1].parent/"progress.json").read_text())["status"] == "complete"
    from lrac.analyze import analyze
    diagnostics = analyze(run, limit=1, device="cpu")
    assert json.loads((diagnostics/"diagnostics.json").read_text())["functional_interaction_estimate"] is None


@pytest.mark.parametrize("changes", [{"groups":2},{"sample_budget":-1},{"keep_fraction":1.5},{"code_budget":1}])
def test_sparse_model_rejects_ambiguous_or_invalid_settings(changes):
    with pytest.raises(ValueError):
        from_dict({"model":{"kind":"sparse_autoencoder","groups":1,**changes}})
