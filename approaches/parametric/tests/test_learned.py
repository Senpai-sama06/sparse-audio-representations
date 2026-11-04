"""Behavioral checks for explicit code budgets, normalization and shared decoding."""
import numpy as np
import pytest
import torch
from approaches.parametric.learned_models import SharedRepresentation,normalize
from approaches.parametric.learned_payload import write_payload,decode_payload


@pytest.mark.parametrize('activation',['sine','snakebeta'])
def test_decoder_needs_only_transmitted_code_and_shared_asset(tmp_path,activation):
    torch.manual_seed(3);torch.set_num_threads(1)
    model=SharedRepresentation(activation,max_dim=8,width=8,features=4)
    x=torch.randn(2,640)*.1+.03;normal,means,gains=normalize(x)
    with torch.no_grad():
        code=model.encode(normal,4);shape=model.decoder(code);expected=(shape*gains+means).sum(0)
    assert torch.count_nonzero(code[:,4:])==0
    asset=tmp_path/'decoder.pt';torch.save(dict(kind='neural',spec=model.decoder.spec,state=model.decoder.state_dict()),asset)
    payload=tmp_path/'payload.json';counts=write_payload(payload,asset,code[:,:4],means,gains)
    reconstructed,rate=decode_payload(asset,payload)
    assert counts['parameters']==2*(4+2)
    assert rate==16000 and np.allclose(reconstructed,expected.numpy(),atol=1e-7)
    assert counts['payload_bytes']==payload.stat().st_size


def test_gain_and_mean_are_invertible():
    x=torch.randn(4,640)*.1+.02;y,mean,gain=normalize(x)
    assert torch.allclose(y*gain+mean,x,atol=1e-7)
    silent=torch.zeros(1,640);y,m,g=normalize(silent)
    assert torch.isfinite(y).all() and torch.equal(y*g+m,silent)


def test_pca_asset_decodes_without_encoder(tmp_path):
    basis=torch.eye(640)[:,:8];asset=tmp_path/'pca.pt';torch.save(dict(kind='pca',basis=basis),asset)
    code=torch.ones(1,4);mean=torch.tensor([[.1]]);gain=torch.tensor([[.2]])
    payload=tmp_path/'code.json';write_payload(payload,asset,code,mean,gain)
    x,_=decode_payload(asset,payload)
    assert np.allclose(x[:4],.3) and np.allclose(x[4:],.1)


def test_decoder_identity_is_checked(tmp_path):
    asset=tmp_path/'pca.pt';torch.save(dict(kind='pca',basis=torch.eye(640)[:,:4]),asset)
    payload=tmp_path/'code.json';write_payload(payload,asset,torch.zeros(1,4),torch.zeros(1,1),torch.ones(1,1))
    torch.save(dict(kind='pca',basis=torch.ones(640,4)),asset)
    with pytest.raises(ValueError,match='identity'):decode_payload(asset,payload)
