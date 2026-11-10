"""Checks for acoustic proxies, confound removal, and actual latent interventions."""
import numpy as np
import torch

from sae.interpret import acoustic_features, association, decode_intervention, select_channels, select_examples
from sae.models import AudioSAE, ModelConfig


def test_pitch_and_spectral_proxies_on_known_tone():
    x=.1*np.sin(2*np.pi*200*np.arange(16000)/16000)
    f,t=acoustic_features(x,63)
    valid=np.isfinite(f[:,0])
    assert not valid[0] and not valid[-1]
    assert np.allclose(np.nanmedian(f[:,4]),200,atol=3)
    assert np.allclose(np.nanmedian(f[:,1]),200,atol=3)
    assert np.nanmax(f[:,2])<.001
    assert np.nanmedian(f[:,3])>.8
    assert np.allclose(np.diff(t),.016)


def test_loudness_control_removes_spurious_association():
    rng=np.random.default_rng(42)
    loud=rng.normal(size=10000)
    x=4*loud+rng.normal(size=10000)
    y=3*loud+rng.normal(size=10000)
    assert association(x,y,loud,control=False)>.8
    assert abs(association(x,y,loud,control=True))<.04
    assert association(np.zeros(100),np.arange(100),np.arange(100)) is None


def test_intervention_only_changes_chosen_channel_and_sham_matches():
    torch.set_num_threads(1)
    torch.manual_seed(0)
    model=AudioSAE(ModelConfig(method='l1',channels=(4,4,4,4),latent_width=8,k=2)).eval()
    x=torch.randn(1,1,1793)*.1
    with torch.inference_mode():
        result=model(x,torch.tensor([1793]))
        z=result['z']; original=z.clone()
        sham=decode_intervention(model,z,0,1793,factor=1)
        assert torch.equal(sham,result['reconstruction'])
        channel=int(z.sum((0,1)).argmax())
        changed=decode_intervention(model,z,channel,1793)
        assert torch.equal(z,original)
        manual=z.clone();manual[...,channel]=0
        assert torch.equal(changed,model.decoder(manual)[...,:1793])
        assert not torch.equal(changed,sham)
        assert changed.shape==x.shape


def test_selection_is_deterministic_and_uses_distinct_cases():
    stats=[dict(channel=i,activity_fraction=0 if i==0 else .2) for i in range(12)]
    assert select_channels(stats,'l1')==select_channels(list(reversed(stats)),'l1')
    assert 0 not in select_channels(stats,'l1')
    codes=[np.array([[i,1],[i+1,0]],dtype=float) for i in range(4)]
    selected=select_examples(codes,0)
    assert [r[1] for r in selected]==[3,2,1]
