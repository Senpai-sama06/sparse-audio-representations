"""Behavioral checks for independent synthesis, budgets and packed streams."""
import numpy as np
import pytest
from approaches.parametric.models import FAMILIES, ModeModel, fit_mode
from approaches.parametric.payload import write_payload, decode_payload
from approaches.parametric.ablation import allocate
from approaches.parametric.backends import make_backend


@pytest.mark.parametrize('family',FAMILIES)
def test_model_budget_and_finite_synthesis(family):
    t=np.linspace(-1,1,128); x=.2*np.cos(12*t+.3)
    model=fit_mode(x,family,48,steps=5)
    assert len(model.parameters)<=48
    assert np.isfinite(model.decode(128)).all()
    if family in ('sinusoid','polynomial','spline'):
        assert np.mean((model.decode(128)-x)**2)<np.mean(x*x)*.1


@pytest.mark.parametrize('bits',[8,12,16,32])
def test_parameter_file_roundtrip(tmp_path,bits):
    p=np.array([.2,.05,-.02,.01])
    frames=[(0,[ModeModel('polynomial',p)])]
    path=tmp_path/'test.par'
    info=write_payload(path,frames,samples=128,window=128,hop=64,bits=bits)
    y,rate=decode_payload(path)
    assert rate==16000 and info['payload_bytes']==path.stat().st_size
    assert info['parameters']==4+(2 if bits!=32 else 0)
    assert np.max(np.abs(y-frames[0][1][0].decode(128)))<.004


def test_budget_includes_all_overlapping_frames():
    modes=np.ones((7,128)); counts=allocate(23,modes,'uniform')
    assert sum(counts)==23 and max(counts)-min(counts)<=1
    assert sum(allocate(0,modes,'energy'))==0


def test_atomic_progress_retries_transient_windows_lock(tmp_path,monkeypatch):
    from approaches.parametric import progress
    replace=progress.os.replace; calls=[]
    def locked_once(source,target):
        calls.append(1)
        if len(calls)==1: raise PermissionError('temporary reader lock')
        return replace(source,target)
    monkeypatch.setattr(progress.os,'replace',locked_once)
    progress.atomic_json(tmp_path/'progress.json',{'status':'running'})
    assert len(calls)==2 and (tmp_path/'progress.json').exists()


@pytest.mark.parametrize('name',['emd','vmd','eemd','ceemdan','svmd'])
def test_backends_keep_odd_length_and_control(name):
    t=np.arange(129); x=np.cos(.2*t)+.1*np.sin(.7*t)
    result=make_backend(name,trials=2).decompose(x)
    assert result.modes.shape[1]==129
    assert np.allclose(result.reconstruct(include_residual=True),x,atol=1e-6)
