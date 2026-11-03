"""Known-answer fitting and report portability, beyond basic execution checks."""
import base64
import gzip
import re
import numpy as np
import pytest
import soundfile as sf
from approaches.parametric.isolated_models import fit,decode,candidates
from approaches.parametric.isolated_payload import save,load
from approaches.parametric.isolated_selection import select
from approaches.parametric.portable_report import export_html


def test_quartic_linear_projection_is_certified():
    t=np.linspace(-1,1,640);x=.2+.05*t-.03*t*t+.01*t**4
    r=fit(x,dict(family='polynomial',degree=4))
    assert r['nmse']<1e-25 and r['parameter_count']==5
    assert r['evidence']=='linear_least_squares_full_rank'


def test_amfm_recovers_known_chirp():
    t=np.linspace(-1,1,640);x=(.3+.05*t)*np.cos(50*t+10*t*t+.3)
    r=fit(x,dict(family='amfm',amplitude_degree=1,phase_degree=2),max_steps=800,restarts=2)
    assert r['nmse']<1e-12 and r['parameter_count']==5
    assert r['evidence']=='best_found_only' and len(r['restarts'])==2


def test_full_dct_and_payload_are_identity(tmp_path):
    x=np.random.default_rng(2).normal(size=640)
    r=fit(x,dict(family='dct',coefficients=640));path=tmp_path/'model.json'
    save(path,[r['model']]);y,rate=load(path)
    assert len(y)==640 and rate==16000 and np.max(np.abs(x-y))<1e-12


def test_sinusoid_refines_off_grid_frequency():
    t=np.linspace(-1,1,640);x=.2*np.cos(123.456*t+.7)
    r=fit(x,dict(family='sinusoid',components=1),max_steps=400,restarts=2)
    assert r['nmse']<1e-12 and r['parameter_count']==3


def test_dp_minimum_cost_for_declared_bound():
    reference=np.ones(640)
    zero=dict(model=dict(spec=dict(family='zero'),parameters=[]),parameter_count=0,error_energy=640.)
    exact=dict(model=dict(spec=dict(family='polynomial',degree=0),parameters=[1.]),parameter_count=1,error_energy=0.)
    selected=select([[zero,exact]],reference,tolerances=(.01,))[0]
    assert selected['parameter_count']==1 and selected['nmse']==0


def test_portable_report_contains_exact_audio_and_no_local_links(tmp_path):
    wav=tmp_path/'test.wav';sf.write(wav,np.sin(np.arange(640)),16000,subtype='DOUBLE')
    (tmp_path/'plot.svg').write_text('<svg/>');(tmp_path/'table.csv').write_text('a,b\n1,2')
    source=tmp_path/'source.html';source.write_text('<title>Test</title><h1>Test</h1><audio controls src="test.wav"></audio><img src="plot.svg"><a href="table.csv">CSV</a>')
    output=tmp_path/'portable.html';export_html(source,output)
    text=output.read_text();asset=re.search(r'type="application/octet-stream" id="[^"]+">([^<]+)',text)[1]
    assert gzip.decompress(base64.b64decode(asset))==wav.read_bytes()
    assert 'src="test.wav"' not in text and 'href="table.csv"' not in text
    assert 'data:image/svg+xml;base64,' in text and 'Load embedded audio' in text


def test_no_orders_chosen_from_zero_crossings():
    assert len(candidates())==53
    assert max(s.get('degree',0) for s in candidates())==128
