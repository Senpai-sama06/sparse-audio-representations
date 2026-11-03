"""Unquantized mathematical model files for exactly one independent block."""
import json
from pathlib import Path
import numpy as np
from .isolated_models import decode


def save(path,models,samples=640):
    obj=dict(version=1,sample_rate=16000,samples=samples,precision='float64_json',models=models)
    Path(path).write_text(json.dumps(obj,separators=(',',':'),allow_nan=False))
    return Path(path).stat().st_size


def load(path):
    obj=json.loads(Path(path).read_text())
    if obj['version']!=1 or obj['samples']!=640 or obj['sample_rate']!=16000:raise ValueError('Invalid isolated-block payload')
    audio=sum((decode(m,obj['samples']) for m in obj['models']),np.zeros(obj['samples']))
    if not np.isfinite(audio).all():raise ValueError('Nonfinite decoded samples')
    return audio,obj['sample_rate']


if __name__=='__main__':
    import argparse
    import soundfile as sf
    p=argparse.ArgumentParser();p.add_argument('payload');p.add_argument('wav');a=p.parse_args()
    audio,rate=load(a.payload);sf.write(a.wav,audio,rate,subtype='DOUBLE')
