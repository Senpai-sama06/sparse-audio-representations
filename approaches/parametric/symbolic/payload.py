"""Evaluate a saved formula using only its explicit structure and constants."""
import argparse
import json
from pathlib import Path
import numpy as np
import soundfile as sf
from .expressions import evaluate,description


def reconstruct(path):
    payload=json.loads(Path(path).read_text())
    if payload['format']!='symbolic-imf-v1':raise ValueError('Unknown expression format')
    if payload['parameters']!=description(payload['tokens'])['parameters']:raise ValueError('Parameter accounting mismatch')
    t=np.linspace(-1,1,payload['samples'])
    return payload['gain']*evaluate(payload['tokens'],payload['constants'],t),payload['sample_rate']


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('expression');p.add_argument('output');a=p.parse_args()
    y,rate=reconstruct(a.expression);sf.write(a.output,y,rate,subtype='FLOAT')
