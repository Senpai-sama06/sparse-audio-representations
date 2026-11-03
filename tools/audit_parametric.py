"""Read-only audit of a parametric run, plus isolated known-answer fits.

Writes diagnostics elsewhere; never alters experiment code, files or processes.
Run: python tools/audit_parametric.py --run RUN --output AUDIT
"""
import argparse
import csv
import json
import struct
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import soundfile as sf
from approaches.parametric.models import fit_mode, synthesize
from approaches.parametric.ablation import allocate


def nmse(x,y):
    return float(np.sum((x-y)**2)/np.sum(x*x))


def main(run,output):
    torch.set_num_threads(1)
    root=Path(run); out=Path(output); out.mkdir(parents=True,exist_ok=True)
    results=[]; signals={}
    t=np.linspace(-1,1,640)
    examples={
        'quartic': ('.2+.1t-.04t^2+.02t^3+.01t^4', .2+.1*t-.04*t*t+.02*t**3+.01*t**4),
        'tone_1000Hz_40ms': ('0.2 cos(2 pi 1000 n/16000 + .4)', .2*np.cos(2*np.pi*1000*np.arange(640)/16000+.4)),
        'amfm_chirp': ('(.3+.05t) cos(50t+10t^2+.3)',(.3+.05*t)*np.cos(50*t+10*t*t+.3)),
    }
    for name,(formula,x) in examples.items():
        for family,budget,steps in [('polynomial',5,80),('sinusoid',3,80),('amfm',6,80),('amfm',6,800)]:
            model=fit_mode(x,family,budget,steps=steps,seed=42); y=model.decode(len(x))
            key=f'{name}_{family}_{steps}'; signals[key]=y
            results.append(dict(signal=name,formula=formula,family=family,budget=budget,steps=steps,
                                parameters=len(model.parameters),nmse=nmse(x,y)))
        if name=='amfm_chirp':
            y=synthesize('amfm',np.array([.3,.05,0,.3,50,10]),len(x))
            results.append(dict(signal=name,formula=formula,family='oracle_amfm',budget=6,steps=0,parameters=6,nmse=nmse(x,y)))
    cache=root/'cache/case0_vmd_40ms.npz'
    with np.load(cache) as data:
        count=len(data['starts'])
        energies=[np.sum(data[f'm{i}']**2) for i in range(count)]
        fi=int(np.argmax(energies)); modes=data[f'm{fi}']; mi=int(np.argmax(np.sum(modes*modes,axis=1)))
        x=modes[mi]; mode_info=dict(frame=fi,mode=mi,start_sample=int(data['starts'][fi]))
        for family in ('polynomial','spline','sinusoid','amfm','sine_network','snakebeta_network'):
            for steps in (80,800) if family in ('amfm','sine_network','snakebeta_network') else (80,):
                model=fit_mode(x,family,18,steps=steps,seed=42)
                results.append(dict(signal='real_vmd_mode',formula=str(mode_info),family=family,budget=18,
                                    steps=steps,parameters=len(model.parameters),nmse=nmse(x,model.decode(len(x)))))
        allocations={}
        for budget in (500,1000,2000,4000,8000):
            counts=allocate(int(budget*5.855)//count,np.vstack([modes,data[f'r{fi}']]),'uniform')
            allocations[budget]=list(map(int,counts))
    payloads=[]
    for family in ('sine_network','snakebeta_network','amfm','sinusoid','polynomial'):
        for budget in (500,8000):
            path=root/f'case0_vmd_40ms_{family}_{budget}_32bit.par'
            if not path.exists(): continue
            raw=path.read_bytes(); length=struct.unpack('<I',raw[4:8])[0]; header=json.loads(raw[8:8+length])
            sizes=[n for _,records in header['frames'] for _,n,_ in records]
            payloads.append(dict(family=family,budget=budget,records=len(sizes),empty=sum(n==0 for n in sizes),
                                 min_parameters=min(sizes),max_parameters=max(sizes),mean_parameters=float(np.mean(sizes))))
    with (out/'fits.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(results[0])); writer.writeheader(); writer.writerows(results)
    (out/'audit.json').write_text(json.dumps(dict(fits=results,selected_real_mode=mode_info,
                                                real_frame_allocations=allocations,payload_records=payloads),indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,1,figsize=(11,7))
    for ax,name in zip(axes,('tone_1000Hz_40ms','amfm_chirp')):
        x=examples[name][1]; ax.plot(t,x,label='target',color='black',linewidth=1)
        for family,steps in [('polynomial',80),('amfm',80),('amfm',800)]:
            ax.plot(t,signals[f'{name}_{family}_{steps}'],label=f'{family}, {steps} steps',alpha=.7)
        ax.set_title(name); ax.legend(fontsize=8); ax.set_xlabel('Normalized time')
    fig.tight_layout(); fig.savefig(out/'known_answer_fits.png',dpi=150)
    print(json.dumps(dict(fits=results,payload_records=payloads,allocations=allocations),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--run',required=True);p.add_argument('--output',required=True)
    main(**vars(p.parse_args()))
