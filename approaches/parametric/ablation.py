"""Resumable whole-utterance mathematical mode coding experiments.

Run python -m approaches.parametric.ablation --help. Python callers may pass
ordinary keyword arguments from main.py. Manifest equality protects resume.
Cached samples are used only by the encoder; every scored result is decoded
from its own stream. An interrupted fit is repeated, completed jobs are reused.
"""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import numpy as np
import soundfile as sf
import torch
from .backends import make_backend
from .models import FAMILIES, fit_mode
from .payload import write_payload, decode_payload
from .progress import Progress, atomic_json
from .reconstruction import WindowedReconstructor
from .experiment import _score, _save_and_read
from .ablation_report import write_ablation_report, write_table
from util.provenance import environment, file_hash, source_fingerprint


def allocate(total, modes, policy):
    """Integer apportionment; sum never exceeds the total frame budget."""
    if not len(modes): return []
    weights = np.ones(len(modes)) if policy=='uniform' else np.array([float(m@m) for m in modes])
    weights = weights/weights.sum() if weights.sum() else np.ones(len(modes))/len(modes)
    exact=total*weights; counts=np.floor(exact).astype(int)
    for i in np.argsort(-(exact-counts))[:total-int(counts.sum())]: counts[i]+=1
    return counts


def run_ablation(audio, output, *, windows=(40,20,80,160), budgets=(500,1000,2000,4000,8000),
                 methods=('emd','vmd','eemd','ceemdan','svmd','direct'), families=FAMILIES,
                 precisions=(32,16,12,8), trials=32, steps=80, seed=42,
                 residual='encode', allocation='uniform', speech_metrics=True):
    """Execute the declared full grid, with explicit fitting limits and provenance."""
    torch.set_num_threads(1)
    if residual not in ('encode','omit') or allocation not in ('uniform','energy'):
        raise ValueError('Invalid residual/allocation policy')
    if any(b<=0 for b in budgets) or any(w<2 for w in windows) or trials<1 or steps<1:
        raise ValueError('Positive budgets, windows >=2 ms, trials and steps required')
    root=Path(output).resolve(); root.mkdir(parents=True,exist_ok=True)
    sources=[Path(p).resolve() for p in ([audio] if isinstance(audio,(str,Path)) else audio)]
    manifest=dict(version=1,sources=[dict(path=str(p),sha256=file_hash(p)) for p in sources],
                  windows=list(windows),budgets=list(budgets),methods=list(methods),families=list(families),
                  precisions=list(precisions),trials=trials,steps=steps,seed=seed,residual=residual,
                  allocation=allocation,speech_metrics=speech_metrics,sample_rate=16000,overlap=.5,
                  scope='Per-mode fits, complete utterances; no corpus-trained decoder; uniform quantizer per model',
                  versions={p:importlib.metadata.version(p) for p in ('EMD-signal','vmdpy','pysdkit')},
                  source=source_fingerprint(Path(__file__).resolve().parents[2]))
    manifest_path=root/'manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text())!=manifest:
        raise ValueError('Resume settings or source code differ; select a new output directory')
    atomic_json(manifest_path,manifest); atomic_json(root/'environment.json',environment())
    jobs=[]
    for ci in range(len(sources)):
        for window in windows:
            for method in methods:
                prefix=f'case{ci}_{method}_{window}ms'
                jobs.append(dict(id=prefix,kind='decomposition',case=ci,window=window,method=method))
                for family in families:
                    for budget in budgets:
                        jobs.append(dict(id=f'{prefix}_{family}_{budget}',kind='fit',case=ci,
                                         window=window,method=method,family=family,budget=budget))
    (root/'jobs').mkdir(exist_ok=True); (root/'cache').mkdir(exist_ok=True)
    atomic_json(root/'jobs.json',jobs)
    def done(job): return (root/'jobs'/f"{job['id']}.json").exists()
    rows=[]
    for job in jobs:
        if done(job): rows.extend(json.loads((root/'jobs'/f"{job['id']}.json").read_text())['rows'])
    progress=Progress(root,len(jobs),sum(done(j) for j in jobs)); completed_this_run=0
    references=[]
    for ci,path in enumerate(sources):
        x,rate=sf.read(path,dtype='float64')
        if rate!=16000 or x.ndim!=1 or len(x)<64 or not np.isfinite(x).all():
            progress.close('failed'); raise ValueError('Inputs must be finite mono 16 kHz audio >=64 samples')
        references.append(_save_and_read(root/f'reference_{ci}.wav',x,rate))
    def refresh():
        write_table(root/'metrics.csv',rows); write_ablation_report(root,rows)
    try:
        for job in jobs:
            if done(job): continue
            started=time.perf_counter(); stage=job['kind']+':'+job.get('family',job['method'])
            x=references[job['case']]; duration=len(x)/16000; size=round(job['window']*16); hop=size//2
            prefix=f"case{job['case']}_{job['method']}_{job['window']}ms"
            cache=root/'cache'/f'{prefix}.npz'
            progress.update(current_job=job['id'],stage=stage,frames_completed=0,error=None)
            print('START',job['id'],flush=True)
            jobrows=[]
            try:
                starts=list(range(-(size//2),len(x),hop)); progress.update(frames_total=len(starts))
                if job['kind']=='decomposition':
                    recon=WindowedReconstructor(size,.5); groups=[]; diagnostics=[]
                    backend=None if job['method']=='direct' else make_backend(job['method'],trials,seed)
                    padded=np.pad(x,(size,size),mode='reflect')
                    modes_audio=np.zeros_like(x); control_audio=np.zeros_like(x); weight=np.zeros_like(x)
                    for fi,start in enumerate(starts):
                        frame=padded[start+size:start+2*size]
                        if backend is None:
                            modes,residue=frame[None,:],np.zeros_like(frame); diagnostics.append({})
                        else:
                            result=backend.decompose(frame); modes,residue=result.modes,result.residual
                            diagnostics.append(result.diagnostics)
                        groups.append((modes,residue))
                        left,right=max(0,start),min(len(x),start+size); sl=slice(left-start,right-start)
                        modes_audio[left:right]+=modes.sum(0)[sl]*recon.weights[sl]
                        control_audio[left:right]+=(modes.sum(0)+residue)[sl]*recon.weights[sl]
                        weight[left:right]+=recon.weights[sl]
                        progress.update(frames_completed=fi+1)
                    temp=cache.with_suffix('.tmp.npz')
                    np.savez(temp,starts=starts,diagnostics=json.dumps(diagnostics),**{f'm{i}':g[0] for i,g in enumerate(groups)},
                             **{f'r{i}':g[1] for i,g in enumerate(groups)})
                    os.replace(temp,cache)
                    for label,estimate in [('sampled_modes',modes_audio/weight),('residual_control',control_audio/weight)]:
                        name=f'{prefix}_{label}.wav'; estimate=_save_and_read(root/name,estimate,16000)
                        jobrows.append(dict(case=job['case'],method=job['method'],family=label,window_ms=job['window'],
                                            status='control',audio=name,**_score(x,estimate,16000,speech_metrics)))
                else:
                    if not cache.exists(): raise RuntimeError('Decomposition cache missing: inspect preceding failure')
                    stream=[]; total_budget=int(job['budget']*duration)
                    with np.load(cache) as data:
                        for fi,start in enumerate(data['starts']):
                            modes=data[f'm{fi}']
                            if residual=='encode' and job['method']!='direct': modes=np.vstack([modes,data[f'r{fi}']])
                            frame_budget=total_budget//len(starts)+(fi<total_budget%len(starts))
                            counts=allocate(frame_budget,modes,allocation)
                            models=[fit_mode(mode,job['family'],count,steps,seed+fi*31+mi)
                                    for mi,(mode,count) in enumerate(zip(modes,counts))]
                            stream.append((int(start),models)); progress.update(frames_completed=fi+1)
                    fit_seconds=time.perf_counter()-started
                    for bits in precisions:
                        name=f"{job['id']}_{bits}bit"; path=root/f'{name}.par'
                        counts=write_payload(path,stream,samples=len(x),window=size,hop=hop,bits=bits)
                        t=time.perf_counter(); estimate,_=decode_payload(path); decode_seconds=time.perf_counter()-t
                        if not (root/'fresh_decode_verified.json').exists():
                            probe=root/'fresh_decode.wav'
                            subprocess.run([sys.executable,'-m','approaches.parametric.payload',str(path),str(probe)],check=True)
                            fresh,_=sf.read(probe)
                            if not np.allclose(fresh,estimate,atol=1e-7,rtol=1e-6): raise RuntimeError('Independent decoder mismatch')
                            atomic_json(root/'fresh_decode_verified.json',dict(payload=path.name,process='fresh Python interpreter'))
                        wav=f'{name}.wav'; estimate=_save_and_read(root/wav,estimate,16000)
                        row=dict(case=job['case'],method=job['method'],family=job['family'],window_ms=job['window'],
                                 budget=job['budget'],bits=bits,**counts,parameters_per_second=counts['parameters']/duration,
                                 parameter_sample_ratio=counts['parameters']/len(x),kbps=8*counts['payload_bytes']/duration/1000,
                                 pcm16_size_ratio=counts['payload_bytes']/(len(x)*2),fit_seconds=fit_seconds,
                                 decode_seconds=decode_seconds,status='ok',audio=wav,payload=path.name,
                                 residual=residual,allocation=allocation,**_score(x,estimate,16000,speech_metrics))
                        jobrows.append(row)
            except Exception as error:
                traceback.print_exc()
                jobrows=[dict(case=job['case'],method=job['method'],family=job.get('family','decomposition'),
                              window_ms=job['window'],status='failed',error=f'{type(error).__name__}: {error}')]
                progress.update(failures=progress.state['failures']+1,error=str(error))
            elapsed=time.perf_counter()-started
            old=progress.durations.get(stage); progress.durations[stage]=elapsed if old is None else .7*old+.3*elapsed
            atomic_json(root/'jobs'/f"{job['id']}.json",dict(seconds=elapsed,rows=jobrows))
            rows.extend(jobrows); completed_this_run+=1
            remaining=[j for j in jobs if not done(j)]
            known=list(progress.durations.values())
            eta=sum(progress.durations.get(j['kind']+':'+j.get('family',j['method']),max(known)) for j in remaining)
            progress.update(jobs_completed=sum(done(j) for j in jobs),eta_seconds=eta,
                            eta_calibrated=all(j['kind']+':'+j.get('family',j['method']) in progress.durations for j in remaining))
            write_table(root/'metrics.csv',rows)
            if completed_this_run%8==0: refresh()
            print('DONE',job['id'],f'{elapsed:.1f}s',flush=True)
        refresh(); progress.close('complete_with_failures' if any(r['status']=='failed' for r in rows) else 'complete')
    except BaseException:
        progress.close('interrupted'); refresh(); raise
    return rows


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audio',nargs='+',required=True); p.add_argument('--output',required=True)
    p.add_argument('--windows',nargs='+',type=int,default=[40,20,80,160])
    p.add_argument('--budgets',nargs='+',type=int,default=[500,1000,2000,4000,8000])
    p.add_argument('--methods',nargs='+',default=['emd','vmd','eemd','ceemdan','svmd','direct'])
    p.add_argument('--families',nargs='+',choices=FAMILIES,default=list(FAMILIES))
    p.add_argument('--precisions',nargs='+',type=int,default=[32,16,12,8])
    p.add_argument('--trials',type=int,default=32); p.add_argument('--steps',type=int,default=80)
    p.add_argument('--seed',type=int,default=42); p.add_argument('--residual',choices=['encode','omit'],default='encode')
    p.add_argument('--allocation',choices=['uniform','energy'],default='uniform'); p.add_argument('--no-speech-metrics',action='store_true')
    a=vars(p.parse_args()); a['speech_metrics']=not a.pop('no_speech_metrics'); run_ablation(**a)
