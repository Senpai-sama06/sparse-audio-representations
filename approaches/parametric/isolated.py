"""Single-block capacity study: no overlap, stitching, quantization or RVQ.

Run python -m approaches.parametric.isolated --source AUDIO --output RUN.
Clip offsets are declared before fitting. Code/source hashes protect resume.
"""
import argparse
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback
import numpy as np
import soundfile as sf
import torch
from .backends import make_backend
from .isolated_models import candidates,identifier,fit,decode
from .isolated_payload import save,load
from .isolated_selection import select
from .isolated_report import write_report
from .ablation_report import write_table
from .progress import Progress,atomic_json
from util.provenance import source_fingerprint,file_hash,environment


def metrics(x,y):
    energy=float(x@x);e=x-y;edge=np.r_[np.arange(64),np.arange(576,640)]
    xf,yf=np.abs(np.fft.rfft(x)),np.abs(np.fft.rfft(y))
    h=np.hanning(640);hx,hy=np.abs(np.fft.rfft(x*h)),np.abs(np.fft.rfft(y*h))
    return dict(nmse=float(e@e/max(energy,1e-30)),spectral_nmse=float(np.sum((xf-yf)**2)/max(xf@xf,1e-30)),
                hann_spectral_nmse=float(np.sum((hx-hy)**2)/max(hx@hx,1e-30)),
                edge_nmse=float(e[edge]@e[edge]/max(x[edge]@x[edge],1e-30)),
                interior_nmse=float(e[64:576]@e[64:576]/max(x[64:576]@x[64:576],1e-30)),
                error_rms=float(np.sqrt(np.mean(e*e))),reconstruction_rms=float(np.sqrt(np.mean(y*y))))


def geometry(x):
    nonzero=x[x!=0];delta=np.diff(x);delta=delta[delta!=0]
    return dict(zero_crossings=int(np.sum(nonzero[1:]*nonzero[:-1]<0)),
                extrema=int(np.sum(delta[1:]*delta[:-1]<0)),energy=float(x@x))


def run(source,output,*,offsets=(.96,1.96,2.96),methods=('emd','vmd','eemd','ceemdan','svmd','direct'),
        trials=32,restarts=3,max_steps=1600,seed=42,library=None):
    from threadpoolctl import threadpool_limits
    pools=threadpool_limits(limits=1)
    torch.set_num_threads(1);root=Path(output).resolve();root.mkdir(parents=True,exist_ok=True)
    signal,rate=sf.read(source,dtype='float64')
    if rate!=16000 or signal.ndim!=1 or not np.isfinite(signal).all():raise ValueError('Need finite mono 16 kHz source')
    starts=[round(float(t)*rate) for t in offsets]
    if not starts or any(s<0 or s+640>len(signal) for s in starts):raise ValueError('Offsets must contain complete 640-sample blocks')
    library=candidates() if library is None else library
    manifest=dict(version=1,source_path=str(Path(source).resolve()),source_sha256=file_hash(source),starts=starts,
                  samples=640,sample_rate=rate,methods=list(methods),trials=trials,restarts=restarts,max_steps=max_steps,
                  seed=seed,library=library,quantization=False,rvq=False,stitching=False,overlap=False,
                  selection='DP minimum count for triangle-inequality error bound over available fitted candidates',
                  boundary='Only 640 original samples passed to backend; native backend extensions allowed and versioned; no neighboring samples',
                  optimizer='Analytic-Jacobian scipy least_squares for AMFM/sinusoids; Adam multi-start for periodic networks; SVD linear projections',
                  versions={p:importlib.metadata.version(p) for p in ('EMD-signal','vmdpy','pysdkit','scipy','torch')},
                  source=source_fingerprint(Path(__file__).resolve().parents[2]))
    if (root/'manifest.json').exists() and json.loads((root/'manifest.json').read_text())!=manifest:raise ValueError('Manifest mismatch; choose a new output')
    atomic_json(root/'manifest.json',manifest);atomic_json(root/'environment.json',environment())
    (root/'jobs').mkdir(exist_ok=True);(root/'cache').mkdir(exist_ok=True)
    references=[]
    for ci,start in enumerate(starts):
        x=signal[start:start+640].copy();references.append(x);sf.write(root/f'reference_{ci}.wav',x,rate,subtype='FLOAT')
    jobs=[]
    for ci in range(len(starts)):
        for method in methods:
            prefix=f'case{ci}_{method}'
            jobs.append(dict(id=prefix+'_decomposition',kind='decomposition',case=ci,method=method))
            for spec in library:jobs.append(dict(id=prefix+'_'+identifier(spec),kind='fit',case=ci,method=method,spec=spec))
    atomic_json(root/'jobs.json',jobs)
    def result_path(job):return root/'jobs'/(job['id']+'.json')
    rows=[]
    for j in jobs:
        if result_path(j).exists():rows.extend(json.loads(result_path(j).read_text()).get('rows',[]))
    progress=Progress(root,len(jobs),sum(result_path(j).exists() for j in jobs));selections=[]
    def update_report():write_report(root,rows,selections)
    def aggregate(ci,method):
        cache=root/'cache'/f'case{ci}_{method}.npz'
        if not cache.exists():return
        with np.load(cache) as data:components=data['components']
        groups=[[dict(model=dict(spec=dict(family='zero'),parameters=[]),parameter_count=0,error_energy=float(x@x))] for x in components]
        group_jobs=[j for j in jobs if j['case']==ci and j['method']==method and j['kind']=='fit']
        complete=True
        for j in group_jobs:
            if not result_path(j).exists():complete=False;continue
            record=json.loads(result_path(j).read_text())
            if record['status']!='ok':complete=False;continue
            for bucket,candidate in zip(groups,record['fits']):bucket.append(candidate)
        for selection in select(groups,references[ci]):
            row={k:v for k,v in selection.items() if k!='models'};row.update(case=ci,method=method,library_complete=complete)
            if selection['status']=='achieved':
                name=f'case{ci}_{method}_selected_{selection["tolerance"]}'
                save(root/f'{name}.json',selection['models']);audio,_=load(root/f'{name}.json');sf.write(root/f'{name}.wav',audio,rate,subtype='FLOAT')
                row['audio']=f'{name}.wav';row['parameter_sample_ratio']=selection['parameter_count']/640
            selections.append(row)
    try:
        for ji,job in enumerate(jobs):
            if result_path(job).exists():continue
            began=time.perf_counter();ci,method=job['case'],job['method'];x=references[ci]
            cache=root/'cache'/f'case{ci}_{method}.npz';stage=job['kind']+':'+job.get('spec',{}).get('family',method)
            progress.update(current_job=job['id'],stage=stage,frames_completed=0,frames_total=1,error=None)
            print('START',job['id'],flush=True)
            try:
                if job['kind']=='decomposition':
                    if method=='direct':components=x[None,:];diagnostics={}
                    else:
                        decomposed=make_backend(method,trials,seed).decompose(x)
                        components=np.vstack([decomposed.modes,decomposed.residual]);diagnostics=decomposed.diagnostics
                    np.savez(cache,components=components,reference=x)
                    reconstructed=components.sum(0);name=f'case{ci}_{method}_sampled_control.wav';sf.write(root/name,reconstructed,rate,subtype='FLOAT')
                    row=dict(case=ci,method=method,family='sampled_control',complexity='',parameter_count=None,
                             parameter_sample_ratio=None,status='control',audio=name,seconds=time.perf_counter()-began,**metrics(x,reconstructed))
                    result=dict(status='ok',rows=[row],geometry=[geometry(v) for v in components],diagnostics=diagnostics)
                else:
                    with np.load(cache) as data:components=data['components']
                    fits=[];progress.update(frames_total=len(components))
                    for mi,component in enumerate(components):
                        fitted=fit(component,job['spec'],restarts=restarts,max_steps=max_steps,seed=seed+ci*1000+mi)
                        fitted['geometry']=geometry(component);fits.append(fitted);progress.update(frames_completed=mi+1)
                    models=[f['model'] for f in fits];name=job['id'];payload=root/f'{name}.json'
                    payload_bytes=save(payload,models);estimate,_=load(payload)
                    if not (root/f'fresh_{job["spec"]["family"]}.json').exists():
                        probe=root/'decoder_probe.wav'
                        subprocess.run([sys.executable,'-m','approaches.parametric.isolated_payload',str(payload),str(probe)],check=True)
                        fresh,_=sf.read(probe,dtype='float64')
                        if not np.array_equal(estimate,fresh):raise RuntimeError('Fresh-process decoding mismatch')
                        atomic_json(root/f'fresh_{job["spec"]["family"]}.json',dict(payload=payload.name,verified=True))
                    wav=f'{name}.wav';sf.write(root/wav,estimate,rate,subtype='FLOAT');saved,_=sf.read(root/wav,dtype='float64')
                    count=sum(f['parameter_count'] for f in fits)
                    row=dict(case=ci,method=method,family=job['spec']['family'],complexity=json.dumps(job['spec']),
                             parameter_count=count,parameter_sample_ratio=count/640,payload_bytes=payload_bytes,
                             seconds=time.perf_counter()-began,status='ok',audio=wav,job_file=str(result_path(job).relative_to(root)).replace('\\','/'),
                             evidence=';'.join(sorted({f['evidence'] for f in fits})),
                             playback_nmse=metrics(x,saved)['nmse'],**metrics(x,estimate))
                    result=dict(status='ok',rows=[row],fits=fits)
            except Exception as error:
                traceback.print_exc();progress.update(failures=progress.state['failures']+1,error=str(error))
                row=dict(case=ci,method=method,family=job.get('spec',{}).get('family','decomposition'),status='failed',error=f'{type(error).__name__}: {error}')
                result=dict(status='failed',rows=[row])
            elapsed=time.perf_counter()-began;result['seconds']=elapsed;atomic_json(result_path(job),result)
            rows.extend(result['rows']);progress.durations[stage]=elapsed if stage not in progress.durations else .6*progress.durations[stage]+.4*elapsed
            remaining=[j for j in jobs if not result_path(j).exists()];costs=progress.durations
            eta=sum(costs.get(j['kind']+':'+j.get('spec',{}).get('family',j['method']),max(costs.values())) for j in remaining)
            progress.update(jobs_completed=len(jobs)-len(remaining),eta_seconds=eta,eta_calibrated=False)
            write_table(root/'metrics.csv',rows)
            if (ji+1)%12==0 or job==jobs[-1]:
                selections=[]
                for c,m in sorted({(r['case'],r['method']) for r in rows}):aggregate(c,m)
                update_report()
            print('DONE',job['id'],f'{elapsed:.2f}s',flush=True)
        selections=[]
        for c in range(len(starts)):
            for m in methods:aggregate(c,m)
        update_report();progress.close('complete_with_failures' if any(r['status']=='failed' for r in rows) else 'complete')
    except BaseException:
        progress.close('interrupted');raise
    return rows


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    p.add_argument('--offsets',nargs='+',type=float,default=[.96,1.96,2.96]);p.add_argument('--methods',nargs='+',default=['emd','vmd','eemd','ceemdan','svmd','direct'])
    p.add_argument('--trials',type=int,default=32);p.add_argument('--restarts',type=int,default=3);p.add_argument('--max-steps',type=int,default=1600);p.add_argument('--seed',type=int,default=42)
    run(**vars(p.parse_args()))
