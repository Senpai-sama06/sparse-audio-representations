"""PitchFlower-style parametric case reports from saved recordings and fits.

No training or refitting. Only saved formulas are evaluated; all other outputs
are existing WAV files. Measurements distinguish waveform, scalar and byte cost.
"""
from pathlib import Path
import base64
import csv
import gzip
import html
import io
import json
import re
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy.signal import stft

from . import tabbed_report as archive
from .symbolic.expressions import evaluate

ROOT, RUNS, OUT, SR = archive.ROOT, archive.RUNS, archive.ARCHIVE, 16000
WORK=ROOT/'tmp/parametric_case_report'
RECORDS=[]
SOURCES={}
IMAGES={}


def rows(path):
    SOURCES[Path(path).relative_to(ROOT).as_posix()]=archive.sha(path)
    with Path(path).open(newline='',encoding='utf-8-sig') as stream:return list(csv.DictReader(stream))


def read(path):
    SOURCES[Path(path).relative_to(ROOT).as_posix()]=archive.sha(path)
    return archive.read(path)


def wav(path):
    SOURCES[Path(path).relative_to(ROOT).as_posix()]=archive.sha(path)
    x,sr=sf.read(path,dtype='float32');assert sr==SR and x.ndim==1
    return x


def fmt(v):return 'Not available' if v is None or v=='' else f'{float(v):.5g}'


def image(x,kind,peak):
    """Use the same native-amplitude and absolute dB scales across a case."""
    key=(x.tobytes(),kind,peak)
    if key in IMAGES:return IMAGES[key]
    fig,ax=plt.subplots(figsize=(4.1,2.1))
    short=len(x)==640;mult=1000 if short else 1
    if kind=='Waveform':
        ax.plot(np.arange(len(x))/SR*mult,x,linewidth=.65,color='black')
        ax.set(ylabel='Native amplitude',ylim=(-peak,peak))
    else:
        win,hop=(128,16) if short else (512,128)
        f,t,z=stft(x,fs=SR,window='hann',nperseg=win,noverlap=win-hop,nfft=win,boundary=None,padded=False,scaling='spectrum')
        ax.pcolormesh(t*mult,f/1000,20*np.log10(np.maximum(abs(z),1e-6)),vmin=-100,vmax=0,cmap='magma',shading='auto')
        ax.set(ylabel='Frequency (kHz)',ylim=(0,8))
    ax.set(xlabel='Time within block (ms)' if short else 'Time (s)',xlim=(0,len(x)/SR*mult))
    fig.tight_layout();buf=io.BytesIO();fig.savefig(buf,format='png',dpi=100);plt.close(fig)
    result='<img width="410" height="210" alt="'+kind+'" src="'+archive.data_uri(buf.getvalue(),'image/png')+'">'
    IMAGES[key]=result
    return result


def table(case,source,outputs,metadata,note):
    """Scores are recalculated on the actual float32 listening samples."""
    source=np.asarray(source,dtype='float32')
    items=[dict(label='Before: source',x=source)]+outputs
    for item in items:
        item['x']=np.asarray(item['x'],dtype='float32')
        assert item['x'].shape==source.shape and np.isfinite(item['x']).all()
    peak=max(max(float(abs(v['x']).max()) for v in items),1e-8)*1.05
    duration=len(source)/SR
    result='<h3>'+html.escape(case)+'</h3><p>'+html.escape(metadata)+'</p><div class="case-scroll"><table border="1"><tr><th>Item</th>'+''.join('<th>'+html.escape(v['label'])+'</th>' for v in items)+'</tr>'
    def row(label,values):return '<tr><th>'+label+'</th>'+''.join('<td>'+str(v)+'</td>' for v in values)+'</tr>'
    result+=row('Audio',[archive.audio(v['x']) for v in items])
    for kind in ('Waveform','Spectrogram'):result+=row(kind,[image(v['x'],kind,peak) for v in items])
    errors=[float(np.sum((v['x'].astype(float)-source)**2)/max(float(np.sum(source.astype(float)**2)),1e-30)) for v in outputs]
    result+=row('Waveform NMSE (embedded audio)',['Reference']+[fmt(v) for v in errors])
    for label,key in [('PESQ-WB (saved full-recording score)','pesq_wb'),('STOI (saved full-recording score)','stoi')]:
        result+=row(label,['Reference']+[fmt(v.get(key)) if len(source)>640 else 'Not applicable to 40 ms / IMF' for v in outputs])
    result+=row('Playback PCM rate (kbit/s)',['512.000']*len(items))
    result+=row('Stored scalar values (includes required gains / means)',['Not applicable']+[fmt(v.get('values')) for v in outputs])
    result+=row('Measured serialized representation rate (kbit/s)',['Not applicable']+[fmt(v.get('encoded_kbps')) for v in outputs])
    result+=row('Representation rate basis',['Source waveform']+[html.escape(v.get('rate_basis','Not measured; sampled reconstruction control')) for v in outputs])
    result+='</table></div><p>'+html.escape(note)+'</p>'
    formulas=[v.get('formula') for v in outputs]
    if any(formulas):result+='<details><summary>Saved fitted mathematical expressions</summary>'+''.join('<p>'+html.escape(v['label'])+': <code>'+html.escape(v.get('formula',''))+'</code></p>' for v in outputs)+'</details>'
    record=dict(case=case,metadata=metadata,samples=len(source),duration_seconds=duration,outputs=[])
    for v,error in zip(outputs,errors):
        if v.get('saved_nmse') is not None:assert abs(error-float(v['saved_nmse']))<2e-5,(case,error,v['saved_nmse'])
        record['outputs'].append({**{k:value for k,value in v.items() if k!='x'},'embedded_pcm_nmse':error})
    RECORDS.append(record)
    return result


def evidence(path,label=None):
    raw=Path(path).read_bytes();SOURCES[Path(path).relative_to(ROOT).as_posix()]=archive.sha(path)
    return '<p><a download="'+Path(path).name+'" href="'+archive.data_uri(raw,'application/octet-stream')+'">'+html.escape(label or Path(path).name)+'</a></p>'


def summary(records,columns):
    return '<table border="1"><tr>'+''.join('<th>'+label+'</th>' for _,label in columns)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(r.get(key,'')))+'</td>' for key,_ in columns)+'</tr>' for r in records)+'</table>'


def page(title,scope,method,body,*,context='',files=()):
    return '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'''+html.escape(title)+'''</title><style>body{font:16px/1.5 system-ui,sans-serif;margin:24px;color:#172c3b}h1{font-size:1.7rem}table{border-collapse:collapse;margin:14px 0}td,th{padding:8px;vertical-align:top;text-align:left;border:1px solid #aaa}th{background:#f4f4f4}audio{width:240px;max-width:100%}.case-scroll{overflow-x:auto}img{max-width:100%;height:auto}td img{min-width:300px}code{overflow-wrap:anywhere}pre{white-space:pre-wrap}details{margin:16px 0}summary{cursor:pointer}select{font:inherit;padding:6px}article{border-top:1px solid #aaa} [hidden]{display:none!important}</style></head><body><h1>'''+html.escape(title)+'''</h1><p><strong>Standalone experiment report.</strong> Audio, images, data records and interpretation are embedded. No network, server, model checkpoint or companion files are needed.</p><h2>Test scope and limits</h2><p>'''+html.escape(scope)+'''</p><h2>Method and signal path</h2><p>'''+html.escape(method)+'''</p><h2>Metric and rate definitions</h2><p>All audio is mono at 16 kHz. Playback WAVs use float32 PCM: 512 kbit/s, excluding WAV headers. Native PCM16 is 256 kbit/s. These playback rates are not compressed representation rates. Waveform NMSE is squared error divided by source energy, calculated on the exact embedded float32 samples, with no alignment or gain fitting. Lower is better. PESQ-WB and STOI are displayed only where saved full-recording evaluations exist; no perceptual score is assigned to 40 ms blocks or individual IMFs.</p><p>Waveforms use a common amplitude scale within each case. Spectrograms use a Hann STFT with 128-sample windows and 16-sample hops for 40 ms signals (8 ms / 1 ms), or 512/128 samples for full recordings. FFT length equals the window; no boundary extension. All spectrograms use -100 to 0 dB relative to amplitude 1 and 0 to 8 kHz. Short windows limit frequency resolution.</p><p>Serialized rates include stored representation headers when available. JSON file rates are debugging/storage rates, not optimized codec bitrates. Scalar counts and reference formula-description bits are separate from serialized rates. Shared neural decoder weights are additional storage and are not counted in per-block code files. A missing encoded rate is shown as Not available.</p>'''+body+('<details><summary>Source recording, crop location, VMD modes and residual</summary>'+context+'</details>' if context else '')+'<h2>Evidence and provenance</h2>'+''.join(evidence(p,label) for p,label in files)+'''<p>All results come from existing runs. Plotting and evaluating saved formulas does not retrain a model, optimize constants, or change test selection. Complete numerical records are included even where only saved listening examples are illustrated.</p><script>function selectCase(s){s.parentElement.parentElement.querySelectorAll(':scope > .signal-case').forEach((p,i)=>{p.hidden=i!==Number(s.value);if(p.hidden)p.querySelectorAll('audio').forEach(a=>a.pause())})}</script></body></html>'''


def build():
    WORK.mkdir(parents=True,exist_ok=True)
    ctx,manifest=archive.contexts()
    docs=[]
    def save(key,title,scope,method,body,files=()):
        p=WORK/(key+'.html');p.write_text(page(title,scope,method,body,context=ctx.get(key,''),files=files),encoding='utf-8');docs.append((key,title,p))
    root=RUNS/'parametric_librispeech_1272_128104_0000'
    cfg=read(root/'settings.json');ref=wav(root/'reference.wav');base=rows(root/'metrics.csv')
    outs=[]
    for method in ('vmd','emd'):
        r=next(r for r in base if r['method']==method and float(r['window_ms'])==40 and r['reconstruction']=='modes')
        outs.append(dict(label='After: VMD sampled modes' if method=='vmd' else 'Control: EMD sampled modes',x=wav(root/f'{method}/window_640_samples/modes.wav'),pesq_wb=r.get('pesq_wb'),stoi=r.get('stoi'),saved_nmse=r['nmse']))
    body='<h2>Summary</h2>'+summary(base,[('method','Method'),('window_ms','Window ms'),('reconstruction','Reconstruction'),('nmse','Saved NMSE'),('pesq_wb','Saved PESQ-WB'),('stoi','Saved STOI')])
    body+=table('Recording 1272-128104-0000: 40 ms analysis windows',ref,outs,'dev-clean; speaker 1272; complete 5.855 s / 93,680 samples. Source recording SHA-256: '+cfg['source_sha256'],'The comparison is on the whole recording, not an isolated 40 ms fragment. Saved full-utterance perceptual scores are retained. Sampled residual controls in the full records restore omitted detail; sampled arrays do not establish compression.')
    save('baseline','1. Sampled EMD/VMD reconstruction','Eight method/window cases on one recording; modes and modes-plus-residual controls. No trainable parameter representation or bitrate experiment.','Recording → overlapping 20/40/80/160 ms windows → EMD or fixed-eight-mode VMD → sampled modes → normalized overlap-add. Fifty percent overlap. A separate sampled residual control is retained.',body,[(root/'metrics.csv','Complete measured baseline results'),(root/'settings.json','Source identity and analysis settings')])
    print('Built baseline',flush=True)
    root=RUNS/'parametric_ablation_librispeech_1272_v2';multi=rows(root/'metrics.csv')
    candidates=[r for r in multi if r['method']=='vmd' and r['status']=='ok'];r=candidates[0]
    binary=root/r['payload'];assert binary.stat().st_size==int(r['payload_bytes'])
    output=dict(label='After: first saved VMD fit',x=wav(root/r['audio']),values=int(r['parameters']),encoded_kbps=binary.stat().st_size*8/(len(ref)/SR)/1000,rate_basis='Measured .par bytes / full recording duration; includes metadata',saved_nmse=r['nmse'],pesq_wb=r['pesq_wb'],stoi=r['stoi'])
    body='<h2>Saved result summary</h2><p>Stopped at 452/744 jobs. Results are incomplete and cannot support a balanced ranking. The table below is the first saved successful VMD job in file order, not a best-quality selection.</p>'
    body+=table('First completed VMD representation example',ref,[output],f"Full dev-clean recording; {r['window_ms']} ms windows; {r['family']}; budget {r['budget']} values/s; {r['bits']}-bit coefficients.",'Low per-mode allocations and inadequate fitting were found in the audit. This case must not be interpreted as the best achievable VMD or polynomial representation. All completed rows, budgets, costs, and scores remain in the embedded CSV.')
    save('multi','2. Multi-window parametric fitting (incomplete)','452 of 744 planned jobs completed on one full utterance; no selected completed job is dropped from the embedded numeric record.','Recording → overlapping windows → decomposition or direct fitting → fitted function coefficients → quantized .par payload → decoded components → overlap-add.',body,[(root/'metrics.csv','All completed sweep rows and measured rates'),(root/'manifest.json','Sweep settings and source identity'),(binary,'Exact saved example payload')])
    root=RUNS/'parametric_isolated_40ms_librispeech';cfg=read(root/'manifest.json');records=rows(root/'metrics.csv');body='<h2>Summary</h2><p>972/972 jobs completed: three independent blocks, five decomposers plus direct fitting, and 53 candidate settings. The comparisons below select the lowest saved NMSE within 128 total values for each arm and block. This retrospective selection is explicit; it does not establish generalization or a global optimum.</p>'
    for i,start in enumerate(cfg['starts']):
        with np.load(root/f'cache/case{i}_vmd.npz') as z:x=z['reference'].copy()
        outputs=[]
        for method in ('vmd','direct','svmd'):
            r=min((r for r in records if int(r['case'])==i and r['method']==method and r['status']=='ok' and int(r['parameter_count'])<=128),key=lambda r:float(r['nmse']))
            payload=root/Path(r['audio']).with_suffix('.json');assert payload.stat().st_size==int(r['payload_bytes'])
            outputs.append(dict(label=('After: VMD fit' if method=='vmd' else 'Control: '+method.upper())+' / '+r['family'],x=wav(root/r['audio']),values=int(r['parameter_count']),encoded_kbps=payload.stat().st_size*8/.04/1000,rate_basis='Measured float64 JSON model file / 0.04 s; storage rate only',saved_nmse=r['playback_nmse'] or r['nmse']))
        body+=table(f'Case {i+1}: independent 40 ms speech block',x,outputs,f'Utterance 1272-128104-0000; speaker 1272; dev-clean; samples [{start},{start+640}), {start/SR:.2f}–{(start+640)/SR:.2f} s. Exactly 640 samples.','Before is the original block. After reconstructs all VMD components plus residual from fitted functions. Controls fit the same block directly or use SVMD. No overlap, stitching, quantization or RVQ. Serialized JSON contains model definitions as well as numbers; its byte rate is not an optimized codec result.')
    save('isolated','3. Isolated 40 ms curve fitting','All three blocks are illustrated. They come from one utterance, not a corpus-wide test. All 972 result rows are embedded.','Exact 640-sample block → eight VMD modes + residual → independent function fitting → sum of decoded components. Direct and SVMD controls use the same total-value cap.',body,[(root/'metrics.csv','All 972 jobs, selection costs and errors'),(root/'manifest.json','Frozen block offsets and settings'),(root/'selections.csv','Conservative bound-based selections')])
    print('Built isolated comparisons',flush=True)
    root=RUNS/'parametric_learned_representation_pilot';data=read(root/'data_manifest.json');test=[r for r in data if r['role']=='test'];metrics=rows(root/'metrics.csv');body='<h2>Held-out summary</h2>'+summary([r for r in metrics if r['role']=='test' and int(r['latent_dimension'])==64],[('method','Domain'),('activation','Model'),('parameters_per_block','Values/block'),('mean_nmse','Mean NMSE'),('blocks','Test blocks')])
    for domain in ('direct','vmd'):
        for i in (0,1):
            row=test[i];x=wav(root/f'examples/{domain}_pca_test_{i}_reference.wav');outputs=[]
            for model in ('sine','snakebeta','pca'):
                stem=root/f'examples/{domain}_{model}_test_{i}_z64';payload=read(stem.with_suffix('.json'))
                count=np.asarray(payload['codes']).size+len(payload['means'])+len(payload['gains'])
                outputs.append(dict(label=('Control: PCA' if model=='pca' else 'After: '+model),x=wav(Path(str(stem)+'_reconstruction.wav')),values=int(count),encoded_kbps=stem.with_suffix('.json').stat().st_size*8/.04/1000,rate_basis='Measured code JSON storage only; shared decoder weights excluded'))
            body+=table(f'{domain.upper()} domain: saved test example {i+1}, code size 64/component',x,outputs,f"{row['utterance']}; speaker {row['speaker']}; {row['split']}; samples [{row['start']},{row['start']+640}); {row['start']/SR:.3f}–{(row['start']+640)/SR:.3f} s.",'These are the first two saved examples by test-data order, not the entire 256-block evaluation. Direct uses 64 codes plus mean/gain; VMD uses nine components, each with 64 codes plus mean/gain. Their whole-block scalar budgets differ. Complete aggregate results across all budgets are embedded.')
    save('shared','4. Learned shared representations','4,096 training, 256 validation, 256 test blocks with separated speakers. Four case tables show two saved test examples in each of two domains at code size 64. Aggregate records cover the full test set and all five code sizes.','Direct waveform OR fixed-eight-mode VMD + residual → shared PCA / sine / SnakeBeta representation → stored codes, means and gains → shared decoder → reconstructed block. This pilot does not produce explicit formulas.',body,[(root/'metrics.csv','All model / budget / partition results'),(root/'manifest.json','Training protocol'),(root/'data_manifest.json','Complete recording and crop manifest')])
    print('Built shared pilot comparisons',flush=True)
    root=RUNS/'parametric_symbolic_imf_pilot';data=read(root/'data_manifest.json')['records'];test=[r for r in data if r['role']=='test'];stats=read(root/'summary.json');body='<h2>Mean held-out result</h2>'+summary([dict(arm=a,nmse=v['best_any_size_mean_nmse'],count=v['evaluated_imfs'],success=v['nmse_thresholds']['0.01']['successful_imfs']) for a,v in stats['arms'].items()],[('arm','Arm'),('nmse','Mean best NMSE'),('count','IMFs'),('success','NMSE ≤ 0.01')])
    with np.load(RUNS/'parametric_learned_representation_pilot/cache/vmd_test.npz') as z:components=z['components'].copy()
    for n,row in enumerate(test):
        x=components[row['cache_block'],row['mode']];outputs=[]
        for arm in ('adapted','pretrained','random'):
            job=read(root/f"jobs/{arm}_{row['item']}.json");best=min(job['records']+job['frontier'],key=lambda r:r['nmse'])
            y=best['gain']*evaluate(best['tokens'],best['constants'],np.linspace(-1,1,640))
            outputs.append(dict(label=('After: adapted' if arm=='adapted' else 'Control: '+arm),x=y,values=best['parameters'],saved_nmse=best['nmse'],rate_basis=f"Reference description: {best['description_bits']} bits / IMF; not a serialized codec",formula=best['formula'],description_bits=best['description_bits']))
        body+=table(f"IMF case {n+1}/64: {row['utterance']}, block {row['cache_block']}, mode {row['mode']+1}/8",x,outputs,f"{row['split']}; speaker {row['speaker']}; parent crop [{row['start']},{row['start']+640}) at {row['start']/SR:.3f} s. The signal here is one VMD IMF, not complete speech.",'The lowest saved error among the fixed search candidates and refinements is shown for each arm; no fitting was rerun. The original IMF is compared with saved explicit formulas. The residual was excluded and no summed full-block expression reconstruction was evaluated. The displayed formulas are evaluated at normalized t in [-1,1].')
        if (n+1)%8==0:print(f'Built symbolic cases {n+1}/64',flush=True)
    save('symbolic','5. Explicit IMF expression discovery','All 64 held-out IMFs from eight 40 ms blocks are included, with random, pretrained, and adapted comparisons. These IMFs are nested within eight blocks and are not 64 independent speakers.','Speech block → fixed VMD → one of eight IMF targets → expression generator / random proposals → saved fitted constants → explicit formula waveform. Residual and full-speech synthesis are outside this evaluation.',body,[(root/'summary.json','Complete held-out summary and success counts'),(root/'data_manifest.json','Source blocks, mode IDs and exclusions'),(root/'manifest.json','Training and search protocol')])
    old=(OUT/'experiment_results_2026_11_04/index.html').read_text(encoding='utf-8');audit=re.search(r'<section id="audit">(.*?)</section>',old,re.S).group(1)
    save('audit','Fitting and quantization audit','Controlled known-function checks; not an additional held-out speech test.','Known mathematical target → declared fitting budget → fitted coefficients; separately inspect finite-precision storage.',audit,[(RUNS/'parametric_implementation_audit/audit.json','All controlled fitting checks'),(RUNS/'parametric_implementation_audit/quantization_check.json','Coefficient and metadata byte accounting')])
    ledger=read(OUT/'experiments.json')
    overview='<h2>Completed work and report coverage</h2>'+summary(ledger,[('id','Experiment'),('status','Saved status'),('jobs_completed','Completed units'),('jobs_total','Planned units')])+'<p>The sampled EMD/VMD baseline is also complete (eight method/window cases). Select an experiment in the menu. Each page uses the same structure: scope, signal path, definitions, summary, Before/After/Control case tables, recording/VMD details, evidence. The complete numerical records are embedded; listening coverage is stated separately.</p><p>Current evidence does not establish reliable general speech compression, a globally minimal representation, or perceptual superiority. The learned pilot and symbolic pilot answer different questions.</p>'
    save('overview','Parametric experiment test reports','Five experiments plus a controlled fitting audit. Each experiment uses its own evaluation units and limits; results are not pooled.','The menu loads one self-contained experiment page at a time. Case audio, waveforms and spectrograms appear together, as in the PitchFlower report.',overview,[(OUT/'experiments.json','Saved experiment ledger')])
    output=OUT/'parametric_experiments_testcases.html';shell=(archive.METHOD_ROOT/'tabbed_report_shell.html').read_text(encoding='utf-8')
    shell='\n'.join(line for line in shell.splitlines() if not line.strip().startswith("if(item.id!=='overview'"))
    docs=sorted(docs,key=lambda r:['overview','baseline','multi','audit','isolated','shared','symbolic'].index(r[0]))
    items=[]
    for key,title,p in docs:
        packed=WORK/(key+'.gz')
        with p.open('rb') as source,gzip.open(packed,'wb',compresslevel=6) as zipped:shutil.copyfileobj(source,zipped)
        items.append(dict(id=key,title=title,source=p.relative_to(ROOT).as_posix(),sha256=archive.sha(p),bytes=p.stat().st_size,packed=packed))
    shell=shell.replace('<!-- MENU -->',''.join(f'<a role="tab" id="tab-{r["id"]}" aria-controls="viewer-panel" aria-selected="false" href="#{r["id"]}">{html.escape(r["title"])}</a>' for r in items))
    shell=shell.replace('/* METADATA */',json.dumps([{k:v for k,v in r.items() if k!='packed'} for r in items]))
    shell=shell.replace('Original experiment results are preserved.','PitchFlower-style case tables. Saved experiment results are preserved.').replace('complete saved report','case report')
    shell=shell.replace('One offline file. Select an experiment to load its report.','One offline file. Select an experiment. Native audio counters can show 0:00 for 40 ms clips.')
    with output.open('w',encoding='utf-8',newline='\n') as stream:
        stream.write(shell)
        for r in items:
            with r['packed'].open('rb') as source:
                for i,chunk in enumerate(iter(lambda:source.read(3*65536),b'')):
                    stream.write(f'<script type="application/octet-stream" data-report="{r["id"]}" id="packed-{r["id"]}-{i}">'+base64.b64encode(chunk).decode()+'</script>\n')
        stream.write('<script>initReport();</script></body></html>')
    verification=dict(file=output.name,bytes=output.stat().st_size,sha256=archive.sha(output),style='PitchFlower-style case tables with experiment tabs',case_tables=len(RECORDS),cases=RECORDS,source_hashes=SOURCES,reports=[{k:v for k,v in r.items() if k!='packed'} for r in items],original_reports_modified=False)
    (OUT/'parametric_experiments_testcases_verification.json').write_text(json.dumps(verification,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'output':str(output),'MB':output.stat().st_size/1e6,'case_tables':len(RECORDS)},indent=2),flush=True)


if __name__=='__main__':build()
