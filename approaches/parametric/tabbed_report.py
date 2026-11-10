"""Combine frozen reports and actual signal context into one offline tabbed HTML.

No training, decomposition, fitting, or historical report rewrites. Cached modes
are visualized as saved. Original reports are gzip-packed and loaded on demand.
"""
from pathlib import Path
import base64
import csv
import gzip
import hashlib
import html
import io
import json
import re
import shutil
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy.signal import stft

from .portable_report import export_html
from .report_paths import ARCHIVE, METHOD_ROOT, REPORT_DIRECTORIES, REPORT_FILENAMES

ROOT = METHOD_ROOT.parents[1]
RUNS = ROOT/'runs'
WORK = ROOT/'tmp/parametric_tabbed_build'
SR = 16000
EVIDENCE = []
CASES = []


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def data_uri(raw, mime):
    return f'data:{mime};base64,'+base64.b64encode(raw).decode()


def audio(x):
    out = io.BytesIO()
    sf.write(out, np.asarray(x), SR, format='WAV', subtype='FLOAT')
    return '<audio controls preload="none" src="'+data_uri(out.getvalue(), 'audio/wav')+'"></audio>'


def figure(fig):
    out = io.BytesIO()
    fig.savefig(out, format='png', dpi=110, bbox_inches='tight')
    plt.close(fig)
    return '<img loading="lazy" alt="Measured waveform and spectrogram" src="'+data_uri(out.getvalue(), 'image/png')+'">'


def plots(signals, labels, *, full=False, highlight=None):
    """Shared native amplitude axes and absolute spectrogram scale within a panel."""
    fig, axes = plt.subplots(len(signals), 2, figsize=(12, 1.8*len(signals)), squeeze=False)
    peak = max(max(float(np.max(np.abs(x))) for x in signals), 1e-8)*1.08
    window, hop = (512, 128) if full else (128, 16)
    for i, (x, label) in enumerate(zip(signals, labels)):
        times = np.arange(len(x))/SR
        axes[i, 0].plot(times, x, linewidth=.6, color='#194b70')
        axes[i, 0].set(title=label, xlabel='Time in signal (s)' if full else 'Time within block (ms)', ylabel='Native amplitude', ylim=(-peak, peak))
        if not full:
            from matplotlib.ticker import FuncFormatter
            axes[i, 0].xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v*1000:g}'))
        if highlight is not None:
            axes[i, 0].axvspan(highlight/SR, (highlight+640)/SR, color='orange', alpha=.5)
        f,t,z = stft(x, fs=SR, window='hann', nperseg=window, noverlap=window-hop,
                     nfft=window, boundary=None, padded=False, scaling='spectrum')
        db = 20*np.log10(np.maximum(np.abs(z),1e-6))
        axes[i, 1].pcolormesh(t if full else t*1000, f/1000, db, shading='auto', vmin=-100, vmax=0, cmap='magma')
        axes[i, 1].set(title=label+' — STFT', xlabel='Time (s)' if full else 'Time within block (ms)', ylabel='Frequency (kHz)', ylim=(0,8))
        axes[i, 1].set_xlim(0, len(x)/SR*(1 if full else 1000))
    fig.suptitle('Left: waveforms, shared native amplitude scale. Right: STFT magnitude, -100 to 0 dB re amplitude 1.', fontsize=10)
    fig.tight_layout(rect=(0,0,1,.985))
    return figure(fig)


def cached(path):
    EVIDENCE.append({'file':Path(path).relative_to(ROOT).as_posix(), 'sha256':sha(path)})
    return np.load(path, allow_pickle=False)


def panel(label, reference, components, row, source_path, *, reconstruction=None, note=''):
    """Authenticate the recording, identify the crop, and show cached VMD modes."""
    source_path = Path(source_path)
    source, rate = sf.read(source_path, dtype='float64')
    assert rate == SR and source.ndim == 1
    digest = sha(source_path)
    assert digest == row['source_sha256'], source_path
    start = int(row['start'])
    assert len(reference)==640 and np.max(np.abs(source[start:start+640]-reference))<2e-7
    assert components.shape==(9,640)
    err = float(np.sum((components.sum(0)-reference)**2)/max(np.sum(reference**2),1e-30))
    meta = {k:row.get(k) for k in ('role','split','speaker','utterance','start','samples','source_sha256')}
    meta.update(label=label, end_sample_exclusive=start+640, seconds=.04, sample_rate=SR,
                mode_count=8, residual_count=1, cached_sum_nmse=err)
    CASES.append(meta)
    signals = [reference,*components,components.sum(0)]
    labels = ['Original 40 ms crop',*[f'VMD mode {i+1}' for i in range(8)],'Residual (separate from eight modes)','Sum of eight modes + residual']
    if reconstruction is not None:
        assert len(reconstruction)==640
        signals.append(reconstruction);labels.append('Saved fitted VMD reconstruction (at most 128 values)')
    body = '<article><h3>'+html.escape(label)+'</h3><p>'+html.escape(note)+'</p>'
    body += '<p><strong>Recording:</strong> '+html.escape(str(row.get('utterance',source_path.stem)))+'; split '+html.escape(str(row.get('split','dev-clean')))+'; speaker '+html.escape(str(row.get('speaker','1272')))+'. '
    body += f'<strong>Crop:</strong> samples [{start}, {start+640}), {start/SR:.5f}–{(start+640)/SR:.5f} s; exactly 640 samples / 40 ms at 16 kHz, mono. Full recording: {len(source)/SR:.3f} s. Source SHA-256: <code>{digest}</code>.</p>'
    body += '<details><summary>Full recording and crop location (orange)</summary>'+audio(source)+plots([source],['Full source recording'],full=True,highlight=start)+'</details>'
    body += '<p><strong>Signal flow:</strong> original recording → exact 40 ms crop → fixed eight-mode VMD + residual → component representation → summed reconstruction. Cached component-sum NMSE: '+f'{err:.3g}'+'. VMD modes are sampled oscillatory components, not spectrogram bins or learned semantic labels.</p>'
    body += plots(signals,labels)
    body += '<details><summary>Listen to the original, each mode, residual, and summed signal</summary>'
    for x,title in zip(signals,labels):body+='<p>'+html.escape(title)+'</p>'+audio(x)
    return body+'</details></article>'


def select_panels(items):
    options=''.join(f'<option value="{i}">{html.escape(name)}</option>' for i,(name,_) in enumerate(items))
    return '<label>Recording / block <select onchange="selectCase(this)">'+options+'</select></label>'+''.join(f'<div class="signal-case" {"hidden" if i else ""}>{body}</div>' for i,(_,body) in enumerate(items))


def contexts():
    shared=RUNS/'parametric_learned_representation_pilot'
    manifest=read(shared/'data_manifest.json')
    test=[r for r in manifest if r['role']=='test']
    with cached(shared/'cache/vmd_test.npz') as z:
        refs=z['reference'].astype('float64');comps=z['components'].astype('float64')
    iso=RUNS/'parametric_isolated_40ms_librispeech'
    settings=read(iso/'manifest.json')
    source=Path(settings['source_path'])
    with (iso/'metrics.csv').open(newline='',encoding='utf-8') as stream:metrics=list(csv.DictReader(stream))
    iso_items=[]
    for i,start in enumerate(settings['starts']):
        with cached(iso/f'cache/case{i}_vmd.npz') as z:r=z['reference'].copy();c=z['components'].copy()
        row=dict(role='exploratory',split='dev-clean',speaker='1272',utterance='1272-128104-0000',start=start,samples=640,source_sha256=settings['source_sha256'])
        fits=[v for v in metrics if int(v['case'])==i and v['method']=='vmd' and v['status']=='ok' and int(v['parameter_count'])<=128]
        best=min(fits,key=lambda v:float(v['nmse']))
        y,rate=sf.read(iso/best['audio']);assert rate==SR
        iso_items.append((f'Case {i}: {start/SR:.2f}–{(start+640)/SR:.2f} s',panel(f'Isolated case {i}',r,c,row,source,reconstruction=y,note='All three study blocks are included. The illustrated saved VMD fit is the lowest-error successful candidate within 128 total values; this is a retrospective selection, not a new fit.')))
    shared_context=select_panels([('First saved test block',panel('Shared representation: first saved test block',refs[0],comps[0],test[0],test[0]['path'],note='Illustration selected by saved data order, not quality. The pilot evaluated 256 test blocks. Eight modes plus residual make nine encoded components. Models learn shared latent decoders; they do not output explicit formulas.'))])
    sym_rows=read(RUNS/'parametric_symbolic_imf_pilot/data_manifest.json')['records']
    blocks=[r for r in sym_rows if r['role']=='test' and r['mode']==0]
    sym_items=[]
    for row in blocks:
        i=row['cache_block']; assert all(row[k]==test[i][k] for k in ('utterance','start','source_sha256'))
        sym_items.append((f"{row['utterance']} at {row['start']/SR:.2f} s",panel(f"Symbolic test block {i}",refs[i],comps[i],row,row['path'],note='All eight held-out source blocks are included. The symbolic experiment fitted the eight IMFs individually (64 IMF targets); it excluded the residual and did not evaluate a summed full-block expression reconstruction. The residual and sum below explain the decomposition only.')))
    # First complete, unpadded 40-ms frame in the original overlapping baseline.
    base=RUNS/'parametric_librispeech_1272_128104_0000';cfg=read(base/'settings.json')
    original,rate=sf.read(base/'reference.wav',dtype='float64');assert rate==SR
    with cached(base/'vmd/window_640_samples/modes/frame_000001.npz') as z:
        start=int(z['start_sample']);bc=np.vstack((z['modes'],z['residual']))
    row=dict(role='exploratory',split='dev-clean',speaker='1272',utterance='1272-128104-0000',start=start,samples=640,source_sha256=cfg['source_sha256'])
    base_context=select_panels([('First unpadded 40 ms window',panel('Sampled baseline: 40 ms window example',original[start:start+640],bc,row,cfg['source'],note='The full experiment reconstructs the 5.855-second utterance using 20/40/80/160 ms windows and 50% overlap. This panel illustrates only the first complete unpadded 40-ms analysis window. Neighboring windows overlap; this is not the isolated-block protocol.'))])
    old=RUNS/'parametric_ablation_librispeech_1272_v2'
    with cached(old/'cache/case0_vmd_40ms.npz') as z:
        i=next(i for i,s in enumerate(z['starts']) if s>=0 and s+640<=len(original));start=int(z['starts'][i]);mc=np.vstack((z[f'm{i}'],z[f'r{i}']))
    row=dict(row,start=start)
    multi=select_panels([('First complete cached 40 ms window',panel('Stopped multi-window sweep: cached example',original[start:start+640],mc,row,cfg['source'],note='This is one saved 40-ms VMD window from the incomplete sweep. Other windows, decomposers, budgets and precisions were also planned; completed evidence covers only 452/744 jobs. The full frozen report below is retained without recomputing fits.'))])
    clean_manifest=[{k:v for k,v in r.items() if k!='path'} for r in manifest]
    return {'baseline':base_context,'multi':multi,'isolated':select_panels(iso_items),'shared':shared_context,'symbolic':select_panels(sym_items)},clean_manifest


def pack(path):
    """Stream compression avoids loading the 591-MB stopped report into Python RAM."""
    digest=sha(path);target=WORK/(digest+'.gz')
    if not target.exists():
        with Path(path).open('rb') as source,target.open('wb') as sink:
            with gzip.GzipFile(filename='',mode='wb',fileobj=sink,mtime=0,compresslevel=6) as zipped:shutil.copyfileobj(source,zipped,1024*1024)
    return target,digest


def main():
    WORK.mkdir(parents=True,exist_ok=True)
    ctx,manifest=contexts()
    base=WORK/'baseline.html'
    export_html(RUNS/'parametric_librispeech_1272_128104_0000/index.html',base,notice='Frozen sampled-decomposition baseline: whole-utterance scores, not a compressed parameter representation.')
    old_overview=(ARCHIVE/'experiment_results_2026_11_04/index.html').read_text(encoding='utf-8')
    audit=re.search(r'<section id="audit">(.*?)</section>',old_overview,re.S).group(1)
    audit_path=WORK/'audit.html';audit_path.write_text('<!doctype html><meta charset="utf-8"><title>Implementation audit</title>'+audit,encoding='utf-8')
    overview=WORK/'overview.html'
    ledger=read(ARCHIVE/'experiments.json')
    overview.write_text('<!doctype html><meta charset="utf-8"><h1>Parametric experiments</h1><p>Use the menu to load one experiment at a time. Each experiment has signal context above its complete frozen report. This file embeds all reports and media; no network or companion files are required. The stopped sweep is large and may take longer to open.</p><p>The source recordings are LibriSpeech mono 16 kHz speech. A 40 ms block contains 640 samples. VMD means variational mode decomposition; IMF means intrinsic mode function; NMSE is squared error divided by source energy. The sampled baseline uses overlapping windows, while the isolated and learned pilots use independent blocks. The symbolic pilot evaluates individual IMFs.</p><p>These are different evaluation units. Their NMSE numbers must not be pooled. Short-fragment listening does not establish speech intelligibility; no PESQ/STOI is assigned to isolated 40 ms signals.</p><table border="1"><tr><th>Study</th><th>Status</th><th>Completed units</th></tr>'+''.join(f'<tr><td>{html.escape(r["id"])}</td><td>{r["status"]}</td><td>{r["jobs_completed"]}/{r["jobs_total"]}</td></tr>' for r in ledger)+'</table><p>The audit is a separate menu item. Learned VMD, RVQ and reliable general compression have not been demonstrated within this experiment sequence.</p>',encoding='utf-8')
    docs=[('overview','Overview',overview),('baseline','1 · Sampled EMD/VMD',base),('multi','2 · Multi-window (stopped)',REPORT_DIRECTORIES['multi_window_pilot']/REPORT_FILENAMES['multi_window_pilot']),('audit','Fitting audit',audit_path),('isolated','3 · Isolated 40 ms',REPORT_DIRECTORIES['isolated_40ms_capacity']/REPORT_FILENAMES['isolated_40ms_capacity']),('shared','4 · Shared representations',REPORT_DIRECTORIES['learned_representation_pilot']/REPORT_FILENAMES['learned_representation_pilot']),('symbolic','5 · IMF expressions',REPORT_DIRECTORIES['symbolic_imf_discovery']/REPORT_FILENAMES['symbolic_imf_discovery'])]
    items=[]
    for key,title,path in docs:
        print('Packing '+title,flush=True);compressed,digest=pack(path)
        items.append(dict(id=key,title=title,source=str(path.relative_to(ROOT)),sha256=digest,bytes=path.stat().st_size,gzip_bytes=compressed.stat().st_size,path=compressed))
    out=ARCHIVE/'parametric_experiments_tabbed.html';temp=WORK/'report.tmp.html'
    shell=(METHOD_ROOT/'tabbed_report_shell.html').read_text(encoding='utf-8')
    shell=shell.replace('<!-- MENU -->',''.join(f'<a role="tab" id="tab-{r["id"]}" aria-controls="viewer-panel" aria-selected="false" href="#{r["id"]}">{html.escape(r["title"])}</a>' for r in items))
    shell=shell.replace('/* METADATA */',json.dumps([{k:v for k,v in r.items() if k!='path'} for r in items]))
    with temp.open('w',encoding='utf-8',newline='\n') as stream:
        stream.write(shell)
        for key,body in ctx.items():stream.write(f'<template id="context-{key}">{body}</template>\n')
        manifest_url=data_uri(json.dumps(manifest).encode(),'application/json')
        stream.write('<template id="context-overview"><p><a download="source_block_manifest.json" href="'+manifest_url+'">Download all source-block metadata for the shared pilot</a>. This includes all train/validation/test recording IDs, speakers, offsets, lengths, roles and source hashes. Only the first saved shared test block is illustrated; all three isolated and all eight symbolic test blocks are illustrated.</p></template>')
        for r in items:
            with r['path'].open('rb') as packed:
                for i, chunk in enumerate(iter(lambda:packed.read(3*65536),b'')):
                    # Keep individual DOM strings below browser string limits,
                    # even for the 591-MB stopped experiment's frozen HTML.
                    stream.write(f'<script type="application/octet-stream" data-report="{r["id"]}" id="packed-{r["id"]}-{i}">')
                    stream.write(base64.b64encode(chunk).decode())
                    stream.write('</script>\n')
        stream.write('<script>initReport();</script></body></html>')
    shutil.copyfile(temp,out)
    verification=dict(file=out.name,bytes=out.stat().st_size,sha256=sha(out),reports=[{k:v for k,v in r.items() if k!='path'} for r in items],signal_cases=CASES,cache_sources=EVIDENCE,source_manifest_records=len(manifest),plots='Source and cached decomposition; no new fitting or decomposition',standalone=True)
    (ARCHIVE/'parametric_experiments_tabbed_verification.json').write_text(json.dumps(verification,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'report':str(out),'MB':out.stat().st_size/1e6,'tabs':len(items),'signal_cases':len(CASES)},indent=2))


if __name__=='__main__':main()
