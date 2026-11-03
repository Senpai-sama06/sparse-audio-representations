"""Offline audio/plot reports for the isolated-block study, with explicit limits."""
import html
import json
from pathlib import Path
import numpy as np
from .portable_report import export_html
from .ablation_report import write_table


def write_report(root,rows,selections):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    write_table(root/'metrics.csv',rows);write_table(root/'selections.csv',selections)
    successful=[r for r in rows if r.get('status')=='ok' and r.get('nmse') is not None]
    fig,ax=plt.subplots(figsize=(10,5))
    for family in sorted({r['family'] for r in successful}):
        group=[r for r in successful if r['family']==family]
        ax.scatter([r['parameter_count'] for r in group],[max(r['nmse'],1e-16) for r in group],label=family,alpha=.6,s=16)
    ax.set(xlabel='Total mathematical parameters for the 640-sample block',ylabel='Whole-block NMSE',yscale='log')
    ax.grid(alpha=.2)
    if successful:ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(root/'complexity_distortion.png',dpi=140);plt.close(fig)
    # Component curves and best-achieved waveform overlays separate basis
    # adequacy from whole-block cancellation and fitting failures.
    keys=sorted({(r['case'],r['method']) for r in successful});images=[]
    for case,method in keys:
        mode_file=root/'cache'/f'case{case}_{method}.npz'
        if not mode_file.exists():continue
        with np.load(mode_file) as data:components=data['components'];x=data['reference']
        fig,axes=plt.subplots(len(components),2,figsize=(12,max(3,2.4*len(components))),squeeze=False)
        for mi,component in enumerate(components):
            by_family={};best=None
            for result in (root/'jobs').glob(f'case{case}_{method}_*.json'):
                record=json.loads(result.read_text())
                if record.get('status')!='ok' or 'fits' not in record:continue
                c=record['fits'][mi]; family=c['model']['spec']['family']
                if c['nmse'] is not None:
                    by_family.setdefault(family,[]).append((c['parameter_count'],c['nmse']))
                    if best is None or c['error_energy']<best['error_energy']:best=c
            for family,points in by_family.items():
                points=sorted(points);axes[mi,0].plot([p[0] for p in points],[max(p[1],1e-16) for p in points],'.-',label=family)
            axes[mi,0].set(yscale='log',title=f'Component {mi}: error versus count',xlabel='Parameters',ylabel='NMSE');axes[mi,0].grid(alpha=.2)
            from .isolated_models import decode
            axes[mi,1].plot(component,label='sampled component',color='black',linewidth=1)
            if best:axes[mi,1].plot(decode(best['model'],len(x)),label='best available fit',alpha=.7)
            axes[mi,1].set(title=f'Component {mi}: best available overlay',xlabel='Sample')
        axes[0,0].legend(fontsize=6);axes[0,1].legend(fontsize=7)
        fig.tight_layout();name=f'case{case}_{method}_components.png';fig.savefig(root/name,dpi=100);plt.close(fig);images.append(name)
    # Convergence histories are retained in every result JSON. Plot one
    # representative nonlinear candidate per family, choosing highest count.
    fig,ax=plt.subplots(figsize=(10,5));histories={}
    for r in successful:
        if r['family'] not in ('sinusoid','amfm','sine_network','snakebeta_network'):continue
        key=r['family']
        if key not in histories or r['parameter_count']>histories[key]['parameter_count']:histories[key]=r
    for family,r in histories.items():
        record=json.loads((root/r['job_file']).read_text());fits=record['fits']
        chosen=max(fits,key=lambda c:c['error_energy'])
        for ri in sorted({h['restart'] for h in chosen['history']}):
            h=[v for v in chosen['history'] if v['restart']==ri]
            ax.plot([v['step'] for v in h],[max(v['nmse'],1e-16) for v in h],label=f'{family}, start {ri}',alpha=.7)
    ax.set(yscale='log',xlabel='Objective evaluations / Adam steps (see family)',ylabel='Component NMSE',title='Example fitting histories; best available candidates, not global certificates')
    if histories:ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(root/'convergence.png',dpi=140);plt.close(fig)
    columns=['case','method','family','complexity','parameter_count','parameter_sample_ratio','nmse','spectral_nmse','edge_nmse','interior_nmse','evidence','seconds','status']
    body=[]
    for r in rows:
        cells=''.join('<td>'+html.escape(str(r.get(k,'')))+'</td>' for k in columns)
        audio=f'<audio controls preload="none" src="{html.escape(r["audio"],quote=True)}"></audio>' if r.get('audio') else ''
        body.append('<tr>'+cells+'<td>'+audio+'</td></tr>')
    source=root/'index.html'
    refs=''.join(f'<p>Case {ci}: original 40 ms <audio controls preload="none" src="reference_{ci}.wav"></audio></p>' for ci in sorted({r['case'] for r in rows}))
    selection_text=html.escape(json.dumps(selections,indent=2))
    source.write_text('''<!doctype html><meta charset="utf-8"><title>Isolated 40 ms representation study</title>
<style>body{font:14px system-ui;margin:2rem}td,th{padding:.4rem;border:1px solid #ddd}table{border-collapse:collapse}img{max-width:1100px;width:100%}audio{width:190px}pre{white-space:pre-wrap}</style>
<h1>Isolated 40 ms representation study</h1>
<p>Each original clip is exactly 640 mono samples at 16 kHz. No overlap, no frame stitching, no cross-fade, no quantization, no RVQ.
PESQ/STOI are not computed on these fragments; plots show waveform/spectral distortion, not validated perceptual quality.
All decomposed components, including the residual, are encoded. Orders come from a declared finite library; zero crossings and extrema are diagnostics only.
Nonlinear results are best found across multiple starts. A local convergence flag is not a global optimum certificate.</p>
<p>Linear certificates apply only to the stated sampled linear space at reported numerical rank. The DCT baseline is a fixed low-frequency prefix, including a full 640-coefficient identity control.
Float64 JSON payload bytes are debugging/storage costs, not an optimized bitrate. Parameter counts exclude fixed model definitions and include every fitted coefficient/weight.
Fits and main errors use float64 decoded parameters. Listening WAVs use float32; playback_nmse in the CSV records their separately measured error.</p>
<p><a href="metrics.csv">Metrics CSV</a> | <a href="manifest.json">Frozen settings</a> | <a href="selections.csv">Threshold selections</a> | <a href="progress.json">Progress snapshot</a></p>
'''+refs+'<img src="complexity_distortion.png"><img src="convergence.png">'+''.join(f'<img src="{name}">' for name in images)+
                      '<h2>Finite-library minimum-count selections under a conservative error bound</h2><p>Dynamic programming minimizes count under the sum-of-component-error-norm bound. It does not certify minimum count for actual waveform distortion, or over all possible functions. Selection uses available fits only; incomplete groups are labeled.</p><pre>'+selection_text+'</pre><h2>Measured block reconstructions</h2><table><tr>'+''.join('<th>'+c+'</th>' for c in columns)+'<th>Audio</th></tr>'+''.join(body)+'</table>',encoding='utf-8')
    export_html(source,root/'isolated_40ms_shareable.html',notice='Share this single file: all audio, plots and linked tables/settings are embedded. Load audio on demand in current Chrome/Edge. This report is an incremental snapshot until progress status is complete.')
