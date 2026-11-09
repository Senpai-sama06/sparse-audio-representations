"""Standalone evidence: generated formulas, fit diagnostics, plots and audio."""
import html
from pathlib import Path
import numpy as np
import soundfile as sf
from ..ablation_report import write_table
from ..portable_report import export_html
from ..progress import atomic_json
from .expressions import evaluate


def report(root,rows,training,summary,examples):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(root);write_table(root/'metrics.csv',rows);write_table(root/'training.csv',training)
    atomic_json(root/'summary.json',summary)
    fig,axes=plt.subplots(1,3,figsize=(16,4))
    for arm in sorted({r['arm'] for r in rows}):
        group=[r for r in rows if r['arm']==arm and r['on_frontier']]
        axes[0].scatter([r['parameters'] for r in group],[max(r['nmse'],1e-12) for r in group],alpha=.3,s=10,label=arm)
    axes[0].set(xlabel='Fitted constants, including output gain',ylabel='Individual IMF NMSE',yscale='log')
    if rows:axes[0].legend()
    pre=[r for r in training if r['stage']=='synthetic_pretraining']
    if pre:axes[1].plot([r['step'] for r in pre],[r['loss'] for r in pre]);axes[1].set(xlabel='Synthetic training step',ylabel='Expression token loss')
    for stage,label in [('speech_adaptation','Training fitted expressions'),('validation','Validation searches')]:
        group=[r for r in training if r['stage']==stage]
        if group:axes[2].plot([r['step'] for r in group],[r['mean_reward'] for r in group],label=label)
    axes[2].set(xlabel='Adaptation update',ylabel='Accuracy/description reward (higher is better)')
    if any(r['stage']!='synthetic_pretraining' for r in training):axes[2].legend(fontsize=7)
    for ax in axes:ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(root/'discovery.png');plt.close(fig)
    body=[]
    for name,y,records in examples:
        t=np.linspace(-1,1,len(y));folder=root/'examples';folder.mkdir(exist_ok=True)
        # Final per-job examples are immutable. Redrawing every previous plot
        # at every progress update would make reporting dominate search time.
        fragment=folder/f'{name}.fragment.html'
        if fragment.exists():body.append(fragment.read_text(encoding='utf-8'));continue
        reference=folder/f'{name}_reference.wav';sf.write(reference,y,16000,subtype='FLOAT')
        fig,axes=plt.subplots(2,1,figsize=(10,5),sharex=True)
        axes[0].plot(t,y,color='black',lw=1,label='VMD IMF')
        cells=[]
        for i,r in enumerate(records):
            predicted=r['gain']*evaluate(r['tokens'],r['constants'],t)
            sf.write(folder/f'{name}_{i}.wav',predicted,16000,subtype='FLOAT')
            axes[0].plot(t,predicted,lw=.8,label=f"{r['arm']}, {r['parameters']} values, NMSE {r['nmse']:.3g}")
            axes[1].plot(t,y-predicted,lw=.8)
            payload={k:r[k] for k in ('tokens','constants','gain','nmse','parameters','description_bits','formula')}
            payload.update(samples=len(y),sample_rate=16000,time_grid='linspace(-1,1,samples)',format='symbolic-imf-v1')
            atomic_json(folder/f'{name}_{i}.json',payload)
            cells.append(f'<p><b>{html.escape(r["arm"])}</b>: {r["parameters"]} values, {r["nodes"]} nodes, NMSE {r["nmse"]:.5g}<br><code>{html.escape(r["formula"])}</code><br><a href="examples/{name}_{i}.json">Expression and constants</a><br><audio controls preload="none" src="examples/{name}_{i}.wav"></audio></p>')
        axes[0].legend(fontsize=7);axes[1].set(xlabel='Normalized time [-1, 1] (40 ms total)',ylabel='Residual');fig.tight_layout()
        fig.savefig(folder/f'{name}.png');plt.close(fig)
        section=f'<details><summary>{html.escape(name)}</summary><p>Original IMF (native amplitude):</p><audio controls preload="none" src="examples/{name}_reference.wav"></audio><img src="examples/{name}.png">'+''.join(cells)+'</details>'
        fragment.write_text(section,encoding='utf-8');body.append(section)
    columns=['arm','item','mode','parameters','nodes','description_bits','nmse','converged','optimality','nfev','formula']
    table=''.join('<tr>'+''.join('<td>'+html.escape(str(r.get(k,'')))+'</td>' for k in columns)+'</tr>' for r in rows if r['on_frontier'])
    document='''<!doctype html><meta charset="utf-8"><title>40 ms VMD IMF expression discovery</title>
<style>body{font:15px system-ui;margin:2rem;max-width:1500px}img{max-width:100%}table{border-collapse:collapse}td,th{border:1px solid #ddd;padding:.4rem}code{overflow-wrap:anywhere}details{margin:1rem 0}</style>
<h1>40 ms VMD IMF expression discovery</h1>
<p>This experiment generates explicit mathematical expressions from elementary operations and fits their numerical constants directly to each IMF. There is no waveform latent decoder, learned VMD, stitching, quantization or RVQ. VMD is fixed; its eight modes are the targets. The residual is not treated as an IMF, and whole-speech reconstruction is outside this pilot.</p>
<p>Arms: random grammar search; synthetically pretrained conditional expression generator; generator adapted on training-split speech IMFs. All use the same proposal count, constant-fitting restarts, node-budget schedule and refinement allowance. Duplicate proposals still consume budget. All held-out IMF samples are available for per-IMF fitting; generator training and checkpoint selection use separate speakers.</p>
<p>Size includes every fitted constant and an explicit output gain. Description bits use a declared fixed-width expression encoding: 16 header bits + 4 per token + 64 per numerical value. Counts are a reference description, not a compressed bitrate measurement. JSON sizes differ. Smallest-parameter and smallest-complete-description frontiers are separate in metrics.csv.</p>
<p>Optimization is nonconvex. Convergence flags are local diagnostics, not proofs of a globally optimal fit or formula. 40 ms waveform/spectral errors do not establish perceptual speech quality. IMF audio uses native amplitude and may be quiet.</p>
<p><a href="manifest.json">Settings and source fingerprint</a> | <a href="data_manifest.json">Frozen source/mode ledger</a> | <a href="metrics.csv">All fitted candidates</a> | <a href="training.csv">Training history</a> | <a href="summary.json">Summary</a> | <a href="known_answers.json">Known-structure fitting audit</a> | <a href="synthetic_discovery.json">Known-function structure discovery</a> | <a href="progress.json">Progress snapshot</a></p>'''
    document+='<pre>'+html.escape(__import__('json').dumps(summary,indent=2))+'</pre><img src="discovery.png">'
    if (root/'synthetic_discovery.json').exists():
        controls=__import__('json').loads((root/'synthetic_discovery.json').read_text())
        document+='<h2>Known-function structure discovery</h2><p>Search sees only sampled curves. Successful known-structure coefficient fitting does not imply successful formula discovery.</p><table><tr><th>Arm</th><th>Function</th><th>Best NMSE</th><th>Values</th><th>Generated formula</th></tr>'
        for control in controls:
            best=control['best'] or {}
            document+='<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in (control['arm'],control['function'],best.get('nmse'),best.get('parameters'),best.get('formula')))+'</tr>'
        document+='</table>'
    document+=''.join(body)
    document+='<h2>Observed error/complete-description frontiers</h2><div style="overflow:auto"><table><tr>'+''.join(f'<th>{k}</th>' for k in columns)+'</tr>'+table+'</table></div>'
    (root/'index.html').write_text(document,encoding='utf-8')
    export_html(root/'index.html',root/'symbolic_imf_shareable.html')
