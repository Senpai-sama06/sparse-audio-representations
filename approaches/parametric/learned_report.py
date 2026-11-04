"""Incremental learned-representation results, with offline audio and plots."""
import html
import json
from .ablation_report import write_table
from .portable_report import export_html


def report(root,history,training,assets):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    write_table(root/'metrics.csv',history);write_table(root/'training.csv',training)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    labels=sorted({(r['method'],r['activation']) for r in history})
    for method,activation in labels:
        group=[r for r in history if r['method']==method and r['activation']==activation and r['role']=='validation']
        dimensions=sorted({r['latent_dimension'] for r in group})
        shown=16 if 16 in dimensions else (dimensions[len(dimensions)//2] if dimensions else None)
        for dimension in dimensions:
            g=[r for r in group if r['latent_dimension']==dimension]
            if dimension==shown:axes[0].plot([r['step'] for r in g],[max(r['mean_nmse'],1e-12) for r in g],'.-',label=f'{method}/{activation}, z={dimension}')
    latest={}
    for r in history:
        key=(r['method'],r['activation'],r['role'],r['latent_dimension']);latest[key]=r
    for method,activation in labels:
        group=[r for r in latest.values() if r['method']==method and r['activation']==activation]
        test=[r for r in group if r['role']=='test'];group=test or [r for r in group if r['role']=='validation']
        group=sorted(group,key=lambda r:r['parameters_per_block'])
        axes[1].plot([r['parameters_per_block'] for r in group],[max(r['mean_nmse'],1e-12) for r in group],'.-',label=f'{method}/{activation} ({"test" if test else "val"})')
    axes[0].set(xlabel='Optimization step',ylabel='Validation mean block NMSE',yscale='log')
    axes[1].set(xlabel='Transmitted scalars per 640-sample block (codes + means/gains)',ylabel='Mean block NMSE',yscale='log')
    for ax in axes:
        ax.grid(alpha=.2)
        if ax.get_legend_handles_labels()[0]:ax.legend(fontsize=6)
    fig.tight_layout();fig.savefig(root/'learning_curves.png',dpi=140);plt.close(fig)
    columns=['method','activation','role','step','latent_dimension','parameters_per_block','parameter_sample_ratio','mean_nmse','median_nmse','spectral_nmse','components','silent_blocks','blocks']
    table=''.join('<tr>'+''.join('<td>'+html.escape(str(r.get(k,'')))+'</td>' for k in columns)+('</tr>') for r in latest.values())
    examples=[]
    for path in sorted((root/'examples').glob('*_reconstruction.wav')):
        original=path.name.rsplit('_z',1)[0]+'_reference.wav'
        examples.append(f'<p>{html.escape(path.stem)}<br>Reference <audio controls preload="none" src="examples/{original}"></audio> Reconstructed <audio controls preload="none" src="examples/{path.name}"></audio></p>')
    content='''<!doctype html><meta charset="utf-8"><title>Learned mode representation pilot</title>
<style>body{font:14px system-ui;margin:2rem}td,th{padding:.4rem;border:1px solid #ddd}table{border-collapse:collapse}img{max-width:1100px;width:100%}audio{width:220px}pre{white-space:pre-wrap}</style>
<h1>Learned mode representation pilot — experiment 1</h1>
<p>First learned-representation arm: PCA basis versus shared sine/SnakeBeta conditional function decoders. Decoder weights are shared assets, not transmitted anew per mode. Means and RMS gains are two transmitted scalars per component. Latent prefixes are explicit budgets; no quantization, RVQ, stitching or learned decomposition.</p>
<p>Speaker-disjoint local LibriSpeech train/validation/test sets, frozen before fitting. Test is evaluated only after selecting checkpoints by mean validation block NMSE across budgets. Training uses normalized component waveform MSE with a small code-norm regularizer. This pilot is not the complete symbolic-function search or proof of the best possible representation.</p>
<p>Both neural decoders have identical trainable time-frequency features, FiLM conditioning and capacity; their hidden activations differ. Prefix masking trains budgets jointly. The uncentered PCA/SVD basis is the optimal training-set through-origin linear subspace for normalized components, not necessarily optimal for held-out block error. Actual per-block counts include all modes and the residual. Float32 JSON payload size and shared decoder storage are reported separately; scalar counts do not establish a bitrate.</p>
<p>40 ms distortion metrics are waveform/spectral diagnostics, not validated perceptual speech scores. No PESQ/STOI is computed. Curves use validation until a model finishes, then test points are explicitly labeled.</p>
<p><a href="manifest.json">Frozen settings</a> | <a href="data_manifest.json">Speaker/source ledger</a> | <a href="metrics.csv">Metrics</a> | <a href="training.csv">Training losses</a> | <a href="progress.json">Progress snapshot</a></p>
<img src="learning_curves.png"><h2>Latest results per model, partition and budget</h2>'''
    content+='<table><tr>'+''.join('<th>'+c+'</th>' for c in columns)+'</tr>'+table+'</table><h2>Shared decoder assets</h2><pre>'+html.escape(json.dumps(assets,indent=2))+'</pre><h2>Independent 40 ms listening examples</h2>'+''.join(examples)
    (root/'index.html').write_text(content,encoding='utf-8')
    export_html(root/'index.html',root/'learned_representation_shareable.html',notice='Incremental training report. This single HTML embeds figures, example audio and linked settings/tables. Click Load embedded audio in current Chrome/Edge. Shared model assets and checkpoints are stored separately in the run directory.')
