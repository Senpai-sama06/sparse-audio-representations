"""Incremental tables, rate-quality plots and Pareto selections from scored files."""
import html
import numpy as np


def write_table(path, rows):
    """Normalize heterogeneous controls/results/failures before shared CSV I/O."""
    from util.io import write_csv
    if rows:
        keys=list(dict.fromkeys(k for r in rows for k in r))
        write_csv(path,[{k:r.get(k) for k in keys} for r in rows])


def write_ablation_report(root, rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    columns = ['case','method','family','window_ms','budget','bits','parameters','payload_bytes',
               'kbps','pesq_wb','stoi','nmse','si_sdr_db','fit_seconds','decode_seconds','status']
    body = []
    for row in rows:
        cells = ''.join('<td>'+html.escape(str(row.get(k,'')))+'</td>' for k in columns)
        audio = html.escape(row.get('audio',''),quote=True)
        body.append(f'<tr>{cells}<td><audio controls preload="none" src="{audio}"></audio></td></tr>')
    ok = [r for r in rows if r.get('status') == 'ok' and r.get('kbps') is not None]
    frontier = []
    for r in ok:
        if r.get('pesq_wb') is None:
            continue
        peers = [s for s in ok if s['case'] == r['case'] and s.get('pesq_wb') is not None]
        if not any(s['kbps'] <= r['kbps'] and s['pesq_wb'] >= r['pesq_wb'] and
                   (s['kbps'] < r['kbps'] or s['pesq_wb'] > r['pesq_wb']) for s in peers):
            frontier.append(r)
    if frontier:
        write_table(root/'pareto.csv',frontier)
    for x,y,label in [('kbps','pesq_wb','PESQ-WB'),('parameters_per_second','pesq_wb','PESQ-WB'),('kbps','stoi','STOI')]:
        fig,ax=plt.subplots(figsize=(9,5))
        for method in sorted({r['method'] for r in ok}):
            group=[r for r in ok if r['method']==method and r.get(y) is not None]
            ax.scatter([r[x] for r in group],[r[y] for r in group],label=method,alpha=.6,s=14)
        ax.set(xlabel=x,ylabel=label); ax.grid(alpha=.25)
        if ok: ax.legend()
        fig.tight_layout(); fig.savefig(root/f'{y}_vs_{x}.png',dpi=140); plt.close(fig)
    # One heatmap per budget/precision/window avoids selecting different budgets
    # inside cells. With multiple testcases, cells are mean available PESQ.
    settings=sorted({(r['budget'],r['bits'],r['window_ms']) for r in ok})
    if settings:
        budget,bits,window=settings[-1]
        selected=[r for r in ok if (r['budget'],r['bits'],r['window_ms'])==(budget,bits,window)]
        methods=sorted({r['method'] for r in selected}); families=sorted({r['family'] for r in selected})
        values=np.full((len(methods),len(families)),np.nan)
        for i,m in enumerate(methods):
            for j,f in enumerate(families):
                scores=[r['pesq_wb'] for r in selected if r['method']==m and r['family']==f and r.get('pesq_wb') is not None]
                if scores: values[i,j]=np.mean(scores)
        fig,ax=plt.subplots(figsize=(10,5)); im=ax.imshow(values,aspect='auto',vmin=1,vmax=4.65)
        ax.set_xticks(range(len(families)),families,rotation=25,ha='right'); ax.set_yticks(range(len(methods)),methods)
        ax.set_title(f'PESQ: budget {budget} params/s, {bits} bit, {window} ms; available cases')
        fig.colorbar(im,ax=ax); fig.tight_layout(); fig.savefig(root/'heatmap.png',dpi=140); plt.close(fig)
    content='''<!doctype html><meta charset="utf-8"><title>Parametric ablation</title>
<style>body{font:14px system-ui;margin:2rem}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:.3rem}audio{width:180px}img{max-width:850px;width:100%}</style>
<h1>Parametric rate-quality ablation</h1><p>Scores use full saved utterances. Payload includes timing and model metadata; parameters includes quantizer scales.
All neural weights are fitted per mode and transmitted. Controls have no compact bitrate. Rankings are exploratory for one testcase.</p>
<p><a href="metrics.csv">All metrics</a> | <a href="pareto.csv">PESQ/rate Pareto frontier per testcase</a> | <a href="progress.json">Progress</a> | <a href="manifest.json">Reproduction settings</a></p>
<img src="pesq_wb_vs_kbps.png"><img src="pesq_wb_vs_parameters_per_second.png"><img src="stoi_vs_kbps.png"><img src="heatmap.png">'''
    header=''.join(f'<th>{k}</th>' for k in columns)+'<th>Listen</th>'
    (root/'index.html').write_text(content+'<table><tr>'+header+'</tr>'+''.join(body)+'</table>',encoding='utf-8')
