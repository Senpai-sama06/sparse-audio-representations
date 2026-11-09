"""Build a results-focused retrospective from saved parametric experiment outputs.

This reads existing runs and writes a new dated report. It neither trains models
nor changes historical metrics, checkpoints or archived listening reports.
"""
from pathlib import Path
from html import escape
import base64
import csv
import hashlib
import json
import os
import shutil
import sys
from collections import Counter
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from approaches.parametric.portable_report import export_html
from approaches.parametric.report_paths import ARCHIVE, REPORT_DIRECTORIES, REPORT_FILENAMES

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RUNS = ROOT / 'runs'
OUT = ARCHIVE / 'experiment_results_2026_11_04'
OUT.mkdir(parents=True, exist_ok=True)
SOURCES = {}


def source(relative):
    path = ARCHIVE / relative.removeprefix("parametric_reports/") if relative.startswith("parametric_reports/") else RUNS / relative
    SOURCES[path.relative_to(ROOT).as_posix()] = {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'bytes': path.stat().st_size}
    return path


def read_json(relative):
    return json.loads(source(relative).read_text(encoding='utf-8-sig'))


def read_csv(relative):
    with source(relative).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(name, records):
    columns = list(dict.fromkeys(k for row in records for k in row))
    with (OUT / name).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, columns)
        writer.writeheader()
        writer.writerows(records)


def num(value, digits=4):
    if value is None or value == '':
        return '-'
    if isinstance(value, (float, np.floating)):
        return f'{value:.{digits}g}'
    return str(value)


def table(rows, columns):
    return '<div class="scroll"><table><thead><tr>' + ''.join('<th>' + escape(label) + '</th>' for key, label in columns) + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join('<td>' + escape(num(row.get(key))) + '</td>' for key, label in columns) + '</tr>' for row in rows) + '</tbody></table></div>'


def save_plot(fig, filename):
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=160, bbox_inches='tight')
    plt.close(fig)
    return f'<figure><img src="{filename}" alt="{filename.replace("_", " ")}"></figure>'


def listen(relative, label):
    src = RUNS / relative
    if not src.is_file():
        raise FileNotFoundError(src)
    dest = OUT / 'audio' / (src.parent.name + '_' + src.name)
    dest.parent.mkdir(exist_ok=True)
    if dest.exists() and dest.read_bytes() != src.read_bytes():
        dest = dest.with_name(hashlib.sha256(str(src).encode()).hexdigest()[:10] + '_' + dest.name)
    shutil.copy2(src, dest)
    return f'<div class="player"><span>{escape(label)}</span><audio controls preload="none" src="audio/{dest.name}"></audio></div>'


BASE = 'parametric_librispeech_1272_128104_0000'
OLD = 'parametric_ablation_librispeech_1272_v2'
ISO = 'parametric_isolated_40ms_librispeech'
SHARED = 'parametric_learned_representation_pilot'
SYM = 'parametric_symbolic_imf_pilot'
AUDIT = 'parametric_implementation_audit'

baseline = read_csv(BASE + '/metrics.csv')
baseline_settings = read_json(BASE + '/settings.json')
baseline_progress = read_json(BASE + '/progress.json')
old = read_csv(OLD + '/metrics.csv')
old_manifest = read_json(OLD + '/manifest.json')
iso = read_csv(ISO + '/metrics.csv')
selections = read_csv(ISO + '/selections.csv')
iso_manifest = read_json(ISO + '/manifest.json')
shared = read_csv(SHARED + '/metrics.csv')
shared_manifest = read_json(SHARED + '/manifest.json')
shared_assets = read_json(SHARED + '/assets.json')
shared_data = read_json(SHARED + '/data_manifest.json')
symbolic_summary = read_json(SYM + '/summary.json')
symbolic_manifest = read_json(SYM + '/manifest.json')
symbolic_data = read_json(SYM + '/data_manifest.json')
symbolic_synthetic = read_json(SYM + '/synthetic_discovery.json')
symbolic_known = read_json(SYM + '/known_answers.json')
symbolic_checkpoint = read_json(SYM + '/selected_checkpoint.json')
read_json(SYM + '/repair_history.json')
audit = read_json(AUDIT + '/audit.json')
quantization = read_json(AUDIT + '/quantization_check.json')
archive = read_json('parametric_reports/experiments.json')

statuses = [{'experiment': 'Sampled EMD/VMD baseline', 'status': baseline_progress['status'], 'completed': '8/8 decomposition cases', 'minutes': baseline_progress['seconds']/60, 'scope': 'One 5.855-second LibriSpeech utterance; sampled modes'}]
labels = {'multi_window_pilot': 'Multi-window parametric pilot', 'isolated_40ms_capacity': 'Isolated 40-ms curve fitting', 'learned_representation_pilot': 'Shared representation pilot (different scope)', 'symbolic_imf_discovery': 'Explicit per-IMF expression discovery'}
for a in archive:
    p = read_json(Path(a['run_directory']).name + '/progress.json')
    statuses.append({'experiment': labels[a['id']], 'status': p['status'], 'completed': f"{p['jobs_completed']}/{p['jobs_total']}", 'minutes': p['elapsed_seconds']/60, 'scope': {'multi_window_pilot':'Full utterance, overlapping windows, quantized coefficients', 'isolated_40ms_capacity':'Three blocks, five decomposers plus direct, 53 candidates each', 'learned_representation_pilot':'4096 train / 256 validation / 256 test blocks', 'symbolic_imf_discovery':'120 eligible train / 32 validation / 64 test IMFs'}[a['id']]})
write_csv('run_status.csv', statuses)

base_rows = [{k: (float(r[k]) if r[k] else None) if k not in ('method','reconstruction') else r[k] for k in ['method','reconstruction','window_ms','nmse','pesq_wb','stoi']} for r in baseline if r['reconstruction']=='modes']
write_csv('sampled_baseline.csv', base_rows)
fig, ax = plt.subplots(figsize=(8,3.6))
for method in ['emd','vmd']:
    group = [r for r in base_rows if r['method']==method]
    ax.plot([r['window_ms'] for r in group], [r['nmse'] for r in group], 'o-', label=method.upper())
ax.set(xlabel='Window length (ms)', ylabel='Whole-utterance NMSE', title='Sampled modes only: one LibriSpeech utterance')
ax.grid(alpha=.25); ax.legend()
base_plot = save_plot(fig, 'sampled_baseline.png')

# Recompute fixed total-budget results from saved successful uniform-family fits.
iso_best=[]
for budget in [16,32,64,128,256,640]:
    for method in iso_manifest['methods']:
        for case in ['0','1','2']:
            candidates=[r for r in iso if r['case']==case and r['method']==method and r['status']=='ok' and r['parameter_count'] and float(r['parameter_count'])<=budget]
            if candidates:
                r=min(candidates,key=lambda x:float(x['nmse']))
                iso_best.append({'budget':budget,'case':int(case),'method':method,'family':r['family'],'parameters':int(r['parameter_count']),'nmse':float(r['nmse']),'edge_nmse':float(r['edge_nmse']),'interior_nmse':float(r['interior_nmse']),'audio':r['audio']})
write_csv('isolated_best_observed_by_budget.csv',iso_best)
fig,ax=plt.subplots(figsize=(9,4.5))
for method in ['direct','emd','vmd','eemd','ceemdan','svmd']:
    points=[]
    for budget in [16,32,64,128,256,640]:
        group=[r for r in iso_best if r['method']==method and r['budget']==budget]
        if len(group)==3:points.append((budget,np.mean([r['nmse'] for r in group])))
    ax.plot([p[0] for p in points],[max(p[1],1e-4) for p in points],'o-',label=method.upper())
ax.set(xlabel='Maximum total fitted scalars per 640-sample block',ylabel='Mean best-observed block NMSE (3 clips)',xscale='log',yscale='log',title='Isolated 40 ms: fixed total budget, uniform-family candidates')
ax.set_ylim(8e-5,1.2)
ax.annotate('Full direct DCT: near machine zero\nshown at the 1e-4 plotting floor',xy=(640,1e-4),xytext=(70,2.5e-4),fontsize=8,arrowprops={'arrowstyle':'->','color':'#526879'})
ax.legend(ncol=3,fontsize=8);ax.grid(alpha=.25)
iso_plot=save_plot(fig,'isolated_budget_error.png')
iso_table=[r for r in iso_best if r['budget']==128]
selection_table=[]
for method in ['direct','emd','vmd','eemd','ceemdan','svmd']:
    row={'method':method.upper()}
    for case in ['0','1','2']:
        r=next(r for r in selections if r['method']==method and r['case']==case and float(r['tolerance'])==.01)
        row['clip'+case]=int(r['parameter_count']);row['error'+case]=float(r['nmse'])
    selection_table.append(row)
write_csv('isolated_selector_at_001.csv',selection_table)
linear_deltas=[]
for r in iso:
    if r['family'] not in ['polynomial','spline','dct'] or r['method']=='direct':continue
    matches=[d for d in iso if d['case']==r['case'] and d['method']=='direct' and d['family']==r['family'] and d['complexity']==r['complexity']]
    if matches:linear_deltas.append(abs(float(r['nmse'])-float(matches[0]['nmse'])))

test_rows=[]
for r in shared:
    if r['role']=='test':
        test_rows.append({'method':r['method'],'model':r['activation'],'code_size':int(r['latent_dimension']),'scalars_per_block':int(r['parameters_per_block']),'mean_nmse':float(r['mean_nmse']),'median_nmse':float(r['median_nmse']),'blocks':int(r['blocks'])})
write_csv('shared_representation_test.csv',test_rows)
fig,axes=plt.subplots(1,2,figsize=(10,4),sharey=True)
for ax,method in zip(axes,['direct','vmd']):
    for model in ['pca','sine','snakebeta']:
        group=[r for r in test_rows if r['method']==method and r['model']==model]
        ax.plot([r['scalars_per_block'] for r in group],[r['mean_nmse'] for r in group],'o-',label=model)
    ax.set(xlabel='Per-block scalars, including means/gains',title=method.upper());ax.grid(alpha=.25);ax.legend(fontsize=8)
axes[0].set_ylabel('Held-out mean block NMSE (256 blocks)')
shared_plot=save_plot(fig,'shared_representation_test.png')

sym_rows=[]
for arm in ['random','pretrained','adapted']:
    a=symbolic_summary['arms'][arm]
    sym_rows.append({'arm':arm,'mean_best_nmse':a['best_any_size_mean_nmse'],'median_nmse_at_16':a['budgets']['16']['median_nmse'],'success_01':a['nmse_thresholds']['0.1']['successful_imfs'],'success_001':a['nmse_thresholds']['0.01']['successful_imfs'],'success_0001':a['nmse_thresholds']['0.001']['successful_imfs'],'test_imfs':a['evaluated_imfs'],'proposals':a['total_proposals'],'unique':a['total_unique'],'search_seconds':a['search_seconds']})
write_csv('symbolic_test.csv',sym_rows)
synthetic_rows=[{'arm':r['arm'],'function':r['function'],'nmse':r['best']['nmse'],'parameters':r['best']['parameters'],'formula':r['best']['formula']} for r in symbolic_synthetic]
write_csv('symbolic_synthetic_discovery.csv',synthetic_rows)
fig,axes=plt.subplots(1,2,figsize=(10,3.8))
arms=[r['arm'] for r in sym_rows]
axes[0].bar(arms,[r['mean_best_nmse'] for r in sym_rows],color=['#8b9aa8','#4f89bd','#25816a'])
axes[0].set(ylabel='Mean best-found IMF NMSE',title='Held-out curve error (64 IMFs)',ylim=(0,1))
for i,r in enumerate(sym_rows):axes[0].text(i,r['mean_best_nmse']+.02,f"{r['mean_best_nmse']:.3f}",ha='center')
x=np.arange(3)
for offset,key,label in [(-.24,'success_01','NMSE <= 0.1'),(0,'success_001','NMSE <= 0.01'),(.24,'success_0001','NMSE <= 0.001')]:
    values=[r[key] for r in sym_rows]
    axes[1].bar(x+offset,values,width=.23,label=label)
axes[1].set(xticks=x,xticklabels=arms,ylabel='Successful test IMFs / 64',title='Low-error coverage',ylim=(0,5));axes[1].legend(fontsize=7)
sym_plot=save_plot(fig,'symbolic_test.png')

CSS='''body{font:16px/1.6 system-ui,sans-serif;color:#192c3b;background:#f3f6fa;margin:0}main{max-width:1120px;margin:auto;padding:32px}h1{font-size:2.4rem;line-height:1.12}h2{margin-top:1.8em}h3{line-height:1.3}section,article{background:white;border:1px solid #d7e0e8;border-radius:12px;padding:24px;margin:22px 0}p{max-width:100ch}a{color:#075a8b}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:10px;text-align:left;border-bottom:1px solid #d7e0e8;vertical-align:top}th{background:#eaf0f5}.scroll{overflow-x:auto}img{width:100%;height:auto}figure{margin:18px 0}nav{display:flex;gap:16px;flex-wrap:wrap}.note{background:#fff3db;border-left:4px solid #b17c1d;padding:14px}.finding{background:#e9f4ef;border-left:4px solid #248167;padding:14px}.muted,small{color:#526879}code{font-size:.9em;background:#edf2f7;padding:2px 5px;overflow-wrap:anywhere}.player{display:inline-flex;flex-direction:column;vertical-align:top;margin:10px 12px 10px 0;max-width:320px;font-size:14px}.player span{min-height:3em}audio{max-width:300px}.tag{color:#285f81;font-weight:650}details{margin-top:16px}summary{cursor:pointer;font-weight:bold}@media(max-width:650px){main{padding:14px}section,article{padding:16px}h1{font-size:1.9rem}td,th{padding:7px}audio{width:250px}}'''
parts=['<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Parametric speech experiments: results to date</title><style>'+CSS+'</style></head><body><main>',
 '<p class="tag">Research results | 4 November 2026 | Saved-output retrospective</p><h1>Parametric speech experiments: results to date</h1>',
 '<p>This report covers the parametric research sequence: sampled decomposition, the stopped multi-window sweep, its implementation audit, isolated 40-ms fitting, the shared representation pilot, and explicit IMF expression discovery. Smoke tests and superseded launches are not treated as independent scientific evidence.</p>',
 '<nav><a href="#overview">Run ledger</a><a href="#baseline">Sampled baseline</a><a href="#audit">Stopped sweep and audit</a><a href="#isolated">40-ms results</a><a href="#shared">Shared representation</a><a href="#symbolic">Expression discovery</a><a href="#conclusions">Conclusions</a><a href="#data">Evidence</a></nav>',
 '<section><h2>What the experiments currently establish</h2><ul><li>Sampled EMD/VMD modes reconstruct the original testcase well; sampled residual controls restore omitted detail. This did not test compression.</li><li>Poor low-budget fitting persists without overlap or stitching. Three isolated blocks show very different difficulty.</li><li>In the recorded conservative 1% error-bound selections, direct fitting used fewer scalars than every decomposition arm on all three blocks. These are finite-library selections, not globally minimal representations.</li><li>The shared representation pilot did not discover mathematical expressions. Its strongest high-budget result was the direct PCA baseline.</li><li>The symbolic learner improved average held-out fitting over random proposals, but only 1/64 adapted IMF fits reached NMSE <= 0.01. Synthetic oscillatory discovery also remained weak.</li></ul><p class="note">We have not demonstrated a broadly reliable compact speech representation or a perceptual-quality improvement. Learned VMD and RVQ have not been trained in this sequence.</p></section>',
 '<section id="overview"><h2>Run ledger</h2>'+table(statuses,[('experiment','Experiment'),('status','Saved status'),('completed','Completed units'),('minutes','Recorded active minutes'),('scope','Data / scope')])+'<p class="muted">Progress units differ: decomposition cases, fitting jobs, data preparation, optimizer updates and search jobs are not comparable workloads. Times are the saved active-time estimates, excluding pauses and unrecorded work. Zero final failures does not mean every candidate fit is accurate or globally optimized. The symbolic run resumed after an HTML replacement error; the report-only repair is preserved in repair_history.json.</p></section>',
 '<section id="baseline"><h2>1. Sampled EMD/VMD baseline</h2><p><strong>Question:</strong> can short-time decomposition and normalized overlap-add reconstruct one LibriSpeech testcase before parameter fitting?</p><p>Utterance 1272-128104-0000: 93,680 samples, 16 kHz, 5.855 seconds. EMD and eight-mode VMD were run at 20/40/80/160 ms with 50% overlap. The table shows modes-only output, before adding a sampled residual.</p>'+base_plot,
 table(base_rows,[('method','Method'),('window_ms','Window ms'),('nmse','NMSE'),('pesq_wb','PESQ-WB'),('stoi','STOI')]),
 '<p class="finding">At 40 ms, modes-only EMD gave NMSE 0.005689 and VMD 0.007516. Adding the sampled residual yielded zero reported float64 NMSE for EMD and near-machine-precision reconstruction for VMD. These controls retained sampled arrays, so their quality cannot be interpreted as a compact parameter result.</p>',
 '<p>PESQ/STOI here describe the full utterance. They are not scores for isolated IMFs or 40-ms clips, and no new listening evaluation was conducted.</p>',
 listen(BASE+'/reference.wav','Original 5.855-second testcase'),listen(BASE+'/emd/window_640_samples/modes.wav','EMD modes only, 40 ms windows'),listen(BASE+'/vmd/window_640_samples/modes.wav','VMD modes only, 40 ms windows'),'</section>',
 '<section id="audit"><h2>2. Stopped multi-window sweep and implementation audit</h2><p>The broad sweep compared EMD, VMD, EEMD, CEEMDAN, SVMD and direct fitting across 20/40/80/160-ms windows. Six function families, budgets of 500-8000 scalars/second and four coefficient precisions were planned. It stopped at 452/744 jobs after 184.25 recorded minutes; available results are incomplete and do not support a balanced global ranking.</p>',
 '<p><strong>Important audit findings:</strong> low per-mode allocations could produce empty nonlinear models; 80-step fitting was inadequate in a known representable AM-FM example; payload metadata could dominate coefficient bytes. Therefore this report does not rank function families by that sweep or use its perceptual-score outliers as evidence.</p>',
 table([r for r in audit['fits'] if r['signal']=='amfm_chirp' and r['family'] in ['amfm','oracle_amfm']],[('family','Known chirp test'),('steps','Fitting steps'),('parameters','Stored values'),('nmse','NMSE')]),
 '<p class="finding">With the same six-value AM-FM representation, the old fitter changed from NMSE 0.6535 at 80 steps to 2.804e-15 at 800 steps. The known correct values gave 6.672e-30. This demonstrates an optimization failure in that test; it does not establish a shortage of representational capacity.</p>',
 '<details><summary>Coefficient precision audit</summary>'+table(quantization,[('bits','Bits / coefficient'),('nmse','NMSE'),('model_parameters','Model values'),('payload_bytes','Total payload bytes'),('metadata_bytes','Metadata bytes')])+'<p>For this one known chirp, reducing coefficients from 32 to 8 bits changed payload size only from 143 to 132 bytes because metadata dominated, while error rose from 1.71e-15 to 0.1128. Later isolated and learned experiments therefore omitted quantization. This is not a general quantization or RVQ result.</p></details><p class="muted">An earlier launch in parametric_ablation_librispeech_1272 failed during a Windows progress-file replacement and was superseded by the v2 directory. Its stale running flag is not used as evidence of an active experiment.</p></section>',
 '<section id="isolated"><h2>3. Isolated 40-ms curve-fitting study</h2><p><strong>Question:</strong> how much fitting error remains when the complete input is just 640 samples and there are no stitches?</p><p>Three blocks from the same utterance start at 0.96, 1.96 and 2.96 seconds. Five decomposers plus a direct-waveform arm received only each isolated block. The 53-candidate library covered Chebyshev polynomial degrees 0-128, spline knot counts 2-256, DCT sizes 1-640, 1-16 sinusoidal atoms, 12 amplitude/phase order pairs, and widths 1/2/4/8 for per-component sine and SnakeBeta networks. Nonlinear fits used three starts and up to 1600 evaluations/steps. All 972 jobs completed.</p><p>Counts below are total fitted numerical values for the whole block, including represented residual components. There is no quantization, overlap, shared decoder or RVQ.</p>'+iso_plot,
 '<h3>Best observed uniform-family result at a maximum of 128 scalars per block</h3>'+table(iso_table,[('case','Clip'),('method','Method'),('family','Fitted family'),('parameters','Actual scalars'),('nmse','Block NMSE'),('edge_nmse','Edge NMSE'),('interior_nmse','Interior NMSE')]),
 '<p class="finding">Errors remain substantial in several arms without stitching. At this budget, the difficult third clip still has NMSE 0.3075 in the best direct candidate, 0.3475 for VMD and 0.2654 for SVMD. The first two direct clips reach 0.002519 and 0.004588. Difficulty varies strongly even within this single utterance.</p>',
 '<p>The curve uses the best saved successful uniform-family fit within each total budget, then averages over all three clips. Missing all-clip coverage would omit a curve point. Values below 1e-4 are displayed at that plotting floor; the full direct DCT is near machine precision. These are retrospective best-observed results, not a trained selector or a generalization benchmark. Edge/interior values diagnose the saved reconstruction; native decomposition endpoint extensions still apply.</p>',
 '<h3>Recorded mixed-representation selections at conservative NMSE bound <= 0.01</h3>'+table(selection_table,[('method','Method'),('clip0','Clip 0 scalars'),('error0','Actual NMSE'),('clip1','Clip 1 scalars'),('error1','Actual NMSE'),('clip2','Clip 2 scalars'),('error2','Actual NMSE')]),
 '<p>The selector minimizes scalar count among available fitted candidates subject to a conservative triangle-inequality error bound. It does not minimize count subject to the exact summed-waveform error. The bound can be more conservative for a multi-component decomposition, so these counts alone do not establish a universal advantage for direct fitting. The 640-value direct result is the full DCT control, not compression.</p>',
 f'<p>A separate saved-result check found a maximum block-NMSE difference of {max(linear_deltas):.3g} between direct fitting and matched polynomial/spline/DCT projections after summing all represented modes and residual. For these shared linear spaces, decomposition repeated the coefficient overhead without improving the summed projection.</p>',
 '<h3>Same-budget listening examples</h3><p>Each pair below uses the best saved candidate at <=128 total scalars. Clips are exactly 40 ms; playback supports artifact inspection but cannot establish speech intelligibility or perceptual quality.</p>']
for case in [0,1,2]:
    parts.append(listen(ISO+f'/reference_{case}.wav',f'Clip {case}: original'))
    for method in ['direct','vmd','svmd']:
        r=next(r for r in iso_best if r['budget']==128 and r['case']==case and r['method']==method)
        parts.append(listen(ISO+'/'+r['audio'],f"Clip {case}: {method.upper()}, {r['parameters']} values, NMSE {r['nmse']:.4g}"))
parts += ['</section>',
 '<section id="shared"><h2>4. Shared representation pilot: a different question</h2><p>This run learned shared encoders/decoders rather than generating explicit IMF functions. It compared PCA, sine and SnakeBeta representations in direct-waveform and fixed-eight-mode VMD domains. VMD also encoded its residual, for nine components. The speaker-disjoint data contain 4096 training, 256 validation and 256 test blocks; each neural model ran 12,000 updates. Validation across code budgets selected checkpoints before test evaluation.</p>'+shared_plot,
 '<h3>Held-out results at code size 64 per component</h3>'+table([r for r in test_rows if r['code_size']==64],[('method','Domain'),('model','Model'),('scalars_per_block','Total block scalars'),('mean_nmse','Mean NMSE'),('median_nmse','Median NMSE'),('blocks','Test blocks')]),
 '<p class="finding">The direct PCA model reached mean NMSE 0.3000 with 66 per-block scalars. Direct sine and SnakeBeta gave 0.5865 and 0.5242 at the same count. VMD-PCA gave 0.3039 using 594 block scalars, with VMD sine/SnakeBeta at 0.6205/0.6039. These results do not demonstrate an advantage for the learned neural representations in this pilot.</p>',
 '<p>Code size 64 is not an equal whole-block budget across domains: direct uses 64+2 values, while VMD uses 9*(64+2). Shared basis/decoder assets are additional storage, not counted in the per-block scalar axis. The run is retained as exploratory evidence and does not answer the requested free-form function-discovery question. No PESQ/STOI was computed on these isolated blocks.</p>']
parts.append(listen(SHARED+'/examples/direct_pca_test_0_reference.wav','Original first saved shared-pilot test block'))
for model in ['pca','sine','snakebeta']:
    parts.append(listen(SHARED+f'/examples/direct_{model}_test_0_z64_reconstruction.wav',f'First saved direct test example: {model}, code 64'))
parts += ['<details><summary>All 30 held-out model/budget results</summary>'+table(test_rows,[('method','Domain'),('model','Model'),('code_size','Code/component'),('scalars_per_block','Total scalars'),('mean_nmse','Mean NMSE'),('median_nmse','Median NMSE')])+'</details></section>',
 '<section id="symbolic"><h2>5. Explicit per-IMF expression-discovery pilot</h2><p><strong>Question:</strong> can an IMF-conditioned model generate a small mathematical expression and fitted constants without a shared waveform decoder?</p><p>Frozen eight-mode VMD caches supply 16 train, four validation and eight test blocks. There are 120 eligible training IMFs after eight near-silent exclusions, 32 validation IMFs and 64 held-out test IMFs. The finite language contains t, adjustable constants, addition, subtraction, multiplication, sin, cos, tanh and square. An MLP/GRU proposes structures; bounded float64 least squares fits constants and output gain. Training used 1500 synthetic updates and 80 adaptation updates. Validation selected adaptation checkpoint 60.</p><p>Random, pretrained and adapted arms each receive 48 proposals per test IMF, scheduled tree limits 7/15/31, two fitting starts, and two frontier refinements with stronger fitting. Duplicates consume proposals. The test consists of 64 IMFs nested within eight chunks, not 64 independent speakers. No residual or full-block synthesis is evaluated here.</p>'+sym_plot,
 table(sym_rows,[('arm','Arm'),('mean_best_nmse','Mean best NMSE'),('median_nmse_at_16','Median NMSE, <=16 values'),('success_01','<=0.1 /64'),('success_001','<=0.01 /64'),('success_0001','<=0.001 /64'),('search_seconds','Test search seconds')]),
 '<p class="finding">Adaptation lowered mean best-found NMSE from 0.8264 (random) to 0.6486, but low-error coverage remained poor: 2/64 reached NMSE <=0.1 and 1/64 reached <=0.01. Synthetic pretraining reached <=0.1 on 3/64, so adaptation did not improve every success measure.</p>',
 '<p>Each arm made 3072 proposals; unique structures were 1982 random, 3048 pretrained and 2876 adapted. Equal proposals are not equal compute: recorded search took 19.23, 75.34 and 56.06 seconds respectively. These are one-seed pilot results without statistical significance claims. Description bits use a reference accounting rule, not an entropy-coded bitrate.</p>',
 '<h3>Discovery controls versus known-structure fitting</h3>'+table(synthetic_rows,[('function','Synthetic target'),('arm','Search arm'),('nmse','Best-found NMSE'),('parameters','Stored values')]),
 f'<p>With its mathematical structure supplied, the chirp fitting audit reached NMSE {symbolic_known["chirped_amfm_nmse"]:.4g}; its maximum derivative-check error was {symbolic_known["jacobian_max_error"]:.4g}. In contrast, free structure discovery left large errors on sinusoid/chirp tests. The adapted model discovered the quadratic accurately but did not reliably discover the oscillatory targets. Successful fitting of a supplied expression must not be reported as successful function discovery.</p>',
 '<p>The displayed formulas in the downloadable synthetic table are the actual best-found programs. Some are unnecessarily complicated; current literal parameter counts are not guaranteed minimal after algebraic simplification.</p>',
 '<details><summary>IMF listening example</summary><p>First saved held-out IMF, adapted arm. These are native-amplitude mode signals, not complete speech.</p>',
 listen(SYM+'/examples/adapted_test_b99_m0_reference.wav','Original first saved IMF'),listen(SYM+'/examples/adapted_test_b99_m0_0.wav','Saved adapted candidate 0'),listen(SYM+'/examples/adapted_test_b99_m0_1.wav','Saved adapted candidate 1'),'</details></section>',
 '<section id="conclusions"><h2>Evidence we can use, and remaining gaps</h2><table><thead><tr><th>Question</th><th>Evidence so far</th></tr></thead><tbody><tr><td>Do sampled modes reconstruct speech?</td><td>Yes on the initial testcase, especially with sampled residual controls.</td></tr><tr><td>Is stitching the sole cause of poor parameterized results?</td><td>No. Substantial fitting errors remain in isolated 40-ms blocks. The experiments do not quantify how much additional error stitching contributes.</td></tr><tr><td>Which decomposition is strongest?</td><td>No corpus-level winner established. Direct and SVMD are useful controls in the three-block study; results depend on budget, library and selection criterion.</td></tr><tr><td>Did a shared neural representation solve function discovery?</td><td>No. That pilot had a different output and did not outperform the strong direct PCA result at its larger budgets.</td></tr><tr><td>Did symbolic learning help?</td><td>Average held-out curve error improved, but accurate-fit coverage and synthetic oscillatory discovery remain inadequate.</td></tr><tr><td>Have we shown speech compression or perceptual superiority?</td><td>No reliable general claim. Scalar count, reference description bits and perceptual quality are distinct measurements.</td></tr><tr><td>Have we trained learned VMD or RVQ?</td><td>No.</td></tr></tbody></table><p>These are small, heterogeneous pilots. Their NMSE values cannot be pooled: some evaluate whole utterances, some isolated blocks, and the symbolic run evaluates individual IMFs. For the next study, function-search reliability and numerical fitting need stronger evidence before learned decomposition is judged through them.</p></section>',
 '<section id="data"><h2>Evidence and downloads</h2><p>All tables and plots above are rebuilt from saved outputs. No new model training, speech fitting, hyperparameter selection or listening study was performed for this report.</p><ul>']
for filename in ['run_status.csv','sampled_baseline.csv','isolated_best_observed_by_budget.csv','isolated_selector_at_001.csv','shared_representation_test.csv','symbolic_test.csv','symbolic_synthetic_discovery.csv','source_ledger.json']:
    parts.append(f'<li><a href="{filename}" download="{filename}">{filename}</a></li>')
parts.append('</ul><p>Original complete listening reports are in their study-specific approaches/parametric/*/report/ folders: preliminary_multi_window_sweep_stopped_after_audit.html, isolated_40ms_capacity_complete.html, shared_representation_pilot_complete.html, and symbolic_imf_pilot_complete.html. The stopped sweep report is about 591 MB; this retrospective embeds only explicitly labeled examples.</p><p class="muted">NMSE is squared waveform error divided by reference squared energy; lower is better. NMSE 0.01 corresponds to -20 dB relative error energy, not a perceptual threshold. No independent human ratings validate the isolated block or IMF scores.</p></section></main></body></html>')

# Prioritized reading guide based on the full local-paper review, with exact PDF sections.
paper_notes=read_json('parametric_reports/local_paper_reading_2026_11_04/review_notes.json')
paper_inventory=read_json('parametric_reports/local_paper_reading_2026_11_04/inventory.json')
papers={p['id']:p for p in paper_notes['papers']}
paths={p['id']:p for p in paper_inventory}
reading=[
 ('P18','1','Read now','PDF p.2, Fig.1 and Section 2.1; pp.3-4 experiments','Why can an exactly suitable oscillator still fail to fit? What does the surrogate improve, and what failures remain?'),
 ('P15','2','Read now','PDF pp.3-7; skim pp.12-13 afterwards','What must we count to call one mathematical description smaller, and what does exhaustive structure search actually guarantee?'),
 ('P08','3','Read now','PDF pp.4-9; Appendix C/D on pp.17-19 as needed','How can the model repair a partially correct formula from the residual, rather than generating everything again?'),
 ('P11','4','Read next','PDF pp.2-5 and experimental comparisons on pp.7-8','Can adaptive shared knots describe two related curves cheaply? Which assumptions belong specifically to periodic 2D contours?'),
 ('P01','5','Before learned decomposition','PDF pp.3-8; conclusion on p.19','Can demodulation make a mode easier to describe, and what happens when ridge-separation assumptions fail?'),
 ('P02','6','Before learned decomposition','PDF pp.2-4, Fig.1; results and conclusion on pp.5-7','Which pieces are learned, what is supervised, and which reconstruction steps remain iterative?'),
]
optional=[('P09','After RAG-SR','PDF pp.3-7 and Appendix C on p.16','A learned guide for remaining formula complexity; inspect training scale and distinguish predicted token length from real coding cost.'),('P16','For localized oscillatory functions','PDF pp.3-5','How localization changes representation and fitting bias; visual results are not direct speech evidence.'),('P06','For activation-language extensions','PDF pp.5-8','Variable-periodic functions and initialization. Prefer this over reading both FINER versions initially.'),('P07','Optional spline-to-symbolic route','PDF pp.3-5 and Appendix M, p.32','Learned edge functions and pruning; symbolification still uses a candidate library.')]
reading_data=[]
guide=['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Prioritized reading for compact IMF functions</title><style>'+CSS+'</style><main><p class="tag">Personal study guide | 4 November 2026</p><h1>Six papers to study first</h1><p>You do not need to work through all 19 papers before forming a direction. The first three address the immediate function-discovery problem. The next three support alternative representations and eventual learned decomposition.</p><p><strong>If you only read three:</strong> sinusoidal optimization, exhaustive SR/MDL, and RAG-SR. The first pass below is about 35 PDF pages; detailed derivations and reference sections can wait. Section references are PDF page numbers, not journal pagination.</p>']
for pid,rank,priority,sections,question in reading:
    p=papers[pid];path=paths[pid];href=quote(Path(os.path.relpath(ROOT/'documentation/literature'/path['filename'], OUT)).as_posix())
    reading_data.append({'rank':rank,'title':p['title'],'venue':p['venue'],'year':p['year'],'priority':priority,'sections':sections,'question':question,'source_pdf':path['source']})
    guide+=['<article><p class="tag">'+rank+' | '+priority+'</p><h2><a href="'+href+'">'+escape(p['title'])+'</a></h2><p>'+escape(p['venue']+' | '+p['year'])+'</p><p><strong>Start with: </strong>'+escape(sections)+'</p><p><strong>Read to answer: </strong>'+escape(question)+'</p><p><strong>Connection to our work: </strong>'+escape(p['proposal'])+'</p><p><strong>Keep in mind: </strong>'+escape(p['limit'])+'</p></article>']
guide+=['<section><h2>Read selectively afterwards</h2>']
for pid,reason,sections,question in optional:
    p=papers[pid];href=quote(Path(os.path.relpath(ROOT/'documentation/literature'/paths[pid]['filename'], OUT)).as_posix())
    guide+=['<p><strong><a href="'+href+'">'+escape(p['title'])+'</a></strong> - '+escape(reason)+'. '+escape(sections)+'. '+escape(question)+'</p>']
guide+=['<p>Keep the broad symbolic-regression survey as a reference. Defer MAPSS until we revisit perceptual evaluation on reconstructed speech in context. The two learnable-EMD papers and DAMD/STVMD are useful comparisons after the function-fitting objective is clearer.</p></section><section><h2>Questions worth writing down while reading</h2><ol><li>What exactly is stored and evaluated: a complete formula, a spline, or shared neural weights?</li><li>Does the method improve the function family, the search, or only numerical optimization?</li><li>What experimental result supports its claim, and on which signals?</li><li>Which assumptions could fail for a 640-sample speech IMF?</li><li>What single controlled ablation would let us accept or reject the transfer?</li></ol></section><p class="muted">Paper links use your repository PDFs. This guide summarizes the supplied papers; proposed transfers are hypotheses, not demonstrated gains or verified novelty claims.</p></main></html>']
(OUT/'reading_guide.html').write_text('\n'.join(guide),encoding='utf-8')
write_csv('reading_order.csv',reading_data)

ledger={'date':'2026-11-04','scope':'Saved parametric experiment results; no new training or fitting','source_files':SOURCES,'derived_checks':{'matched_linear_projection_max_nmse_difference':max(linear_deltas),'isolated_csv_rows':len(iso),'old_sweep_csv_rows':len(old),'shared_test_rows':len(test_rows),'symbolic_test_imfs':64,'symbolic_synthetic_targets':len(synthetic_rows)}}
(OUT/'source_ledger.json').write_text(json.dumps(ledger,indent=2),encoding='utf-8')
(OUT/'index.html').write_text('\n'.join(parts),encoding='utf-8')
result=export_html(OUT/'index.html',OUT/'experiment_results_shareable.html',notice='Standalone report: plots, result tables, downloadable evidence and audio are embedded. In current Chrome/Edge, click Load embedded audio. This report summarizes saved experiments; no new run was started.')
print(json.dumps({'report':result,'reading_guide':str(OUT/'reading_guide.html'),'linear_projection_max_nmse_difference':max(linear_deltas),'derived_tables':7},indent=2))


if __name__ == '__main__':
    pass
