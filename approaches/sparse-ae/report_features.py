"""Build and audit a standalone channel-inspection report from completed analysis."""
import argparse
import base64
import hashlib
import html
import io
import json
from pathlib import Path
import re
from html.parser import HTMLParser

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.signal import stft
import soundfile as sf

from sae.interpret import FEATURES
from sae.metrics import audio_metrics, summarize

REPO=Path(__file__).resolve().parents[2]


def esc(value):
    return html.escape(str(value),quote=True)


def num(value):
    return 'Not available' if value is None else f'{value:.6f}'


def uri(raw,mime):
    return f'data:{mime};base64,'+base64.b64encode(raw).decode('ascii')


def figure_uri(fig):
    stream=io.BytesIO();fig.savefig(stream,format='png',dpi=90);plt.close(fig)
    return uri(stream.getvalue(),'image/png')


class Inspector(HTMLParser):
    def __init__(self):
        super().__init__();self.audio=[];self.images=[];self.external=[];self.cases=0
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if tag=='audio':self.audio.append(a['src'])
        if tag=='img':self.images.append(a['src'])
        if tag=='table' and a.get('class')=='case':self.cases+=1
        for key in ('src','href'):
            if key in a and not a[key].startswith(('data:','#')):self.external.append(a[key])


def build(root,destination):
    record=json.loads((root/'analysis.json').read_text())
    assert record['checks']['completed_baseline']==40 and record['checks']['completed_interventions']==48
    destination.mkdir(parents=True,exist_ok=True)
    cache={};metric_checks=0
    def asset(path):
        if path in cache:return cache[path]
        raw=(root/path).read_bytes()
        x,rate=sf.read(io.BytesIO(raw),dtype='float32')
        info=sf.info(io.BytesIO(raw))
        assert rate==16000 and x.ndim==1 and info.subtype=='PCM_16' and np.isfinite(x).all()
        fig,ax=plt.subplots(figsize=(5,1.9))
        f,t,z=stft(x,fs=rate,window='hann',nperseg=512,noverlap=384,boundary=None)
        im=ax.pcolormesh(t,f/1000,20*np.log10(np.maximum(np.abs(z),1e-7)),vmin=-80,vmax=0,shading='auto',cmap='magma')
        ax.set(xlabel='Time (s)',ylabel='Frequency (kHz)',xlim=(0,len(x)/rate),ylim=(0,8))
        fig.colorbar(im,ax=ax,label='Magnitude (dB re 1)');fig.tight_layout()
        result=dict(audio=uri(raw,'audio/wav'),image=figure_uri(fig),wave=x)
        cache[path]=result;return result
    def validate(source,path,expected):
        nonlocal metric_checks
        x,y=asset(source)['wave'],asset(path)['wave']
        actual=audio_metrics(x,y,speech=True)
        for k in ('mse','nmse','si_sdr','pesq_wb','stoi'):
            a,b=actual[k],expected[k]
            assert (a is None and b is None) or (a is not None and b is not None and abs(a-b)<1e-8),(path,k,a,b)
        assert actual['statuses']==expected['statuses']
        metric_checks+=1
    parts=['<!doctype html><html lang="en"><meta charset="utf-8"><title>Sparse autoencoder feature inspection</title>',
        '<style>body{margin:20px}table{border-collapse:collapse;margin:12px 0}th,td{border:1px solid #888;padding:6px;vertical-align:top}img{max-width:100%;height:auto}audio{width:100%;min-width:180px}table.case{width:100%;table-layout:fixed}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style>',
        '<h1>Sparse autoencoder feature inspection</h1>',
        '<p>This exploratory study asks what activates a latent channel and what changes when that channel is removed. '
        'It compares the trained L1 sparse autoencoder (SAE) with a dense autoencoder. No model was retrained.</p>',
        '<p><strong>Scope:</strong> 40 full LibriSpeech dev-clean utterances, one per speaker, totaling 267.735 s. '
        'The utterances were excluded from the saved training, validation-selection and calibration-selection lists. '
        'Their speakers were used during development. These results are not independent speaker-held-out confirmation.</p>',
        '<p>All 256 channels per model were measured. Eight channels per model were selected by a fixed hash rule among channels '
        'active on 1–99% of latent frames. Three peak examples from different speakers were selected per channel. '
        'This gives 48 ablation examples of 1.024 s each. Every baseline case and every selected ablation example is shown below. '
        'There are 40 completed baseline cases and 48 completed ablation cases; none failed or were excluded.</p>',
        '<p>Perceptual Evaluation of Speech Quality (PESQ) uses wideband mode at 16 kHz. Short-Time Objective Intelligibility (STOI) '
        'uses the standard, non-extended setting at 16 kHz. Scale-invariant signal-to-distortion ratio (SI-SDR) uses zero-mean signals. '
        'Normalized mean square error (NMSE) is squared error divided by reference energy. Higher PESQ, STOI and SI-SDR are better; lower NMSE is better. '
        'These scores alone do not establish listening quality or intelligibility.</p>',
        '<h2>Mean reconstruction scores</h2><p>Unweighted means across the same 40 full utterances. Valid score counts are shown.</p>',
        '<table><tr><th>Model</th><th>PESQ-WB</th><th>STOI</th><th>SI-SDR (dB)</th><th>NMSE</th><th>Active latent entries (%)</th></tr>']
    for method in ('l1','dense'):
        summary=summarize([c['models'][method] for c in record['baseline_cases']])
        m=record['models'][method]
        parts.append(f'<tr><td>{method}</td>'+''.join(f'<td>{num(summary[k])} (n={summary[k+"_valid"]})</td>' for k in ('pesq_wb','stoi','si_sdr','nmse'))+f'<td>{100*m["active_fraction"]:.3f}</td></tr>')
    parts.append('</table><h2>Measured observations</h2>')
    for method in ('l1','dense'):
        m=record['models'][method]
        eligible=[r for r in m['channels'] if .01<=r['activity_fraction']<=.99]
        stable=[]
        for r in m['channels']:
            for name,a in r['associations'].items():
                if a['r'] is not None and abs(a['r'])>=.3 and a['speaker_valid']>=30 and a['same_sign_speaker_fraction']>=.75:
                    stable.append((r['channel'],name,a))
        effect=[r for r in record['interventions'] if r['method']==method]
        changes=[r['after']['nmse']-r['control']['nmse'] for r in effect]
        parts.append(f'<p>{method}: {len(eligible)} of 256 channels meet the activity selection rule. '
            f'{sum(r["activity_fraction"]==0 for r in m["channels"])} channels never activate on this set. '
            f'{len({r[0] for r in stable})} channels have at least one proxy association with |r| ≥ 0.30, '
            'a valid within-speaker correlation on at least 30 speakers, and matching signs on at least 75% of those speakers. '
            f'For its 24 selected peak examples, the mean NMSE change after ablation is {np.mean(changes):+.6f}; '
            f'{sum(v>0 for v in changes)} of 24 examples have increased NMSE.</p>')
        valid=[r for r in m['channels'] if r['strongest_proxy'] is not None]
        parts.append(f'<p>Among the {len(valid)} channels with a defined proxy correlation, '
            f'{sum(r["strongest_proxy"]=="log_rms" for r in valid)} have log RMS as their strongest inspected proxy. '
            'A channel with no activation has no defined correlation on this set.</p>')
    parts.append('<h3>Candidate associations beyond signal level</h3><p>For each model, the following table shows the five channels '
        'with the largest absolute non-level proxy correlation. Only one proxy is listed per channel. '
        'This ranking is exploratory and uses the same data as the correlations. A channel can still have a stronger signal-level association.</p>'
        '<table><tr><th>Model</th><th>Channel</th><th>Proxy</th><th>Level-controlled r</th><th>Same-sign speakers</th><th>Ablation included</th></tr>')
    for method in ('l1','dense'):
        candidates=[]
        for r in record['models'][method]['channels']:
            valid=[(abs(a['r']),name,a) for name,a in r['associations'].items() if name!='log_rms' and a['r'] is not None]
            if valid:
                strength,name,a=max(valid,key=lambda v:v[0]);candidates.append((strength,r['channel'],name,a))
        for _,channel,name,a in sorted(candidates,key=lambda v:-v[0])[:5]:
            included=channel in record['models'][method]['selected_channels']
            parts.append(f'<tr><td>{method}</td><td>{channel}</td><td>{name}</td><td>{num(a["r"])}</td>'
                f'<td>{num(a["same_sign_speaker_fraction"])} (n={a["speaker_valid"]})</td><td>{"Yes" if included else "No"}</td></tr>')
    parts.append('</table><p>L1 channel 70 has a high-band power association after linear level control. '
        'It was not selected by the fixed hash rule for ablation. Dense channel 50 has a similar candidate association and was selected. '
        'This suggests a later targeted test of both channels; the channel numbers do not identify matched features across models.</p>')
    parts += ['<p>These thresholds are descriptive summaries. Frames are dependent, and many channels and proxies were inspected. '
        'No significance test or semantic feature count is claimed. The two models have different channel identities and activity levels.</p>',
        '<h2>Interpretation and limits</h2><p>Association shows co-occurrence. Ablation measures the effect of one latent intervention in the current decoder. '
        'The decoder is nonlinear. Channel effects can depend on other channels. Setting a channel to zero can create an uncommon code combination. '
        'No channel is labelled as a phoneme detector. Acoustic proxies are not phoneme annotations. The pitch proxy can make octave errors.</p>',
        '<p>These results can suggest candidates for later testing. They do not establish that sparse channels are more semantic than dense channels. '
        'The strongest proxy and peak examples were found on this same exploration set. Confirmation needs frozen hypotheses and speakers absent from training and development.</p>',
        '<h2>All-channel measurements</h2><p>Activity includes all valid latent frames. Correlations exclude incomplete acoustic windows and RMS below 0.0001. '
        'Root mean square (RMS) measures signal amplitude. All correlations except log RMS remove a linear log-RMS effect first. '
        'Spectral centroid is a power-weighted mean frequency. High-band fraction is the power fraction above 3 kHz. '
        'Periodicity is the strongest local autocorrelation peak from 50–400 Hz. Pitch is used only when that peak is at least 0.6. '
        'Speaker sign agreement compares within-recording correlation signs with the pooled sign; it is not an independent test.</p>']
    for method in ('l1','dense'):
        rows=record['models'][method]['channels']
        fig,ax=plt.subplots(figsize=(12,3))
        matrix=np.array([[r['associations'][f]['r'] if r['associations'][f]['r'] is not None else np.nan for r in rows] for f in FEATURES])
        im=ax.imshow(matrix,aspect='auto',vmin=-1,vmax=1,cmap='coolwarm',interpolation='nearest')
        ax.set(xlabel='Latent channel',title=f'{method}: acoustic proxy associations',yticks=range(5),yticklabels=FEATURES)
        fig.colorbar(im,ax=ax,label='Pearson r (unitless)');fig.tight_layout()
        parts.append(f'<img width="1100" src="{figure_uri(fig)}" alt="{method} association heatmap">')
        parts.append(f'<details><summary>{method}: all 256 channels</summary><table><tr><th>Channel</th><th>Active (%)</th><th>Strongest inspected proxy</th><th>Pooled r</th><th>Median within-speaker r</th><th>Same-sign speakers</th></tr>')
        for r in rows:
            a=r['associations'].get(r['strongest_proxy'],{})
            parts.append(f'<tr><td>{r["channel"]}</td><td>{100*r["activity_fraction"]:.3f}</td><td>{esc(r["strongest_proxy"] or "Not available")}</td><td>{num(a.get("r"))}</td><td>{num(a.get("median_speaker_r"))}</td><td>{num(a.get("same_sign_speaker_fraction"))} (n={a.get("speaker_valid",0)})</td></tr>')
        parts.append('</table></details>')
    parts += ['<h2>Method and provenance</h2>',
        '<p>The latent hop is 256 samples (16 ms). The encoder receptive field is 1,276 samples (79.75 ms). '
        'A plotted activation time uses the nominal center at 256t + 128 samples; it is not an exact event boundary. '
        'Acoustic proxies use 1,024-sample (64 ms) windows. This window is shorter than the encoder receptive field.</p>',
        '<p>Ablations decode full-utterance latents before cropping. Only the specified channel changes. '
        'The control is the same model with all channels retained. A sham intervention with multiplier one reproduces the baseline decoder output '
        'within 0.000001 sample amplitude. No external vocoder is used. Checkpoints use the shared WaveformDecoder implementation.</p>',
        '<p>All playback audio is 16-kHz mono PCM16 WAV. All displayed scores use its exact decoded samples. '
        'No gain fitting, normalization, resampling or alignment was used. PCM16 can clip values outside [-1, 1); clipping counts are in the saved records. '
        'Spectrograms use a 512-sample Hann window, 128-sample hop, 0–8 kHz range and -80 to 0 dB magnitude relative to amplitude 1. '
        'Compared signals in each table have the same time and colour scales.</p>',
        '<p>Playback data rate is 16,000 samples/s × 16 bits/sample × 1 channel = 256,000 bit/s (256 kbit/s). '
        'Here 1 kbit/s is 1,000 bit/s. WAV headers are excluded. Encoded bitstream rate is not available. '
        'No quantized bitstream was created. Model weights and metadata are not included in a codec-rate claim.</p>',
        '<details><summary>Run settings, source hashes and checkpoints</summary><pre>'+esc(json.dumps({k:record[k] for k in ('run_id','code_revision','source_hashes','packages','protocol','device')},indent=2))+'</pre>']
    for method in ('l1','dense'):
        m=record['models'][method]
        parts.append('<pre>'+esc(json.dumps({k:v for k,v in m.items() if k!='channels'},indent=2))+'</pre>')
    parts.append('</details><h2>Full-utterance reconstruction cases</h2>')
    def case_table(key,paths,scores):
        parts.append(f'<table class="case" id="{esc(key)}"><tr><th>Item</th><th>Before: source</th><th>After</th><th>Control</th></tr>')
        parts.append('<tr><th>Audio</th>'+''.join(f'<td><audio controls preload="none" src="{asset(p)["audio"]}"></audio></td>' for p in paths)+'</tr>')
        parts.append('<tr><th>Spectrogram</th>'+''.join(f'<td><img width="450" alt="Spectrogram" src="{asset(p)["image"]}"></td>' for p in paths)+'</tr>')
        for metric,title in [('pesq_wb','PESQ-WB'),('stoi','STOI'),('si_sdr','SI-SDR (dB)'),('nmse','NMSE')]:
            parts.append(f'<tr><th>{title}</th><td>Reference</td>'+''.join('<td>'+num(s[metric])+((' — '+esc(s['statuses'].get(metric,''))) if s[metric] is None else '')+'</td>' for s in scores)+'</tr>')
        parts.append('<tr><th>Playback audio data rate</th><td>256 kbit/s</td><td>256 kbit/s</td><td>256 kbit/s</td></tr><tr><th>Encoded data rate</th><td>Not applicable</td><td>Not available</td><td>Not available</td></tr></table>')
    for c in record['baseline_cases']:
        parts.append(f'<h3>{esc(c["id"])}; speaker {esc(c["speaker"])}</h3><p>Full recording; source and output duration {c["samples"]/16000:.6f} s. Start sample 0. After: L1 SAE. Control: dense autoencoder.</p>')
        paths=[c['source_audio'],c['models']['l1']['audio'],c['models']['dense']['audio']]
        for method in ('l1','dense'):validate(c['source_audio'],c['models'][method]['audio'],c['models'][method])
        case_table(c['id'],paths,[c['models']['l1'],c['models']['dense']])
        parts.append('<p>The table measures reconstruction with every latent channel retained. It does not measure encoded bitrate.</p>')
    parts.append('<h2>Selected channel interventions</h2><p>After: one channel set to zero. Control: unchanged reconstruction from the same model. '
        'All three examples for each selected channel are included. Selection uses activation strength, not ablation score. '
        'The change-energy concentration is the fraction of output-change energy in the highest-energy 10% of 16-ms blocks, rounded up.</p>')
    for r in record['interventions']:
        stats=record['models'][r['method']]['channels'][r['channel']]
        a=stats['associations'][stats['strongest_proxy']]
        parts.append(f'<h3>{esc(r["method"])} channel {r["channel"]}, example {r["rank"]}: {esc(r["id"])}; speaker {esc(r["speaker"])}</h3>'
            f'<p>Source crop start {r["start_sample"]/16000:.6f} s ({r["start_sample"]} samples). Source and output duration {r["samples"]/16000:.6f} s. '
            f'Peak activation {r["peak_activation"]:.6f} (model units). Strongest inspected proxy: {esc(stats["strongest_proxy"])}; '
            f'pooled r={num(a["r"])}; same-sign speaker fraction={num(a["same_sign_speaker_fraction"])} (n={a["speaker_valid"]}). This is a candidate association, not a feature label.</p>')
        fig,ax=plt.subplots(figsize=(8,1.8));ax.plot(r['trace_times'],r['trace_values'])
        ax.set(xlabel='Time in crop (s)',ylabel='Activation\n(model units)',xlim=(0,r['samples']/16000));fig.tight_layout()
        parts.append(f'<img width="720" src="{figure_uri(fig)}" alt="Channel activation trace">')
        paths=[r['source_audio'],r['after_audio'],r['control_audio']]
        validate(paths[0],paths[1],r['after']);validate(paths[0],paths[2],r['control'])
        case_table(r['key'],paths,[r['after'],r['control']])
        change=r['after']['nmse']-r['control']['nmse']
        parts.append(f'<p>Observed NMSE change: {change:+.6f}. Output-change energy / control energy: {r["change_energy_relative_to_control"]:.6f}. '
            f'Highest-energy 10% block concentration: {r["change_energy_top_10_percent_frames"]:.6f}. '
            'These values apply to this selected example. They do not establish that the channel is necessary for a speech category.</p>')
    parts.append('<h2>Next action</h2><p>Use these measurements to choose a small set of candidate hypotheses. Freeze the channels, proxy directions, '
        'thresholds and intervention tests. Test them on speakers absent from training and development. Add reliable phoneme annotations before making phoneme-specific claims.</p>')
    record['report_builder_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    embedded=json.dumps(record,allow_nan=False).replace('</','<\\/')
    parts.append('<script id="records" type="application/json">'+embedded+'</script></html>')
    path=destination/'sparse_ae_feature_inspection.html';path.write_text('\n'.join(parts),encoding='utf-8')
    # Audit the actual standalone file, not just its input assets.
    parsed=Inspector();parsed.feed(path.read_text(encoding='utf-8'))
    assert parsed.cases==88 and len(parsed.audio)==264 and not parsed.external
    for link in parsed.audio:
        x,sr=sf.read(io.BytesIO(base64.b64decode(link.split(',',1)[1])),dtype='float32')
        assert sr==16000 and x.ndim==1 and np.isfinite(x).all()
    for link in parsed.images:
        with Image.open(io.BytesIO(base64.b64decode(link.split(',',1)[1]))) as im:im.verify()
    verification=dict(case_tables=parsed.cases,expected_case_tables=88,audio_elements=len(parsed.audio),
        images=len(parsed.images),metric_pairs_recomputed=metric_checks,exact_embedded_audio_metrics_matched=True,
        all_embedded_audio_decoded=True,all_embedded_images_decoded=True,external_dependencies=parsed.external,
        html_bytes=path.stat().st_size,html_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        offline_browser_visual_check='Pending',audio_playback_browser_check='Pending')
    (destination/'sparse_ae_feature_inspection_metrics.json').write_text(json.dumps(record,indent=2,allow_nan=False))
    (destination/'sparse_ae_feature_inspection_verification.json').write_text(json.dumps(verification,indent=2))
    print(json.dumps(verification,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',type=Path,default=REPO/'runs/sparse_ae_features_20261110')
    parser.add_argument('--report-dir',type=Path,default=REPO/'approaches/sparse-ae/reports')
    args=parser.parse_args();build(args.inputs.resolve(),args.report_dir.resolve())
