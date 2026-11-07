"""Build a portable, plain HTML report for 40 held-out PitchFlower RVQ crops."""
import base64, hashlib, html, io, json, math, os, sys, time
from pathlib import Path
os.environ.setdefault("HF_HUB_OFFLINE", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly, stft
from pesq import pesq
from pystoi import stoi
from vocos import Vocos
from approaches.pitchflower.training.campaign import TrainingModule

OUT = ROOT / "runs/pitchflower_best_report"
SR = 24000
# Keep source handles open during evaluation. Some local archive handles take
# a long time to close. Decode each archive from RAM instead.
SOURCE_HANDLES = []

def uri(data, mime):
    return "data:"+mime+";base64,"+base64.b64encode(data).decode("ascii")

def wav(x):
    stream = io.BytesIO()
    sf.write(stream, x, SR, format="WAV", subtype="PCM_16")
    return stream.getvalue()

def spectrum(x, duration):
    f,t,z = stft(x, fs=SR, nperseg=1024, noverlap=768, boundary=None)
    db = 20*np.log10(np.maximum(abs(z), 1e-6))
    fig,ax = plt.subplots(figsize=(5.4,2.6))
    im=ax.pcolormesh(t, f/1000, db, vmin=-90, vmax=0, shading="auto", cmap="magma")
    ax.set(xlabel="Time (s)", ylabel="Frequency (kHz)", xlim=(0,duration), ylim=(0,12))
    fig.colorbar(im,ax=ax,label="Magnitude (dB; reference amplitude = 1)")
    fig.tight_layout()
    stream=io.BytesIO(); fig.savefig(stream,format="png",dpi=100); plt.close(fig)
    return uri(stream.getvalue(),"image/png")

def score(x,y):
    n=min(len(x),len(y)); a=resample_poly(x[:n],2,3); b=resample_poly(y[:n],2,3)
    result={"stoi":float(stoi(a,b,16000,extended=False))}
    try:
        result["pesq_wb"]=float(pesq(16000,a,b,"wb"))
        result["pesq_error"]=None
    except Exception as e:
        result["pesq_wb"]=None; result["pesq_error"]=str(e)
    return result

def num(v):
    return "Not available" if v is None else f"{v:.4f}"

def load_crop(row,settings):
    digest=hashlib.sha256((row["id"]+":"+str(settings["seed"])+":pitchflower24k-v1").encode()).hexdigest()
    path=ROOT/settings["cache_dir"]/(digest+".npz")
    stream=path.open("rb"); SOURCE_HANDLES.append(stream)
    data=stream.read()
    with np.load(io.BytesIO(data),allow_pickle=False) as archive:
        a=archive["original"].copy(); b=archive["perturbed"].copy(); pitch=archive["pitch"].copy()
    samples=settings["samples"]; frames=samples//256+1
    if len(a)<samples:
        count=int(np.ceil(samples/len(a))); a=np.tile(a,count); b=np.tile(b,count)
        pitch=np.tile(pitch,int(np.ceil(frames/len(pitch))))
    a=a[:samples].copy(); b=b[:samples].copy(); pitch=pitch[:frames].copy()
    a/=max(float(abs(a).max()),1e-8); b/=max(float(abs(b).max()),1e-8)
    return a,b,pitch

def make_report(records,checkpoint_sha,selection):
    mean=lambda name:float(np.mean([r[name] for r in records if r[name] is not None]))
    means={k:mean(k) for k in ["stoi","pesq_wb","vocoder_stoi","vocoder_pesq_wb"]}
    parts=['<!doctype html><html lang="en"><head><meta charset="utf-8"><title>PitchFlower RVQ test report</title></head><body>',
    '<h1>PitchFlower RVQ test report</h1>',
    '<p>This file contains all audio and images. You can use this file without a network connection.</p>',
    '<h2>Test limits</h2>',
    f'<p>This report contains {len(records)} test cases from dev-clean. There is one test case for each selected speaker. Each source crop is 1.504 seconds long. These files were not used for training. These utterances were not used to select the checkpoint or to calibrate the model.</p>',
    '<p>This is a short-crop test. It is not a test of all 2,703 dev-clean files. It is not a test-clean result. The test does not measure full-utterance quality.</p>',
    '<p>The model uses the checkpoint at step 91,500. The flow solver uses 10 steps. The random seed is 0 plus the test case index. The input to the content encoder is the cached WORLD signal. The decoder also receives the source pitch. There is no requested pitch change.</p>',
    '<p>The output of the vocoder alone is a control. This control uses the source mel spectrogram. It does not use the RVQ encoder or the flow model.</p>',
    '<p>Before means the source waveform. After means the output of the model and the vocoder. All audio is mono at 24 kHz. The embedded WAV files use 16-bit PCM. The scores use those same audio samples after PCM conversion.</p>',
    '<p>PESQ means perceptual evaluation of speech quality. STOI means short-time objective intelligibility. Higher scores show better agreement with the source. PESQ uses wideband mode at 16 kHz. STOI uses standard mode at 16 kHz. The test resamples both signals from 24 kHz to 16 kHz. The test does not change the gain or time alignment to improve the scores. If the sample counts differ, the test uses their common length.</p>',
    '<h2>Bitrate limits</h2>',
    '<p>The PCM audio data rate is 384 kbit/s: 24,000 samples/s times 16 bits/sample. This rate excludes the WAV header. The before and after playback files have this same format.</p>',
    '<p>The RVQ content rate is an estimate. It uses 8 codebooks and 512 entries per codebook. Each index requires 9 bits. The estimate uses the actual latent frame count for each crop. The pitch rate assumes 32 bits for each pitch value. The combined estimate is the sum of the content and pitch rates. It excludes headers, codebook storage, model weights, gain data, and other metadata. This test does not write or decode a complete compressed bitstream. A measured codec bitrate is not available.</p>',
    '<h2>Mean scores</h2><table border="1"><tr><th>Output</th><th>PESQ-WB</th><th>STOI</th></tr>',
    f'<tr><td>Model and vocoder</td><td>{means["pesq_wb"]:.4f}</td><td>{means["stoi"]:.4f}</td></tr>',
    f'<tr><td>Vocoder alone</td><td>{means["vocoder_pesq_wb"]:.4f}</td><td>{means["vocoder_stoi"]:.4f}</td></tr></table>',
    '<h2>Results</h2>',
    '<p>The scores below apply only to these short crops. Use the audio to check the result. The scores do not prove that the output is suitable for a specific use.</p>']
    for i,r in enumerate(records):
        parts += [f'<h3>Test case {i+1}: {html.escape(r["id"])}</h3>',
        f'<p>Speaker: {html.escape(r["speaker"])}. Source duration: {r["source_samples"]/SR:.3f} s. Output duration: {r["output_samples"]/SR:.3f} s.</p>',
        '<table border="1"><tr><th>Item</th><th>Before</th><th>After</th><th>Vocoder alone</th></tr>',
        '<tr><th>Audio</th>'+''.join(f'<td><audio controls preload="none" src="{r[k+"_audio"]}"></audio></td>' for k in ["before","after","control"])+'</tr>',
        '<tr><th>Spectrogram</th>'+''.join(f'<td><img width="540" height="260" alt="{label} spectrogram for test case {i+1}" src="{r[k+"_spectrum"]}"></td>' for k,label in [("before","Source"),("after","Model output"),("control","Vocoder control")])+'</tr>',
        f'<tr><th>PESQ-WB</th><td>Reference</td><td>{num(r["pesq_wb"])}</td><td>{num(r["vocoder_pesq_wb"])}</td></tr>',
        f'<tr><th>STOI</th><td>Reference</td><td>{num(r["stoi"])}</td><td>{num(r["vocoder_stoi"])}</td></tr>',
        '<tr><th>Playback PCM rate (kbit/s)</th><td>384.000</td><td>384.000</td><td>384.000</td></tr>',
        f'<tr><th>Estimated RVQ content rate (kbit/s)</th><td>Not applicable</td><td>{r["content_kbps"]:.3f}</td><td>Not applicable</td></tr>',
        f'<tr><th>Assumed float32 pitch rate (kbit/s)</th><td>Not applicable</td><td>{r["pitch_kbps"]:.3f}</td><td>Not applicable</td></tr>',
        f'<tr><th>Estimated content and pitch rate (kbit/s)</th><td>Not applicable</td><td>{r["content_kbps"]+r["pitch_kbps"]:.3f}</td><td>Not applicable</td></tr>',
        '<tr><th>Measured codec bitrate</th><td>Not applicable</td><td>Not available</td><td>Not applicable</td></tr></table>',
        '<p>All spectrograms use the same frequency range and colour scale. A time shift can reduce the waveform scores.</p>']
        if r["pesq_error"]:
            parts.append('<p>PESQ did not return a score: '+html.escape(r["pesq_error"])+'</p>')
    parts += ['<h2>Record</h2>',f'<p>Checkpoint SHA-256: <code>{checkpoint_sha}</code>.</p>',
        '<p>The test case list is fixed before inference. The list excludes the 120 checkpoint-selection utterances and the 40 calibration utterances. It uses the first remaining utterance for each speaker, in source ID order. The test uses the first crop of each utterance. The source and WORLD input crops each use peak normalization, as in the training data loader.</p>',
        '<p>The language uses short, direct technical sentences. This report does not claim certified ASD-STE100 compliance.</p>',
        '</body></html>']
    return "\n".join(parts),means

def main():
    torch.set_num_threads(1); OUT.mkdir(exist_ok=True)
    run=ROOT/"runs/pitchflower_hardware"
    settings=json.loads((run/"settings.json").read_text()); audit=json.loads((run/"data_audit.json").read_text())
    checkpoint=run/"rvq/checkpoints/best_step=91500.ckpt"
    print("Load checkpoint.",flush=True)
    # Read once. Keep the source handle until report generation is complete.
    handle=checkpoint.open("rb"); SOURCE_HANDLES.append(handle); checkpoint_bytes=handle.read()
    checkpoint_sha=hashlib.sha256(checkpoint_bytes).hexdigest()
    payload=torch.load(io.BytesIO(checkpoint_bytes),map_location="cpu",weights_only=False)
    del checkpoint_bytes
    model=TrainingModule(settings,"rvq"); model.load_state_dict(payload["state_dict"],strict=True); del payload
    model.cuda().eval(); vocoder=Vocos.from_pretrained("charactr/vocos-mel-24khz").cuda().eval()
    rows=[json.loads(l) for l in (ROOT/"pitchflower_gs_standalone/data/manifests/validation.jsonl").read_text().splitlines() if l.strip()]
    excluded=set(audit["validation_ids"]+audit["calibration_ids"]); selected={}
    for row in sorted(rows,key=lambda r:r["id"]):
        if row["id"] not in excluded: selected.setdefault(str(row["speaker"]),row)
    records=[]
    with torch.inference_mode():
        for i,row in enumerate(selected.values()):
            print(f"Read case {i+1}/{len(selected)}: {row['id']}",flush=True)
            a,b,pitch=load_crop(row,settings)
            source=torch.from_numpy(a)[None].cuda(); content=torch.from_numpy(b)[None].cuda(); f0=torch.from_numpy(pitch)[None].cuda()
            torch.manual_seed(settings["seed"]+i)
            latent=model.net.latents(content); latent_frames=latent.shape[-1]
            mel=model.net.reconstruct_mel(content,f0,steps=10)
            y=vocoder.decode(mel).flatten().cpu().numpy()
            control=vocoder.decode(model.net.mel(source).clamp_min(1e-7).log()).flatten().cpu().numpy()
            if not all(np.isfinite(z).all() for z in [a,y,control]): raise ValueError("Nonfinite waveform")
            # Measure the actual PCM audio embedded in the report.
            audio_bytes=[wav(z) for z in [a,y,control]]
            decoded=[sf.read(io.BytesIO(z),dtype="float32")[0] for z in audio_bytes]
            metric=score(decoded[0],decoded[1]); baseline=score(decoded[0],decoded[2])
            duration=len(a)/SR
            r={"id":row["id"],"speaker":str(row["speaker"]),"source_samples":len(a),"output_samples":len(y),
               "stoi":metric["stoi"],"pesq_wb":metric["pesq_wb"],"pesq_error":metric["pesq_error"],
               "vocoder_stoi":baseline["stoi"],"vocoder_pesq_wb":baseline["pesq_wb"],
               "latent_frames":latent_frames,"pitch_frames":len(pitch),
               "content_kbps":latent_frames*8*9/duration/1000,"pitch_kbps":len(pitch)*32/duration/1000}
            for k,z,blob in zip(["before","after","control"],decoded,audio_bytes):
                r[k+"_audio"]=uri(blob,"audio/wav"); r[k+"_spectrum"]=spectrum(z,len(z)/SR)
            records.append(r)
            print(f"Done case {i+1}: PESQ {r['pesq_wb']}; STOI {r['stoi']}",flush=True)
    report,means=make_report(records,checkpoint_sha,list(selected.values()))
    destination=OUT/"pitchflower_best_testcases.html"
    destination.write_text(report,encoding="utf-8")
    compact=[{k:v for k,v in r.items() if not k.endswith(("_audio","_spectrum"))} for r in records]
    summary={"checkpoint_sha256":checkpoint_sha,"checkpoint_step":91500,"cases":len(records),"means":means,"cases_data":compact}
    (OUT/"metrics.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print("REPORT_READY "+str(destination),flush=True)
    print(json.dumps({"cases":len(records),"means":means,"html_bytes":destination.stat().st_size}),flush=True)
    # Do not wait for delayed archive closes after all artifacts are saved.
    sys.stdout.flush()
    os._exit(0)

if __name__=="__main__": main()
