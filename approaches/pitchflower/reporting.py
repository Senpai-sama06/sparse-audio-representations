"""Build a portable, plain HTML report for 40 held-out PitchFlower RVQ crops."""
import argparse, base64, hashlib, html, io, json, math, os, sys, wave
from html.parser import HTMLParser
from pathlib import Path
from statistics import fmean
os.environ.setdefault("HF_HUB_OFFLINE", "1")
METHOD_ROOT = Path(__file__).resolve().parent
ROOT = METHOD_ROOT.parents[1]
sys.path.insert(0, str(ROOT))

OUT = METHOD_ROOT / "reports"
SR = 24000
PCM_BITS = 16
CODEBOOKS, CODEBOOK_SIZE = 8, 512
INDEX_BITS = math.ceil(math.log2(CODEBOOK_SIZE))
MEL_BANDS, MEL_HOP, FLOAT_BITS = 100, 256, 32
# Keep source handles open during evaluation. Some local archive handles take
# a long time to close. Decode each archive from RAM instead.
SOURCE_HANDLES = []

def uri(data, mime):
    return "data:"+mime+";base64,"+base64.b64encode(data).decode("ascii")

def wav(x):
    import soundfile as sf

    stream = io.BytesIO()
    sf.write(stream, x, SR, format="WAV", subtype="PCM_16")
    return stream.getvalue()

def spectrum(x, duration):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.signal import stft

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
    from scipy.signal import resample_poly
    from pesq import pesq
    from pystoi import stoi

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
    import numpy as np

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

def bitrate_budget(record):
    """Fixed-width payload accounting, not a measured serialized codec rate."""
    duration = record["source_samples"] / SR
    # The saved report predates mel frame logging. The centered mel transform in
    # PitchFlowerModel has floor(samples / 256) + 1 frames and 100 bands.
    mel_frames = record.get("source_mel_frames", record["source_samples"] // MEL_HOP + 1)
    content_bits = record["latent_frames"] * CODEBOOKS * INDEX_BITS
    pitch_bits = record["pitch_frames"] * FLOAT_BITS
    mel_bits = mel_frames * MEL_BANDS * FLOAT_BITS
    return {
        "duration_seconds": duration,
        "source_mel_frames": mel_frames,
        "source_mel_frames_basis": record.get("source_mel_frames_basis", "derived from centered mel transform"),
        "content_bits": content_bits,
        "pitch_bits": pitch_bits,
        "combined_bits": content_bits + pitch_bits,
        "raw_mel_bits": mel_bits,
        "content_kbps": content_bits / duration / 1000,
        "pitch_kbps": pitch_bits / duration / 1000,
        "combined_kbps": (content_bits + pitch_bits) / duration / 1000,
        "raw_mel_kbps": mel_bits / duration / 1000,
    }


def bitrate_summary(records):
    budgets = [bitrate_budget(r) for r in records]
    seconds = sum(r["duration_seconds"] for r in budgets)
    rates = {key + "_kbps": sum(r[key + "_bits"] for r in budgets) / seconds / 1000
             for key in ("content", "pitch", "combined", "raw_mel")}
    pcm = SR * PCM_BITS / 1000
    combined = rates["combined_kbps"]
    return {
        "status": "estimated fixed-width payload; no serialized codec bitstream",
        "aggregation": "total payload bits / total source duration",
        "pcm_kbps": pcm,
        **rates,
        "pcm_to_combined_ratio": pcm / combined,
        "reduction_vs_pcm_percent": 100 * (1 - combined / pcm),
        "raw_mel_to_combined_ratio": rates["raw_mel_kbps"] / combined,
        "reduction_vs_raw_mel_percent": 100 * (1 - combined / rates["raw_mel_kbps"]),
        "pitch_share_percent": 100 * rates["pitch_kbps"] / combined,
        "assumptions": {
            "sample_rate": SR, "pcm_bits_per_sample": PCM_BITS,
            "rvq_codebooks": CODEBOOKS, "entries_per_codebook": CODEBOOK_SIZE,
            "bits_per_index": INDEX_BITS, "bits_per_pitch_value": FLOAT_BITS,
            "mel_bands": MEL_BANDS, "mel_hop_samples": MEL_HOP,
            "bits_per_raw_mel_value": FLOAT_BITS,
            "raw_mel_baseline": "hypothetical uncompressed float32 source mel; not a tested mel codec",
            "excluded": ["headers and packetization", "byte padding", "gain and other metadata",
                         "shared model weights and codebooks"],
        },
    }


def measure_playback(record):
    """Measure the actual embedded WAVs, independently of codec payload estimates."""
    measurements = {}
    for name in ("before", "after", "control"):
        blob = base64.b64decode(record[name + "_audio"].split(",", 1)[1], validate=True)
        with wave.open(io.BytesIO(blob), "rb") as audio:
            duration = audio.getnframes() / audio.getframerate()
            pcm_bits = audio.getnframes() * audio.getnchannels() * audio.getsampwidth() * 8
            measurements[name] = {
                "wav_bytes": len(blob), "duration_seconds": duration,
                "pcm_bits": pcm_bits, "pcm_kbps": pcm_bits / duration / 1000,
                "wav_file_kbps": len(blob) * 8 / duration / 1000,
            }
    return measurements


def playback_summary(records):
    result = {}
    for name in ("before", "after", "control"):
        clips = [r["playback"][name] for r in records]
        duration = sum(c["duration_seconds"] for c in clips)
        result[name] = {
            "total_wav_bytes": sum(c["wav_bytes"] for c in clips),
            "duration_seconds": duration,
            "pcm_kbps": sum(c["pcm_bits"] for c in clips) / duration / 1000,
            "wav_file_kbps": sum(c["wav_bytes"] for c in clips) * 8 / duration / 1000,
        }
    return result


def quality_rate_section(records, means):
    rate = bitrate_summary(records)
    playback = playback_summary(records)
    return [
        '<h2>After versus Vocoder alone: bitrate and quality</h2>',
        f'<p>For these {len(records)} short dev-clean crops, the vocoder alone has higher reconstruction scores, '
        'but it receives the full source mel spectrogram. '
        'It is a reconstruction control, not a codec tested at the same bitrate as PitchFlower. '
        'These results do not establish which system gives better quality at a fixed bitrate.</p>',
        '<table border="1"><tr><th>Comparison</th><th>After (PitchFlower and vocoder)</th><th>Vocoder alone</th></tr>',
        '<tr><th>Decoder input representation</th><td>RVQ indices plus float32 source pitch</td><td>Full source mel, raw float32</td></tr>',
        f'<tr><th>Estimated decoder-input payload rate (kbit/s)</th><td>{rate["combined_kbps"]:.3f}</td><td>{rate["raw_mel_kbps"]:.3f}</td></tr>',
        f'<tr><th>Measured output PCM rate (kbit/s)</th><td>{playback["after"]["pcm_kbps"]:.3f}</td><td>{playback["control"]["pcm_kbps"]:.3f}</td></tr>',
        f'<tr><th>Measured output WAV file rate, including headers (kbit/s)</th><td>{playback["after"]["wav_file_kbps"]:.3f}</td><td>{playback["control"]["wav_file_kbps"]:.3f}</td></tr>',
        f'<tr><th>Mean PESQ-WB</th><td>{means["pesq_wb"]:.4f}</td><td>{means["vocoder_pesq_wb"]:.4f}</td></tr>',
        f'<tr><th>Mean STOI</th><td>{means["stoi"]:.4f}</td><td>{means["vocoder_stoi"]:.4f}</td></tr></table>',
        f'<p><strong>After versus Vocoder alone:</strong> After needs an estimated '
        f'{rate["raw_mel_to_combined_ratio"]:.2f} times fewer decoder-input payload bits '
        f'({rate["reduction_vs_raw_mel_percent"]:.3f}% reduction), under the fixed-width RVQ and float32 assumptions. '
        'The output WAVs use the same PCM format, so their PCM bitrates are equal. '
        'The compression benefit concerns the representation sent to the decoder; it does not reduce the bitrate of the decoded WAV.</p>',
        f'<p>The estimated PitchFlower payload is {rate["content_kbps"]:.3f} kbit/s of content plus '
        f'{rate["pitch_kbps"]:.3f} kbit/s of pitch, for {rate["combined_kbps"]:.3f} kbit/s total. '
        f'This is {rate["pcm_to_combined_ratio"]:.2f}:1 relative to source PCM '
        f'({rate["reduction_vs_pcm_percent"]:.2f}% fewer payload bits), and '
        f'{rate["raw_mel_to_combined_ratio"]:.2f}:1 relative to raw float32 mel '
        f'({rate["reduction_vs_raw_mel_percent"]:.2f}% fewer payload bits). '
        'These are representation-size estimates, not measured end-to-end compression ratios. '
        'The PCM comparison uses the report\'s 24 kHz reference, not the original dataset files or another audio codec.</p>',
        f'<p>The quality cost relative to the full-source-mel control is '
        f'{means["pesq_wb"] - means["vocoder_pesq_wb"]:+.4f} PESQ-WB and '
        f'{means["stoi"] - means["vocoder_stoi"]:+.4f} STOI in the mean. '
        'The scores show lower reconstruction fidelity for PitchFlower at this operating point. '
        'Whether that loss is acceptable depends on the bitrate target and listening results.</p>',
        f'<p>Pitch accounts for {rate["pitch_share_percent"]:.2f}% of the estimated PitchFlower payload. '
        'The content-only rate omits a decoder input and must not be reported as the total codec rate. '
        'Reducing pitch precision would require a new quality evaluation.</p>',
        '<p>The raw mel equivalent assumes 100 float32 values per mel frame; it is not the measured rate '
        'of a compressed mel codec. A fair codec ranking needs complete bitstreams, all required side '
        'information, and quality measurements at comparable bitrates. The embedded WAV sizes describe '
        'playback audio, not the transmitted representation.</p>',
    ]


def make_report(records,checkpoint_sha):
    for record in records:
        record["playback"] = measure_playback(record)
    mean=lambda name:fmean(r[name] for r in records if r[name] is not None)
    means={k:mean(k) for k in ["stoi","pesq_wb","vocoder_stoi","vocoder_pesq_wb"]}
    parts=['<!doctype html><html lang="en"><head><meta charset="utf-8"><title>PitchFlower RVQ test report</title></head><body>',
    '<h1>PitchFlower RVQ test report</h1>',
    '<p><strong>Standalone report:</strong> Share this HTML file itself. Save it on the receiving computer and open it in a browser. All 120 audio clips, 120 spectrograms, scores, bitrate comparisons, and calculation assumptions are contained in this file. No internet connection, repository, dataset, model checkpoint, Python installation, or companion JSON files are needed to view and discuss these precomputed results.</p>',
    *quality_rate_section(records, means),
    '<h2>Test limits</h2>',
    f'<p>This report contains {len(records)} test cases from dev-clean. There is one test case for each selected speaker. Each source crop is 1.504 seconds long. These files were not used for training. These utterances were not used to select the checkpoint or to calibrate the model.</p>',
    '<p>This is a short-crop test. It is not a test of all 2,703 dev-clean files. It is not a test-clean result. The test does not measure full-utterance quality.</p>',
    '<p>The model uses the checkpoint at step 91,500. The flow solver uses 10 steps. The random seed is 0 plus the test case index. The input to the content encoder is the cached WORLD signal. The decoder also receives the source pitch. There is no requested pitch change.</p>',
    '<p>The output of the vocoder alone is a control. This control uses the source mel spectrogram. It does not use the RVQ encoder or the flow model.</p>',
    '<p>Both paths use the same pretrained Vocos vocoder, charactr/vocos-mel-24khz. RVQ means residual vector quantization. After uses RVQ content and source pitch to generate a mel spectrogram, then Vocos generates the waveform. Vocoder alone gives Vocos the source mel spectrogram directly.</p>',
    '<p>Before means the source waveform. After means the output of the model and the vocoder. All audio is mono at 24 kHz. The embedded WAV files use 16-bit PCM. The scores use those same audio samples after PCM conversion.</p>',
    '<p>PESQ means perceptual evaluation of speech quality. STOI means short-time objective intelligibility. Higher scores show better agreement with the source. PESQ uses wideband mode at 16 kHz. STOI uses standard mode at 16 kHz. The test resamples both signals from 24 kHz to 16 kHz. The test does not change the gain or time alignment to improve the scores. If the sample counts differ, the test uses their common length.</p>',
    '<h2>Bitrate limits</h2>',
    '<p>The PCM audio data rate is 384 kbit/s: 24,000 samples/s times 16 bits/sample. This rate excludes the WAV header. The before and after playback files have this same format.</p>',
    '<p>The RVQ content rate is an estimate. It uses 8 codebooks and 512 entries per codebook. Each index requires 9 bits. The estimate uses the actual latent frame count for each crop. The pitch rate assumes 32 bits for each pitch value. The combined estimate is the sum of the content and pitch rates. It excludes headers, codebook storage, model weights, gain data, and other metadata. This test does not write or decode a complete compressed bitstream. A measured codec bitrate is not available.</p>',
    '<p>For a crop of duration T seconds, content bits = latent frames times 8 times 9; pitch bits = pitch frames times 32; raw mel bits = source mel frames times 100 times 32. Divide bits by T and by 1,000 to obtain kbit/s. Source mel frames are measured during inference or derived for saved cases as floor(source samples / 256) + 1 from the centered mel transform. Summary rates use total bits divided by total source duration. Fixed-width estimates assume packed indices and exclude byte padding. Model weights and codebooks are assumed to be shared at the decoder.</p>',
    '<h2>Results</h2>',
    '<p>The scores below apply only to these short crops. Use the audio to check the result. The scores do not prove that the output is suitable for a specific use.</p>']
    for i,r in enumerate(records):
        budget = bitrate_budget(r)
        parts += [f'<h3>Test case {i+1}: {html.escape(r["id"])}</h3>',
        f'<p>Speaker: {html.escape(r["speaker"])}. Source duration: {r["source_samples"]/SR:.3f} s. Output duration: {r["output_samples"]/SR:.3f} s.</p>',
        '<table border="1"><tr><th>Item</th><th>Before</th><th>After</th><th>Vocoder alone</th></tr>',
        '<tr><th>Audio</th>'+''.join(f'<td><audio controls preload="none" src="{r[k+"_audio"]}"></audio></td>' for k in ["before","after","control"])+'</tr>',
        '<tr><th>Spectrogram</th>'+''.join(f'<td><img width="540" height="260" alt="{label} spectrogram for test case {i+1}" src="{r[k+"_spectrum"]}"></td>' for k,label in [("before","Source"),("after","Model output"),("control","Vocoder control")])+'</tr>',
        f'<tr><th>PESQ-WB</th><td>Reference</td><td>{num(r["pesq_wb"])}</td><td>{num(r["vocoder_pesq_wb"])}</td></tr>',
        f'<tr><th>STOI</th><td>Reference</td><td>{num(r["stoi"])}</td><td>{num(r["vocoder_stoi"])}</td></tr>',
        '<tr><th>Playback PCM rate (kbit/s)</th><td>384.000</td><td>384.000</td><td>384.000</td></tr>',
        '<tr><th>Measured WAV file bytes, including headers</th>'+''.join(f'<td>{r["playback"][name]["wav_bytes"]:,}</td>' for name in ("before","after","control"))+'</tr>',
        '<tr><th>Measured WAV file rate, including headers (kbit/s)</th>'+''.join(f'<td>{r["playback"][name]["wav_file_kbps"]:.3f}</td>' for name in ("before","after","control"))+'</tr>',
        f'<tr><th>Estimated decoder-input payload rate (kbit/s)</th><td>Not applicable</td><td>{budget["combined_kbps"]:.3f} (RVQ + pitch)</td><td>{budget["raw_mel_kbps"]:.3f} (raw float32 mel)</td></tr>',
        f'<tr><th>Estimated decoder-input reduction vs Vocoder alone</th><td>Not applicable</td><td>{100*(1-budget["combined_kbps"]/budget["raw_mel_kbps"]):.3f}% ({budget["raw_mel_kbps"]/budget["combined_kbps"]:.2f}:1)</td><td>Reference</td></tr>',
        f'<tr><th>Estimated RVQ content rate (kbit/s)</th><td>Not applicable</td><td>{r["content_kbps"]:.3f}</td><td>Not applicable</td></tr>',
        f'<tr><th>Assumed float32 pitch rate (kbit/s)</th><td>Not applicable</td><td>{r["pitch_kbps"]:.3f}</td><td>Not applicable</td></tr>',
        f'<tr><th>Estimated content and pitch rate (kbit/s)</th><td>Not applicable</td><td>{budget["combined_kbps"]:.3f}</td><td>Not applicable</td></tr>',
        f'<tr><th>Raw float32 source mel equivalent (kbit/s)</th><td>Not applicable</td><td>Not applicable</td><td>{budget["raw_mel_kbps"]:.3f}</td></tr>',
        f'<tr><th>Estimated payload reduction vs source PCM</th><td>Reference</td><td>{100*(1-budget["combined_kbps"]/(SR*PCM_BITS/1000)):.2f}% ({SR*PCM_BITS/1000/budget["combined_kbps"]:.2f}:1)</td><td>Not a codec comparison</td></tr>',
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
    import numpy as np
    import soundfile as sf
    import torch
    from vocos import Vocos
    from approaches.pitchflower.training.campaign import TrainingModule

    torch.set_num_threads(1); OUT.mkdir(parents=True, exist_ok=True)
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
            source_mel=model.net.mel(source).clamp_min(1e-7).log()
            control=vocoder.decode(source_mel).flatten().cpu().numpy()
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
               "source_mel_frames":source_mel.shape[-1],"source_mel_frames_basis":"measured during inference",
               "content_kbps":latent_frames*CODEBOOKS*INDEX_BITS/duration/1000,"pitch_kbps":len(pitch)*FLOAT_BITS/duration/1000}
            for k,z,blob in zip(["before","after","control"],decoded,audio_bytes):
                r[k+"_audio"]=uri(blob,"audio/wav"); r[k+"_spectrum"]=spectrum(z,len(z)/SR)
            records.append(r)
            print(f"Done case {i+1}: PESQ {r['pesq_wb']}; STOI {r['stoi']}",flush=True)
    report,means=make_report(records,checkpoint_sha)
    destination=OUT/"pitchflower_best_testcases.html"
    destination.write_text(report,encoding="utf-8")
    compact=[{k:v for k,v in r.items() if not k.endswith(("_audio","_spectrum"))} for r in records]
    summary={"checkpoint_sha256":checkpoint_sha,"checkpoint_step":91500,"cases":len(records),"means":means,"bitrate":bitrate_summary(records),"playback":playback_summary(records),"cases_data":compact}
    (OUT/"pitchflower_best_metrics.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print("REPORT_READY "+str(destination),flush=True)
    print(json.dumps({"cases":len(records),"means":means,"html_bytes":destination.stat().st_size}),flush=True)
    # Do not wait for delayed archive closes after all artifacts are saved.
    sys.stdout.flush()
    os._exit(0)

class SavedReport(HTMLParser):
    """Read embedded assets and case identities without decoding or changing them."""
    def __init__(self):
        super().__init__()
        self.assets = []
        self.headings = []
        self.in_heading = False

    def handle_starttag(self, tag, attrs):
        if tag in ("audio", "img"):
            self.assets.append((tag, dict(attrs)["src"]))
        if tag == "h3":
            self.in_heading = True
            self.headings.append("")

    def handle_data(self, data):
        if self.in_heading:
            self.headings[-1] += data

    def handle_endtag(self, tag):
        if tag == "h3":
            self.in_heading = False


def refresh_existing(report_path, metrics_path, verification_path=None):
    """Refresh interpretation from saved metrics and assets; no GPU or models needed."""
    summary = json.loads(metrics_path.read_text(encoding="utf-8"))
    records = [dict(r) for r in summary["cases_data"]]
    saved = SavedReport()
    saved.feed(report_path.read_text(encoding="utf-8"))
    expected_headings = [f"Test case {i+1}: {r['id']}" for i, r in enumerate(records)]
    expected_tags = [tag for _ in records for tag in ("audio",)*3 + ("img",)*3]
    if saved.headings != expected_headings or [tag for tag, _ in saved.assets] != expected_tags:
        raise ValueError("Report cases/assets do not match the supplied metric records")
    for i, record in enumerate(records):
        for suffix, offset in (("audio", 0), ("spectrum", 3)):
            for j, name in enumerate(("before", "after", "control")):
                record[name + "_" + suffix] = saved.assets[6*i + offset + j][1]
    report, means = make_report(records, summary["checkpoint_sha256"])
    if any(not math.isclose(means[k], summary["means"][k], abs_tol=1e-12, rel_tol=0) for k in means):
        raise ValueError("Saved summary scores do not match the case records")
    refreshed = SavedReport()
    refreshed.feed(report)
    if refreshed.assets != saved.assets or refreshed.headings != saved.headings:
        raise ValueError("Report refresh changed case identities or embedded assets")
    summary["bitrate"] = bitrate_summary(records)
    summary["playback"] = playback_summary(records)
    summary["cases_data"] = [{k:v for k,v in r.items() if not k.endswith(("_audio", "_spectrum"))} for r in records]
    report_path.write_text(report, encoding="utf-8")
    metrics_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if verification_path:
        verification = json.loads(verification_path.read_text(encoding="utf-8"))
        verification.update(html_bytes=report_path.stat().st_size, bitrate=summary["bitrate"], playback=summary["playback"])
        verification["embedded_assets_preserved"] = True
        verification_path.write_text(json.dumps(verification, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary["bitrate"], indent=2))


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-existing", type=Path, metavar="HTML", help="reuse saved audio, images and scores")
    parser.add_argument("--metrics", type=Path, metavar="JSON", help="metric records for the saved report")
    parser.add_argument("--verification", type=Path, metavar="JSON", help="optional saved verification record to update")
    args = parser.parse_args()
    if args.refresh_existing:
        if not args.metrics:
            parser.error("--refresh-existing requires --metrics")
        refresh_existing(args.refresh_existing, args.metrics, args.verification)
    elif args.metrics or args.verification:
        parser.error("--metrics and --verification require --refresh-existing")
    else:
        main()
