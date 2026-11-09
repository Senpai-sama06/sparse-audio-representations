"""Compile saved adaptive-STFT results according to documentation/REPORTS_README.md.

No model inference or training. Decode saved ASD1 files, embed FLOAT WAV audio,
recalculate scores from those exact WAV samples, and verify all selected cases.
Keep the earlier run reports intact. Outputs use approaches/representations/reports/adaptive_stft_dictionary/.
"""
from __future__ import annotations

import base64
import csv
import hashlib
import html
from html.parser import HTMLParser
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.signal import stft
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/adaptive_stft_dictionary_librispeech_20261107"
EVAL = RUN / "bitrate_quality_test_64"
OUT = ROOT / "approaches/representations/reports/adaptive_stft_dictionary"
PREFIX = "adaptive_stft_dictionary"


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def embed(data, mime):
    return f"data:{mime};base64," + base64.b64encode(data).decode("ascii")


def wav_bytes(x):
    stream = io.BytesIO()
    sf.write(stream, x, 16000, format="WAV", subtype="FLOAT")
    encoded = stream.getvalue()
    decoded, rate = sf.read(io.BytesIO(encoded), dtype="float32")
    assert rate == 16000 and np.array_equal(x, decoded)
    info = sf.info(io.BytesIO(encoded))
    assert info.channels == 1 and info.subtype == "FLOAT"
    return encoded, decoded


def spectrum(x):
    f, t, z = stft(x, fs=16000, window="hann", nperseg=512, noverlap=384,
                   nfft=512, boundary="zeros", padded=False, scaling="spectrum")
    db = 20 * np.log10(np.maximum(np.abs(z), 10 ** (-90 / 20)))
    fig, ax = plt.subplots(figsize=(4.6, 2.4))
    image = ax.pcolormesh(t, f / 1000, db, shading="auto", cmap="magma", vmin=-90, vmax=0)
    ax.set(xlabel="Time (s)", ylabel="Frequency (kHz)", xlim=(0, len(x) / 16000), ylim=(0, 8))
    fig.colorbar(image, ax=ax, label="Magnitude (dB; reference amplitude = 1)")
    fig.tight_layout()
    stream = io.BytesIO()
    fig.savefig(stream, format="png", dpi=100)
    plt.close(fig)
    raw = stream.getvalue()
    with Image.open(io.BytesIO(raw)) as decoded:
        decoded.verify()
    return raw


def training_figure(validation):
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot([r["step"] for r in validation], [r["score"] for r in validation], color="black")
    ax.set(xlabel="Optimizer updates (count)", ylabel="Composite validation loss (dimensionless)",
           title="Fixed validation excerpts")
    fig.tight_layout()
    stream = io.BytesIO(); fig.savefig(stream, format="png", dpi=110); plt.close(fig)
    return stream.getvalue()


def num(value, digits=4):
    return "Not available" if value is None else f"{value:.{digits}f}"


class PortableAudit(HTMLParser):
    def __init__(self):
        super().__init__(); self.audio = 0; self.images = 0; self.case_tables = 0
        self.external_assets = []; self.ids = []; self.links = []; self.style = 0

    def handle_starttag(self, tag, attributes):
        a = dict(attributes)
        if tag == "table" and a.get("data-case"):
            self.case_tables += 1
        if tag == "style": self.style += 1
        if "id" in a: self.ids.append(a["id"])
        if a.get("href", "").startswith("#"): self.links.append(a["href"][1:])
        if tag in {"audio", "img"}:
            src = a.get("src", "")
            if not src.startswith("data:"):
                self.external_assets.append(src); return
            raw = base64.b64decode(src.split(",", 1)[1], validate=True)
            if tag == "audio":
                info = sf.info(io.BytesIO(raw))
                assert info.samplerate == 16000 and info.channels == 1 and info.subtype == "FLOAT"
                sf.read(io.BytesIO(raw), dtype="float32")
                self.audio += 1
            else:
                with Image.open(io.BytesIO(raw)) as image: image.verify()
                self.images += 1
        if tag in {"script", "link", "iframe", "source"}:
            address = a.get("src") or a.get("href")
            if address and not address.startswith("data:"): self.external_assets.append(address)


def main():
    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = read(EVAL / "protocol.json")
    completion = read(RUN / "completion.json")
    validation = [json.loads(line) for line in (RUN / "validation.jsonl").read_text().splitlines()]
    saved_summary = read(EVAL / "summary.json")
    environment = read(RUN / "environment.json")
    old_metrics = list(csv.DictReader((EVAL / "metrics.csv").open(newline="", encoding="utf-8")))
    lookup = {(row["utterance"], row["mode"]): row for row in old_metrics}
    selected = protocol["selection"]
    expected = {(row["utterance"], mode) for row in selected for mode in protocol["modes"]}
    assert set(lookup) == expected and len(old_metrics) == len(expected)
    model_path = ROOT / "approaches/representations/adaptive_stft_dictionary.py"
    assert sha(model_path) == protocol["model_source_sha256"]
    checkpoint = RUN / "checkpoints/best.pt"
    assert sha(checkpoint) == protocol["checkpoint_sha256"]
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert state["step"] == protocol["checkpoint_step"]
    spec = importlib.util.spec_from_file_location("adaptive_saved_payload_decoder", ROOT / "tools/evaluate_adaptive_stft_dictionary.py")
    decoder = importlib.util.module_from_spec(spec); sys.modules[spec.name] = decoder; spec.loader.exec_module(decoder)
    build_revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    record = dict(run_id=RUN.name, training_git_revision="Not recorded in the saved training artifacts",
                  report_build_git_revision=build_revision, checkpoint_step=state["step"],
                  checkpoint_path=str(checkpoint.relative_to(ROOT)), checkpoint_sha256=sha(checkpoint),
                  model_source_sha256=sha(model_path), report_builder_sha256=sha(Path(__file__)),
                  saved_evaluation_source_sha256=protocol["evaluation_source_sha256"],
                  current_decoder_source_sha256=sha(ROOT / "tools/evaluate_adaptive_stft_dictionary.py"),
                  training_environment=environment, evaluation_environment=protocol["environment"],
                  metric_versions=protocol["metric_versions"], seed=protocol["training_settings"]["seed"],
                  settings=protocol["training_settings"], selection_rule="Hash-order speakers and utterances; round-robin speakers; deterministic hashed excerpt offsets",
                  source_split="test-clean", selected_utterances=len(selected), speakers=len({r["speaker"] for r in selected}),
                  selected_cases=len(expected), report_specification="documentation/REPORTS_README.md",
                  case_definition="One utterance ID plus one inference mode", selection=selected)
    full = OUT / f"{PREFIX}_testcases.html"
    rows = []; max_score_difference = 0.; examples_completed = 0
    started = time.monotonic()
    with full.open("w", encoding="utf-8") as report:
        def write(s): report.write(s + "\n")
        write("<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>Adaptive STFT dictionary: completed run and test results</title></head><body>")
        write("<h1>Adaptive STFT dictionary: completed run and test results</h1>")
        write("<p>STFT means short-time Fourier transform. This report describes the completed training run and its fixed test group.</p><p>All required audio, images, and records are inside this file.</p>")
        write("<h2>Scope and limits</h2><p>The group contains 64 test-clean utterances from 40 speakers. Each utterance has two cases: a one-second excerpt and a complete recording.</p>")
        write("<p>The report contains 128 selected cases. All cases completed inference. No selected case was excluded. Both metrics returned valid scores for all cases.</p>")
        write("<p>This is a selected test group. It is not the complete test-clean partition. Training used train-clean-360. Checkpoint selection used fixed dev-clean excerpts.</p>")
        write("<p>The model trained on one-second excerpts. Complete-recording inference extends the input context. It uses one dictionary per recording and no excerpt stitching.</p>")
        write("<p>PESQ means perceptual evaluation of speech quality. STOI means short-time objective intelligibility. Both compare reconstructed audio with source audio.</p>")
        write("<p>WB means wideband. PESQ uses wideband mode at 16 kHz. STOI uses standard mode at 16 kHz.</p><p>A score alone does not establish listening quality or intelligibility.</p>")
        write("<h2>Mean scores and rates</h2><p>Score means are arithmetic means across the 64 cases in each mode. Rate totals use all payload bits divided by total source duration.</p>")
        write("<p>PCM means pulse-code modulation. Playback uses the float32 PCM representation.</p>")
        write("<table border='1' cellpadding='6'><tr><th>Mode</th><th>Cases</th><th>PESQ-WB</th><th>STOI</th><th>Raw payload rate (kbit/s)</th><th>Playback PCM rate (kbit/s)</th></tr>")
        for mode, s in saved_summary.items():
            write(f"<tr><td>{mode}</td><td>{s['examples']}</td><td>{s['metrics']['pesq_wb']['mean']:.4f}</td><td>{s['metrics']['stoi']['mean']:.4f}</td><td>{s['duration_weighted_payload_kbps']:.3f}</td><td>512.000</td></tr>")
        write("</table><h2>Rate definitions</h2>")
        write("<p>One kbit/s means 1,000 bit/s. Playback uses mono 16 kHz WAV audio with 32-bit floating-point samples.</p>")
        write("<p>Playback PCM rate = 16,000 samples/s times 32 bits/sample times one channel = 512 kbit/s. This excludes WAV headers.</p>")
        write("<p>JSON means JavaScript Object Notation. ASD1 is the saved payload format.</p><p>ASD1 files contain an uncompressed float32 representation. Their rate includes the dictionary, coefficients, gain, format marker, length field, and JSON header.</p>")
        write("<p>Each complex value has two real scalars. The dictionary and coefficients are recording-specific. Their storage is included.</p>")
        write("<p>Raw payload rate = ASD1 file bytes times 8 divided by source duration. This is a measured serialized rate.</p>")
        write("<p>A compressed codec bitstream is not available. No entropy coding or quantization is applied. No pitch stream or codebook is required.</p>")
        write("<p>The rate excludes program storage, file-system overhead, and training checkpoints. The standalone decoder needs no learned model weights.</p>")
        write("<p>For comparison, mono PCM16 at 16 kHz requires 256 kbit/s. Both measured raw payload rates exceed this reference.</p>")
        write("<h2>Training evidence</h2>")
        write(f"<p>Training completed {completion['step']:,} updates. Early stopping was disabled. The best checkpoint occurs at update {state['step']:,}.</p>")
        write(f"<p>The composite validation loss decreased from {validation[0]['score']:.4f} to {completion['best_validation']:.4f}. Lower values are better for this loss.</p>")
        write("<p>This loss combines complex STFT error, waveform L1 error, and multiresolution spectral error. It is not a perceptual score.</p>")
        write(f"<p>Active training and validation time was {completion['elapsed_active_seconds']/60:.2f} min. This excludes the initial data scan.</p>")
        write(f"<img alt='Composite validation loss across optimizer updates' width='660' height='330' src='{embed(training_figure(validation),'image/png')}'>")
        write("<h2>Test method</h2><p>The test loads best.pt. Its step and SHA-256 are verified against the saved evaluation protocol.</p>")
        write("<p>The model has 32 complex spectral atoms, 128 encoder features, four attention heads, and four joint update stages.</p>")
        write("<p>It uses four initial coefficient steps. Dictionary and coefficient correction networks share their weights across the update stages.</p>")
        write("<p>STFT means short-time Fourier transform. The model uses a 512-sample periodic Hann window and a 128-sample hop.</p>")
        write("<p>The analysis uses centered frames and constant boundary padding. The inverse restores the original sample count and stored RMS gain.</p>")
        write("<p>RMS means root mean square. One gain applies to both complex channels. The test applies no resampling, time alignment, or output gain fit.</p>")
        write("<p>Each payload is decoded independently. The exact embedded FLOAT WAV samples are used to calculate the displayed metrics again.</p>")
        write("<p>There is no vocoder. The ASD1 decoder uses the inverse STFT. The method record gives the software versions and source hashes.</p>")
        write("<p>FFT means fast Fourier transform. All spectrograms use a 512-sample periodic Hann window, a 128-sample hop, and a 512-point FFT.</p>")
        write("<p>The plots use zero boundary extension and spectrum scaling. Magnitude is divided by the window sum.</p>")
        write("<p>The magnitude reference amplitude is 1. All plots use -90 to 0 dB and 0 to 8 kHz.</p>")
        write("<p>Before and after plots use the same time range for each case. Their axes show seconds, kilohertz, and decibels.</p>")
        write("<h2>Test cases</h2><p>Before is the source audio. After is the independently decoded ASD1 audio. No additional control audio was saved for this group.</p>")
        for selected_row in selected:
            path = Path(protocol["dataset_root"]) / selected_row["relative"]
            source_sha = sha(path)
            source, rate = sf.read(path, dtype="float32")
            assert rate == 16000 and source.ndim == 1
            for mode in protocol["modes"]:
                old = lookup[(selected_row["utterance"], mode)]
                assert source_sha == old["source_sha256"]
                offset = selected_row["offset"] if mode == "excerpt" else 0
                before = source[offset:offset+16000] if mode == "excerpt" else source
                payload = EVAL / "payloads" / f"{selected_row['utterance']}_{mode}.asd"
                after, header = decoder.decode(payload)
                assert len(before) == len(after) == header["samples"]
                bwav, before = wav_bytes(before); awav, after = wav_bytes(after)
                scores = decoder.quality(before, after, 16000)
                for key in ("pesq_wb", "stoi"):
                    if old[key]:
                        assert scores[key] is not None, scores
                        difference = abs(scores[key]-float(old[key])); max_score_difference = max(max_score_difference,difference)
                        assert difference < 1e-6, (selected_row["utterance"],mode,key,difference)
                    else:
                        assert scores[key] is None
                duration = len(before)/16000
                size = payload.stat().st_size
                scalars = 1+2*header["bins"]*header["atoms"]+2*header["atoms"]*header["frames"]
                header_bytes = size-4*scalars
                assert header_bytes > 8 and size == int(old["payload_bytes"])
                rate_kbps = size*8/duration/1000
                assert abs(rate_kbps-float(old["raw_payload_kbps"])) < 1e-8
                row = dict(case_id=f"{selected_row['utterance']}:{mode}", utterance=selected_row["utterance"],
                           speaker=selected_row["speaker"], mode=mode, status="complete", offset_samples=offset,
                           samples=len(before), source_seconds=duration, output_seconds=len(after)/16000,
                           sample_rate=16000, channels=1, playback_format="FLOAT WAV (32-bit)",
                           playback_pcm_kbps=512, raw_payload_bytes=size, header_bytes=header_bytes,
                           raw_payload_kbps=rate_kbps, dictionary_scalars=int(old["dictionary_scalars"]),
                           coefficient_scalars=int(old["coefficient_scalars"]), gain_scalars=1,
                           source_sha256=source_sha,payload_sha256=sha(payload),before_wav_sha256=hashlib.sha256(bwav).hexdigest(),
                           after_wav_sha256=hashlib.sha256(awav).hexdigest(), **scores)
                rows.append(row)
                case = len(rows)
                write(f"<h3 id='case-{case}'>Case {case}: {html.escape(row['utterance'])}; {mode}</h3>")
                write(f"<p>Speaker: {row['speaker']}. Status: complete. Source duration: {duration:.5f} s. Output duration: {duration:.5f} s.</p>")
                write(f"<p>Source start: {offset} samples ({offset/16000:.5f} s). Sample count: {len(before)}. Sample rate: 16,000 Hz.</p>")
                write(f"<table border='1' cellpadding='6' data-case='{case}'><tr><th>Item</th><th>Before</th><th>After</th><th>Control, if available</th></tr>")
                write(f"<tr><th>Audio</th><td><audio controls preload='none' src='{embed(bwav,'audio/wav')}'></audio></td><td><audio controls preload='none' src='{embed(awav,'audio/wav')}'></audio></td><td>Not available</td></tr>")
                write(f"<tr><th>Spectrogram</th><td><img width='460' height='240' alt='Source spectrogram for case {case}' src='{embed(spectrum(before),'image/png')}'></td><td><img width='460' height='240' alt='Output spectrogram for case {case}' src='{embed(spectrum(after),'image/png')}'></td><td>Not available</td></tr>")
                for key,label in (("pesq_wb","PESQ-WB"),("stoi","STOI")):
                    value = num(scores[key]) if scores[key] is not None else "Not available: "+html.escape(scores[key+"_status"])
                    write(f"<tr><th>{label}</th><td>Reference</td><td>{value}</td><td>Not available</td></tr>")
                write("<tr><th>Playback PCM data rate</th><td>512.000 kbit/s</td><td>512.000 kbit/s</td><td>Not available</td></tr>")
                write(f"<tr><th>Encoded data rate: raw ASD1 payload</th><td>Not applicable</td><td>{rate_kbps:.3f} kbit/s; uncompressed</td><td>Not available</td></tr>")
                write("<tr><th>Compressed codec bitstream rate</th><td>Not applicable</td><td>Not available</td><td>Not available</td></tr></table>")
                write(f"<p>Observation: the decoded audio gives PESQ-WB {num(scores['pesq_wb'])} and STOI {num(scores['stoi'])}. These values apply to this case.</p>")
                write("<p>Possible inference: differences can involve phase, atom representation, or coefficient estimation. This table does not identify the cause.</p>")
            examples_completed += 1
            print(f"Compiled {examples_completed}/{len(selected)} utterances; {len(rows)} case tables",flush=True)
        write("<h2>Observations and unresolved questions</h2>")
        write("<p>The validation loss decreased during training. The test scores show remaining differences from the source. Listening assessment remains necessary.</p>")
        write("<p>Complete-recording scores are lower on average than excerpt scores. This comparison changes both input context and evaluated audio duration.</p>")
        write("<p>The cause is not confirmed. The current evidence does not isolate a context-length effect.</p>")
        write("<p>The full-recording payload has a lower aggregate rate because one dictionary serves a longer signal. The dictionary size remains constant.</p>")
        write("<p>The raw payload exceeds the PCM16 reference rate. These results do not establish compression.</p>")
        write("<p>Next action: compare stronger coefficient inference using the saved atoms. Then evaluate a fixed-dictionary model with the same test group.</p>")
        write("<h2>Method record</h2><p>The original training artifacts do not record a Git revision. The script SHA-256 identifies the actual model source.</p>")
        write("<p>The report-build revision identifies the current checkout. It is not a claim about the training revision.</p>")
        write("<pre>"+html.escape(json.dumps(record,indent=2))+"</pre>")
        write("<h2>Embedded metric records</h2><p>The following records contain each case and its measured rates and scores.</p>")
        write("<script type='application/json' id='metric-records'>"+json.dumps(rows,ensure_ascii=True).replace("<","\\u003c")+"</script>")
        write("<p>The text follows the ASD-STE100 writing target. No certified language compliance is claimed.</p></body></html>")
    metrics_path = OUT / f"{PREFIX}_metrics.json"
    metrics_path.write_text(json.dumps(dict(method=record, summary=saved_summary,cases=rows),indent=2,allow_nan=False),encoding="utf-8")
    audit = PortableAudit()
    with full.open(encoding="utf-8") as stream:
        for chunk in iter(lambda:stream.read(65536),""): audit.feed(chunk)
    assert audit.case_tables == len(expected)
    assert audit.audio == 2*len(expected) and audit.images == 2*len(expected)+1
    assert not audit.external_assets and audit.style == 0
    assert not set(audit.links)-set(audit.ids)
    checks = dict(selected_utterances=len(selected),selected_cases=len(expected),completed_cases=len(rows),
                  failed_cases=0,excluded_selected_cases=0,case_tables=audit.case_tables,
                  audio_clips=audit.audio,spectrograms=2*len(rows),other_figures=1,
                  source_and_output_samples_verified=True,all_embedded_audio_decoded=True,
                  all_embedded_images_decoded=True,metrics_recomputed_from_embedded_samples=True,
                  maximum_score_difference_from_saved_metrics=max_score_difference,
                  all_raw_rates_checked=True,external_asset_dependencies=audit.external_assets,
                  style_elements=audit.style,html_bytes=full.stat().st_size,html_sha256=sha(full),
                  browser_visual_check="pending",native_audio_controls_present=True,
                  audible_playback_check="Not performed; audio bytes decoded, but no listening assessment was made",
                  build_seconds=time.monotonic()-started)
    (OUT / f"{PREFIX}_verification.json").write_text(json.dumps(checks,indent=2),encoding="utf-8")
    print(json.dumps(checks,indent=2))
    print(full)


if __name__ == "__main__":
    main()
