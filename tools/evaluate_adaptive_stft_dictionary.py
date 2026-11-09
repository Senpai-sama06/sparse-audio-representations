"""Measure an adaptive dictionary's raw payload rate and decoded PESQ/STOI.

No training or dataset downloads. The frozen test selection is saved before
inference. Both modes use the same utterances: excerpt uses the saved one-second
crop protocol; utterance applies the fully convolutional model to the complete
recording, with one dictionary and no excerpt stitching. Utterance mode is an
inference-context extension beyond training, not the matched training protocol.

The ASD1 payload stores little-endian float32 gain, complex dictionary, complex
coefficients and a JSON header. Both complex parts are counted. No quantization,
entropy coding or shared neural decoder is used. Decode needs only this file,
NumPy, PyTorch and SoundFile; it does not load a model checkpoint or source audio.

  python tools/evaluate_adaptive_stft_dictionary.py --run RUN --examples 64
  python tools/evaluate_adaptive_stft_dictionary.py --decode FILE.asd --wav OUT.wav
"""
from __future__ import annotations

import argparse
import base64
import csv
from dataclasses import asdict, replace
import hashlib
import html
import importlib.util
import io
import json
from pathlib import Path
import struct
import sys
import time
import warnings

import numpy as np
import soundfile as sf
import torch


def encode(path, result, cfg, samples):
    d = result.dictionaries[-1][0].detach().cpu().numpy()
    a = result.coefficients[-1][0].detach().cpu().numpy()
    gain = result.gain.detach().cpu().numpy().reshape(-1)
    header = dict(format="ASD1", dtype="little-endian float32", rate=cfg.sample_rate,
                  samples=samples, n_fft=cfg.n_fft, hop=cfg.hop, bins=d.shape[0],
                  atoms=d.shape[1], frames=a.shape[1], center=True,
                  window="periodic Hann", padding="constant", normalized=False)
    raw = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    with path.open("wb") as stream:
        stream.write(b"ASD1" + struct.pack("<I", len(raw)) + raw)
        for values in (gain, d.real, d.imag, a.real, a.imag):
            stream.write(np.asarray(values, dtype="<f4").tobytes(order="C"))
    return header


@torch.no_grad()
def decode(path):
    raw = path.read_bytes()
    if raw[:4] != b"ASD1" or len(raw) < 8:
        raise ValueError("Invalid ASD1 payload")
    header_length = struct.unpack("<I", raw[4:8])[0]
    h = json.loads(raw[8:8 + header_length])
    f, k, t = h["bins"], h["atoms"], h["frames"]
    values = np.frombuffer(raw, dtype="<f4", offset=8 + header_length).copy()
    if len(values) != 1 + 2 * f * k + 2 * k * t:
        raise ValueError("Payload scalar count differs from header")
    gain = values[0]
    offset = 1
    dr, di = values[offset:offset + f*k], values[offset + f*k:offset + 2*f*k]
    offset += 2*f*k
    ar, ai = values[offset:offset + k*t], values[offset + k*t:]
    d = torch.complex(torch.from_numpy(dr.reshape(f, k)), torch.from_numpy(di.reshape(f, k)))
    a = torch.complex(torch.from_numpy(ar.reshape(k, t)), torch.from_numpy(ai.reshape(k, t)))
    s = d @ a
    imag = s.imag.clone()
    imag[0] = imag[-1] = 0
    s = torch.complex(s.real, imag)
    wave = torch.istft(s, h["n_fft"], hop_length=h["hop"], win_length=h["n_fft"],
                       window=torch.hann_window(h["n_fft"], periodic=True), center=True,
                       normalized=False, onesided=True, length=h["samples"]) * float(gain)
    return wave.numpy(), h


def quality(reference, estimate, rate):
    from pesq import pesq
    from pystoi import stoi
    output = {}
    for name, operation in (("pesq_wb", lambda: pesq(rate, reference, estimate, "wb")),
                            ("stoi", lambda: stoi(reference, estimate, rate, extended=False))):
        try:
            with warnings.catch_warnings(record=True) as emitted:
                warnings.simplefilter("always")
                value = float(operation())
            if emitted:
                raise ValueError("; ".join(str(w.message) for w in emitted))
            if not np.isfinite(value):
                raise ValueError("Nonfinite result")
            output[name] = value
            output[name + "_status"] = "ok"
        except Exception as error:
            output[name] = None
            output[name + "_status"] = f"{type(error).__name__}: {error}"
    return output


def summary(rows):
    output = {}
    for mode in sorted({row["mode"] for row in rows}):
        group = [row for row in rows if row["mode"] == mode]
        metrics = {}
        for key in ("pesq_wb", "stoi", "nmse", "si_sdr_db", "raw_payload_kbps", "encode_seconds"):
            valid = [row[key] for row in group if row.get(key) is not None and np.isfinite(row[key])]
            metrics[key] = dict(mean=float(np.mean(valid)) if valid else None,
                                median=float(np.median(valid)) if valid else None,
                                valid=len(valid), failed=len(group)-len(valid))
        output[mode] = dict(examples=len(group), seconds=sum(row["seconds"] for row in group),
                            duration_weighted_payload_kbps=sum(row["payload_bytes"]*8 for row in group)/sum(row["seconds"] for row in group)/1000,
                            pcm16_kbps=256, metrics=metrics)
    return output


def audio_uri(wave, rate):
    stream = io.BytesIO()
    sf.write(stream, wave, rate, format="WAV", subtype="FLOAT")
    return "data:audio/wav;base64," + base64.b64encode(stream.getvalue()).decode()


def report(out, rows, examples, settings):
    totals = summary(rows)
    (out / "summary.json").write_text(json.dumps(totals, indent=2, allow_nan=False), encoding="utf-8")
    with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    parts = ["<!doctype html><html lang='en'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>",
             "<title>Adaptive STFT dictionary: rate and quality</title><body><h1>Payload rate and speech quality</h1>",
             "<p>The frozen best checkpoint is evaluated on a fixed test subset. No training occurs.</p>",
             "<p>Excerpt mode uses the one-second training context. Utterance mode uses each complete recording and one dictionary. It has no excerpt stitching. This is an extension beyond the training context.</p>",
             "<p>The rate is an actual uncompressed ASD1 file rate. It includes the float32 dictionary, complex coefficients, gain and header. No entropy coding is applied. It is not a compressed-codec bitrate.</p>",
             "<p>Metrics use independently decoded payload audio at 16 kHz. No alignment or amplitude fitting is applied. PESQ uses wideband mode; STOI uses extended=False. Invalid scores are excluded with counts and reasons in the CSV.</p>",
             "<p>PCM16 at 16 kHz mono is 256 kb/s. The payload decoder needs no source audio or learned model weights.</p>",
             "<h2>Summary</h2><pre>" + html.escape(json.dumps(totals, indent=2)) + "</pre><h2>Listening</h2>"]
    for example in examples:
        parts.append(f"<h3>{html.escape(example['id'])}: {example['mode']}</h3>")
        for label in ("original", "decoded"):
            parts.append(f"<p>{label}</p><audio controls preload='none' src='{audio_uri(example[label],16000)}'></audio>")
    parts.append("<h2>Protocol</h2><pre>" + html.escape(json.dumps(settings, indent=2)) + "</pre></body></html>")
    (out / "report.html").write_text("\n".join(parts), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--examples", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--decode", type=Path)
    parser.add_argument("--wav", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.decode:
        if not args.wav:
            parser.error("--decode requires --wav")
        wave, metadata = decode(args.decode)
        sf.write(args.wav, wave, metadata["rate"], subtype="FLOAT")
        return
    if not args.run or args.examples < 1:
        parser.error("Supply --run and a positive --examples")
    # Metrics must be installed before any output or model evaluation starts.
    import pesq, pystoi
    script = Path(__file__).resolve().parents[1] / "approaches/representations/adaptive_stft_dictionary.py"
    spec = importlib.util.spec_from_file_location("adaptive_stft_quality_model", script)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    run = args.run.resolve()
    state_path = run / "checkpoints/best.pt"
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    cfg = module.Settings(**state["settings"])
    if state["source_sha256"] != module.file_hash(script):
        raise ValueError("Training script differs from checkpoint")
    manifest = json.loads((run / "data_manifest.json").read_text())
    if state["data_fingerprint"] != manifest["fingerprint"]:
        raise ValueError("Data identity differs")
    selected = module.select_excerpts(manifest["records"][cfg.test_split], args.examples, cfg)
    out = args.output or run / f"bitrate_quality_test_{len(selected)}"
    out.mkdir(parents=True, exist_ok=False)
    (out / "payloads").mkdir()
    protocol = dict(checkpoint_sha256=module.file_hash(state_path), checkpoint_step=state["step"],
                    training_settings=asdict(cfg), modes=["excerpt", "utterance"], selection=selected,
                    dataset_root=manifest["root"], model_source_sha256=module.file_hash(script),
                    evaluation_source_sha256=module.file_hash(Path(__file__)), environment=module.environment(),
                    metric_versions={"pesq": module.importlib.metadata.version("pesq"), "pystoi": module.importlib.metadata.version("pystoi")})
    module.atomic_json(out / "protocol.json", protocol)
    model = module.AdaptiveSTFTDictionary(cfg).to(args.device).eval()
    model.load_state_dict(state["model"])
    # Same package call, unmodified input: an explicit metric sanity control.
    first = selected[0]
    original, _ = sf.read(Path(manifest["root"]) / first["relative"], dtype="float32")
    protocol["identity_quality_control"] = quality(original, original.copy(), cfg.sample_rate)
    if protocol["identity_quality_control"]["pesq_wb"] is None or protocol["identity_quality_control"]["stoi"] is None:
        raise RuntimeError("Identity metric control failed")
    module.atomic_json(out / "protocol.json", protocol)
    rows, examples = [], []
    started = time.monotonic()
    for index, record in enumerate(selected):
        path = Path(manifest["root"]) / record["relative"]
        if path.stat().st_size != record["bytes"] or path.stat().st_mtime_ns != record["mtime_ns"]:
            raise ValueError(f"Source changed: {path}")
        full, rate = sf.read(path, dtype="float32")
        if full.ndim != 1 or rate != 16000 or len(full) != record["frames"]:
            raise ValueError(f"Invalid source: {path}")
        for mode in ("excerpt", "utterance"):
            x = full[record["offset"]:record["offset"]+cfg.samples] if mode == "excerpt" else full
            # Only the length contract changes; weights, operations and update counts stay fixed.
            model.cfg = replace(cfg, seconds=len(x)/cfg.sample_rate)
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            tic = time.monotonic()
            with torch.no_grad():
                result = model(torch.from_numpy(x.copy())[None].to(args.device))
            payload = out / "payloads" / f"{record['utterance']}_{mode}.asd"
            encode(payload, result, cfg, len(x))
            elapsed = time.monotonic()-tic
            reconstructed, metadata = decode(payload)
            native = result.audio[0].detach().cpu().numpy()
            decoder_nmse = float(np.sum((native.astype(float)-reconstructed)**2)/max(np.sum(native.astype(float)**2),1e-12))
            if decoder_nmse > 1e-9:
                raise ValueError(f"Independent decoder differs: {decoder_nmse}")
            duration = len(x)/rate
            d_values = 2*metadata["bins"]*metadata["atoms"]
            a_values = 2*metadata["atoms"]*metadata["frames"]
            row = dict(utterance=record["utterance"], speaker=record["speaker"], mode=mode,
                       seconds=duration, payload_bytes=payload.stat().st_size,
                       raw_payload_kbps=payload.stat().st_size*8/duration/1000,
                       dictionary_scalars=d_values, coefficient_scalars=a_values, gain_scalars=1,
                       encode_seconds=elapsed, decoder_vs_native_nmse=decoder_nmse,
                       source_sha256=module.file_hash(path), **quality(x,reconstructed,rate),
                       **module.waveform_metrics(torch.from_numpy(x),torch.from_numpy(reconstructed),cfg.n_fft//2))
            rows.append(row)
            if index < 3:
                examples.append(dict(id=record["utterance"], mode=mode, original=x.copy(), decoded=reconstructed.copy()))
            del result
        module.atomic_json(out / "progress.json", dict(status="evaluating", utterances_completed=index+1,
                           total_utterances=len(selected), elapsed_seconds=time.monotonic()-started))
        module.append_json(out / "rows.jsonl", {"rows": rows[-2:]})
        print(f"{index+1}/{len(selected)} {record['utterance']} PESQ={rows[-1]['pesq_wb']} STOI={rows[-1]['stoi']}",flush=True)
    report(out,rows,examples,protocol)
    module.atomic_json(out / "progress.json", dict(status="complete", utterances_completed=len(selected),
                       total_utterances=len(selected), elapsed_seconds=time.monotonic()-started))
    print(json.dumps(summary(rows),indent=2))
    print(f"Report: {out / 'report.html'}",flush=True)


if __name__ == "__main__":
    main()
