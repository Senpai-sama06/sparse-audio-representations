"""Score the seven SAE checkpoints on the fixed PitchFlower dev-clean crops.

This is a reproduction comparison set, not a fresh test set: the SAE campaign
used dev-clean speakers for validation/calibration. The report reference clips
are normalized PitchFlower PCM crops resampled to 16 kHz.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from sae.metrics import audio_metrics, summarize
from sae.models import AudioSAE, ModelConfig


METHODS = ("l1", "kl", "topk", "jumprelu", "gated", "batchtopk", "dense")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True, help="Folder containing cases.json and float WAV references")
    parser.add_argument("--campaign", required=True, help="Completed campaign directory containing jobs/<method>/attempt_01")
    parser.add_argument("--output", required=True, help="New evaluation output folder")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args(argv)
    inputs, campaign, output = map(lambda value: Path(value).resolve(),
                                   (args.inputs, args.campaign, args.output))
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    if output.exists() and any(output.iterdir()):
        parser.error(f"Output is not empty: {output}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is not available")
    torch.set_num_threads(2)
    cases = json.loads((inputs / "cases.json").read_text(encoding="utf-8"))
    if len(cases) != 40 or len({case["id"] for case in cases}) != 40:
        parser.error("Expected exactly 40 unique reference cases")
    references = []
    for case in cases:
        audio, rate = sf.read(inputs / case["path"], dtype="float32", always_2d=True)
        if rate != 16000 or audio.shape[1] != 1 or len(audio) != case["samples"]:
            raise ValueError(f"Invalid reference WAV for {case['id']}")
        if not np.isfinite(audio).all():
            raise ValueError(f"Nonfinite reference audio for {case['id']}")
        references.append(audio[:, 0])

    output.mkdir(parents=True, exist_ok=True)
    by_method = {}
    for method in METHODS:
        run = campaign / "jobs" / method / "attempt_01"
        checkpoint = run / "best_weights.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        config = ModelConfig(**payload["model_config"])
        if config.method != method:
            raise ValueError(f"Checkpoint method mismatch for {method}: {config.method}")
        model = AudioSAE(config)
        model.load_state_dict(payload["model_state"], strict=True)
        model.to(device).eval()
        method_dir = output / method
        method_dir.mkdir()
        rows = []
        with torch.inference_mode():
            for start in range(0, len(cases), args.batch_size):
                batch_cases = cases[start:start + args.batch_size]
                batch_refs = references[start:start + args.batch_size]
                lengths = torch.tensor([len(ref) for ref in batch_refs], device=device)
                wave = torch.from_numpy(np.stack(batch_refs))[:, None].to(device)
                result = model(wave, lengths)
                for offset, (case, reference) in enumerate(zip(batch_cases, batch_refs)):
                    estimate = result["reconstruction"][offset, 0, :len(reference)].float().cpu().numpy()
                    if not np.isfinite(estimate).all():
                        raise ValueError(f"Nonfinite reconstruction: {method}/{case['id']}")
                    code = result["z"][offset, result["valid"][offset]]
                    filename = f"{start + offset:02d}_{case['id'].replace(':', '_')}_reconstruction.wav"
                    output_path = method_dir / filename
                    sf.write(output_path, estimate, 16000, subtype="PCM_16")
                    playback, playback_rate = sf.read(output_path, dtype="float32", always_2d=True)
                    if playback_rate != 16000 or playback.shape != (len(reference), 1):
                        raise ValueError(f"Invalid PCM16 playback WAV: {method}/{case['id']}")
                    record = {
                        "id": case["id"], "speaker": str(case["speaker"]),
                        "samples": len(reference), "duration_seconds": len(reference) / 16000,
                        "active_coefficients": int((code != 0).sum().item()),
                        **audio_metrics(reference, playback[:, 0], speech=True),
                    }
                    record["reconstruction_wav"] = f"{method}/{filename}"
                    rows.append(record)
                print(f"{method}: {min(start + len(batch_cases), len(cases))}/{len(cases)}", flush=True)
        history_path = run / "history.csv"
        with history_path.open(newline="", encoding="utf-8") as stream:
            history = list(csv.DictReader(stream))
        best = json.loads((run / "summary.json").read_text(encoding="utf-8"))
        by_method[method] = {
            "cases": rows,
            "summary": summarize(rows),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256(checkpoint),
            "best_epoch": best.get("best_epoch"),
            "stop_reason": best.get("stop_reason"),
            "history_rows": len(history),
            "model_config": payload["model_config"],
        }
        del model, payload

    result = {
        "title": "Sparse autoencoders on PitchFlower's fixed LibriSpeech dev-clean crops",
        "cases": cases,
        "methods": by_method,
        "evaluation": {
            "sample_rate": 16000,
            "crop_samples": sorted({case["samples"] for case in cases}),
            "crop_duration_seconds": sorted({case["samples"] / 16000 for case in cases}),
            "reference": "PitchFlower report embedded PCM16 source crop at 24 kHz, decoded and resampled to 16 kHz",
            "speech_metrics": "PESQ-WB and STOI; SI-SDR, NMSE, MSE computed by sae.metrics",
            "bitrate": None,
            "bitrate_note": "These are unquantized reconstructions; no operational bitstream or bitrate was measured.",
            "test_status": "Historical dev-clean comparison only. All dev-clean speakers were used by the SAE campaign for validation or BatchTopK calibration; not a fresh held-out test.",
            "normalization_note": "The reference crops follow PitchFlower's independent peak normalization; SAE training used native LibriSpeech amplitude. This fixed-set inference comparison has an input-gain distribution difference for SAE.",
        },
    }
    (output / "scores.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (output / "scores.csv").open("w", newline="", encoding="utf-8") as stream:
        keys = ["method", "id", "speaker", "samples", "active_coefficients", "mse", "nmse", "si_sdr", "pesq_wb", "stoi", "statuses", "reconstruction_wav"]
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        for method, data in by_method.items():
            for row in data["cases"]:
                writer.writerow({"method": method, **{key: json.dumps(row[key]) if key == "statuses" else row[key]
                                                       for key in keys if key != "method"}})
    print(json.dumps({method: data["summary"] for method, data in by_method.items()}, indent=2), flush=True)


if __name__ == "__main__":
    main()
