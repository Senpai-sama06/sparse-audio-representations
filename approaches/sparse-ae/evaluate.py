"""Evaluate exported best weights on full utterances; no bitrate claim."""
import argparse
import csv
import json
from pathlib import Path

import soundfile as sf
import torch

from sae.data import SAMPLE_RATE, SpeechDataset, digest, speaker_limit, write_json
from sae.metrics import audio_metrics, speech_dependencies, summarize
from sae.models import AudioSAE, ModelConfig


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="best_weights.pt, not the optimizer/RNG checkpoint")
    parser.add_argument("--manifest", required=True, help="Prepared test.jsonl, validation.jsonl, or other declared split")
    parser.add_argument("--output", required=True, help="New/empty evaluation directory")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0, help="0 evaluates all; otherwise speaker-round-robin")
    parser.add_argument("--save-audio", type=int, default=8, help="Number of first cases with reference/reconstruction WAVs")
    parser.add_argument("--speech-metrics", action="store_true")
    args = parser.parse_args(argv)
    if args.limit < 0 or args.save_audio < 0 or args.threads < 1:
        parser.error("limits must be nonnegative and threads positive")
    if args.speech_metrics:
        speech_dependencies()
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("Evaluation output is not empty; choose a new directory")
    torch.set_num_threads(args.threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = AudioSAE(ModelConfig(**payload["model_config"]))
    model.load_state_dict(payload["model_state"])
    model.to(device).eval()
    manifest = Path(args.manifest).resolve()
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        parser.error("Evaluation manifest is empty")
    rows = speaker_limit(rows, args.limit)
    for row in rows:
        if digest(row["path"]) != row["sha256"]:
            raise ValueError(f"Evaluation audio changed: {row['path']}")
    output.mkdir(parents=True, exist_ok=True)
    dataset = SpeechDataset(rows)
    records = []
    with torch.no_grad():
        for index in range(len(dataset)):
            audio, identifier = dataset[index]
            x = audio[None].to(device)
            result = model(x, torch.tensor([audio.shape[-1]], device=device))
            reference = audio[0].numpy()
            estimate = result["reconstruction"][0, 0].cpu().numpy()
            code = result["z"][result["valid"]]
            record = dict(id=identifier, speaker=rows[index]["speaker"], samples=len(reference),
                          active_coefficients=int((code != 0).sum()),
                          **audio_metrics(reference, estimate, args.speech_metrics))
            if index < args.save_audio:
                for label, signal in (("reference", reference), ("reconstruction", estimate)):
                    name = f"{index:05d}_{label}.wav"
                    sf.write(output / name, signal, SAMPLE_RATE, subtype="FLOAT")
                    record[label + "_wav"] = name
            records.append(record)
            print(f"Evaluated {index + 1}/{len(dataset)}: {identifier}", flush=True)
    summary = dict(**summarize(records), checkpoint=str(Path(args.checkpoint).resolve()),
                   checkpoint_sha256=digest(args.checkpoint), manifest=str(manifest),
                   manifest_sha256=digest(manifest), method=payload["model_config"]["method"],
                   stage="native_objective_unquantized", bitrate=None,
                   note="Unquantized reconstruction. Operational rate-distortion requires a saved bitstream.")
    write_json(output / "summary.json", summary)
    write_json(output / "cases.json", records)
    with open(output / "metrics.csv", "w", newline="", encoding="utf-8") as stream:
        keys = ["id", "speaker", "samples", "active_coefficients", "mse", "nmse", "si_sdr", "pesq_wb", "stoi", "statuses"]
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows({key: json.dumps(row[key]) if key == "statuses" else row[key] for key in keys} for row in records)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
