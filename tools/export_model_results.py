"""Export the completed three-model study for the LaTeX report (standard library only)."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def export(study_path, output):
    root = Path(__file__).resolve().parents[1]
    study = json.loads(study_path.read_text(encoding="utf-8"))
    if len(study["runs"]) != 3:
        raise ValueError("Expected three runs in S-NL, D-NL, S-AF order")
    rows, records, coverage, training_manifests = [], [], set(), set()
    table = [r"\begin{table}[p]", r"\centering\small", r"\begin{tabular}{llrrrrr}",
             r"\toprule", r"Model & Retained & Values/s & NMSE & SNR (dB) & STOI & PESQ-WB \\", r"\midrule"]
    for label, item in zip(["S-NL", "D-NL", "S-AF"], study["runs"]):
        run = root / item["path"]
        progress = json.loads((run / "progress.json").read_text())
        if progress["status"] != "complete":
            raise ValueError(f"Run is incomplete: {run.name}")
        config = json.loads((run / "config.json").read_text())
        dense_training = config["train"]["keep_fractions"] == [1.0]
        affine = config["model"].get("activation", "leaky_relu") == "identity"
        if (dense_training, affine) != {"S-NL": (False, False), "D-NL": (True, False), "S-AF": (False, True)}[label]:
            raise ValueError(f"Run order does not match the report labels: {run.name}")
        training_manifests.add(file_hash(root / config["data"]["manifest_dir"] / config["data"]["train_manifest"]))
        source = json.loads((run / "source.json").read_text())
        summaries = sorted(run.glob("evaluation/validation/limit_200/fraction_*/summary.json"),
                           key=lambda p: float(p.parent.name.split("_")[1]))
        if len(summaries) != 8:
            raise ValueError(f"Expected eight evaluated budgets: {run.name}")
        score_files = {}
        for path in summaries:
            summary = json.loads(path.read_text())
            fraction = float(path.parent.name.split("_")[1])
            if summary["examples"] != 200 or summary["split"] != "validation":
                raise ValueError("Expected the common 200-example development probe")
            coverage.add(summary["coverage_sha256"])
            metrics = summary["metrics"]
            row = {"model": label, "keep_fraction": fraction, "examples": summary["examples"],
                   "checkpoint_step": summary["checkpoint_step"]}
            for metric, stats in metrics.items():
                for name, value in stats.items():
                    if isinstance(value, list):
                        row[f"{metric}_{name}_low"], row[f"{metric}_{name}_high"] = value
                    else:
                        row[f"{metric}_{name}"] = value
            rows.append(row)
            table.append(f"{label} & {100*fraction:g}\\% & "
                         f"{metrics['active_coefficients_per_second']['mean']:,.1f} & "
                         f"{metrics['nmse']['mean']:.4f} & {metrics['snr_db']['median']:.2f} & "
                         f"{metrics['stoi']['mean']:.3f} & {metrics['pesq_wb']['mean']:.3f} " + r"\\")
            score_files[path.parent.name] = file_hash(path)
        table.append(r"\midrule")
        search = json.loads((summaries[0].parent.parent / "support_search.json").read_text())
        records.append({
            "model": label, "description": item["name"], "run_name": run.name,
            "train_name": config["train"]["name"], "training_status": progress,
            "training_statistics": json.loads((run / "training_statistics.json").read_text()),
            "environment": json.loads((run / "environment.json").read_text()),
            "training_source_combined_sha256": source["combined_sha256"],
            "checkpoint_sha256": file_hash(run / "checkpoints/best.pt"),
            "checkpoint_step": summary["checkpoint_step"],
            "validation_manifest_sha256": summary["manifest_sha256"],
            "validation_coverage_sha256": summary["coverage_sha256"],
            "evaluation_summary_sha256": score_files,
            "speech_metric_statuses": summary["speech_metric_statuses"],
            "support_search": search,
        })
    if len(coverage) != 1 or len(training_manifests) != 1:
        raise ValueError("The runs used different training manifests or evaluated different recording coverage")
    table[-1] = r"\bottomrule"
    table.extend([r"\end{tabular}",
                  r"\caption{Common development probe: 200 full utterances, 40 speakers. "
                  r"Values/s, NMSE, STOI, and PESQ-WB are utterance means; SNR is the utterance median. "
                  r"Retained is the requested fraction of valid latent capacity. "
                  r"All scores use best checkpoints selected on this same probe.}",
                  r"\label{tab:results}", r"\end{table}"])
    output.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (output / "sparse_autoencoder_results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (output / "sparse_autoencoder_results.tex").write_text("\n".join(table) + "\n", encoding="utf-8")
    provenance = {"date": "2026-10-27", "scope": "development; not a final test or sparsity certificate",
                  "training_manifest_sha256": next(iter(training_manifests)),
                  "model_order": ["S-NL", "D-NL", "S-AF"], "runs": records}
    (output / "sparse_autoencoder_runs.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(f"Exported {len(rows)} rows, the LaTeX table, and three run records to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=Path("runs/concurrent_training.json"))
    parser.add_argument("--output", type=Path, default=Path("documentation"))
    args = parser.parse_args()
    export(args.study, args.output)
