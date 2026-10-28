"""Export the completed three-model study for the LaTeX report (standard library only)."""
import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def export(study_path, output, split="validation"):
    root = Path(__file__).resolve().parents[1]
    study = json.loads(study_path.read_text(encoding="utf-8"))
    subset = "limit_200" if split == "validation" else "complete"
    stem = "sparse_autoencoder" if split == "validation" else "sparse_autoencoder_test"
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
        manifest_name = config["data"]["val_manifest"] if split == "validation" else config["data"]["test_manifests"][split]
        manifest = root / config["data"]["manifest_dir"] / manifest_name
        expected = 200 if split == "validation" else sum(bool(line.strip()) for line in manifest.read_text().splitlines())
        summaries = sorted(run.glob(f"evaluation/{split}/{subset}/fraction_*/summary.json"),
                           key=lambda p: float(p.parent.name.split("_")[1]))
        if len(summaries) != 8:
            raise ValueError(f"Expected eight evaluated budgets: {run.name}")
        score_files = {}
        for path in summaries:
            summary = json.loads(path.read_text())
            fraction = float(path.parent.name.split("_")[1])
            if summary["examples"] != expected or summary["split"] != split or summary["limited"] != (split == "validation"):
                raise ValueError(f"Expected the common {expected}-example {split} evaluation")
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
            f"{split}_manifest_sha256": summary["manifest_sha256"],
            f"{split}_coverage_sha256": summary["coverage_sha256"],
            "evaluation_summary_sha256": score_files,
            "speech_metric_statuses": summary["speech_metric_statuses"],
            "support_search": search,
        })
    if len(coverage) != 1 or len(training_manifests) != 1:
        raise ValueError("The runs used different training manifests or evaluated different recording coverage")
    table[-1] = r"\bottomrule"
    if split == "clean":
        table[0] = r"\begin{table}[!htbp]"
    caption = "Common development probe: 200 full utterances, 40 speakers. " if split == "validation" else f"Full held-out test-clean: {expected:,} utterances, 40 speakers. "
    selection = "All scores use best checkpoints selected on this same probe." if split == "validation" else "All scores use frozen checkpoints previously selected on development data."
    table.extend([r"\end{tabular}",
                  r"\caption{" + caption +
                  r"Values/s, NMSE, STOI, and PESQ-WB are utterance means; SNR is the utterance median. "
                  r"Retained is the requested fraction of valid latent capacity. "
                  + selection + "}",
                  r"\label{tab:results}" if split == "validation" else r"\label{tab:test-results}", r"\end{table}"])
    output.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (output / f"{stem}_results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (output / f"{stem}_results.tex").write_text("\n".join(table) + "\n", encoding="utf-8")
    scope = "development; not a final test or sparsity certificate" if split == "validation" else "full held-out test-clean; fixed models and budgets; no minimum-support certificate"
    provenance = {"date": datetime.now(timezone.utc).date().isoformat(), "scope": scope,
                  "training_manifest_sha256": next(iter(training_manifests)),
                  "model_order": ["S-NL", "D-NL", "S-AF"], "runs": records}
    protocol = root / "runs/final_test_evaluation.json"
    if split == "clean" and protocol.exists():
        frozen = json.loads(protocol.read_text())
        frozen_runs = {Path(r["run"]).name: r for r in frozen["runs"]}
        for record in records:
            expected_run = frozen_runs[record["run_name"]]
            if record["checkpoint_sha256"] != expected_run["checkpoint_sha256"] or record["clean_manifest_sha256"] != expected_run["test_manifest_sha256"]:
                raise ValueError("Checkpoint or test manifest changed since protocol freeze")
        provenance["frozen_test_protocol"] = frozen
    (output / f"{stem}_runs.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(f"Exported {len(rows)} rows, the LaTeX table, and three run records to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=Path("runs/concurrent_training.json"))
    parser.add_argument("--output", type=Path, default=Path("documentation"))
    parser.add_argument("--split", choices=["validation", "clean"], default="validation")
    args = parser.parse_args()
    export(args.study, args.output, args.split)
