"""Publish selected L1 study artifacts without committing audio or bitstreams.

Usage: python tools/export_l1_results.py --run runs/l1_librispeech_<timestamp>
The original run remains the provenance source; copied tables are review artifacts.
"""
import argparse
import json
from pathlib import Path
import shutil

from lrac.utils import file_hash, write_csv, write_json


def export(run, output, synthetic_run=None):
    """Verify recorded artifacts before copying a completed study into a report."""
    run, output = Path(run), Path(output)
    completion = json.loads((run/"completion.json").read_text(encoding="utf-8"))
    config = json.loads((run/"config.json").read_text(encoding="utf-8"))
    if completion["status"] != "complete" or config["kind"] != "l1":
        raise ValueError("Export requires a completed L1 study")
    mapping = {"metrics.csv": "l1_metrics.csv", "comparison.csv": "l1_comparison.csv",
               "paired_rates.csv": "l1_paired_rates.csv", "reference_metrics.csv": "l1_reference_metrics.csv",
               "rate_distortion.png": "l1_rate_distortion.png", "summary.json": "l1_results.json"}
    # Check source integrity before overwriting any published artifact. Completion
    # stores hashes from the actual run, rather than hashes of newly regenerated data.
    checked = [*mapping, "cohort.json", "config.json", "source.json", "environment.json"]
    checked += [p.relative_to(run).as_posix() for p in sorted((run/"diagnostics").glob("*.json"))]
    for filename in checked:
        if file_hash(run/filename) != completion["artifacts"][filename]["sha256"]:
            raise ValueError(f"Run artifact changed: {filename}")
    output.mkdir(parents=True, exist_ok=True)
    for source, destination in mapping.items():
        shutil.copyfile(run/source, output/destination)
    cohort = json.loads((run/"cohort.json").read_text(encoding="utf-8"))
    # Absolute cache paths are machine-specific. Preserve source IDs, source file
    # and crop hashes, extents and speaker roles in the portable protocol instead.
    for row in cohort["examples"]:
        row.pop("path", None)
    write_json(output/"l1_cohort.json", cohort)
    fits = []
    for path in sorted((run/"diagnostics").glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        for fit in record["l1"]:
            fits.append(dict(id=record["reference"]["id"], speaker=record["reference"]["speaker"],
                role=record["reference"]["role"], **{k: v for k, v in fit.items() if k != "trials"},
                lambda_trials=len(fit["trials"]),
                unconverged_subproblems=sum(not trial["converged"] for trial in fit["trials"])))
    write_csv(output/"l1_solver.csv", fits)
    provenance = dict(run_name=run.name, completion=completion, exporter_sha256=file_hash(Path(__file__)),
        source=json.loads((run/"source.json").read_text(encoding="utf-8")),
        environment=json.loads((run/"environment.json").read_text(encoding="utf-8")),
        config={k: v for k, v in config.items() if k != "project_root"})
    if synthetic_run:
        synthetic = Path(synthetic_run)
        recorded = json.loads((synthetic/"completion.json").read_text(encoding="utf-8"))
        if file_hash(synthetic/"summary.json") != recorded["artifacts"]["summary.json"]["sha256"]:
            raise ValueError("Synthetic summary changed after completion")
        provenance["synthetic_run"] = dict(name=synthetic.name,
            summary=json.loads((synthetic/"summary.json").read_text(encoding="utf-8")))
    write_json(output/"l1_runs.json", provenance)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--synthetic-run")
    parser.add_argument("--output", default="documentation")
    args = parser.parse_args()
    export(args.run, args.output, args.synthetic_run)
