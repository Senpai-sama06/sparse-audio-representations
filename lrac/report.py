from pathlib import Path
from datetime import datetime, timezone
import html
import json
import itertools
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lrac.utils import write_json
from lrac.analyze import compare_stability


def report(runs, output="runs/reports/comparison"):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    entries, rows = [], []
    for run in map(Path, runs):
        config = json.loads((run / "config.json").read_text(encoding="utf-8"))
        for path in sorted((run / "evaluation").rglob("summary.json")):
            value = json.loads(path.read_text(encoding="utf-8"))
            if "coverage_sha256" not in value:
                continue  # Do not mix development artifacts without coverage provenance.
            snr = value["metrics"].get("snr_db", {}).get("median")
            active = value["metrics"].get("active_fraction", {}).get("mean")
            label = f"{run.name} ({value.get('checkpoint', 'best.pt')})"
            setting = value.get("representation_setting", value["code_budget"])
            entries.append({"run": label, "model": config["model"]["kind"], "summary": str(path.resolve()), **value})
            rows.append(f"| {label} | {value['split']} | {value.get('study_stage', 'main')} / {'limited' if value['limited'] else 'complete dataset'} | {setting} | {snr if snr is not None else 'silence'} | {active} |")
    # Separate datasets and limited/complete runs: never plot unlike test conditions together.
    conditions = sorted({(v["split"], v["coverage_sha256"], v["study_stage"]) for v in entries})
    for split, coverage, stage in conditions:
        fig, ax = plt.subplots(figsize=(8, 5))
        for name in sorted({v["run"] for v in entries}):
            points = [v for v in entries if v["run"] == name and v["split"] == split and v["coverage_sha256"] == coverage and v["study_stage"] == stage and "snr_db" in v["metrics"]]
            points.sort(key=lambda v: v["metrics"]["active_coefficients_per_second"]["mean"])
            if points:
                ax.plot([p["metrics"]["active_coefficients_per_second"]["mean"] for p in points], [p["metrics"]["snr_db"]["median"] for p in points], "o-", label=name)
        ax.set(xlabel="Active coordinates per second (continuous values)", ylabel="Median waveform SNR (dB)", title=f"{split} | {stage} | coverage {coverage[:8]}")
        if ax.lines:
            ax.legend(fontsize=6)
        ax.grid(alpha=.2)
        fig.tight_layout()
        fig.savefig(output / f"reconstruction_{split}_{stage}_{coverage[:8]}.png", dpi=150)
        plt.close(fig)
    stability = []
    for a, b in itertools.combinations(map(Path, runs), 2):
        if (a / "analysis/validation/activations.npz").exists() and (b / "analysis/validation/activations.npz").exists():
            stability.append({"a": a.name, "b": b.name, **compare_stability(a, b)})
    write_json(output / "results.json", {"created_utc": datetime.now(timezone.utc).isoformat(), "evaluations": entries, "stability": stability})
    text = "# Reconstructive feature discovery\n\nLimited evaluations and smoke-stage runs are not full benchmark evidence. Plots are separated by actual example coverage and study stage.\n\n| Run / checkpoint | Split | Stage / coverage | Representation setting | Median SNR dB | Active fraction |\n|---|---|---|---|---:|---:|\n" + "\n".join(rows)
    text += "\n\nInterpret reconstruction, feature stability, and separation independently. Low functional interaction with inactive groups is not evidence of recovered structure. Synthetic transfer supports claims only for the evaluated generators.\n"
    (output / "report.md").write_text(text, encoding="utf-8")
    images = "".join(f'<figure><img src="{html.escape(p.name)}" style="max-width:100%"><figcaption>{html.escape(p.stem)}</figcaption></figure>' for p in sorted(output.glob("*.png")))
    (output / "index.html").write_text('<!doctype html><meta charset="utf-8"><title>LRAC results</title><main style="max-width:1100px;margin:40px auto;font:16px system-ui"><h1>Reconstructive feature discovery</h1><pre style="white-space:pre-wrap">' + html.escape(text) + "</pre>" + images + "</main>", encoding="utf-8")
    print(f"REPORT={output.resolve()}", flush=True)
    return output
