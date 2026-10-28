"""Write the held-out three-model comparison from completed evaluation files."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from export_model_results import export


def compare(study_path, output):
    export(study_path, output, split="clean")
    study = json.loads(study_path.read_text(encoding="utf-8"))
    labels = ["S-NL", "D-NL", "S-AF"]
    names = dict(zip(labels, ["Sparse nonlinear", "Dense-trained nonlinear", "Sparse affine"]))
    summaries, rows, searches, support_rows = {}, {}, {}, {}
    for label, run in zip(labels, study["runs"]):
        root = Path(run["path"]) / "evaluation/clean/complete"
        summaries[label], rows[label] = {}, {}
        for folder in sorted(root.glob("fraction_*")):
            fraction = float(folder.name.split("_")[1])
            summaries[label][fraction] = json.loads((folder / "summary.json").read_text())
            with (folder / "metrics.csv").open(newline="", encoding="utf-8") as stream:
                rows[label][fraction] = {r["id"]: r for r in csv.DictReader(stream)}
        searches[label] = json.loads((root / "support_search.json").read_text())["targets"]
        with (root / "support_search.csv").open(newline="", encoding="utf-8") as stream:
            support_rows[label] = {(r["id"], float(r["target_nmse"])): r for r in csv.DictReader(stream)}
    expected_ids = set(rows[labels[0]][0.0])
    for model_rows in rows.values():
        for result in model_rows.values():
            if set(result) != expected_ids:
                raise ValueError("Paired comparison requires identical utterance IDs")
    # Paired bootstrap resamples speakers, preserving the same recordings/models.
    paired = []
    for a, b in [("S-NL", "D-NL"), ("S-AF", "S-NL"), ("S-AF", "D-NL")]:
        for fraction in [.01, .05, .1, 1.0]:
            for metric in ["snr_db", "stoi", "pesq_wb"]:
                groups = {}
                wins = []
                for identifier in sorted(expected_ids):
                    ar, br = rows[a][fraction][identifier], rows[b][fraction][identifier]
                    if ar[metric] and br[metric]:
                        delta = float(ar[metric]) - float(br[metric])
                        groups.setdefault(ar["speaker"], []).append(delta)
                        wins.append(delta > 0)
                means = np.array([np.mean(v) for v in groups.values()])
                rng = np.random.default_rng(0)
                boot = rng.choice(means, size=(2000, len(means)), replace=True).mean(axis=1)
                paired.append({"a": a, "b": b, "keep_fraction": fraction, "metric": metric,
                               "speaker_macro_mean_difference": float(means.mean()),
                               "paired_speaker_bootstrap_ci95": np.percentile(boot, [2.5, 97.5]).tolist(),
                               "utterance_win_fraction": float(np.mean(wins)),
                               "speaker_win_fraction": float(np.mean(means > 0)), "speakers": len(means)})
    paired_support = []
    for a, b in [("S-NL", "D-NL"), ("S-AF", "S-NL"), ("S-AF", "D-NL")]:
        for target in [.1, .01, .001]:
            both, only_a, only_b, neither, ratios, less = 0, 0, 0, 0, [], 0
            for identifier in sorted(expected_ids):
                ar, br = support_rows[a][identifier, target], support_rows[b][identifier, target]
                am, bm = ar["met"] == "True", br["met"] == "True"
                if am and bm:
                    both += 1
                    ka, kb = int(ar["active_coefficients"]), int(br["active_coefficients"])
                    less += ka < kb
                    if kb:
                        ratios.append(ka / kb)
                elif am:
                    only_a += 1
                elif bm:
                    only_b += 1
                else:
                    neither += 1
            paired_support.append({"a": a, "b": b, "target_nmse": target,
                "both_succeed": both, "only_a_succeeds": only_a, "only_b_succeeds": only_b,
                "neither_succeeds": neither, "a_uses_fewer_values_among_both": less,
                "median_support_ratio_a_over_b_among_both": float(np.median(ratios)) if ratios else None})
    (output / "sparse_autoencoder_test_paired.json").write_text(json.dumps(
        {"quality": paired, "support": paired_support}, indent=2) + "\n")
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    definitions = [("snr_db", "median", "Median waveform SNR (dB)"),
                   ("nmse", "mean", "Mean waveform NMSE"),
                   ("stoi", "mean", "Mean STOI"), ("pesq_wb", "mean", "Mean wideband PESQ")]
    for ax, (metric, statistic, title) in zip(axes.flat, definitions):
        for label in labels:
            fractions = sorted(f for f in summaries[label] if f > 0)
            ax.plot(np.array(fractions) * 100,
                    [summaries[label][f]["metrics"][metric][statistic] for f in fractions],
                    marker="o", label=names[label])
        ax.set(xscale="log", xlabel="Retained latent capacity (%)", ylabel=title)
        ax.set_xticks([1, 2, 5, 10, 25, 50, 100], ["1", "2", "5", "10", "25", "50", "100"])
        ax.grid(alpha=.25)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Held-out LibriSpeech test-clean: 2,620 utterances, 40 speakers")
    fig.savefig(output / "sparse_autoencoder_test_results.png", dpi=180)
    plt.close(fig)
    lines = ["# Held-out sparse autoencoder comparison", "",
             "Full LibriSpeech test-clean: 2,620 recordings, 40 speakers, approximately 5.40 hours. "
             "Frozen best checkpoints from update 20,000; no test-time fitting or test-driven checkpoint selection. "
             "All models have 1,002,369 trainable parameters and were trained on the same ten-hour subset.", "",
             "S-NL = sparse nonlinear; D-NL = dense-trained nonlinear; S-AF = sparse affine. "
             "The affine encoder/decoder retain biases, and support selection is nonlinear.", "",
             "![Held-out support-quality curves](sparse_autoencoder_test_results.png)", "",
             "SNR is the utterance median; NMSE, STOI and PESQ are utterance means. "
             "Retained fraction refers to valid latent capacity, not a measured bitrate.", "",
             "| Model | Retained | Values/s | NMSE | SNR dB | STOI | PESQ-WB |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for fraction in [.01, .02, .05, .1, .25, .5, 1.]:
        for label in labels:
            m = summaries[label][fraction]["metrics"]
            lines.append(f"| {label} | {fraction:.0%} | {m['active_coefficients_per_second']['mean']:.1f} | "
                         f"{m['nmse']['mean']:.4f} | {m['snr_db']['median']:.2f} | "
                         f"{m['stoi']['mean']:.3f} | {m['pesq_wb']['mean']:.3f} |")
    lines += ["", "## Smallest tested support at specified accuracy", "",
              "Counts mean that at least one of the eight tested codes meets the target. "
              "A failure does not prove that no other code or representation could succeed.", "",
              "| Target NMSE | S-NL successes | D-NL successes | S-AF successes |", "|---|---:|---:|---:|"]
    for i, target in enumerate([.1, .01, .001]):
        lines.append(f"| {target:g} | " + " | ".join(f"{searches[label][i]['met']}/2620" for label in labels) + " |")
    lines += ["", "### Matched-recording support comparison", "",
              "Each row uses only recordings where both models meet the specified tolerance somewhere "
              "in the fixed sweep. Ratios compare their smallest tested supports on the same recording; "
              "success coverage above must accompany these conditional comparisons. A ratio below one "
              "means A retained fewer values. The paired JSON also records one-sided successes and failures.", "",
              "| A / B | NMSE target | Both succeed | Median support ratio A/B | A uses fewer values |",
              "|---|---:|---:|---:|---:|"]
    for item in paired_support:
        ratio = item["median_support_ratio_a_over_b_among_both"]
        if ratio is not None:
            lines.append(f"| {item['a']} / {item['b']} | {item['target_nmse']:g} | {item['both_succeed']} | "
                         f"{ratio:.3f} | {item['a_uses_fewer_values_among_both']} |")
    def score(label, fraction, metric="snr_db", statistic="median"):
        return summaries[label][fraction]["metrics"][metric][statistic]
    lines += ["", "## Findings", "",
              f"At 1% support, sparse nonlinear training gives {score('S-NL',.01):.2f} dB median SNR, "
              f"versus {score('D-NL',.01):.2f} dB for dense training followed by pruning. "
              "This directly tests the value of training the representation to survive sparsification.", "",
              f"At 10% support the affine control reaches {score('S-AF',.1):.2f} dB, "
              f"versus {score('S-NL',.1):.2f} dB for sparse nonlinear and {score('D-NL',.1):.2f} dB for dense-trained nonlinear. "
              "This is a comparison of these trained models, not a proof that affine representations are universally best.", "",
              "At full support, median SNR is " + ", ".join(f"{score(label,1.):.2f} dB ({label})" for label in labels) + ". "
              "Any full-support reconstruction error exists before discarding latent values; "
              "it cannot be attributed solely to an insufficient sparse budget.", "",
              "**Architecture limitation:** for a two-second, 32,000-sample input, encoder stage E4 has "
              "128 channels x 125 time steps = 16,000 values. A final 1x1 projection expands them to "
              "32,000 latent slots; the decoder starts by projecting 256 channels back to 128. "
              "Thus all three baselines compress information before top-k. The affine analysis and "
              "synthesis operators have rank at most 16,000 at this input length. This rules out a "
              "general lossless dense reference, although it does not prove a particular speech-error floor. "
              "The earlier interpretation that only sparsification creates a bottleneck was incomplete.", "",
              "The fewest successful scalar values among a finite sweep are achieved supports. "
              "No minimum-support certificate, optimized sparse bitstream, or guarantee over all representations is provided. "
              "The three runs use one training seed and one clean English speech corpus. "
              "Tests involve unseen speakers but do not establish transfer to all audio or signals."]
    lines += ["", "## Uncertainty and interpretation", "",
              "Paired differences and 95% percentile bootstrap intervals are in "
              "[sparse_autoencoder_test_paired.json](sparse_autoencoder_test_paired.json). "
              "They resample 40 speakers with 2,000 draws under seed 0, after averaging matched "
              "utterance differences within each speaker. These are intervals for a mean difference, "
              "not for the median SNR or variability across training seeds.", "",
              "PESQ/STOI validity counts, checkpoint hashes, original training source fingerprints, "
              "and the frozen test protocol are in [the provenance record](sparse_autoencoder_test_runs.json). "
              "ViSQOL was not integrated. Inference times are contended and are not used for model-speed claims.", "",
              "Reproduce with `python tools/compare_models.py`. This also regenerates the numeric "
              "export and LaTeX table from the completed local runs; datasets and checkpoints stay local."]
    (output / "sparse_autoencoder_test_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote held-out comparison, paired uncertainty, and figure to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=Path("runs/concurrent_training.json"))
    parser.add_argument("--output", type=Path, default=Path("documentation"))
    args = parser.parse_args()
    compare(args.study, args.output)
