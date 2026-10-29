"""L1/DCT inference, real sparse bitstreams, and exploratory rate-distortion curves.

Reusable math lives in linear.py/l1.py; codec.py knows nothing about the source
waveform. This module only assembles cohorts, runs the grid and reports results.
"""
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import time

import numpy as np
import soundfile as sf
import torch

from lrac.codec import encode, decode
from lrac.data.l1_examples import synthetic_examples, speech_examples, read_example
from lrac.l1 import solve_l1, threshold_dct
from lrac.linear import CosineDictionary
from lrac.metrics import waveform_metrics, speech_quality_metrics
from lrac.utils import write_json, write_csv


ROOT = Path(__file__).resolve().parents[1]
METRICS = ("bitrate_kbps", "nmse", "snr_db", "pesq_wb", "stoi", "active_coefficients",
           "bits_per_sample", "encoding_seconds", "decoding_seconds")


def _path(value):
    """Study input paths refer to the checkout; home/absolute paths also work."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT/path


def _measure(task):
    """Process one source on CPU; reuse inference across coefficient precisions."""
    source, settings, output = task
    output = Path(output)
    torch.set_num_threads(1)  # Avoid one BLAS/OpenMP team per process and metric.
    x, digest = read_example(source, settings["sample_rate"])
    name = source["id"].replace(":", "_")
    reference = dict(id=source["id"], speaker=str(source["speaker"]), role=source["role"],
                     waveform_sha256=digest, samples=len(x), sample_rate=settings["sample_rate"])
    if settings["speech_metrics"]:
        reference.update(speech_quality_metrics(x, x, settings["sample_rate"]))
    else:
        reference.update(pesq_wb=None, stoi=None, pesq_wb_status="not speech", stoi_status="not speech",
                         visqol=None, visqol_status="not integrated; no score computed")
    # FLOAT WAV is an inspectable reference, not the coded payload or a PCM-rate
    # claim. It preserves amplitudes outside [-1,1] without clipping.
    sf.write(output/"audio"/f"{name}_reference.wav", x, settings["sample_rate"], subtype="FLOAT")
    points, diagnostics = [], []
    energy = float(np.dot(x, x))
    for target in settings["targets"]:
        # Reserve a declared 1% of the distortion budget for quantization. Both
        # methods get the same margin; final feasibility is still measured.
        solver_target = target*(1-settings["quantization_margin"])
        for method in ("l1", "dct"):
            started = time.perf_counter()
            if method == "l1":
                fit = solve_l1(x, solver_target, max_iterations=settings["max_iterations"],
                               search_steps=settings["search_steps"], gap_tolerance=settings["gap_tolerance"],
                               constrained_tolerance=settings["constrained_tolerance"])
                z = fit.pop("coefficients")
                trials = fit.pop("trials")
                diagnostics.append(dict(target_nmse=target, solver_target=solver_target, trials=trials, **fit))
                solver_converged, solver_gap = fit["converged"], fit["constrained_relative_gap"]
                iterations = fit["iterations"]
            else:
                z = threshold_dct(x, solver_target)
                solver_converged, solver_gap, iterations = True, 0.0, 0
            solver_seconds = time.perf_counter()-started
            bank = CosineDictionary(len(x), "impulse_dct" if method == "l1" else "dct")
            residual = x-bank.synthesis(z)
            pre_nmse = float(np.dot(residual, residual)/energy) if energy else None
            for bits in settings["quantization_bits"]:
                started = time.perf_counter()
                payload = encode(z, samples=len(x), sample_rate=settings["sample_rate"], dictionary=bank.kind, bits=bits)
                packing_seconds = time.perf_counter()-started
                filename = f"{name}_{method}_e{target:g}_b{bits}.sac"
                (output/"bitstreams"/filename).write_bytes(payload)
                # The evaluated waveform comes from disk bytes, never z. Disk
                # I/O is excluded from CPU encode/decode timing for both methods.
                saved_payload = (output/"bitstreams"/filename).read_bytes()
                started = time.perf_counter()
                reconstructed, coding = decode(saved_payload)
                decoding_seconds = time.perf_counter()-started
                metrics = waveform_metrics(torch.from_numpy(x)[None, None],
                                           torch.from_numpy(reconstructed)[None, None], settings["sample_rate"])
                quality = speech_quality_metrics(x, reconstructed, settings["sample_rate"]) if settings["speech_metrics"] else {
                    key: reference[key] for key in ("pesq_wb", "stoi", "pesq_wb_status", "stoi_status", "visqol", "visqol_status")}
                duration = len(x)/settings["sample_rate"]
                points.append(dict(id=source["id"], speaker=str(source["speaker"]), role=source["role"],
                    method=method, target_nmse=target, solver_target=solver_target,
                    pre_nmse=pre_nmse, pre_nonzeros=int(np.count_nonzero(z)), pre_l1_norm=float(np.abs(z).sum()),
                    solver_converged=solver_converged, constrained_relative_gap=solver_gap, solver_iterations=iterations,
                    solver_seconds=solver_seconds, packing_seconds=packing_seconds,
                    encoding_seconds=solver_seconds+packing_seconds, decoding_seconds=decoding_seconds,
                    encoding_rtf=(solver_seconds+packing_seconds)/duration, decoding_rtf=decoding_seconds/duration,
                    bitstream=f"bitstreams/{filename}", compression_ratio_pcm16=16*len(x)/coding["total_bits"],
                    active_coefficients_per_second=coding["active_coefficients"]/duration,
                    final_meets_target=metrics["nmse"] is not None and metrics["nmse"] <= target,
                    **coding, **metrics, **quality))
                if source.get("save_example") and bits == max(settings["quantization_bits"]):
                    sf.write(output/"audio"/filename.replace(".sac", ".wav"), reconstructed,
                             settings["sample_rate"], subtype="FLOAT")
    write_json(output/"diagnostics"/f"{name}.json", dict(reference=reference, l1=diagnostics))
    return points, reference


def operating_points(rows, targets, rates):
    """Select measured grid points, without interpolation or assumed bit costs.

    Each clip can choose its own encoder setting: the resulting bitstream already
    contains everything needed for decoding. Distortion is the selection metric;
    PESQ/STOI are reported afterward rather than independently cherry-picked.
    """
    groups = defaultdict(list)
    for row in rows:
        if row["nmse"] is not None:
            groups[row["id"], row["method"]].append(row)
    selected = []
    for candidates in groups.values():
        for mode, limits in (("nmse", targets), ("kbps", rates)):
            for limit in limits:
                feasible = [r for r in candidates if r["nmse" if mode == "nmse" else "bitrate_kbps"] <= limit]
                if not feasible:
                    continue
                best = min(feasible, key=lambda r: (r["bitrate_kbps"], r["nmse"]) if mode == "nmse"
                           else (r["nmse"], r["bitrate_kbps"]))
                selected.append(dict(mode=mode, limit=limit, **best))
    return selected


def _macro(rows, key):
    """Average clips within speakers first; missing optional scores stay missing."""
    speakers = defaultdict(list)
    for row in rows:
        value = row.get(key)
        if value is not None and np.isfinite(value):
            speakers[row["speaker"]].append(value)
    return float(np.mean([np.mean(v) for v in speakers.values()])) if speakers else None


def _report(output, rows, references, settings):
    """Build paired exploratory comparisons and a compact scientific figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected = operating_points(rows, settings["targets"], settings["comparison_rates"])
    if selected:
        write_csv(output/"operating_points.csv", selected)
    comparison, paired = [], []
    for role in sorted({r["role"] for r in rows}):
        for mode, limits in (("nmse", settings["targets"]), ("kbps", settings["comparison_rates"])):
            for limit in limits:
                candidates = [r for r in selected if r["role"] == role and r["mode"] == mode and r["limit"] == limit]
                identities = {m: {r["id"] for r in candidates if r["method"] == m} for m in ("l1", "dct")}
                common = identities["l1"] & identities["dct"]
                pair_rows = [r for r in candidates if r["id"] in common]
                for method in ("l1", "dct"):
                    subset = [r for r in pair_rows if r["method"] == method]
                    comparison.append(dict(role=role, mode=mode, limit=limit, method=method,
                        available_clips=len(identities[method]), paired_clips=len(common),
                        paired_speakers=len({r["speaker"] for r in subset}),
                        pesq_valid=sum(r["pesq_wb"] is not None for r in subset),
                        stoi_valid=sum(r["stoi"] is not None for r in subset),
                        **{key: _macro(subset, key) for key in METRICS}))
                if mode == "nmse" and common:
                    by_source = {(r["id"], r["method"]): r for r in pair_rows}
                    speaker_rates = defaultdict(list)
                    for identity in sorted(common):
                        a, b = by_source[identity, "l1"], by_source[identity, "dct"]
                        speaker_rates[a["speaker"]].append((a["bitrate_kbps"], b["bitrate_kbps"]))
                    values = np.asarray([np.mean(v, axis=0) for v in speaker_rates.values()])
                    rng = np.random.default_rng(settings["seed"])
                    resampled = values[rng.integers(len(values), size=(1000, len(values)))].mean(axis=1)
                    savings = 100*(1-resampled[:, 0]/resampled[:, 1])
                    means = values.mean(axis=0)
                    paired.append(dict(role=role, target_nmse=limit, paired_clips=len(common),
                        paired_speakers=len(values), l1_kbps=float(means[0]), dct_kbps=float(means[1]),
                        l1_rate_saving_percent=float(100*(1-means[0]/means[1])),
                        ci95_low=float(np.percentile(savings, 2.5)), ci95_high=float(np.percentile(savings, 97.5))))
    write_csv(output/"comparison.csv", comparison)
    if paired:
        write_csv(output/"paired_rates.csv", paired)
    l1_fits = [r for r in rows if r["method"] == "l1" and r["bits"] == min(settings["quantization_bits"])]
    summary = dict(settings=settings, clips=len(references),
        roles={role: dict(clips=sum(r["role"] == role for r in references),
                         speakers=len({r["speaker"] for r in references if r["role"] == role}))
               for role in sorted({r["role"] for r in references})},
        encoded_points=len(rows), l1_fits=len(l1_fits),
        l1_converged=sum(r["solver_converged"] for r in l1_fits),
        worst_constrained_relative_gap=max(r["constrained_relative_gap"] for r in l1_fits),
        final_target_passes=sum(r["final_meets_target"] for r in rows),
        reference_pesq=_macro(references, "pesq_wb"), reference_stoi=_macro(references, "stoi"),
        comparison=comparison, paired_rates=paired,
        scope="Exploratory fixed-family/grid comparison; no minimum-L0/bitrate or practical-codec superiority claim")
    write_json(output/"summary.json", summary)
    # Each curve fixes the inference target and varies quantization precision.
    # Points summarize measured settings, not an interpolated achievable frontier.
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout="constrained")
    role = "measurement" if any(r["role"] == "measurement" for r in rows) else "synthetic"
    for method, color in (("l1", "tab:blue"), ("dct", "tab:orange")):
        for target in settings["targets"]:
            values = []
            for bits in settings["quantization_bits"]:
                subset = [r for r in rows if r["role"] == role and r["method"] == method
                          and r["target_nmse"] == target and r["bits"] == bits]
                values.append([_macro(subset, k) for k in ("bitrate_kbps", "nmse", "pesq_wb", "stoi")])
            for index, ax in enumerate(axes):
                available = [(v[0], v[index+1]) for v in values if v[index+1] is not None]
                if available:
                    ax.plot(*np.asarray(available).T, "o-", color=color, alpha=.7,
                            label=f"{method}, pre-target {target:g}", markersize=3)
    for ax, label in zip(axes, ("Decoded NMSE (lower better)", "PESQ-WB", "STOI")):
        ax.set(xlabel="Actual kb/s, including overhead", ylabel=label)
        ax.grid(alpha=.2)
    axes[0].set_yscale("log")
    axes[0].legend(fontsize=7)
    fig.suptitle(f"{role}: equal-speaker means | fixed whole-clip dictionaries | simple sparse codec")
    fig.savefig(output/"rate_distortion.png", dpi=160)
    plt.close(fig)
    lines = ["# L1 sparse-coding pilot", "", summary["scope"], "",
        f"Clips: {summary['clips']}; encoded points: {len(rows)}. "
        f"L1 solves within {settings['constrained_tolerance']:.1%} constrained relative dual gap: "
        f"{summary['l1_converged']}/{summary['l1_fits']}; worst gap {summary['worst_constrained_relative_gap']:.6g}.", "",
        "Every score uses saved, decoded bytes. Rate includes the 28-byte header, support, values and padding. "
        "The analytic dictionaries have no stored fitted matrix. Raw 16-kHz mono PCM16 is a 256-kb/s reference.", "",
        f"Original-versus-itself reference: PESQ-WB {summary['reference_pesq']}, STOI {summary['reference_stoi']}.", "",
        "## Rate at matched final waveform distortion", "",
        "Only clips with feasible candidates for both methods enter each paired comparison. "
        "Selection chooses the smallest measured stream meeting the limit, from this finite grid. "
        "Intervals resample paired speakers, with 1,000 draws. Positive savings favor L1.", "",
        "| Role | NMSE limit | Paired clips | L1 kb/s | DCT kb/s | L1 rate saving %, 95% CI |",
        "|---|---:|---:|---:|---:|---:|"]
    for row in paired:
        lines.append(f"| {row['role']} | {row['target_nmse']:g} | {row['paired_clips']} | {row['l1_kbps']:.3f} | "
                     f"{row['dct_kbps']:.3f} | {row['l1_rate_saving_percent']:+.2f} [{row['ci95_low']:+.2f}, {row['ci95_high']:+.2f}] |")
    lines += ["", "![Rate and quality](rate_distortion.png)", "", "## Interpretation limits", "",
        "L1 minimizes coefficient magnitude, not support or bits. DCT thresholding and L1 also use different "
        "dictionaries, so this does not isolate the inference algorithm alone. The codec has no entropy coding; "
        "no comparison with a conventional audio codec is implied. Optional speech metrics retain validity statuses.", "",
        "The speech split is exploratory development data. No settings were selected using measurement-speaker scores; "
        "both speaker groups used the same declared grid. They are not a fresh confirmation set. "
        "Reported timings charge the full inference solve to each independently encoded candidate; I/O and "
        "multi-candidate exploration/report time are separate from per-candidate CPU timing.", "",
        "See config.json and cohort.json for protocol and source identities, diagnostics/ for every solver path, "
        "metrics.csv for all decoded scores and bit components, reference_metrics.csv for self-reference quality, "
        "comparison.csv for matched-rate/quality summaries, and bitstreams/ for the decodable files.", ""]
    (output/"report.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def run(output, *, dataset="synthetic", manifest="data/manifests/librispeech_cached/validation.jsonl",
        exclusions=None, sample_rate=16000, seconds=2.0, speakers=40, clips_per_speaker=3,
        seed=0, targets=(.1, .01, .001), quantization_bits=(4, 6, 8, 12, 16),
        comparison_rates=(16, 32, 64, 128), workers=4, max_iterations=2000, search_steps=18,
        gap_tolerance=1e-5, constrained_tolerance=.01, quantization_margin=.01):
    """Run the frozen pilot grid; source selection precedes all model responses."""
    output = Path(output)
    samples = round(seconds*sample_rate)
    if dataset not in {"synthetic", "librispeech"} or sample_rate != 16000 or samples < 512:
        raise ValueError("Use synthetic/librispeech at 16 kHz, at least 512 samples")
    if workers < 1 or not targets or any(not 0 < t < 1 for t in targets):
        raise ValueError("Require positive workers and targets between zero and one")
    if not quantization_bits or any(b not in {4, 6, 8, 12, 16} for b in quantization_bits):
        raise ValueError("Unsupported quantization grid")
    if not 0 <= quantization_margin < 1 or not comparison_rates or min(comparison_rates) <= 0:
        raise ValueError("Invalid quantization margin or comparison rates")
    settings = dict(dataset=dataset, sample_rate=sample_rate, seed=seed, targets=list(targets),
        quantization_bits=list(quantization_bits), comparison_rates=list(comparison_rates),
        max_iterations=max_iterations, search_steps=search_steps, gap_tolerance=gap_tolerance,
        constrained_tolerance=constrained_tolerance, quantization_margin=quantization_margin,
        speech_metrics=dataset == "librispeech")
    if dataset == "synthetic":
        cohort, provenance = synthetic_examples(samples, sample_rate, seed), dict(generator="lrac/data/l1_examples.py")
    else:
        if exclusions is None:
            raise ValueError("Speech pilot requires the checkpoint-selection metrics CSV for exclusions")
        cohort, provenance = speech_examples(_path(manifest), _path(exclusions), samples=samples,
            sample_rate=sample_rate, speakers=speakers, clips_per_speaker=clips_per_speaker, seed=seed)
    for folder in ("bitstreams", "audio", "diagnostics"):
        (output/folder).mkdir(parents=True, exist_ok=True)
    # Store identities/crops and hashes before executing; arrays are represented
    # by content hashes, rather than bloating a JSON file with waveform samples.
    for role in {r["role"] for r in cohort}:
        next(r for r in cohort if r["role"] == role)["save_example"] = True
    serialized = []
    for row in cohort:
        x, digest = read_example(row, sample_rate)
        if len(x) != samples:
            raise ValueError("Selected crop length differs from the protocol")
        serialized.append({**{k: v for k, v in row.items() if k != "waveform"}, "waveform_sha256": digest})
    write_json(output/"cohort.json", dict(selection="Centered crops in sorted unused utterance IDs; shuffled disjoint speaker roles",
        samples=samples, settings=settings, provenance=provenance, examples=serialized))
    tasks = [(row, settings, str(output)) for row in cohort]
    rows, references = [], []
    def collect(result):
        points, reference = result
        rows.extend(points)
        references.append(reference)
        write_json(output/"progress.json", dict(status="running", stage="coding", completed=len(references), total=len(cohort)))
        print(f"L1 {len(references)}/{len(cohort)} {reference['id']}", flush=True)
    if workers == 1:
        for task in tasks:
            collect(_measure(task))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for future in as_completed([pool.submit(_measure, task) for task in tasks]):
                collect(future.result())
    rows.sort(key=lambda r: (r["role"], r["speaker"], r["id"], r["method"], r["target_nmse"], r["bits"]))
    references.sort(key=lambda r: (r["role"], r["speaker"], r["id"]))
    write_csv(output/"metrics.csv", rows)
    write_csv(output/"reference_metrics.csv", references)
    summary = _report(output, rows, references, settings)
    print(f"L1 COMPLETE: {len(rows)} decoded points; {summary['l1_converged']}/{summary['l1_fits']} L1 fits converged", flush=True)
