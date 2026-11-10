"""Validate scores and build a portable listening report for the PitchFlower crop set."""
import argparse
import base64
import hashlib
import html
import json
import math
import re
from pathlib import Path

import numpy as np
import soundfile as sf

from sae.metrics import audio_metrics, summarize


METHODS = ("l1", "kl", "topk", "jumprelu", "gated", "batchtopk", "dense")
LABELS = {
    "pitchflower": "PitchFlower RVQ",
    "l1": "L1 SAE",
    "kl": "KL SAE",
    "topk": "TopK SAE",
    "jumprelu": "JumpReLU SAE",
    "gated": "Gated SAE",
    "batchtopk": "BatchTopK SAE",
    "dense": "Dense control",
}
METRICS = ("pesq_wb", "stoi", "si_sdr", "nmse", "mse")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_audio(path, samples):
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    if (rate != 16000 or sf.info(path).subtype != "PCM_16"
            or audio.shape != (samples, 1) or not np.isfinite(audio).all()):
        raise ValueError(f"Invalid 16-kHz mono reference/reconstruction: {path}")
    return audio[:, 0]


def read_pitchflower_pcm(path):
    from scipy.signal import resample_poly

    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    if rate != 24000 or sf.info(path).subtype != "PCM_16" or audio.shape != (36096, 1):
        raise ValueError(f"Invalid 24-kHz PitchFlower PCM16 WAV: {path}")
    return resample_poly(audio[:, 0], 2, 3).astype(np.float32)


def score_matches(actual, expected, tolerance=1e-6):
    mismatches = {}
    for key in METRICS:
        if key not in expected:
            continue
        a, b = actual.get(key), expected.get(key)
        if a is None and b is None:
            continue
        if a is None or b is None or not math.isclose(float(a), float(b), rel_tol=0, abs_tol=tolerance):
            mismatches[key] = {"recomputed": a, "saved": b}
    if "statuses" in expected and actual.get("statuses", {}) != expected.get("statuses", {}):
        mismatches["statuses"] = {"recomputed": actual.get("statuses", {}), "saved": expected.get("statuses", {})}
    return mismatches


def mean_metric(rows, metric):
    values = [row[metric] for row in rows if row.get(metric) is not None]
    return {"mean": float(np.mean(values)) if values else None, "valid": len(values)}


def data_uri(path, mime):
    return f"data:{mime};base64," + base64.b64encode(Path(path).read_bytes()).decode("ascii")


def spectrogram_uri(audio, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.signal import stft

    f, t, z = stft(audio, fs=16000, nperseg=512, noverlap=384, boundary=None)
    db = 20 * np.log10(np.maximum(np.abs(z), 1e-7))
    fig, ax = plt.subplots(figsize=(6.0, 2.2))
    im = ax.pcolormesh(t, f / 1000, db, vmin=-80, vmax=0, shading="auto", cmap="magma")
    ax.set(xlabel="Time (s)", ylabel="Frequency (kHz)", xlim=(0, len(audio) / 16000), ylim=(0, 8))
    fig.colorbar(im, ax=ax, label="Magnitude (dB; reference amplitude = 1)")
    fig.tight_layout()
    fig.savefig(path, format="png", dpi=90)
    plt.close(fig)
    return data_uri(path, "image/png")


def number(value):
    return "Not available" if value is None else f"{value:.4f}"


def build(args):
    root = Path(__file__).resolve().parents[2]
    source_report = root / "approaches/pitchflower/reports/pitchflower_best_metrics.json"
    source_verification = root / "approaches/pitchflower/reports/pitchflower_best_verification.json"
    input_root = Path(args.inputs).resolve()
    result_root = input_root / "results_pcm16_exact"
    report_dir = Path(args.report_dir).resolve()
    official = json.loads(source_report.read_text(encoding="utf-8"))
    source_audit = json.loads(source_verification.read_text(encoding="utf-8"))
    scores = json.loads((result_root / "scores.json").read_text(encoding="utf-8"))
    cases = json.loads((input_root / "cases.json").read_text(encoding="utf-8"))
    expected_ids = [case["id"] for case in official["cases_data"]]
    ids = [case["id"] for case in cases]
    remote_case_keys = [(case.get("id"), str(case.get("speaker")), case.get("samples"))
                        for case in scores.get("cases", [])]
    local_case_keys = [(case.get("id"), str(case.get("speaker")), case.get("samples"))
                       for case in cases]
    if (len(cases) != 40 or len(set(ids)) != 40 or ids != expected_ids
            or remote_case_keys != local_case_keys):
        raise ValueError("Case identity, order, or count differs from the PitchFlower report")
    if len({case["speaker"] for case in cases}) != 40:
        raise ValueError("Expected one case for each of 40 distinct speakers")
    metadata = input_root / "results" / "metadata"
    case_ids = {case["id"].removeprefix("librispeech:") for case in cases}
    selections = {method: json.loads((metadata / f"{method}_data_selection.json").read_text(encoding="utf-8"))
                  for method in METHODS}
    shared_ids = {key: set(selections["l1"][key]) for key in ("train_ids", "validation_ids", "calibration_ids")}
    for method in METHODS[1:]:
        for key, ids_for_split in shared_ids.items():
            if set(selections[method][key]) != ids_for_split:
                raise ValueError(f"The {method} run used a different {key} selection")
    split_overlap = {key: case_ids & ids_for_split for key, ids_for_split in shared_ids.items()}
    if (split_overlap["train_ids"] or len(split_overlap["validation_ids"]) != 20
            or len(split_overlap["calibration_ids"]) != 20
            or split_overlap["validation_ids"] & split_overlap["calibration_ids"]):
        raise ValueError("Unexpected overlap between the comparison cases and the SAE data selections")
    manifests = {name: [json.loads(line) for line in (metadata / f"{name}.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
                 for name in ("train", "validation", "calibration")}
    manifest_speakers = {name: {row["speaker"] for row in rows} for name, rows in manifests.items()}
    if (manifest_speakers["train"] & manifest_speakers["validation"]
            or manifest_speakers["train"] & manifest_speakers["calibration"]
            or manifest_speakers["validation"] & manifest_speakers["calibration"]):
        raise ValueError("The saved SAE manifests have speaker leakage")
    case_speakers = {case["speaker"] for case in cases}
    if (case_speakers & manifest_speakers["train"]
            or len(case_speakers & manifest_speakers["validation"]) != 20
            or len(case_speakers & manifest_speakers["calibration"]) != 20):
        raise ValueError("The comparison speakers do not match the expected speaker-disjoint dev partition")
    case_use = {identifier: ("validation" if identifier in split_overlap["validation_ids"] else "BatchTopK calibration")
                for identifier in case_ids}
    checks = source_audit.get("refresh_checks", {})
    standalone = source_audit.get("standalone_checks", {})
    if not (source_audit.get("embedded_assets_preserved") is True
            and checks.get("original_case_metrics_preserved") is True
            and checks.get("wav_headers_and_sample_counts_validated") == 160
            and checks.get("case_quality_and_rate_rows_match_records") == 40
            and standalone.get("single_file_relocation_checked") is True
            and standalone.get("external_dependencies") == []
            and standalone.get("script_elements") == 0):
        raise ValueError("The original PitchFlower artifact does not pass its recorded verification checks")

    reference_wavs, pitchflower_wavs, pitchflower_audio_paths = [], [], []
    for case in cases:
        reference16 = read_pitchflower_pcm(input_root / "references" / case["reference_24k_path"])
        inference_input, rate = sf.read(input_root / "references" / case["path"], dtype="float32", always_2d=True)
        if rate != 16000 or inference_input.shape != (case["samples"], 1) or not np.isfinite(inference_input).all():
            raise ValueError(f"Invalid 16-kHz inference reference: {case['id']}")
        if not np.allclose(inference_input[:, 0], reference16, rtol=0, atol=1e-7):
            raise ValueError(f"Inference input differs from the embedded source reference: {case['id']}")
        reference_wavs.append(reference16)
        pitchflower_wavs.append(read_pitchflower_pcm(input_root / "references" / case["pitchflower_path"]))
        pitchflower_audio_paths.append(input_root / "references" / case["pitchflower_path"])
    saved_methods = scores["methods"]
    expected_summaries = {}
    recomputed_pitchflower = []
    max_error = {method: {metric: 0.0 for metric in METRICS} for method in ("pitchflower", *METHODS)}
    for index, case in enumerate(cases):
        pf = audio_metrics(reference_wavs[index], pitchflower_wavs[index], speech=True)
        expected_pf = official["cases_data"][index]
        mismatch = score_matches(pf, {"pesq_wb": expected_pf["pesq_wb"], "stoi": expected_pf["stoi"]}, tolerance=1e-4)
        if mismatch:
            raise ValueError(f"PitchFlower embedded audio does not reproduce its saved scores for {case['id']}: {mismatch}")
        for metric in METRICS:
            a = pf.get(metric)
            b = {"pesq_wb": expected_pf.get("pesq_wb"), "stoi": expected_pf.get("stoi")}.get(metric)
            if a is not None and b is not None:
                max_error["pitchflower"][metric] = max(max_error["pitchflower"][metric], abs(a - b))
        recomputed_pitchflower.append({"id": case["id"], "speaker": case["speaker"], **pf})
        for method in METHODS:
            if method not in saved_methods:
                raise ValueError(f"Missing method result: {method}")
            rows = saved_methods[method]["cases"]
            if len(rows) != 40 or rows[index]["id"] != case["id"]:
                raise ValueError(f"Case order/count mismatch in {method}")
            row = rows[index]
            output = read_audio(result_root / row["reconstruction_wav"], case["samples"])
            recalculated = audio_metrics(reference_wavs[index], output, speech=True)
            mismatch = score_matches(recalculated, row)
            if mismatch:
                raise ValueError(f"Saved and recomputed scores disagree for {method}/{case['id']}: {mismatch}")
            for metric in METRICS:
                if recalculated.get(metric) is not None:
                    max_error[method][metric] = max(max_error[method][metric], abs(recalculated[metric] - row[metric]))
    # Check PitchFlower report's stored means against the scores of its embedded audio.
    for metric in ("pesq_wb", "stoi"):
        computed = mean_metric(recomputed_pitchflower, metric)["mean"]
        if not math.isclose(computed, official["means"][metric], rel_tol=0, abs_tol=1e-4):
            raise ValueError(f"PitchFlower summary mean mismatch for {metric}: {computed} vs {official['means'][metric]}")

    method_summaries = {"pitchflower": summarize(recomputed_pitchflower)}
    for method in METHODS:
        rows = saved_methods[method]["cases"]
        summary = summarize(rows)
        if summary != saved_methods[method]["summary"]:
            for metric in METRICS:
                a, b = summary[metric], saved_methods[method]["summary"].get(metric)
                if a is None and b is None:
                    continue
                if a is None or b is None or not math.isclose(a, b, rel_tol=0, abs_tol=1e-12):
                    raise ValueError(f"Stored {method} aggregate does not match its case scores for {metric}")
        if summary["cases"] != 40 or any(summary[f"{metric}_valid"] != 40 for metric in METRICS):
            raise ValueError(f"Invalid or missing metric values in {method}")
        method_summaries[method] = summary
        expected_summaries[method] = summary

    case_score_rows = []
    for index, case in enumerate(cases):
        case_entry = {"id": case["id"], "speaker": str(case["speaker"]), "systems": {}}
        pitchflower_row = recomputed_pitchflower[index]
        case_entry["systems"]["pitchflower"] = {key: pitchflower_row.get(key) for key in (*METRICS, "statuses")}
        for method in METHODS:
            row = saved_methods[method]["cases"][index]
            case_entry["systems"][method] = {key: row.get(key) for key in (*METRICS, "statuses", "active_coefficients")}
        case_score_rows.append(case_entry)

    report_dir.mkdir(parents=True, exist_ok=True)
    rows_html = []
    audio_validation = 0
    png_validation = 0
    for method in ("pitchflower", *METHODS):
        summary = method_summaries[method]
        if method == "pitchflower":
            rate = "6.420 kbit/s estimated payload; not a measured bitstream"
            checkpoint = f"RVQ checkpoint step {official['checkpoint_step']}"
            activity = "—"
        else:
            record = saved_methods[method]
            rate = "Not measured (unquantized continuous latents)"
            checkpoint = f"best epoch {record['best_epoch']}; SHA-256 {record['checkpoint_sha256']}"
            activity = f"{np.mean([case['active_coefficients'] for case in record['cases']]):.1f} nonzero latent coefficients/utterance"
        rows_html.append(
            f"<tr><th>{html.escape(LABELS[method])}</th>"
            f"<td>{number(summary['pesq_wb'])} ({summary['pesq_wb_valid']}/40)</td>"
            f"<td>{number(summary['stoi'])} ({summary['stoi_valid']}/40)</td>"
            f"<td>{number(summary['si_sdr'])} dB</td><td>{number(summary['nmse'])}</td>"
            f"<td>{html.escape(rate)}</td><td>{html.escape(activity)}</td>"
            f"<td><small>{html.escape(checkpoint)}</small></td></tr>"
        )
    parts = [
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">",
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\"><title>Sparse AE and PitchFlower comparison</title>",
        "<style>body{font:16px/1.5 system-ui,sans-serif;max-width:1500px;margin:2rem auto;padding:0 1rem;color:#18212b}table{border-collapse:collapse;width:100%;margin:1rem 0 2rem}th,td{border:1px solid #cbd5e1;padding:.55rem;text-align:left;vertical-align:top}thead,th{background:#f1f5f9}audio{width:190px}img{width:100%;max-width:360px}.case{border-top:3px solid #64748b;margin-top:2rem;padding-top:1rem}.note{background:#fff7ed;padding:1rem;border-left:4px solid #ea580c}small{overflow-wrap:anywhere}</style></head><body>",
        "<h1>Sparse autoencoders and PitchFlower: fixed LibriSpeech crop comparison</h1>",
        "<p>This standalone report contains the 40 source crops, PitchFlower reconstructions, seven SAE reconstructions, and per-case scores. Source and PitchFlower audio retain the original 24-kHz, 16-bit PCM WAVs; SAE output is 16-kHz, 16-bit PCM. Quality metrics are computed at 16 kHz from the declared resampling path and from the decoded PCM16 SAE outputs. It can be opened without companion files or an internet connection.</p>",
        "<div class=\"note\"><strong>Evaluation limits.</strong> These are 40 short 1.504-second dev-clean crops from the PitchFlower report. There is no training-utterance overlap, and the saved SAE manifests have speaker-disjoint train, validation, and calibration partitions. However, 20 exact case utterances appear among the 256 validation utterances used for checkpoint selection; the other 20 exact cases appear among the 256 calibration utterances used to fit the BatchTopK inference threshold. These scores are therefore a reproduction/diagnostic comparison, not an unbiased held-out test. The source crops follow PitchFlower's peak-normalized reference level; SAE training used native LibriSpeech amplitude. The inference input gain differs from SAE training. SAE outputs are unquantized; no sparse-AE bitrate or rate-distortion point has been measured. PitchFlower's 6.420 kbit/s value is a fixed-width decoder-input payload estimate, not a measured serialized bitstream.</div>",
        "<h2>Mean scores across 40 cases</h2>",
        "<p>Higher PESQ-WB, STOI, and SI-SDR are better; lower NMSE is better. These are separate metrics, not one combined score. SI-SDR is computed on the aligned waveforms with no per-case time-shift search; a vocoder reconstruction can score lower on waveform SI-SDR even when PESQ/STOI indicate better perceptual similarity. All displayed metric values are valid for 40/40 cases.</p>",
        "<table><thead><tr><th>System</th><th>PESQ-WB</th><th>STOI</th><th>SI-SDR</th><th>NMSE</th><th>Rate</th><th>Latent activity</th><th>Checkpoint</th></tr></thead><tbody>",
        *rows_html,
        "</tbody></table>",
        "<h2>Per-case audio and scores</h2>",
        "<p>The same reference crop is used for every reconstruction. Spectrograms share a 16-kHz sample rate and color range.</p>",
    ]
    for index, case in enumerate(cases):
        normalized_id = case["id"].removeprefix("librispeech:")
        parts += [f"<section class=\"case\"><h3>{index + 1}. {html.escape(case['id'])}</h3>",
                  f"<p>Speaker {html.escape(case['speaker'])}; 1.504 s; {case['samples']:,} samples at 16 kHz. This exact utterance was included in the SAE {html.escape(case_use[normalized_id])} selection.</p>",
                  "<table><thead><tr><th>System</th><th>Audio</th><th>Spectrogram</th><th>PESQ-WB</th><th>STOI</th><th>SI-SDR</th><th>NMSE</th></tr></thead><tbody>"]
        variants = [("reference", None, reference_wavs[index], input_root / "references" / case["reference_24k_path"])]
        variants.append(("pitchflower", recomputed_pitchflower[index], pitchflower_wavs[index], pitchflower_audio_paths[index]))
        for method in METHODS:
            row = saved_methods[method]["cases"][index]
            audio = read_audio(result_root / row["reconstruction_wav"], case["samples"])
            variants.append((method, row, audio, result_root / row["reconstruction_wav"]))
        for method, row, audio, wav_path in variants:
            audio_uri = data_uri(wav_path, "audio/wav")
            image_path = report_dir / f".sparse-ae-spectrum-{index:02d}-{method}.png"
            image_uri = spectrogram_uri(audio, image_path)
            image_path.unlink()
            audio_validation += 1
            png_validation += 1
            scores_html = "<td>Reference</td><td>—</td><td>—</td><td>—</td>" if row is None else (
                f"<td>{number(row['pesq_wb'])}</td><td>{number(row['stoi'])}</td>"
                f"<td>{number(row['si_sdr'])} dB</td><td>{number(row['nmse'])}</td>")
            parts.append(f"<tr><th>{html.escape(LABELS.get(method, method))}</th><td><audio controls preload=\"none\" src=\"{audio_uri}\"></audio></td>"
                         f"<td><img alt=\"{html.escape(LABELS.get(method, method))} spectrogram, case {index + 1}\" src=\"{image_uri}\"></td>{scores_html}</tr>")
        parts.append("</tbody></table></section>")
    parts += ["<h2>Reproducibility record</h2>",
              f"<p>PitchFlower report checkpoint SHA-256: <code>{html.escape(official['checkpoint_sha256'])}</code>. SAE campaign: <code>/home/kiran/sparse-ae-work/sparse-ae/runs/librispeech_10h_parallel</code>.</p>",
              "<p>Each SAE checkpoint was the saved best validation-loss checkpoint. We recomputed MSE, NMSE, SI-SDR, PESQ-WB, and STOI for every output and compared them with the server's saved per-case scores. The PitchFlower scores were recomputed from its embedded PCM output and checked against its saved per-case scores and means. See the companion JSON verification record for hashes and checks.</p>",
              "</body></html>"]
    report_path = report_dir / "sparse_ae_pitchflower_best_testcases.html"
    report_path.write_text("\n".join(parts), encoding="utf-8")
    report_text = report_path.read_text(encoding="utf-8")
    report_checks = {
        "html_bytes": report_path.stat().st_size,
        "embedded_audio_elements": report_text.count("<audio controls"),
        "embedded_spectrogram_images": report_text.count("<img alt="),
        "expected_audio_elements": 40 * 9,
        "expected_spectrogram_images": 40 * 9,
        "external_dependencies": bool(re.search(r"(?:src|href)=\"https?://", report_text)),
        "script_elements": bool(re.search(r"<script\b", report_text, re.I)),
        "all_audio_and_spectrograms_embedded": audio_validation == 360 and png_validation == 360,
        "embedded_audio_format": "Original reference/PitchFlower are 24-kHz mono PCM16; SAE outputs are 16-kHz mono PCM16. Scores use the declared 16-kHz resampling path and decoded SAE PCM16.",
    }
    if (report_checks["embedded_audio_elements"] != report_checks["expected_audio_elements"]
            or report_checks["embedded_spectrogram_images"] != report_checks["expected_spectrogram_images"]
            or report_checks["external_dependencies"] or report_checks["script_elements"]):
        raise ValueError(f"Standalone report validation failed: {report_checks}")
    final_metrics = {
        "cases": 40,
        "unique_speakers": 40,
        "crop_samples": 24064,
        "sample_rate": 16000,
        "means": method_summaries,
        "case_scores": case_score_rows,
        "pitched_crops_source_report_sha256": sha256(source_report),
        "pitchflower_checkpoint_sha256": official["checkpoint_sha256"],
        "sae_checkpoints": {method: {"sha256": saved_methods[method]["checkpoint_sha256"],
                                     "best_epoch": saved_methods[method]["best_epoch"]} for method in METHODS},
        "rate_claim": "No SAE bitrate measured; no serialized sparse-AE bitstream.",
        "evaluation_limit": scores["evaluation"]["test_status"],
        "data_overlap_audit": {
            "exact_case_utterances_in_train": len(split_overlap["train_ids"]),
            "exact_case_utterances_in_validation_selection": len(split_overlap["validation_ids"]),
            "exact_case_utterances_in_calibration_selection": len(split_overlap["calibration_ids"]),
            "case_speakers_in_train": len(case_speakers & manifest_speakers["train"]),
            "case_speakers_in_validation_partition": len(case_speakers & manifest_speakers["validation"]),
            "case_speakers_in_calibration_partition": len(case_speakers & manifest_speakers["calibration"]),
            "train_validation_calibration_speakers_disjoint": True,
        },
    }
    metrics_path = report_dir / "sparse_ae_pitchflower_best_metrics.json"
    metrics_path.write_text(json.dumps(final_metrics, indent=2), encoding="utf-8")
    case_csv = report_dir / "sparse_ae_pitchflower_best_case_scores.csv"
    with case_csv.open("w", newline="", encoding="utf-8") as stream:
        import csv
        fields = ["system", "id", "speaker", *METRICS, "active_coefficients", "statuses"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for case in case_score_rows:
            for method, row in case["systems"].items():
                writer.writerow({"system": method, "id": case["id"], "speaker": case["speaker"],
                                 **{key: row.get(key) for key in METRICS},
                                 "active_coefficients": row.get("active_coefficients"),
                                 "statuses": json.dumps(row.get("statuses", {}), sort_keys=True)})
    verification = {
        "cases_and_speakers_validated": True,
        "per_case_metrics_recomputed_and_matched": True,
        "pitchflower_embedded_pcm_scores_reproduced": True,
        "pitchflower_summary_means_matched": True,
        "case_score_rows": len(case_score_rows) * 8,
        "pitchflower_metric_tolerance": 1e-4,
        "sae_metric_tolerance": 1e-6,
        "all_score_values_valid": True,
        "max_score_recomputation_abs_error": max_error,
        "checkpoint_sha256_values_present": all(saved_methods[method]["checkpoint_sha256"] for method in METHODS),
        "wav_headers_and_sample_counts_validated": audio_validation,
        "spectrograms_generated_and_embedded": png_validation,
        "standalone_report": report_checks,
        "source_pitchflower_verification_passed": True,
        "sae_speaker_partition_leakage": False,
        "exact_case_utterance_overlap_audit": final_metrics["data_overlap_audit"],
    }
    verification_path = report_dir / "sparse_ae_pitchflower_best_verification.json"
    verification_path.write_text(json.dumps(verification, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(report_path), "metrics": str(metrics_path),
                      "verification": str(verification_path), "report_checks": report_checks,
                      "means": method_summaries}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", default="tmp/sparse_ae_pitchflower_comparison")
    parser.add_argument("--report-dir", default="approaches/sparse-ae/reports")
    args = parser.parse_args()
    build(args)


if __name__ == "__main__":
    main()
