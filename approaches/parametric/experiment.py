"""Single-testcase EMD/VMD window-length sweep, without dataset training."""
from datetime import datetime, timezone
import importlib.metadata
import json
import math
from pathlib import Path
import time

import numpy as np
import soundfile as sf
import torch

from approaches.parametric.decomposition import EMDDecomposer, VMDDecomposer
from approaches.parametric.reconstruction import WindowedReconstructor
from approaches.parametric.report import write_report
from util.data.preprocessing import load_audio
from util.io import write_csv, write_json
from util.metrics import speech_quality_metrics, waveform_metrics
from util.provenance import environment, file_hash, source_fingerprint


def _score(reference, estimate, sample_rate, speech_metrics):
    """Score aligned full clips, with no fitted gain, time shift or normalization."""
    # Shared spectral metrics expect (batch, channels, samples), even though
    # this experiment processes one mono clip at a time.
    metrics = waveform_metrics(torch.from_numpy(reference)[None, None],
                               torch.from_numpy(estimate)[None, None], sample_rate)
    if speech_metrics:
        metrics.update(speech_quality_metrics(reference, estimate, sample_rate))
    else:
        metrics.update(pesq_wb=None, stoi=None, pesq_wb_status="disabled", stoi_status="disabled")
    return metrics


def _save_and_read(path, audio, sample_rate):
    """Measure the saved FLOAT WAV that is also presented for listening."""
    sf.write(path, audio, sample_rate, subtype="FLOAT")
    saved, actual_rate = sf.read(path, dtype="float64")
    if actual_rate != sample_rate or saved.shape != audio.shape or not np.isfinite(saved).all():
        raise RuntimeError("Saved reconstruction has invalid samples or timing")
    return saved


def run(audio, *, window_ms=(20.0, 40.0, 80.0, 160.0), overlap=0.5,
        sample_rate=16000, emd_max_imfs=-1, emd_max_iterations=1000,
        vmd_modes=8, vmd_alpha=2000.0, vmd_tau=0.0, vmd_tolerance=1e-7,
        start_seconds=0.0, duration_seconds=None, output=None,
        save_modes=False, speech_metrics=True):
    """Compare window lengths on one mono recording using ordinary arguments.

    EMD keeps its native residue separate. VMD reports input minus mode sum as
    a residual control. Neither condition is a compressed representation; all
    modes remain sampled waveforms. PESQ/STOI are objective speech proxies and
    are scored on the full reconstructed testcase, never individual short
    windows. The report includes reference audio for subjective comparison.

    Args:
        audio: Local mono audio file; no corpus download occurs.
        window_ms: Requested frame durations in milliseconds.
        overlap: Fraction of overlap; realized integer hops are recorded.
        sample_rate: Analysis rate in Hz; PESQ-WB requires 16000.
        start_seconds: Beginning of the selected crop in the source file.
        duration_seconds: Optional crop duration; None selects the remaining file.
        output: New output directory; defaults to a timestamped run in runs/.
        save_modes: Store per-frame sampled modes/residue for later investigation.
        speech_metrics: Enable optional PESQ/STOI, recording unavailable statuses.
    """
    source = Path(audio).expanduser().resolve()
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate < 2000:
        raise ValueError("sample_rate must be an integer >= 2000 Hz")
    windows = tuple(float(value) for value in window_ms)
    if not windows or any(not math.isfinite(value) or value <= 0 for value in windows):
        raise ValueError("window_ms must contain positive finite durations")
    samples = [round(value * sample_rate / 1000) for value in windows]
    if len(set(samples)) != len(samples):
        raise ValueError("Window lengths must produce distinct sample counts")
    reconstructors = [WindowedReconstructor(size, overlap) for size in samples]
    if not math.isfinite(start_seconds) or start_seconds < 0:
        raise ValueError("start_seconds must be finite and nonnegative")
    if duration_seconds is not None and (not math.isfinite(duration_seconds) or duration_seconds <= 0):
        raise ValueError("duration_seconds must be finite and positive")
    info = sf.info(source)
    if info.channels != 1:
        raise ValueError("The testcase must be mono; prepare a mono source explicitly")
    start = round(start_seconds * info.samplerate)
    stop = info.frames if duration_seconds is None else min(info.frames, start + round(duration_seconds * info.samplerate))
    if start >= stop:
        raise ValueError("Selected source crop contains no samples")
    signal = load_audio({"corpus": "local", "id": source.stem, "path": str(source),
                         "num_samples": info.frames, "start": start, "stop": stop}, sample_rate).astype(np.float64)
    # Import both optional backends before creating a run, so missing packages
    # fail with an installation hint rather than leave misleading partial runs.
    backends = [EMDDecomposer(emd_max_imfs, emd_max_iterations),
                VMDDecomposer(vmd_modes, vmd_alpha, vmd_tau, vmd_tolerance)]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = Path(output).expanduser().resolve() if output is not None else Path("runs") / f"parametric_{stamp}"
    destination.mkdir(parents=True, exist_ok=False)
    began = time.perf_counter()
    settings = {
        "source": str(source), "source_sha256": file_hash(source),
        "source_sample_rate": info.samplerate, "source_start_sample": start, "source_stop_sample": stop,
        "sample_rate": sample_rate, "samples": len(signal), "window_ms": list(windows),
        "window_samples": samples, "hop_samples": [item.hop_samples for item in reconstructors],
        "overlap": overlap, "boundary_padding": "reflection within selected clip",
        "synthesis": "positive Hann normalized overlap-add; rectangular if overlap=0",
        "emd_max_imfs": emd_max_imfs, "emd_max_iterations": emd_max_iterations,
        "vmd_modes": vmd_modes, "vmd_alpha": vmd_alpha, "vmd_tau": vmd_tau,
        "vmd_tolerance": vmd_tolerance, "vmd_init": "uniform", "vmd_dc": False,
        "speech_metrics": speech_metrics, "save_modes": save_modes,
        "scope": "Sampled decomposition/windowing baseline. No parameter fitting, quantization, bitrate or training.",
        "residual_control": "EMD native residue; VMD raw input-minus-mode-sum residual. Controls are not compressed.",
    }
    write_json(destination / "settings.json", settings)
    runtime = environment()
    runtime["decomposition_packages"] = {name: importlib.metadata.version(name) for name in ("EMD-signal", "vmdpy")}
    write_json(destination / "environment.json", runtime)
    write_json(destination / "source.json", source_fingerprint(Path(__file__).resolve().parents[2]))
    print(f"RUN={destination}", flush=True)
    rows = []
    try:
        reference = _save_and_read(destination / "reference.wav", signal, sample_rate)
        duration = len(reference) / sample_rate
        shared = {"frames": 0, "mean_modes": None, "max_modes": None, "decomposition_seconds": 0.0,
                  "decomposition_rtf": 0.0, "window_ms": None, "actual_window_ms": None,
                  "window_samples": None, "hop_samples": None}
        rows.append({"method": "reference", "reconstruction": "self_reference", **shared,
                     "audio": "reference.wav", **_score(reference, reference, sample_rate, speech_metrics)})
        total = len(backends) * len(windows)
        completed = 0
        for backend in backends:
            for requested_ms, reconstructor in zip(windows, reconstructors):
                size = reconstructor.window_samples
                folder = destination / backend.name / f"window_{size}_samples"
                folder.mkdir(parents=True)

                def persist_frame(index, frame_start, decomposition):
                    np.savez_compressed(folder / "modes" / f"frame_{index:06d}.npz",
                                        start_sample=frame_start, modes=decomposition.modes,
                                        residual=decomposition.residual)

                if save_modes:
                    (folder / "modes").mkdir()
                def progress_frame(index, frame_start, decomposition):
                    if save_modes:
                        persist_frame(index, frame_start, decomposition)
                    if index % 25 == 0:
                        write_json(destination / "progress.json", {
                            "status": "running", "method": backend.name, "window_samples": size,
                            "frame": index, "cases_completed": completed, "total_cases": total,
                        })

                print(f"DECOMPOSE method={backend.name} window_ms={requested_ms:g}", flush=True)
                result = reconstructor.reconstruct(reference, backend, on_frame=progress_frame)
                frame_rows = [{**row, "diagnostics": json.dumps(row["diagnostics"], sort_keys=True)} for row in result.frames]
                write_csv(folder / "frames.csv", frame_rows)
                counts = [row["mode_count"] for row in result.frames]
                shared = {"window_ms": requested_ms, "actual_window_ms": 1000 * size / sample_rate,
                          "window_samples": size, "hop_samples": reconstructor.hop_samples,
                          "frames": len(counts), "mean_modes": float(np.mean(counts)), "max_modes": max(counts),
                          "decomposition_seconds": result.decomposition_seconds,
                          "decomposition_rtf": result.decomposition_seconds / duration}
                for label, waveform in (("modes", result.modes),
                                        ("modes_plus_residual_control", result.modes_plus_residual_control)):
                    target = folder / f"{label}.wav"
                    saved = _save_and_read(target, waveform, sample_rate)
                    rows.append({"method": backend.name, "reconstruction": label, **shared,
                                 "audio": target.relative_to(destination).as_posix(),
                                 **_score(reference, saved, sample_rate, speech_metrics)})
                completed += 1
                write_csv(destination / "metrics.csv", rows)
                write_report(destination, rows)
        write_json(destination / "summary.json", {"settings": settings, "results": rows,
                   "metric_notes": "PESQ-WB estimates quality; STOI estimates intelligibility. Missing values retain explicit statuses. Scores are full-clip, without gain/alignment fitting.",
                   "runtime_notes": "Modes and residual control share the same decomposition; reported fit time is repeated, not a separate fit."})
        write_json(destination / "progress.json", {"status": "complete", "cases_completed": completed,
                   "total_cases": total, "seconds": time.perf_counter() - began})
    except (Exception, KeyboardInterrupt) as error:
        write_json(destination / "progress.json", {"status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                   "error": f"{type(error).__name__}: {error}", "seconds": time.perf_counter() - began})
        raise
    print(f"REPORT={destination / 'index.html'}", flush=True)
    return destination
