import numpy as np
import torch
import warnings
from lrac.losses import spectral_loss


def speech_quality_metrics(reference, estimate, sample_rate):
    """Optional speech metrics; missing/invalid scores have explicit reasons."""
    x = reference.detach().cpu().numpy().reshape(-1)
    y = estimate.detach().cpu().numpy().reshape(-1)
    result = {"pesq_wb": None, "stoi": None, "visqol": None, "visqol_status": "not integrated; no score computed"}
    for key in ["pesq_wb", "stoi"]:
        try:
            if len(x) < sample_rate // 4 or np.mean(x.astype(np.float64) ** 2) < 1e-10:
                raise ValueError("reference is too short or silent")
            if key == "pesq_wb":
                from pesq import pesq
                if sample_rate != 16000:
                    raise ValueError("wideband PESQ requires 16 kHz")
                value = pesq(sample_rate, x, y, "wb")
            else:
                from pystoi import stoi
                with warnings.catch_warnings(record=True) as emitted:
                    warnings.simplefilter("always")
                    value = stoi(x, y, sample_rate, extended=False)
                if emitted:
                    raise ValueError("; ".join(str(item.message) for item in emitted))
            if not np.isfinite(value):
                raise ValueError("metric returned a nonfinite value")
            result[key], result[f"{key}_status"] = float(value), "ok"
        except (ImportError, ValueError, RuntimeError) as error:
            result[f"{key}_status"] = f"{type(error).__name__}: {error}"
    return result


@torch.no_grad()
def waveform_metrics(reference, estimate, sample_rate):
    x, y = reference.reshape(-1).double(), estimate.reshape(-1).double()
    error = y - x
    energy, error_energy = float(x.square().sum()), float(error.square().sum())
    power = energy / max(x.numel(), 1)
    base = {"mse": float(error.square().mean()), "mae": float(error.abs().mean())}
    if power < 1e-10:
        return {**base, "nmse": None, "log_nmse_db": None, "complex_stft_nmse": None, "silence": True, "snr_db": None, "nrmse": None, "si_sdr_db": None, "mrstft": None, "log_spectral_distance": None, "transient_nrmse": None, "high_band_nrmse": None, "error_rms": float(error.square().mean().sqrt())}
    eps = 1e-20
    snr = 10 * np.log10(max(energy, eps) / max(error_energy, eps))
    xc, yc = x - x.mean(), y - y.mean()
    projection = xc * ((xc @ yc) / xc.square().sum().clamp_min(eps))
    sisdr = 10 * torch.log10(projection.square().sum().clamp_min(eps) / (yc - projection).square().sum().clamp_min(eps))
    n = 512
    if x.numel() < n:
        xp, yp = torch.nn.functional.pad(x, (0, n - x.numel())), torch.nn.functional.pad(y, (0, n - y.numel()))
    else:
        xp, yp = x, y
    w = torch.hann_window(n, device=x.device, dtype=x.dtype)
    ac = torch.stft(xp, n, n // 4, window=w, return_complex=True)
    bc = torch.stft(yp, n, n // 4, window=w, return_complex=True)
    complex_error = float((ac - bc).abs().square().sum() / ac.abs().square().sum().clamp_min(eps))
    a, b = ac.abs(), bc.abs()
    lsd = ((20 * torch.log10(a + 1e-5) - 20 * torch.log10(b + 1e-5)).square().mean(0).sqrt()).mean()
    changes = torch.nn.functional.pad(x.diff().abs(), (1, 0))
    event_mask = changes >= torch.quantile(changes, .9)
    event_error = error[event_mask].square().sum() / x[event_mask].square().sum().clamp_min(eps)
    xf, ef = torch.fft.rfft(x), torch.fft.rfft(error)
    frequencies = torch.fft.rfftfreq(x.numel(), 1 / sample_rate, device=x.device)
    high = frequencies >= sample_rate / 4
    high_error = ef[high].abs().square().sum() / xf[high].abs().square().sum().clamp_min(eps)
    return {**base, "nmse": error_energy / energy, "log_nmse_db": -float(snr), "complex_stft_nmse": complex_error, "silence": False, "snr_db": float(snr), "nrmse": float(np.sqrt(error_energy / energy)), "si_sdr_db": float(sisdr), "mrstft": float(spectral_loss(reference.float(), estimate.float())), "log_spectral_distance": float(lsd), "transient_nrmse": float(event_error.sqrt()), "high_band_nrmse": float(high_error.sqrt()), "error_rms": float(error.square().mean().sqrt())}


def summarize(rows, bootstrap=1000, seed=0):
    metrics = ["mse", "mae", "nmse", "log_nmse_db", "complex_stft_nmse", "snr_db", "nrmse", "si_sdr_db", "mrstft", "log_spectral_distance", "transient_nrmse", "high_band_nrmse", "active_fraction", "coefficients_per_second", "active_coefficients_per_second", "active_coefficients", "active_atom_types", "active_feature_dictionaries", "tail_coefficients", "pesq_wb", "stoi", "inference_seconds", "values_per_input_sample"]
    summary = {"examples": len(rows), "silence_examples": sum(bool(r["silence"]) for r in rows), "metrics": {}, "per_speaker": {}}
    speakers = {}
    for row in rows:
        speakers.setdefault(row.get("speaker") or row["id"], []).append(row)
    rng = np.random.default_rng(seed)
    for key in metrics:
        valid = [r[key] for r in rows if r.get(key) is not None and np.isfinite(r[key])]
        if not valid:
            continue
        means = []
        for speaker, records in speakers.items():
            values = [r[key] for r in records if r.get(key) is not None and np.isfinite(r[key])]
            if values:
                value = float(np.mean(values))
                means.append(value)
                summary["per_speaker"].setdefault(speaker, {})[key] = value
        boot = [float(np.mean(rng.choice(means, len(means), replace=True))) for _ in range(bootstrap)]
        summary["metrics"][key] = {"mean": float(np.mean(valid)), "median": float(np.median(valid)), "p05": float(np.percentile(valid, 5)), "p95": float(np.percentile(valid, 95)), "speaker_macro_mean": float(np.mean(means)), "speaker_bootstrap_ci95": np.percentile(boot, [2.5, 97.5]).tolist() if boot else None}
    return summary
