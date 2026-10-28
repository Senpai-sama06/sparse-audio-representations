"""Controlled pitch interventions and measurements for the frozen affine model.

WORLD is imported only by the preparation functions; ordinary training does not
need it. Channel identifiers are zero based throughout.
"""
import numpy as np
import torch
from scipy import signal

SR = 16000
SAMPLES = 12288
CORE = (4096, 8192)
RADIUS = 1020
SHIFTS = np.array([0, 13, 29, 61, 103, 157, 199, 239])
TARGETS = np.arange(80., 361., 5.)
SUPPORTS = (.01, .1, 1.)


def normalize(x, rms=.05):
    """Common 10-ms edge fade and exact whole-excerpt RMS normalization."""
    x = np.asarray(x, dtype=np.float64).copy()
    fade = np.sin(np.linspace(0, np.pi / 2, 160)) ** 2
    x[:160] *= fade
    x[-160:] *= fade[::-1]
    return (x * rms / max(np.sqrt(np.mean(x*x)), 1e-12)).astype(np.float32)


def interior_indices(shift, samples=SAMPLES, core=CORE, radius=RADIUS):
    """Complete encoder receptive field must lie in the same source core."""
    centers = np.arange((samples + 255) // 256) * 256 - shift
    return np.flatnonzero((centers - radius >= core[0]) &
                          (centers + radius < core[1]))


def pitch_check(x):
    """Independent pYIN check, not the WORLD F0 supplied to synthesis."""
    import librosa
    from lrac.channel_analysis import PYIN
    f0, voiced, _ = librosa.pyin(x, sr=SR, fill_na=np.nan, **PYIN)
    centers = np.arange(len(f0)) * 256
    mask = (centers >= CORE[0]) & (centers < CORE[1])
    usable = mask & voiced & np.isfinite(f0)
    return dict(pyin_f0_hz=float(np.median(f0[usable])) if usable.any() else None,
                pyin_voiced_fraction=float(voiced[mask].mean()))


def coarse_envelope(x):
    """Gain-free log-power shape, cepstrally smoothed below 2-ms quefrency."""
    _, p = signal.welch(x[CORE[0]:CORE[1]], SR, nperseg=2048,
                        noverlap=1536, detrend=False, nfft=4096)
    cepstrum = np.fft.irfft(10*np.log10(np.maximum(p, 1e-16)))
    cepstrum[33:-32] = 0
    envelope = np.fft.rfft(cepstrum).real
    f = np.fft.rfftfreq(4096, 1/SR)
    shape = envelope[(f >= 300) & (f <= 6000)]
    return shape - shape.mean()


def analysis_filters(model):
    """Exact interior encoder rows by differentiation of the affine encoder.

    Return cross-correlation kernels and the (interior) affine offset. This is
    different from the decoder's unit-impulse synthesis waveforms.
    """
    device = next(model.parameters()).device
    x = torch.zeros(1, 1, SAMPLES, device=device, requires_grad=True)
    with torch.backends.cudnn.flags(enabled=False, allow_tf32=False):
        z = model.encoder(x)
        t = SAMPLES // 512
        filters = []
        for c in range(z.shape[1]):
            derivative, = torch.autograd.grad(z[0, c, t], x, retain_graph=True)
            full = derivative[0, 0].detach().cpu().numpy()
            outside = full.copy()
            outside[t*256-RADIUS:t*256+RADIUS+1] = 0
            if np.any(outside != 0):
                raise ValueError("Unexpected encoder receptive field")
            filters.append(full[t*256-RADIUS:t*256+RADIUS+1])
        return np.stack(filters), z[0, :, t].detach().cpu().numpy()


def spectral_prediction(x, filters, envelope=None, f0=None):
    """Phase-averaged stationary power predictions, without fitting model weights.

    PSD overlap uses the observed core waveform. Harmonic overlap instead uses
    ONLY the fixed WORLD envelope and target F0. Both ignore local nonstationarity
    and phase cross terms. The harmonic model additionally ignores aperiodicity.
    """
    nfft = 16384
    f = np.fft.rfftfreq(nfft, 1/SR)
    gain = np.abs(np.fft.rfft(filters, n=nfft, axis=-1))**2
    freq, psd = signal.periodogram(x[CORE[0]:CORE[1]], SR, window="boxcar",
                                  detrend=False, nfft=nfft, scaling="density")
    observed = np.trapezoid(gain * psd, freq, axis=-1)
    if envelope is None:
        return observed
    harmonics = np.arange(f0, SR/2, f0)
    ef = np.linspace(0, SR/2, len(envelope))
    weights = np.interp(harmonics, ef, envelope)
    harmonic_gain = np.stack([np.interp(harmonics, f, g) for g in gain])
    # Whole-source core power, not a separately fitted per-channel gain.
    weights *= np.mean(x[CORE[0]:CORE[1]].astype(float)**2) / max(weights.sum(), 1e-20)
    return observed, harmonic_gain @ weights


@torch.no_grad()
def measure(model, x, bias, filters=None):
    """Equal phase averaging of RMS power/rates; top-k spans the full excerpt."""
    device = next(model.parameters()).device
    waveforms = np.stack([np.roll(x, int(s)) for s in SHIFTS])
    with torch.backends.cudnn.flags(enabled=False, allow_tf32=False):
        rep = model.encode_dense(torch.from_numpy(waveforms[:, None]).to(device))
        z = rep.concatenate().cpu().numpy()
        power, rates = [], []
        for fraction in SUPPORTS:
            sparse = model.sparsify(rep, keep_fraction=fraction).concatenate().cpu().numpy()
            power.append(np.mean([np.mean(sparse[i, interior_indices(s)]**2, axis=0)
                                  for i, s in enumerate(SHIFTS)], axis=0))
            rates.append(np.mean([np.mean(sparse[i, interior_indices(s)] != 0, axis=0)
                                  for i, s in enumerate(SHIFTS)], axis=0))
    centered = np.mean([np.mean((z[i, interior_indices(s)]-bias)**2, axis=0)
                        for i, s in enumerate(SHIFTS)], axis=0)
    error = None
    if filters is not None:
        indices = interior_indices(0)
        windows = np.stack([x[t*256-RADIUS:t*256+RADIUS+1] for t in indices])
        predicted = windows.astype(float) @ filters.astype(float).T + bias
        error = float(np.max(np.abs(predicted - z[0, indices])))
    return dict(power=np.stack(power), rate=np.stack(rates), centered_power=centered,
                filter_check_max_abs=error)


def prepare_speaker(task):
    """Choose sources without examining the network, then synthesize the sweep."""
    import pyworld as pw
    from lrac.data.preprocessing import load_audio
    from lrac.utils import write_json
    from pathlib import Path
    speaker, rows, destination = task
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    voiced_source = unvoiced_source = None
    for row in rows[:30]:
        x = load_audio(row, SR).astype(np.float64)
        raw, times = pw.dio(x, SR, f0_floor=50, f0_ceil=600, frame_period=5)
        f0 = pw.stonemask(x, raw, times, SR)
        for center in np.arange(.4, len(x)/SR-.4, .032):
            start = int(round(center*SR)) - SAMPLES//2
            if start < 0 or start+SAMPLES > len(x):
                continue
            core = (times >= (start+CORE[0])/SR) & (times < (start+CORE[1])/SR)
            f = f0[core]
            kind = None
            if voiced_source is None and (f > 0).mean() >= .98:
                positive = f[f > 0]
                if np.std(positive)/np.mean(positive) <= .15:
                    kind = "voiced"
            if unvoiced_source is None and (f == 0).mean() >= .98:
                kind = "unvoiced"
            if kind is None:
                continue
            excerpt = x[start:start+SAMPLES]
            core_rms = np.sqrt(np.mean(excerpt[CORE[0]:CORE[1]]**2))
            minimum = .25 if kind == "unvoiced" else .1
            if core_rms < max(1e-4, np.sqrt(np.mean(x*x))*minimum):
                continue
            checked = pitch_check(excerpt.astype(np.float32))
            item = dict(id=row["id"], speaker=speaker, start_sample=start, **checked)
            if kind == "voiced" and checked["pyin_voiced_fraction"] >= .9:
                if checked["pyin_f0_hz"] is not None and abs(checked["pyin_f0_hz"]/np.median(f)-1) <= .05:
                    # DIO can change its voicing decision after cropping. Check
                    # the exact excerpt before accepting it, then continue the
                    # same source-only search if it fails (no network involved).
                    crop_raw, crop_times = pw.dio(excerpt, SR, f0_floor=50, f0_ceil=600, frame_period=5)
                    crop_f0 = pw.stonemask(excerpt, crop_raw, crop_times, SR)
                    crop_core = (crop_times >= CORE[0]/SR) & (crop_times < CORE[1]/SR)
                    if (crop_f0[crop_core] > 0).mean() >= .95:
                        voiced_source = (excerpt, item)
            elif kind == "unvoiced" and checked["pyin_voiced_fraction"] <= .1:
                unvoiced_source = (excerpt, item)
            if voiced_source is not None and unvoiced_source is not None:
                break
        if voiced_source is not None and unvoiced_source is not None:
            break
    if voiced_source is None:
        raise ValueError(f"No eligible voiced source for speaker {speaker}")
    x, item = voiced_source
    raw, times = pw.dio(x, SR, f0_floor=50, f0_ceil=600, frame_period=5)
    f0 = pw.stonemask(x, raw, times, SR)
    sp = pw.cheaptrick(x, f0, times, SR, fft_size=2048, f0_floor=50)
    ap = pw.d4c(x, f0, times, SR, fft_size=2048)
    core = (times >= CORE[0]/SR) & (times < CORE[1]/SR)
    if (f0[core] > 0).mean() < .95:
        raise ValueError(f"Voicing not stable after cropping: {speaker}")
    item["world_f0_hz"] = float(np.median(f0[core][f0[core] > 0]))
    variants, checks = [normalize(x)], []
    names = ["original", "sham", "flat_reference", *[f"f0_{f:g}" for f in TARGETS], "aperiodic"]
    for name in names[1:]:
        new_f0, new_ap = f0.copy(), ap
        if name == "flat_reference":
            new_f0[f0 > 0] = item["world_f0_hz"]
        elif name.startswith("f0_"):
            new_f0[f0 > 0] = float(name[3:])
        elif name == "aperiodic":
            new_f0[:] = 0
            new_ap = np.ones_like(ap)
        y = pw.synthesize(new_f0, sp, new_ap, SR, frame_period=5)
        if len(y) < SAMPLES:
            raise ValueError("WORLD returned a short waveform")
        variants.append(normalize(y[:SAMPLES]))
    envelope_reference = coarse_envelope(variants[1])
    for name, y in zip(names, variants):
        check = pitch_check(y)
        measured = check["pyin_f0_hz"]
        target = float(name[3:]) if name.startswith("f0_") else None
        error = abs(measured/target-1) if target is not None and measured is not None else None
        drift = float(np.sqrt(np.mean((coarse_envelope(y)-envelope_reference)**2)))
        pitch_pass = error is not None and error <= .05 and check["pyin_voiced_fraction"] >= .8
        checks.append(dict(speaker=speaker, source_id=item["id"], variant=name,
            target_f0_hz=target, **check, relative_f0_error=error,
            envelope_shape_rmse_db=drift, samples=len(y),
            rms=float(np.sqrt(np.mean(y.astype(float)**2))),
            core_rms=float(np.sqrt(np.mean(y[CORE[0]:CORE[1]].astype(float)**2))),
            pitch_pass=bool(pitch_pass), strict_pass=bool(pitch_pass and drift <= 3.)))
    negative = normalize(unvoiced_source[0]) if unvoiced_source is not None else np.zeros(SAMPLES, np.float32)
    np.savez_compressed(destination/f"{speaker}.npz", waveforms=np.stack(variants),
        unvoiced=negative, envelope=sp[core].mean(0), world_f0=f0, sp=sp, ap=ap)
    result = dict(speaker=speaker, voiced=item,
                  unvoiced=unvoiced_source[1] if unvoiced_source else None, checks=checks,
                  variants=names)
    write_json(destination/f"{speaker}.json", result)
    return result
