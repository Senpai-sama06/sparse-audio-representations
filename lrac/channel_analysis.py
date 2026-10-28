"""Measurements for individual channels of a frozen affine sparse autoencoder."""
import numpy as np
import torch
from scipy import ndimage, signal
from lrac.data.preprocessing import load_audio


PYIN = dict(fmin=50, fmax=600, frame_length=1024, hop_length=256,
            n_thresholds=100, beta_parameters=(2, 18), boltzmann_parameter=2,
            resolution=.1, max_transition_rate=35.92, switch_prob=.01,
            no_trough_prob=.01, center=True, pad_mode="constant")


def reference_voicing(row, sample_rate):
    """Signal-derived pYIN labels, aligned to the encoder's 256-sample hop."""
    import librosa
    x = load_audio(row, sample_rate)
    f0, voiced, probability = librosa.pyin(x, sr=sample_rate, fill_na=np.nan, **PYIN)
    rms = librosa.feature.rms(y=x, frame_length=1024, hop_length=256,
                             center=True, pad_mode="constant")[0]
    steps = (len(x) + 255) // 256
    threshold = max(1e-4, float(rms.max()) * 10 ** (-35 / 20))
    audible = rms[:steps] >= threshold
    return dict(f0=f0[:steps], voiced=voiced[:steps], probability=probability[:steps],
                rms=rms[:steps], audible=audible, threshold=threshold)


@torch.no_grad()
@torch.backends.cudnn.flags(enabled=False, allow_tf32=False)
def affine_contributions(model, representation, channels, chunk_size=2):
    """Return g(0), g(z), and g(z restricted to c)-g(0), for one recording.

    Bias is included exactly once in the full reconstruction. No top-k mask is
    recomputed when isolating or removing a channel.
    Native FP32 convolutions (cuDNN and TF32 disabled) preserve numerical affinity;
    the caller's encoder precision flag is restored when the function returns.
    """
    if model.config.activation != "identity" or representation.concatenate().shape[0] != 1:
        raise ValueError("Expected one recording and the affine activation control")
    if torch.backends.cuda.matmul.allow_tf32:
        raise ValueError("Disable CUDA matmul TF32 before affine attribution")
    values = representation.concatenate().transpose(1, 2)
    length = int(representation.lengths[0])
    bias = model.decoder(torch.zeros_like(values))[..., :length]
    full = model.decoder(values)[..., :length]
    parts = []
    for start in range(0, len(channels), chunk_size):
        selected = channels[start:start + chunk_size]
        isolated = torch.zeros(len(selected), *values.shape[1:], device=values.device, dtype=values.dtype)
        for index, channel in enumerate(selected):
            isolated[index, channel] = values[0, channel]
        parts.append((model.decoder(isolated)[..., :length] - bias)[:, 0].cpu().numpy())
    return bias[0, 0].cpu().numpy(), full[0, 0].cpu().numpy(), np.concatenate(parts)


def voicing_counts(active, labels):
    """2x2 counts and a recording/RMS-stratified activation-rate difference.

    Matching strata are 3-dB bins relative to this recording's peak frame RMS.
    Require at least five frames in each class and weight by nV*nU/(nV+nU).
    """
    voiced, audible = labels["voiced"], labels["audible"]
    v, u = voiced & audible, ~voiced & audible
    av, au = active[v].sum(0), active[u].sum(0)
    stable_v = ndimage.minimum_filter1d(v.astype(int), 9, mode="constant", cval=0).astype(bool)
    stable_u = ndimage.minimum_filter1d(u.astype(int), 9, mode="constant", cval=0).astype(bool)
    confidence_v = v & (labels["probability"] >= .5)
    confidence_u = u & (labels["probability"] <= .1)
    rms = labels["rms"]
    bins = np.floor(-20 * np.log10(np.maximum(rms, 1e-12) / max(float(rms.max()), 1e-12)) / 3).astype(int)
    weighted = np.zeros(active.shape[1], dtype=float)
    weight, matched_v, matched_u = 0., 0, 0
    for index in np.unique(bins[audible]):
        bv, bu = v & (bins == index), u & (bins == index)
        nv, nu = int(bv.sum()), int(bu.sum())
        if min(nv, nu) < 5:
            continue
        w = nv * nu / (nv + nu)
        weighted += w * (active[bv].mean(0) - active[bu].mean(0))
        weight += w
        matched_v += nv
        matched_u += nu
    return dict(v=int(v.sum()), u=int(u.sum()), low=int((~audible).sum()), av=av, au=au,
                stable_v=int(stable_v.sum()), stable_u=int(stable_u.sum()),
                stable_av=active[stable_v].sum(0), stable_au=active[stable_u].sum(0),
                confident_v=int(confidence_v.sum()), confident_u=int(confidence_u.sum()),
                confident_av=active[confidence_v].sum(0), confident_au=active[confidence_u].sum(0),
                matched_weight=weight, matched_difference_sum=weighted,
                matched_v=matched_v, matched_u=matched_u)


def binary_correlation(av, au, nv, nu):
    """Point-biserial/phi correlation between retained indicator and voiced flag."""
    active, total = av + au, nv + nu
    denominator = np.sqrt(float(active) * (total - active) * nv * nu)
    return float((av * nu - au * nv) / denominator) if denominator else None


def contribution_spectrum(contributions, sample_rate):
    frequencies, psd = signal.welch(contributions, sample_rate, window="hann", nperseg=2048,
                                   noverlap=1536, detrend=False, scaling="density", axis=-1)
    return frequencies, psd


def spectral_summary(frequencies, power):
    total = float(power.sum())
    if total <= 0:
        return {}
    p = power / total
    cdf = np.cumsum(p)
    result = {"peak_hz": float(frequencies[np.argmax(power)]),
              "peak_non_dc_hz": float(frequencies[1 + np.argmax(power[1:])]),
              "centroid_hz": float(frequencies @ p),
              "energy05_hz": float(frequencies[np.searchsorted(cdf, .05)]),
              "energy95_hz": float(frequencies[min(np.searchsorted(cdf, .95), len(cdf) - 1)]),
              "dc_bin_fraction": float(p[0])}
    peaks, _ = signal.find_peaks(power, height=power.max()*10**(-12/10), distance=3)
    peaks = sorted(set(peaks.tolist()+[int(np.argmax(power))]), key=lambda i: power[i], reverse=True)
    result["peaks_within12db_hz"] = " / ".join(f"{frequencies[i]:.1f}" for i in peaks)
    for lo, hi, name in [(0, 300, "below300"), (300, 1000, "300to1000"),
                         (1000, 3000, "1000to3000"), (3000, 8001, "above3000")]:
        result[f"power_fraction_{name}"] = float(p[(frequencies >= lo) & (frequencies < hi)].sum())
    return result
