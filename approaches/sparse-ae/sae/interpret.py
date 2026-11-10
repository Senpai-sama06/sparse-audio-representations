"""Exploratory channel associations and direct latent interventions.

The analysis does not assign phoneme labels or claim held-out confirmation.
"""
import hashlib

import numpy as np
from scipy.signal import correlate, find_peaks
import torch

RATE = 16000
HOP = 256
FEATURES = ('log_rms', 'centroid_hz', 'high_band_fraction', 'periodicity', 'pitch_hz')


def acoustic_features(wave, steps):
    """64-ms Hann windows at nominal encoder centers, with boundary exclusion.

    Encoder stride is 256 samples; its receptive field is 1276 samples.
    Features are local acoustic proxies, not phonetic ground truth.
    """
    centers = np.arange(steps) * HOP + 128
    offsets = np.arange(1024) - 512
    indices = centers[:, None] + offsets
    complete = (indices[:, 0] >= 0) & (indices[:, -1] < len(wave))
    frames = wave[np.clip(indices, 0, len(wave)-1)].astype(np.float64)
    rms = np.sqrt(np.mean(frames**2, axis=1))
    frames -= frames.mean(axis=1, keepdims=True)
    power = np.abs(np.fft.rfft(frames * np.hanning(1024), axis=1))**2
    freq = np.fft.rfftfreq(1024, 1/RATE)
    energy = power.sum(axis=1)
    centroid = (power*freq).sum(axis=1)/np.maximum(energy, 1e-30)
    high = power[:, freq >= 3000].sum(axis=1)/np.maximum(energy, 1e-30)
    periodicity, pitch = [], []
    for frame in frames:
        ac = correlate(frame, frame, mode='full', method='fft')[1023:]
        normalized = ac/np.maximum(ac[0], 1e-30)
        peaks, _ = find_peaks(normalized[40:321])
        lag = int(peaks[np.argmax(normalized[40:321][peaks])]+40) if len(peaks) else 40
        confidence = float(normalized[lag])
        periodicity.append(confidence)
        pitch.append(RATE/lag if confidence >= .6 else np.nan)
    values = np.column_stack((np.log10(np.maximum(rms, 1e-8)), centroid, high, periodicity, pitch))
    # Discard incomplete and effectively silent windows from association analysis.
    usable = complete & (rms >= 1e-4)
    values[~usable] = np.nan
    return values, centers/RATE


def pearson(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = np.asarray(x)[mask], np.asarray(y)[mask]
    if len(x) < 20 or np.std(x) < 1e-10 or np.std(y) < 1e-10:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def association(x, y, loudness, control=True):
    """Descriptive Pearson r, optionally after removing a linear loudness effect."""
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(loudness)
    x, y, loudness = x[mask], y[mask], loudness[mask]
    if len(x) < 20:
        return None
    if control:
        design = np.column_stack((np.ones(len(x)), loudness))
        x = x - design @ np.linalg.lstsq(design, x, rcond=None)[0]
        y = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
    return pearson(x, y)


def channel_statistics(codes, acoustics):
    z, features = np.concatenate(codes), np.concatenate(acoustics)
    records = []
    for channel in range(z.shape[1]):
        x = z[:, channel]
        associations = {}
        for index, name in enumerate(FEATURES):
            r = association(x, features[:, index], features[:, 0], control=index != 0)
            speaker_rs = [association(c[:, channel], f[:, index], f[:, 0], control=index != 0)
                          for c, f in zip(codes, acoustics)]
            valid = [v for v in speaker_rs if v is not None]
            associations[name] = dict(r=r, speaker_valid=len(valid),
                median_speaker_r=float(np.median(valid)) if valid else None,
                same_sign_speaker_fraction=float(np.mean([np.sign(v)==np.sign(r) for v in valid]))
                if valid and r is not None else None)
        available = [(abs(v['r']), name) for name,v in associations.items() if v['r'] is not None]
        strongest = max(available)[1] if available else None
        records.append(dict(channel=channel, activity_fraction=float(np.mean(x != 0)),
            mean_activation=float(x.mean()), max_activation=float(x.max()),
            strongest_proxy=strongest, associations=associations))
    return records


def select_channels(stats, method, count=8):
    eligible = [r['channel'] for r in stats if .01 <= r['activity_fraction'] <= .99]
    if len(eligible) < count:
        raise ValueError('Too few eligible channels; revise scope explicitly')
    return sorted(eligible, key=lambda c: hashlib.sha256(f'sae-features-v1:{method}:{c}'.encode()).hexdigest())[:count]


def select_examples(codes, channel, count=3):
    # One utterance per speaker in the declared input set. No score-based selection.
    candidates = [(float(z[:,channel].max()), i, int(z[:,channel].argmax())) for i,z in enumerate(codes)]
    return sorted(candidates, key=lambda r: (-r[0], r[1]))[:count]


@torch.inference_mode()
def decode_intervention(model, code, channel, samples, factor=0.0):
    changed = code.clone()
    changed[...,channel] *= factor
    return model.decoder(changed)[..., :samples]
