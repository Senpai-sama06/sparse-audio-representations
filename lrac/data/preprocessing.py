from math import gcd
import numpy as np
from scipy.signal import resample_poly
import soundfile as sf
from lrac.data.synthetic import generate


def load_audio(row, sample_rate):
    if row["corpus"] == "synthetic":
        audio, _, _ = generate(row)
        original_rate = row["sample_rate"]
    else:
        start, stop = int(row.get("start", 0)), int(row.get("stop", row["num_samples"]))
        audio, original_rate = sf.read(row["path"], start=start, stop=stop, dtype="float32", always_2d=True)
        if audio.shape[1] != 1:
            raise ValueError(f"Expected mono recording: {row['id']}")
        audio = audio[:, 0]
    if original_rate != sample_rate:
        divisor = gcd(original_rate, sample_rate)
        audio = resample_poly(audio, sample_rate // divisor, original_rate // divisor)
    if not len(audio) or not np.isfinite(audio).all():
        raise ValueError(f"Empty or nonfinite audio: {row['id']}")
    return np.asarray(audio, dtype=np.float32)
