"""Reference-based metrics. Invalid cases stay explicit rather than becoming zero."""
import math
import warnings
import numpy as np


def speech_dependencies():
    try:
        from pesq import pesq
        from pystoi import stoi
    except ImportError as error:
        raise RuntimeError('Speech metrics need: pip install -e ".[speech-metrics]"') from error
    return pesq, stoi


def audio_metrics(reference, estimate, speech=False):
    x = np.asarray(reference, dtype=np.float64)
    y = np.asarray(estimate, dtype=np.float64)
    error = float(np.sum((x - y) ** 2))
    energy = float(np.sum(x ** 2))
    result = dict(mse=error / len(x), nmse=None, si_sdr=None,
                  pesq_wb=None, stoi=None, statuses={})
    if energy > 1e-12:
        result["nmse"] = error / energy
    else:
        result["statuses"]["nmse"] = "silent_reference"
    a, b = x - x.mean(), y - y.mean()
    denominator = float(a @ a)
    if denominator > 1e-12:
        target = float(b @ a) / denominator * a
        numerator, residual = float(target @ target), float(np.sum((b - target) ** 2))
        if numerator > 1e-20:
            result["si_sdr"] = 10 * math.log10(numerator / max(residual, 1e-20))
        else:
            result["statuses"]["si_sdr"] = "zero_projection"
    else:
        result["statuses"]["si_sdr"] = "silent_or_constant_reference"
    if speech:
        pesq, stoi = speech_dependencies()
        if energy <= 1e-12:
            result["statuses"].update(pesq_wb="silent_reference", stoi="silent_reference")
            return result
        for name, function in (("pesq_wb", lambda: pesq(16000, x, y, "wb")),
                               ("stoi", lambda: stoi(x, y, 16000, extended=False))):
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    value = float(function())
                failures = [str(item.message) for item in caught
                            if issubclass(item.category, (UserWarning, RuntimeWarning))]
                if failures:
                    raise ValueError("; ".join(failures))
                if not math.isfinite(value):
                    raise ValueError("nonfinite result")
                result[name] = value
            except Exception as error:
                result["statuses"][name] = f"{type(error).__name__}: {error}"
    else:
        result["statuses"].update(pesq_wb="disabled", stoi="disabled")
    return result


def summarize(rows):
    summary = {}
    for name in ("mse", "nmse", "si_sdr", "pesq_wb", "stoi"):
        values = [row[name] for row in rows if row[name] is not None]
        summary[name] = float(np.mean(values)) if values else None
        summary[name + "_valid"] = len(values)
    summary["cases"] = len(rows)
    return summary
