import math
import torch
from torch.nn import functional as F


def dct_basis(n, dtype=torch.float32):
    samples = torch.arange(n, dtype=dtype)[:, None]
    frequencies = torch.arange(n, dtype=dtype)[None, :]
    result = math.sqrt(2 / n) * torch.cos(math.pi / n * (samples + .5) * frequencies)
    result[:, 0] /= math.sqrt(2)
    return result


def frame(x, window, hop):
    n = window.numel()
    left = n - hop
    right = left + (-(x.shape[-1] + 2 * left - n)) % hop
    padded = F.pad(x, (left, right))
    values = padded[:, 0].unfold(-1, n, hop) * window
    return values, {"left": left, "right": right, "padded_samples": padded.shape[-1], "hop": hop, "frame_length": n}


def unframe(values, window, timing, original_samples):
    n, hop = window.numel(), timing["hop"]
    signal = F.fold((values * window).transpose(1, 2), (1, timing["padded_samples"]), kernel_size=(1, n), stride=(1, hop))[:, :, 0]
    weights = window.square().view(1, n, 1).expand(1, n, values.shape[1])
    denominator = F.fold(weights, (1, timing["padded_samples"]), kernel_size=(1, n), stride=(1, hop))[:, :, 0]
    signal = signal / denominator.clamp_min(torch.finfo(values.dtype).eps)
    start = timing["left"]
    return signal[..., start:start + original_samples]
