"""Native SAE losses, with waveform-specific auxiliary reconstruction targets."""
import torch
from torch.nn import functional as F


def waveform_mse(reference, estimate, lengths):
    mask = torch.arange(reference.shape[-1], device=reference.device)[None, None, :] < lengths[:, None, None]
    return (((reference - estimate).square() * mask).sum((1, 2)) / lengths).mean()


def multiresolution_stft(reference, estimate, lengths):
    losses = []
    for x, y, length in zip(reference[:, 0], estimate[:, 0], lengths):
        x, y = x[:int(length)], y[:int(length)]
        resolutions = []
        for fft in (256, 512, 1024):
            # A zero-extended short signal remains a valid numerical smoke input.
            a, b = F.pad(x, (0, max(0, fft - x.numel()))), F.pad(y, (0, max(0, fft - y.numel())))
            window = torch.hann_window(fft, device=x.device, dtype=x.dtype)
            a = torch.stft(a, fft, fft // 4, window=window, return_complex=True).abs()
            b = torch.stft(b, fft, fft // 4, window=window, return_complex=True).abs()
            convergence = (a - b).norm() / a.norm().clamp_min(1e-5)
            log_error = ((a + 1e-5).log() - (b + 1e-5).log()).abs().mean()
            resolutions.append(convergence + log_error)
        losses.append(torch.stack(resolutions).mean())
    return torch.stack(losses).mean()


def objective(model, result, x, lengths, config):
    estimate = result["reconstruction"]
    mse = waveform_mse(x, estimate, lengths)
    reconstruction = mse / config["mse_scale"] ** 2
    spectral = reconstruction.new_zeros(())
    if config["spectral_weight"]:
        spectral = multiresolution_stft(x, estimate, lengths)
        reconstruction = reconstruction + config["spectral_weight"] * spectral
    penalty = result["penalty"]
    if model.config.method == "kl":
        mean = result["z"][result["valid"]].mean(0).clamp(1e-6, 1 - 1e-6)
        target = config["kl_target"]
        penalty = (target * (torch.log(mean.new_tensor(target)) - mean.log()) +
                   (1 - target) * (torch.log(mean.new_tensor(1 - target)) - torch.log1p(-mean))).sum()
    auxiliary = reconstruction.new_zeros(())
    if result["gate_code"] is not None and config["gated_aux_weight"]:
        auxiliary = waveform_mse(x, model.detached_decode(result["gate_code"], x.shape[-1]), lengths)
        auxiliary = auxiliary / config["mse_scale"] ** 2 * config["gated_aux_weight"]
    if result["aux_code"] is not None and config["aux_weight"]:
        # A nonlinear waveform decoder has a nonzero zero-code response. Subtract
        # that response so this branch predicts an additive residual contribution.
        auxiliary_estimate = (model.decoder(result["aux_code"]) -
                              model.decoder(torch.zeros_like(result["aux_code"])))[:, :, :x.shape[-1]]
        target = (x - estimate).detach()
        auxiliary = waveform_mse(target, auxiliary_estimate, lengths)
        auxiliary = auxiliary / config["mse_scale"] ** 2 * config["aux_weight"]
    loss = reconstruction + config["sparsity_weight"] * penalty + auxiliary
    valid_z = result["z"][result["valid"]]
    return dict(loss=loss, reconstruction=reconstruction, mse=mse, spectral=spectral,
                sparsity=penalty, auxiliary=auxiliary,
                active_per_step=(valid_z != 0).float().sum(-1).mean(),
                latent_rms=valid_z.square().mean().sqrt())
