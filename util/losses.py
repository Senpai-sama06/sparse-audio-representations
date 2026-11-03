"""Waveform/spectral reconstruction and functional-separation losses."""
import torch
from torch.nn import functional as F


def masked_mse(reference, estimate, lengths):
    mask = torch.arange(reference.shape[-1], device=reference.device)[None, None, :] < lengths[:, None, None]
    return ((reference - estimate).square() * mask).sum() / mask.sum().clamp_min(1)


def spectral_loss(reference, estimate):
    losses = []
    for n in [256, 512, 1024]:
        x, y = reference.flatten(0, 1), estimate.flatten(0, 1)
        if x.shape[-1] < n:
            x, y = F.pad(x, (0, n - x.shape[-1])), F.pad(y, (0, n - y.shape[-1]))
        window = torch.hann_window(n, device=x.device, dtype=x.dtype)
        a = torch.stft(x, n, hop_length=n // 4, window=window, return_complex=True).abs()
        b = torch.stft(y, n, hop_length=n // 4, window=window, return_complex=True).abs()
        convergence = (a - b).flatten(1).norm(dim=1) / a.flatten(1).norm(dim=1).clamp_min(1e-5)
        log_error = ((a + 1e-5).log() - (b + 1e-5).log()).abs().flatten(1).mean(1)
        losses.append((convergence + log_error).mean())
    return torch.stack(losses).mean()


def reconstruction_loss(reference, estimate, lengths, train_power, spectral=True, spectral_weight=0.1):
    mse = masked_mse(reference, estimate, lengths) / max(float(train_power), 1e-8)
    if not spectral or spectral_weight == 0:
        return mse
    # Crop each item to its valid extent; padded silence cannot improve spectral scores.
    terms = [spectral_loss(x[..., :int(n)][None], y[..., :int(n)][None]) for x, y, n in zip(reference, estimate, lengths)]
    return mse + spectral_weight * torch.stack(terms).mean()




def functional_interaction(synthesis, latent, groups, probes=4, epsilon=1e-12, generator=None):
    if latent.shape[-1] % groups:
        raise ValueError("Last latent dimension must divide into equal groups")
    width = latent.shape[-1] // groups
    responses, energies = [], []
    for k in range(groups):
        items = []
        for _ in range(probes):
            direction = torch.zeros_like(latent)
            direction[..., k * width:(k + 1) * width] = torch.randn(latent[..., k * width:(k + 1) * width].shape, device=latent.device, dtype=latent.dtype, generator=generator)
            _, response = torch.func.jvp(synthesis, (latent,), (direction,))
            items.append(response.reshape(-1))
        values = torch.stack(items)
        responses.append(values)
        energies.append(values.square().sum(1).mean())
    loss = latent.sum() * 0
    for i in range(groups):
        for j in range(i + 1, groups):
            numerator = (responses[i] @ responses[j].T).square().mean()
            loss = loss + numerator / (energies[i] * energies[j] + epsilon)
    return loss


def exact_interaction(synthesis, latent, groups, epsilon=1e-12):
    jacobian = torch.func.jacrev(synthesis)(latent).reshape(-1, latent.numel())
    indices = torch.arange(latent.numel(), device=latent.device).reshape(latent.shape)
    width = latent.shape[-1] // groups
    blocks = [jacobian[:, indices[..., k * width:(k + 1) * width].reshape(-1)] for k in range(groups)]
    loss = latent.sum() * 0
    for i in range(groups):
        for j in range(i + 1, groups):
            loss += (blocks[i].T @ blocks[j]).square().sum() / (blocks[i].square().sum() * blocks[j].square().sum() + epsilon)
    return loss
