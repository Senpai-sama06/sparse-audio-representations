"""Gaussian-splat bottleneck fitting and fixed-grid synthesis."""
import torch
from torch import nn
from torch.nn import functional as F


def render_gaussians(weights, scales, frames):
    """Fixed grid centers; one positive width and C amplitudes per primitive."""
    count = weights.shape[-1]
    t = torch.arange(frames, device=weights.device, dtype=weights.dtype)
    centers = torch.linspace(0, frames-1, count, device=weights.device, dtype=weights.dtype)
    basis = torch.exp(-.5*((t[None, None, :]-centers[None, :, None])/scales[..., None])**2)
    return torch.bmm(weights, basis)


class GaussianBottleneck(nn.Module):
    """Fit detached latents, then use the paper's STE and commitment penalty.

    Density 80/225 transfers the paper's 80 primitives per 225 frames to
    PitchFlower's different latent frame rate. No channel normalization/bias or
    quantization is applied during training; centers are the frozen-grid variant.
    """
    def __init__(self, density=80/225, iterations=100, learning_rate=.4):
        super().__init__()
        self.density, self.iterations, self.learning_rate = density, iterations, learning_rate
        self.last_parameters = None

    def fit(self, latent, iterations=None):
        """All inner fitting is FP32 even when the outer network uses autocast."""
        with torch.enable_grad(), torch.autocast(latent.device.type, enabled=False):
            target = latent.detach().float()
            batch, channels, frames = target.shape
            count = max(2, round(frames*self.density))
            initial_scale = frames/count/2
            indices = torch.linspace(0, frames-1, count, device=target.device).long()
            weights = (target[:, :, indices]/max(1., count/frames*initial_scale*2.5)).clone().requires_grad_()
            scales = target.new_full((batch, count), initial_scale, requires_grad=True)
            optimizer = torch.optim.Adam([weights, scales], lr=self.learning_rate, fused=target.is_cuda)
            for _ in range(iterations or self.iterations):
                optimizer.zero_grad(set_to_none=True)
                error = F.mse_loss(render_gaussians(weights, scales, frames), target)
                error.backward()
                optimizer.step()
                # These positivity/width safeguards match the released GS fitter.
                with torch.no_grad():
                    scales.clamp_(.5, max(.5, frames/4))
            return weights.detach(), scales.detach()

    def forward(self, latent):
        weights, scales = self.fit(latent)
        rendered = render_gaussians(weights, scales, latent.shape[-1]).to(latent.dtype)
        self.last_parameters = {"weights": weights, "scales": scales}
        commitment = F.mse_loss(latent.float(), rendered.detach().float())
        # The forward value is the fitted signal; the derivative to the encoder
        # is identity, as specified in GS-Codec section 3.1.2.
        return latent+(rendered-latent).detach(), commitment


