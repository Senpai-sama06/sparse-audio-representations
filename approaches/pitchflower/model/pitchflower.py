"""Official PitchFlower backbone with RVQ or the GS-Codec bottleneck.

References are pinned by tools/setup_pitchflower.py. We import the official
backbone rather than silently replacing its ConvNeXt/attention/flow architecture.
The GS equations and Adam initialization follow Aluf et al., Appendix A/code.
"""
from pathlib import Path
import sys
from types import MethodType

import torch
from torch import nn
from torch.nn import functional as F

from approaches.pitchflower.model.bottlenecks.gaussian import GaussianBottleneck, render_gaussians
from approaches.pitchflower.model.bottlenecks.gaussian_adapter import GaussianBottleneckAdapter
from approaches.pitchflower.model.bottlenecks.rvq import RVQBottleneck


REFERENCE_COMMIT = "4cef72ac9630cab632f13bd1d26b22cdae81cf17"


def configure_runtime():
    """Avoid CPU thread oversubscription and favor timely CUDA kernel submission.

    Above-normal Windows priority applies only to this process; it does not stop
    or alter the exhaustive L0 search. CUDA work is asynchronous, but its launch
    thread still needs CPU time when all logical processors are occupied.
    """
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = True
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        kernel.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel.SetPriorityClass.restype = wintypes.BOOL
        if not kernel.SetPriorityClass(kernel.GetCurrentProcess(), 0x8000):
            raise OSError(ctypes.get_last_error(), "Unable to set training process priority")


def sdpa_attention(self, x):
    """Upstream single-head attention, using PyTorch's fused-kernel dispatcher.

    RoPE, normalization, projections, scale and residual are unchanged. Keeping
    this as an instance adapter preserves every checkpoint parameter name and
    leaves the pinned reference checkout untouched. Backend selection depends
    on dtype/head width; SDPA does not imply Flash Attention for every block.
    """
    h = self.norm(x)
    q = self.rope.rotate_queries_or_keys(self.q(h).transpose(1, 2).unsqueeze(1))
    k = self.rope.rotate_queries_or_keys(self.k(h).transpose(1, 2).unsqueeze(1))
    v = self.v(h).transpose(1, 2).unsqueeze(1)
    attended = F.scaled_dot_product_attention(q.contiguous(), k.contiguous(), v.contiguous(),
                                            dropout_p=0., is_causal=False)
    return x+self.proj_out(attended.squeeze(1).transpose(1, 2))


def reference_path():
    """Make the pinned, locally installed reference importable in spawned workers."""
    root = Path(__file__).resolve().parents[3]/"runs/vendor/PitchFlower/src"
    if not root.exists():
        raise RuntimeError("Run python tools/setup_pitchflower.py first")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


class PitchFlowerModel(nn.Module):
    """Identical upstream backbone for both bottlenecks; original waveform target."""
    def __init__(self, bottleneck="rvq", gs_density=80/225, gs_iterations=100,
                 hidden=512, flow_channels=256, ae_layers=6, flow_layers=8, gs_channels=32,
                 optimized_attention=True):
        super().__init__()
        reference_path()
        from pitchflower.modules import AutoEncoderRVQ
        from pitchflower.flow_matcher import FlowMatcher
        from torchaudio.transforms import MelSpectrogram
        self.autoencoder = AutoEncoderRVQ(hidden_dim=hidden, n_convs=[ae_layers]*2,
            strides=[1, 2], stride_kernel_sizes=[None, 7], mels=100, n_q=8,
            codebook_size=512, codebook_dim=256 if bottleneck == "rvq" else gs_channels, semantic_dim=768)
        if bottleneck not in {"rvq", "gs"}:
            raise ValueError("Expected rvq or gs")
        self.kind = bottleneck
        if bottleneck == "gs":
            # Keep the encoder/decoder backbone, while adopting the GS paper's
            # C=32 projection. RVQ retains PitchFlower's 256-dimensional codes.
            quantizer = GaussianBottleneck(gs_density, gs_iterations)
            self.autoencoder.quantizer = quantizer
            self.bottleneck = GaussianBottleneckAdapter(quantizer)
        else:
            self.bottleneck = RVQBottleneck(self.autoencoder.quantizer)
        self.flow = FlowMatcher(sigma_min=1e-4, mels=100, channels=flow_channels,
                                n_convs=[flow_layers]*4, strides=[1, 1, 1, 2])
        if optimized_attention:
            from pitchflower.transformer import AttnBlock
            for module in self.modules():
                if isinstance(module, AttnBlock):
                    module.forward = MethodType(sdpa_attention, module)
        self.mel = MelSpectrogram(sample_rate=24000, n_fft=1024, n_mels=100,
                                  hop_length=256, center=True, power=1.)

    def latents(self, perturbed):
        """Continuous latents before either bottleneck, also used by predictor fitting."""
        mel = self.mel(perturbed.float()).clamp_min(1e-7).log()
        return self.autoencoder.quantizer_in(self.autoencoder.encoder(mel))

    def condition(self, latent):
        result = self.bottleneck.encode(latent, training=self.training)
        decoded = self.autoencoder.decoder(self.autoencoder.quantizer_out(result.values))
        return decoded, result.commitment

    def forward(self, original, perturbed, pitch, drop_probability=.1, commitment_weight=None):
        """Paper CFM: xt=t*x1+[1-(1-sigma)t]*x0, target x1-(1-sigma)x0.

        Both conditions independently drop with probability 0.1. Upstream's
        pitch mask is a KEEP mask, whereas masked_fill uses a DROP mask.
        """
        target = self.mel(original.float()).clamp_min(1e-7).log()
        condition, commitment = self.condition(self.latents(perturbed))
        time = torch.rand(target.shape[0], device=target.device)
        noise = torch.randn_like(target)
        interpolated = time[:, None, None]*target+(1-(1-self.flow.sigma_min)*time[:, None, None])*noise
        velocity = target-(1-self.flow.sigma_min)*noise
        pitch_keep = torch.rand(target.shape[0], device=target.device) >= drop_probability
        content_mask = torch.rand(target.shape[0], device=target.device) < drop_probability
        condition = condition.masked_fill(content_mask[:, None, None], 0.)
        prediction = self.flow(interpolated, condition, pitch, time, f0_mask=pitch_keep)
        flow_loss = F.mse_loss(prediction.float(), velocity.float())
        weight = commitment_weight if commitment_weight is not None else self.bottleneck.default_commitment_weight
        return flow_loss+weight*commitment, {"flow_loss": flow_loss.detach(), "commitment_loss": commitment.detach()}

    @torch.no_grad()
    def reconstruct_mel(self, perturbed, pitch, steps=10, guidance=1.0, predictor=None):
        """Midpoint flow integration; pitch is log2(Hz), with zero meaning unvoiced."""
        latent = self.latents(perturbed)
        if predictor is None:
            condition, _ = self.condition(latent)
        else:
            if not self.bottleneck.supports_predictor:
                raise ValueError("A Gaussian predictor requires the GS bottleneck")
            count = max(2, round(latent.shape[-1]*self.bottleneck.density))
            weights, scales = predictor(latent, count)
            rendered = render_gaussians(weights, scales, latent.shape[-1])
            condition = self.autoencoder.decoder(self.autoencoder.quantizer_out(rendered))
        state = torch.randn_like(condition)
        def velocity(x, t):
            times = x.new_full((x.shape[0],), t)
            predicted = self.flow(x, condition, pitch, times)
            if guidance != 1:
                unconditioned = self.flow(x, torch.zeros_like(condition), pitch, times,
                    f0_mask=torch.zeros(x.shape[0], dtype=torch.bool, device=x.device))
                predicted = unconditioned+guidance*(predicted-unconditioned)
            return predicted
        for index in range(steps):
            t, dt = index/steps, 1/steps
            middle = state+.5*dt*velocity(state, t)
            state = state+dt*velocity(middle, t+.5*dt)
        return state
