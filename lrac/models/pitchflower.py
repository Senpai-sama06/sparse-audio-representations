"""Official PitchFlower backbone with RVQ or the GS-Codec bottleneck.

References are pinned by tools/setup_pitchflower.py. We import the official
backbone rather than silently replacing its ConvNeXt/attention/flow architecture.
The GS equations and Adam initialization follow Aluf et al., Appendix A/code.
"""
from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F


REFERENCE_COMMIT = "4cef72ac9630cab632f13bd1d26b22cdae81cf17"


def reference_path():
    """Make the pinned, locally installed reference importable in spawned workers."""
    root = Path(__file__).resolve().parents[2]/"runs/vendor/PitchFlower/src"
    if not root.exists():
        raise RuntimeError("Run python tools/setup_pitchflower.py first")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


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


class GaussianPredictor(nn.Module):
    """Released GS predictor, imported in isolation from its pinned MIT source.

    Four dilated blocks, positional encodings and one attention decoder follow
    the authors' implementation. Frozen centers bypass the position head; the
    shared query supports changing primitive count at a fixed segment length.
    """
    def __init__(self, channels=32, width=128, frames=71, count=25):
        super().__init__()
        import importlib.util
        path = Path(__file__).resolve().parents[2]/"runs/vendor/gs-codec/gscodec/quantization/amortized_predictor.py"
        if not path.exists():
            raise RuntimeError("Run tools/setup_pitchflower.py to fetch the pinned GS source")
        name = "_lrac_gs_reference_predictor"
        if name not in sys.modules:
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
        module = sys.modules[name]
        self.reference = module.AmortizedGSPredictor(in_dim=channels, n_gaussians=count,
            T=frames, d_model=width, n_heads=4, n_blocks=4, n_decoder_layers=1,
            variable_ng=True)
        self.reference.head_pos.requires_grad_(False)

    def forward(self, latent, count):
        parameters = self.reference(latent, n_gaussians=count, frozen_mode=True)
        return parameters["weights"], parameters["scales"].squeeze(1)


class PitchFlowerModel(nn.Module):
    """Identical upstream backbone for both bottlenecks; original waveform target."""
    def __init__(self, bottleneck="rvq", gs_density=80/225, gs_iterations=100,
                 hidden=512, flow_channels=256, ae_layers=6, flow_layers=8, gs_channels=32):
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
            self.autoencoder.quantizer = GaussianBottleneck(gs_density, gs_iterations)
        self.flow = FlowMatcher(sigma_min=1e-4, mels=100, channels=flow_channels,
                                n_convs=[flow_layers]*4, strides=[1, 1, 1, 2])
        self.mel = MelSpectrogram(sample_rate=24000, n_fft=1024, n_mels=100,
                                  hop_length=256, center=True, power=1.)

    def latents(self, perturbed):
        """Continuous latents before either bottleneck, also used by predictor fitting."""
        mel = self.mel(perturbed.float()).clamp_min(1e-7).log()
        return self.autoencoder.quantizer_in(self.autoencoder.encoder(mel))

    def condition(self, latent):
        if self.kind == "gs":
            encoded, penalty = self.autoencoder.quantizer(latent)
        else:
            encoded, _, penalty, _ = self.autoencoder.quantizer(latent, n_q=8)
        return self.autoencoder.decoder(self.autoencoder.quantizer_out(encoded)), penalty

    def forward(self, original, perturbed, pitch, drop_probability=.1, commitment_weight=None):
        """Paper CFM: xt=t*x1+[1-(1-sigma)t]*x0, target x1-(1-sigma)x0.

        The upstream loss inverts its pitch dropout comparison and omits content
        dropout. Here both conditions independently drop with probability 0.1,
        matching the paper rather than reproducing that implementation discrepancy.
        """
        target = self.mel(original.float()).clamp_min(1e-7).log()
        condition, commitment = self.condition(self.latents(perturbed))
        time = torch.rand(target.shape[0], device=target.device)
        noise = torch.randn_like(target)
        interpolated = time[:, None, None]*target+(1-(1-self.flow.sigma_min)*time[:, None, None])*noise
        velocity = target-(1-self.flow.sigma_min)*noise
        pitch_mask = torch.rand(target.shape[0], device=target.device) < drop_probability
        content_mask = torch.rand(target.shape[0], device=target.device) < drop_probability
        condition = condition.masked_fill(content_mask[:, None, None], 0.)
        prediction = self.flow(interpolated, condition, pitch, time, f0_mask=pitch_mask)
        flow_loss = F.mse_loss(prediction.float(), velocity.float())
        weight = commitment_weight if commitment_weight is not None else (.25 if self.kind == "rvq" else 20.)
        return flow_loss+weight*commitment, {"flow_loss": flow_loss.detach(), "commitment_loss": commitment.detach()}

    @torch.no_grad()
    def reconstruct_mel(self, perturbed, pitch, steps=10, guidance=1.0, predictor=None):
        """Midpoint flow integration; pitch is log2(Hz), with zero meaning unvoiced."""
        latent = self.latents(perturbed)
        if predictor is None:
            condition, _ = self.condition(latent)
        else:
            count = max(2, round(latent.shape[-1]*self.autoencoder.quantizer.density))
            weights, scales = predictor(latent, count)
            rendered = render_gaussians(weights, scales, latent.shape[-1])
            condition = self.autoencoder.decoder(self.autoencoder.quantizer_out(rendered))
        state = torch.randn_like(condition)
        def velocity(x, t):
            times = x.new_full((x.shape[0],), t)
            predicted = self.flow(x, condition, pitch, times)
            if guidance != 1:
                unconditioned = self.flow(x, torch.zeros_like(condition), pitch, times,
                    f0_mask=torch.ones(x.shape[0], dtype=torch.bool, device=x.device))
                predicted = unconditioned+guidance*(predicted-unconditioned)
            return predicted
        for index in range(steps):
            t, dt = index/steps, 1/steps
            middle = state+.5*dt*velocity(state, t)
            state = state+dt*velocity(middle, t+.5*dt)
        return state
