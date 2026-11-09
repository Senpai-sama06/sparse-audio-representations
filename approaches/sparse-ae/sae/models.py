"""Shared waveform backbone with six distinct sparse bottlenecks.

These are audio adaptations, not reproductions of an LM activation experiment.
JumpReLU uses the rectangular-kernel threshold estimators from arXiv:2407.14435.
Gated reconstruction uses a detached nonlinear waveform decoder (2404.16014).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F
from . import METHODS


@dataclass
class ModelConfig:
    method: str = "topk"
    channels: tuple = (32, 64, 128, 128)
    latent_width: int = 256
    k: int = 16
    threshold: float = 0.1
    bandwidth: float = 0.1
    dead_after_steps: int = 100
    aux_k: int = 32

    def validate(self):
        if self.method not in METHODS or len(self.channels) != 4 or min(self.channels) < 1:
            raise ValueError("Need a supported method and four positive channel widths")
        if self.latent_width < 2 or not 1 <= self.k <= self.latent_width:
            raise ValueError("Require latent_width >= 2 and 1 <= k <= latent_width")
        if (not math.isfinite(self.threshold) or self.threshold <= 0 or
                not math.isfinite(self.bandwidth) or self.bandwidth <= 0):
            raise ValueError("Threshold and bandwidth must be finite and positive")
        if self.dead_after_steps < 1 or self.aux_k < 1:
            raise ValueError("dead_after_steps and aux_k must be positive")


class Jump(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, threshold, bandwidth):
        ctx.save_for_backward(x, threshold)
        ctx.bandwidth = bandwidth
        return x * (x > threshold)

    @staticmethod
    def backward(ctx, gradient):
        x, threshold = ctx.saved_tensors
        kernel = ((x - threshold).abs() < ctx.bandwidth / 2).to(x.dtype)
        dx = gradient * (x > threshold)
        dt = (-gradient * threshold * kernel / ctx.bandwidth).sum_to_size(threshold.shape)
        return dx, dt, None


class Step(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, threshold, bandwidth):
        ctx.save_for_backward(x, threshold)
        ctx.bandwidth = bandwidth
        return (x > threshold).to(x.dtype)

    @staticmethod
    def backward(ctx, gradient):
        x, threshold = ctx.saved_tensors
        kernel = ((x - threshold).abs() < ctx.bandwidth / 2).to(x.dtype)
        dt = (-gradient * kernel / ctx.bandwidth).sum_to_size(threshold.shape)
        return torch.zeros_like(x), dt, None


class Residual(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.net = nn.Sequential(nn.LeakyReLU(.2), nn.Conv1d(width, width, 3, padding=1),
                                 nn.LeakyReLU(.2), nn.Conv1d(width, width, 1))

    def forward(self, x):
        return x + self.net(x)


class WaveformDecoder(nn.Module):
    def __init__(self, channels, latent_width):
        super().__init__()
        self.input = nn.Conv1d(latent_width, channels[-1], 1)
        blocks = []
        for i in range(3, -1, -1):
            source, target = channels[i], channels[i - 1] if i else 1
            blocks.extend([Residual(source), nn.LeakyReLU(.2),
                           nn.ConvTranspose1d(source, target, 8, stride=4, padding=2)])
        self.blocks = nn.Sequential(*blocks)

    def forward(self, z):
        return self.blocks(self.input(z.transpose(1, 2)))


class AudioSAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        config.validate()
        self.config = config
        blocks, source = [], 1
        for width in config.channels:
            blocks.extend([nn.Conv1d(source, width, 8, stride=4, padding=2),
                           nn.LeakyReLU(.2), Residual(width)])
            source = width
        self.encoder = nn.Sequential(*blocks)
        self.feature_norm = nn.LayerNorm(source, elementwise_affine=False)
        self.project = nn.Linear(source, config.latent_width)
        nn.init.zeros_(self.project.bias)
        self.decoder = WaveformDecoder(config.channels, config.latent_width)
        if config.method == "gated":
            self.log_magnitude_scale = nn.Parameter(torch.zeros(config.latent_width))
            self.magnitude_bias = nn.Parameter(torch.zeros(config.latent_width))
        if config.method == "jumprelu":
            self.log_threshold = nn.Parameter(torch.full((config.latent_width,), math.log(config.threshold)))
        self.register_buffer("inference_threshold", torch.tensor(0.0))
        self.register_buffer("threshold_calibrated", torch.tensor(False))
        self.register_buffer("inactive_steps", torch.zeros(config.latent_width, dtype=torch.long))
        self.constrain_weights()

    @torch.no_grad()
    def constrain_weights(self):
        # Fix the obvious encoder-row/decoder-column scale freedom.
        self.project.weight.copy_(F.normalize(self.project.weight, dim=1))
        self.decoder.input.weight.copy_(F.normalize(self.decoder.input.weight, dim=0))
        if self.config.method == "jumprelu":
            self.log_threshold.clamp_(-12, 8)
        if self.config.method == "gated":
            self.log_magnitude_scale.clamp_(-8, 8)

    def preactivations(self, x, lengths):
        padded = F.pad(x, (0, (-x.shape[-1]) % 256))
        features = self.feature_norm(self.encoder(padded).transpose(1, 2))
        logits = self.project(features)
        valid = torch.arange(logits.shape[1], device=x.device)[None, :] * 256 < lengths[:, None]
        return features, logits, valid

    @staticmethod
    def topk(values, k):
        selected, indices = values.topk(k, dim=-1)
        return torch.zeros_like(values).scatter(-1, indices, selected)

    def batch_selection(self, positive, valid):
        flat = positive[valid].reshape(-1)
        count = int(valid.sum()) * self.config.k
        values, indices = flat.topk(count)
        packed = torch.zeros_like(flat).scatter(0, indices, values)
        result = torch.zeros_like(positive)
        result[valid] = packed.reshape(-1, self.config.latent_width)
        return result

    def forward(self, x, lengths, update_activity=False):
        features, logits, valid = self.preactivations(x, lengths)
        method, positive = self.config.method, logits.relu()
        zero = logits.sum() * 0
        gate_code, aux_code = None, None
        if method in {"dense", "l1"}:
            z = positive
            penalty = z[valid].sum(-1).mean() if method == "l1" else zero
        elif method == "kl":
            z, penalty = logits.sigmoid(), zero  # KL is computed with the configured target in the objective.
        elif method == "topk":
            z, penalty = self.topk(positive, self.config.k), zero
        elif method == "batchtopk":
            if self.training:
                z = self.batch_selection(positive, valid)
            else:
                if not bool(self.threshold_calibrated):
                    raise RuntimeError("BatchTopK needs development calibration before inference")
                z = positive * (positive > self.inference_threshold)
            penalty = zero
        elif method == "jumprelu":
            threshold = self.log_threshold.exp()
            z = Jump.apply(positive, threshold, self.config.bandwidth)
            penalty = Step.apply(positive, threshold, self.config.bandwidth)[valid].sum(-1).mean()
        elif method == "gated":
            magnitude = F.linear(features, self.project.weight * self.log_magnitude_scale.exp()[:, None],
                                 self.magnitude_bias).relu()
            z = (logits > 0) * magnitude
            gate_code = positive * valid[..., None]
            penalty = positive[valid].sum(-1).mean()
        else:
            raise ValueError(method)
        z = z * valid[..., None]
        if method in {"topk", "batchtopk"}:
            dead = self.inactive_steps >= self.config.dead_after_steps
            if bool(dead.any()):
                available = positive.masked_fill(~dead[None, None, :], -torch.inf)
                aux_code = self.topk(available, min(self.config.aux_k, int(dead.sum())))
                aux_code = aux_code.nan_to_num(neginf=0) * valid[..., None]
        if self.training and update_activity:
            with torch.no_grad():
                active = (z[valid] != 0).any(0)
                self.inactive_steps.add_(1)
                self.inactive_steps[active] = 0
        reconstruction = self.decoder(z)[..., :x.shape[-1]]
        return dict(reconstruction=reconstruction, z=z, valid=valid, penalty=penalty,
                    gate_code=gate_code, aux_code=aux_code)

    def detached_decode(self, code, samples):
        parameters = {name: value.detach() for name, value in self.decoder.named_parameters()}
        return torch.func.functional_call(self.decoder, parameters, (code,))[..., :samples]

    @torch.no_grad()
    def calibrate(self, loader, device, progress=None):
        if self.config.method != "batchtopk":
            return
        previous = self.training
        self.eval()
        thresholds = []
        for index, (x, lengths, _) in enumerate(loader, 1):
            _, logits, valid = self.preactivations(x.to(device), lengths.to(device))
            selected = self.batch_selection(logits.relu(), valid)
            positive = selected[selected > 0]
            thresholds.append(positive.min() if positive.numel() else selected.new_zeros(()))
            if progress:
                progress.pulse(batch=index, batches=len(loader))
        if not thresholds:
            raise ValueError("Empty BatchTopK calibration loader")
        self.inference_threshold.copy_(torch.stack(thresholds).mean())
        self.threshold_calibrated.fill_(True)
        self.train(previous)

    def specification(self):
        return asdict(self.config)
