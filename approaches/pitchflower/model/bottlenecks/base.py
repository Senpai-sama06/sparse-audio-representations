"""Shared result contract for latent bottlenecks used by PitchFlower."""
from dataclasses import dataclass
import torch


@dataclass
class BottleneckOutput:
    """Quantized or approximated latents and their auxiliary commitment loss."""

    values: torch.Tensor
    commitment: torch.Tensor


class LatentBottleneck:
    """Interface for mapping encoder latents to decoder-ready latents.

    Implementations may delegate to a registered PyTorch module owned by the
    autoencoder. This adapter itself stays a plain Python object so it does not
    register the same module twice or change checkpoint parameter names.
    """

    default_commitment_weight = 0.0
    supports_predictor = False

    def encode(self, latent, *, training):
        raise NotImplementedError
