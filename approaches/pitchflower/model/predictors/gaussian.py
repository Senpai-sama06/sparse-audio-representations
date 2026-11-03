"""Amortized predictor for Gaussian-splat parameters."""
from pathlib import Path
import sys
import torch
from torch import nn


class GaussianPredictor(nn.Module):
    """Released GS predictor, imported in isolation from its pinned MIT source.

    Four dilated blocks, positional encodings and one attention decoder follow
    the authors' implementation. Frozen centers bypass the position head; the
    shared query supports changing primitive count at a fixed segment length.
    """
    def __init__(self, channels=32, width=128, frames=71, count=25):
        super().__init__()
        import importlib.util
        path = Path(__file__).resolve().parents[4]/"runs/vendor/gs-codec/gscodec/quantization/amortized_predictor.py"
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


