"""Interchangeable latent bottleneck adapters and implementations."""

from approaches.pitchflower.model.bottlenecks.base import BottleneckOutput, LatentBottleneck
from approaches.pitchflower.model.bottlenecks.gaussian_adapter import GaussianBottleneckAdapter
from approaches.pitchflower.model.bottlenecks.rvq import RVQBottleneck
from approaches.pitchflower.model.bottlenecks.gaussian import GaussianBottleneck

__all__ = [
    "BottleneckOutput",
    "LatentBottleneck",
    "GaussianBottleneck",
    "GaussianBottleneckAdapter",
    "RVQBottleneck",
]
