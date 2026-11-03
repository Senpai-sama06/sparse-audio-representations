"""Adapt the iterative Gaussian fitter to the shared latent bottleneck API."""
from approaches.pitchflower.model.bottlenecks.base import BottleneckOutput, LatentBottleneck


class GaussianBottleneckAdapter(LatentBottleneck):
    """Expose Gaussian fitting while leaving its trainable state in the codec."""

    default_commitment_weight = 20.0
    supports_predictor = True

    def __init__(self, quantizer):
        self.quantizer = quantizer

    @property
    def density(self):
        return self.quantizer.density

    def encode(self, latent, *, training):
        values, commitment = self.quantizer(latent)
        return BottleneckOutput(values=values, commitment=commitment)
