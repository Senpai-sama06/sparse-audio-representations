"""Adapter for PitchFlower's pinned residual vector quantizer."""
import torch

from approaches.pitchflower.model.bottlenecks.base import BottleneckOutput, LatentBottleneck


class RVQBottleneck(LatentBottleneck):
    """Adapt the upstream RVQ return format to the project's bottleneck API.

    Training keeps the upstream four-example EMA update cadence used by the
    existing campaign. Evaluation encodes the full batch in one quantizer call.
    """

    default_commitment_weight = 0.25

    def __init__(self, quantizer, *, codebooks=8, ema_batch=4):
        self.quantizer = quantizer
        self.codebooks = codebooks
        self.ema_batch = ema_batch

    def encode(self, latent, *, training):
        chunks = latent.split(self.ema_batch) if training else (latent,)
        results = [self.quantizer(part, n_q=self.codebooks) for part in chunks]
        values = torch.cat([result[0] for result in results], dim=0)
        commitment = sum(
            result[2] * len(part) / len(latent)
            for part, result in zip(chunks, results)
        )
        return BottleneckOutput(values=values, commitment=commitment)
