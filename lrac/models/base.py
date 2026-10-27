from dataclasses import dataclass, field, replace
import torch
from torch import nn


@dataclass
class Representation:
    groups: dict[str, torch.Tensor]
    lengths: torch.Tensor
    sample_rate: int
    original_samples: int
    timing: dict = field(default_factory=dict)
    version: int = 1
    coefficient_mask: torch.Tensor | None = None
    selected_mask: torch.Tensor | None = None

    def concatenate(self):
        return torch.cat(list(self.groups.values()), dim=-1)

    def with_values(self, values):
        sizes = [v.shape[-1] for v in self.groups.values()]
        return replace(self, groups=dict(zip(self.groups, values.split(sizes, dim=-1))), timing=dict(self.timing))


class ReconstructiveModel(nn.Module):
    is_dictionary = False
    needs_fit = False

    def forward(self, x, lengths=None):
        representation = self.encode(x, lengths)
        return self.decode(representation), representation

    def post_step(self):
        pass

    def make_representation(self, values, x, lengths, timing):
        if lengths is None:
            lengths = torch.full((x.shape[0],), x.shape[-1], device=x.device, dtype=torch.long)
        pieces = values.chunk(self.config.groups, dim=-1)
        return Representation({f"group_{i}": v for i, v in enumerate(pieces)}, lengths, self.sample_rate, x.shape[-1], timing)


def budget(values, count):
    if count <= 0 or count >= values.shape[-1]:
        return values
    indices = values.abs().topk(count, dim=-1).indices
    return torch.zeros_like(values).scatter(-1, indices, values.gather(-1, indices))
