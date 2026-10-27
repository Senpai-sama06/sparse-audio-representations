"""Convolutional autoencoder with an explicit whole-example sparse budget."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from lrac.models.autoencoder import GroupedAutoencoder


class SparseAutoencoder(GroupedAutoencoder):
    uses_global_budget = True

    def __init__(self, config, sample_rate):
        super().__init__(config, sample_rate)
        if config.activation == "identity":
            for module in self.modules():
                for name, child in list(module.named_children()):
                    if isinstance(child, nn.LeakyReLU):
                        setattr(module, name, nn.Identity())

    def encode_dense(self, x, lengths=None):
        if x.ndim != 3 or x.shape[1] != 1:
            raise ValueError("Expected audio shaped (batch, 1, samples)")
        if lengths is None:
            lengths = torch.full((x.shape[0],), x.shape[-1], device=x.device, dtype=torch.long)
        if lengths.shape != (x.shape[0],) or bool(((lengths <= 0) | (lengths > x.shape[-1])).any()):
            raise ValueError("Invalid waveform lengths")
        # Group equal lengths so batch padding cannot change a shorter example's
        # boundary activations. The usual fixed-length training batch is one group.
        encoded = [None] * x.shape[0]
        for length in lengths.unique().tolist():
            indices = (lengths == length).nonzero(as_tuple=True)[0]
            signal = F.pad(x[indices, :, :length], (0, (-length) % 256))
            features = self.encoder(signal).transpose(1, 2)
            for index, values in zip(indices.tolist(), features):
                encoded[index] = values
        steps = max(value.shape[0] for value in encoded)
        values = torch.stack([F.pad(value, (0, 0, 0, steps - value.shape[0])) for value in encoded])
        valid = (torch.arange(steps, device=x.device)[None, :, None] * 256 < lengths[:, None, None]).expand_as(values)
        representation = self.make_representation(values, x, lengths, {"hop": 256, "budget_scope": "whole recording"})
        representation.coefficient_mask = valid
        return representation

    def sparsify(self, representation, *, keep_fraction=None, sample_budget=None):
        if sample_budget is not None:
            if isinstance(sample_budget, bool) or not isinstance(sample_budget, int) or sample_budget < 0:
                raise ValueError("sample_budget must be a nonnegative integer")
        elif keep_fraction is None or not math.isfinite(keep_fraction) or not 0 <= keep_fraction <= 1:
            raise ValueError("keep_fraction must be between zero and one")
        values = representation.concatenate()
        valid = representation.coefficient_mask
        selected = torch.zeros_like(valid)
        for index in range(values.shape[0]):
            positions = valid[index].reshape(-1).nonzero(as_tuple=True)[0]
            count = min(sample_budget, positions.numel()) if sample_budget is not None else math.ceil(keep_fraction * positions.numel())
            if count == positions.numel():
                selected[index] = valid[index]
            elif count:
                largest = values[index].reshape(-1)[positions].abs().topk(count).indices
                selected[index].reshape(-1)[positions[largest]] = True
        result = representation.with_values(values * selected)
        result.selected_mask = selected
        result.timing.update(keep_fraction=keep_fraction, sample_budget=sample_budget)
        return result

    def encode(self, x, lengths=None):
        return self.sparsify(self.encode_dense(x, lengths), keep_fraction=self.config.keep_fraction, sample_budget=self.config.sample_budget)

    def decode(self, representation):
        values = representation.concatenate()
        decoded = [None] * values.shape[0]
        for length in representation.lengths.unique().tolist():
            indices = (representation.lengths == length).nonzero(as_tuple=True)[0]
            steps = (length + 255) // 256
            signals = self.decoder(values[indices, :steps].transpose(1, 2))[..., :length]
            for index, signal in zip(indices.tolist(), signals):
                decoded[index] = F.pad(signal, (0, representation.original_samples - length))
        return torch.stack(decoded)
