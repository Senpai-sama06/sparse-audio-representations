"""Global l0 coding in an orthonormal block transform, with Procrustes learning.

The support optimum is conditional on the current dictionary and squared error.
The dictionary learning problem itself is nonconvex. No claim of globally optimal
learned features or perceptual-metric optimality is made.
"""
import math
import torch
from torch import nn
from lrac.models.base import ReconstructiveModel, Representation
from lrac.models.transforms import dct_basis


def select_coefficients(values, mask, fraction=0.125, budget=None, target_nmse=None):
    """Select across the entire recording, excluding padded coordinates.

    budget=0 retains all coefficients. A positive budget is an absolute count.
    With target_nmse, sort residual energies and find the smallest feasible
    support. Reverse cumulative sums avoid subtractive cancellation near zero.
    """
    if budget is not None and budget < 0 or not 0 <= fraction <= 1:
        raise ValueError("Invalid coefficient budget")
    if target_nmse is not None and target_nmse < 0:
        raise ValueError("NMSE tolerance cannot be negative")
    if not torch.isfinite(values).all():
        raise ValueError("Coefficients must be finite")
    output = torch.zeros_like(values)
    selected = torch.zeros_like(mask)
    diagnostics = []
    for i in range(values.shape[0]):
        valid_indices = mask[i].reshape(-1).nonzero().flatten()
        valid = values[i].reshape(-1)[valid_indices]
        order = torch.argsort(valid.abs(), descending=True, stable=True)
        ordered = valid[order]
        energies = ordered.square()
        residuals = torch.cat([energies.flip(0).cumsum(0).flip(0), energies.new_zeros(1)])
        total = residuals[0]
        nonzeros = int(torch.count_nonzero(valid))
        if budget is not None:
            count = nonzeros if budget == 0 else min(budget, nonzeros)
        elif target_nmse is not None:
            count = int((residuals <= target_nmse * total).nonzero()[0])
        else:
            count = min(math.ceil(fraction * len(valid)), nonzeros)
        indices = valid_indices[order[:count]]
        output[i].reshape(-1)[indices] = ordered[:count]
        selected[i].reshape(-1)[indices] = True
        denominator = float(total)
        diagnostics.append({
            "valid_coefficients": len(valid),
            "active_coefficients": count,
            "predicted_nmse": float(residuals[count] / total) if denominator else 0.0,
            "one_fewer_nmse": float(residuals[count - 1] / total) if count and denominator else None,
            "target_nmse": target_nmse if budget is None else None,
            "optimality_scope": "minimum support for this fixed orthonormal transform under waveform NMSE" if budget is None and target_nmse is not None else "minimum squared error at this support budget for this fixed orthonormal transform",
        })
    return output, selected, diagnostics


class BlockOrthogonalDictionary(ReconstructiveModel):
    is_dictionary = True
    uses_procrustes = True

    def __init__(self, config, sample_rate):
        super().__init__()
        self.config, self.sample_rate = config, sample_rate
        n = config.frame_length
        if config.initialization == "dct":
            basis = dct_basis(n, dtype=torch.float64)
        elif config.initialization == "random":
            basis = torch.linalg.qr(torch.randn(n, n, dtype=torch.float64))[0]
        elif config.initialization == "identity":
            basis = torch.eye(n, dtype=torch.float64)
        else:
            raise ValueError("Unknown block dictionary initialization")
        self.register_buffer("basis", basis)

    def matrix(self):
        return self.basis

    def _transform(self, x, lengths):
        n = self.config.frame_length
        blocks = math.ceil(x.shape[-1] / n)
        values = self.basis.new_zeros(x.shape[0], blocks, n)
        mask = torch.zeros_like(values, dtype=torch.bool)
        full_counts, tail_lengths = [], []
        for i, length in enumerate(lengths.tolist()):
            full, tail = divmod(length, n)
            full_counts.append(full)
            tail_lengths.append(tail)
            if full:
                frames = x[i, 0, :full * n].reshape(full, n).to(self.basis.dtype)
                values[i, :full] = frames @ self.basis
                mask[i, :full] = True
            if tail:
                # A true r-dimensional transform, not a padded n-dimensional one.
                tail_basis = dct_basis(tail, dtype=self.basis.dtype).to(x.device)
                values[i, full, :tail] = x[i, 0, full * n:length].to(self.basis.dtype) @ tail_basis
                mask[i, full, :tail] = True
        return values, mask, full_counts, tail_lengths

    def encode(self, x, lengths=None):
        if x.ndim != 3 or x.shape[1] != 1 or x.shape[-1] == 0:
            raise ValueError("Expected nonempty (batch, 1, samples) audio")
        if lengths is None:
            lengths = torch.full((x.shape[0],), x.shape[-1], device=x.device, dtype=torch.long)
        if (lengths <= 0).any() or (lengths > x.shape[-1]).any():
            raise ValueError("Invalid audio lengths")
        values, mask, full, tails = self._transform(x, lengths)
        values, selected, diagnostics = select_coefficients(
            values, mask, self.config.keep_fraction, self.config.sample_budget, self.config.target_nmse,
        )
        representation = self.make_representation(values, x, lengths, {
            "hop": self.config.frame_length,
            "block_length": self.config.frame_length,
            "transform": "nonoverlapping_orthonormal_blocks",
            "full_blocks": full,
            "tail_lengths": tails,
            "tail_transform": "fixed orthonormal DCT of actual tail length",
            "sparse_coding": diagnostics,
        })
        representation.coefficient_mask = mask
        representation.selected_mask = selected
        return representation

    def decode(self, representation):
        n = self.config.frame_length
        values = representation.concatenate()
        output = values.new_zeros(values.shape[0], 1, representation.original_samples)
        for i, length in enumerate(representation.lengths.tolist()):
            full, tail = divmod(length, n)
            if full:
                output[i, 0, :full * n] = (values[i, :full] @ self.basis.T).reshape(-1)
            if tail:
                tail_basis = dct_basis(tail, dtype=self.basis.dtype).to(values.device)
                output[i, 0, full * n:length] = values[i, full, :tail] @ tail_basis.T
        return output

    @torch.no_grad()
    def update_dictionary(self, x, lengths):
        """One exact fixed-code Procrustes step on this batch, then recode."""
        if self.config.target_nmse is not None:
            raise ValueError("Procrustes fitting requires a fixed support budget")
        before, representation = self(x, lengths)
        codes = representation.concatenate()
        cross = torch.zeros_like(self.basis)
        for i, full in enumerate(representation.timing["full_blocks"]):
            if full:
                frames = x[i, 0, :full * self.config.frame_length].reshape(full, self.config.frame_length).to(self.basis.dtype)
                cross += frames.T @ codes[i, :full]
        mask = torch.arange(x.shape[-1], device=x.device)[None, None] < lengths[:, None, None]
        def mse(estimate):
            return float(((estimate - x).square() * mask).sum() / mask.sum())
        if bool(torch.count_nonzero(cross)):
            # Small audio batches often give rank-deficient cross-covariances.
            # Request the robust QR-based CUDA driver directly instead of first
            # exhausting Jacobi iterations and triggering an implicit fallback.
            options = {"driver": "gesvd"} if cross.is_cuda else {}
            u, _, vh = torch.linalg.svd(cross, full_matrices=False, **options)
            self.basis.copy_(u @ vh)
        fixed_codes = self.decode(representation)
        after, _ = self(x, lengths)
        return {
            "mse_before": mse(before),
            "mse_fixed_codes": mse(fixed_codes),
            "mse_after": mse(after),
            "orthogonality_error": float((self.basis.T @ self.basis - torch.eye(self.config.frame_length, device=x.device, dtype=self.basis.dtype)).norm()),
        }
