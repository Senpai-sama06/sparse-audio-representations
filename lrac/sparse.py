"""Sparse inference with a checked Lipschitz bound and objective convergence."""
import torch


def soft_threshold(values, threshold):
    return values.sign() * (values.abs() - threshold).clamp_min(0)


def sparse_group_prox(values, threshold, group_threshold=0.0, groups=1):
    values = soft_threshold(values, threshold)
    if group_threshold:
        shape = values.shape
        grouped = values.reshape(*shape[:-1], groups, shape[-1] // groups)
        norm = grouped.norm(dim=-1, keepdim=True)
        grouped = grouped * (1 - group_threshold / norm.clamp_min(1e-12)).clamp_min(0)
        values = grouped.reshape(shape)
    return values


@torch.no_grad()
def fista(target, analysis, synthesis, code_shape, threshold, iterations=200, tolerance=1e-5, group_threshold=0.0, groups=1):
    # All operator pairs must be true adjoints, including their boundaries.
    probe = torch.randn(code_shape, device=target.device, dtype=target.dtype)
    probe /= probe.norm().clamp_min(1e-12)
    for _ in range(16):
        transformed = analysis(synthesis(probe))
        probe = transformed / transformed.norm().clamp_min(1e-12)
    lipschitz = max(float((probe * analysis(synthesis(probe))).sum()) * 1.05, 1e-6)
    current = torch.zeros(code_shape, device=target.device, dtype=target.dtype)
    extrapolated = current.clone()
    momentum = 1.0
    previous_objective = None
    for step in range(iterations):
        residual = synthesis(extrapolated) - target
        gradient = analysis(residual)
        smooth = .5 * residual.square().sum()
        # Backtracking protects against a power iteration underestimating the norm.
        for _ in range(30):
            candidate = sparse_group_prox(extrapolated - gradient / lipschitz, threshold / lipschitz, group_threshold / lipschitz, groups)
            delta = candidate - extrapolated
            candidate_residual = synthesis(candidate) - target
            bound = smooth + (gradient * delta).sum() + .5 * lipschitz * delta.square().sum()
            if .5 * candidate_residual.square().sum() <= bound + 1e-5 * smooth.abs().clamp_min(1):
                break
            lipschitz *= 2
        else:
            raise RuntimeError("Sparse inference could not find a valid step size")
        next_momentum = (1 + (1 + 4 * momentum * momentum) ** .5) / 2
        extrapolated = candidate + ((momentum - 1) / next_momentum) * (candidate - current)
        current, momentum = candidate, next_momentum
        if step % 10 == 0 or step + 1 == iterations:
            objective = .5 * candidate_residual.square().sum() + threshold * candidate.abs().sum()
            if group_threshold:
                objective += group_threshold * candidate.reshape(*candidate.shape[:-1], groups, -1).norm(dim=-1).sum()
            value = float(objective)
            if previous_objective is not None and abs(previous_objective - value) / max(abs(previous_objective), 1e-12) < tolerance:
                break
            previous_objective = value
    return current
