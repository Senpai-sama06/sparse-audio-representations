"""L1 inference with explicit numerical certificates, independent of file I/O.

The penalized subproblem uses accelerated proximal gradient (FISTA). A lambda
search approaches a requested squared-error constraint. Both the penalized and
constrained dual gaps are reported: an iteration limit is never called optimal.
"""
import time

import numpy as np

from lrac.linear import CosineDictionary


def _signal(waveform):
    x = np.asarray(waveform, dtype=np.float64)
    if x.ndim != 1 or not x.size or not np.isfinite(x).all():
        raise ValueError("Expected a finite, nonempty one-dimensional signal")
    return x


def solve_lasso(waveform, dictionary, penalty, *, initial=None, max_iterations=2000,
                gap_tolerance=1e-5):
    """Minimize 0.5||x-Phi z||^2 + penalty||z||_1 and return a dual gap.

    ``penalty`` is in the units of the supplied signal. The dual feasible vector
    is a scaled residual, so its objective is a lower bound at every iteration.
    """
    x = _signal(waveform)
    if penalty <= 0 or not np.isfinite(penalty) or max_iterations < 1 or gap_tolerance <= 0:
        raise ValueError("Require positive penalty, iteration limit and gap tolerance")
    z = np.zeros(dictionary.size) if initial is None else np.array(initial, dtype=np.float64, copy=True)
    if z.shape != (dictionary.size,) or not np.isfinite(z).all():
        raise ValueError("Invalid initial coefficients")
    v, momentum = z.copy(), 1.0
    step = 1.0 / dictionary.frame_bound
    converged = False
    for iteration in range(1, max_iterations + 1):
        gradient = dictionary.adjoint(dictionary.synthesis(v) - x)
        proposal = v - step * gradient
        updated = np.sign(proposal) * np.maximum(np.abs(proposal) - step * penalty, 0)
        next_momentum = (1 + np.sqrt(1 + 4 * momentum * momentum)) / 2
        extrapolated = updated + (momentum - 1) / next_momentum * (updated - z)
        # Restart accelerated momentum when it points against the proximal step.
        # This changes speed, not the objective or its optimality certificate.
        if np.dot(v - updated, updated - z) > 0:
            next_momentum, extrapolated = 1.0, updated.copy()
        z, v, momentum = updated, extrapolated, next_momentum
        if iteration == 1 or iteration % 10 == 0 or iteration == max_iterations:
            residual = x - dictionary.synthesis(z)
            primal = .5 * np.dot(residual, residual) + penalty * np.abs(z).sum()
            max_correlation = float(np.abs(dictionary.adjoint(residual)).max())
            dual_vector = residual * min(1.0, penalty / max(max_correlation, 1e-300))
            dual = np.dot(x, dual_vector) - .5 * np.dot(dual_vector, dual_vector)
            gap = max(0.0, float(primal - dual))
            relative_gap = gap / max(float(primal), 1e-300)
            if relative_gap <= gap_tolerance:
                converged = True
                break
    return dict(coefficients=z, iterations=iteration, converged=converged,
                objective=float(primal), dual_objective=float(dual),
                dual_gap=gap, relative_gap=relative_gap)


def solve_l1(waveform, target_nmse, *, max_iterations=2000, search_steps=18,
             gap_tolerance=1e-5, constrained_tolerance=.01):
    """Approximately solve min ||z||_1 subject to NMSE <= target in [I,Q].

    Internal unit-energy scaling improves numerical comparability across clips;
    returned coefficients restore the original scale. This is not waveform gain
    normalization in the codec, and no extra inverse gain is needed to decode.
    """
    x = _signal(waveform)
    if not 0 < target_nmse < 1 or search_steps < 1 or constrained_tolerance <= 0:
        raise ValueError("Require 0 < target_nmse < 1 and positive search settings")
    started = time.perf_counter()
    dictionary = CosineDictionary(x.size)
    norm = float(np.linalg.norm(x))
    if norm == 0:
        return dict(coefficients=np.zeros(dictionary.size), nmse=0.0, penalty=0.0,
                    converged=True, constrained_relative_gap=0.0, l1_lower_bound=0.0,
                    l1_norm=0.0, iterations=0, seconds=time.perf_counter()-started, trials=[])
    unit = x / norm
    low, high = 0.0, float(np.abs(dictionary.adjoint(unit)).max())
    # Identity synthesis is always a feasible fallback. It is not asserted to be
    # the L1 optimum if the solver cannot improve it within the declared budget.
    best = np.concatenate((unit, np.zeros(x.size)))
    best_penalty, best_gap, best_lower = 0.0, 1.0, 0.0
    initial, trials, total_iterations = None, [], 0
    for _ in range(search_steps):
        penalty = (low + high) / 2
        result = solve_lasso(unit, dictionary, penalty, initial=initial,
                             max_iterations=max_iterations, gap_tolerance=gap_tolerance)
        initial = result.pop("coefficients")
        residual = unit - dictionary.synthesis(initial)
        nmse = float(np.dot(residual, residual))
        total_iterations += result["iterations"]
        # BPDN dual: max x.u - epsilon||u||, subject to ||Phi.T u||_inf <= 1.
        # Any residual can supply a feasible u; it need not come from convergence.
        correlation = float(np.abs(dictionary.adjoint(residual)).max())
        dual_vector = residual / max(correlation, 1e-300)
        lower = max(0.0, float(np.dot(unit, dual_vector) - np.sqrt(target_nmse) * np.linalg.norm(dual_vector)))
        primal = float(np.abs(initial).sum())
        constrained_gap = max(0.0, primal - lower) / max(primal, 1e-300)
        trials.append(dict(penalty=penalty, nmse=nmse, constrained_relative_gap=constrained_gap, **result))
        if nmse <= target_nmse:
            # For an inexact path, keep the best feasible L1 value, not merely
            # the most recent lambda. Feasibility is checked on actual synthesis.
            if primal < np.abs(best).sum():
                best, best_penalty, best_gap, best_lower = initial.copy(), penalty, constrained_gap, lower
            low = penalty
            if result["converged"] and constrained_gap <= constrained_tolerance:
                break
        else:
            high = penalty
    residual = unit - dictionary.synthesis(best)
    return dict(coefficients=best * norm, nmse=float(np.dot(residual, residual)),
                penalty=best_penalty * norm, converged=best_gap <= constrained_tolerance,
                constrained_relative_gap=best_gap, l1_lower_bound=best_lower * norm,
                l1_norm=float(np.abs(best).sum() * norm), iterations=total_iterations,
                seconds=time.perf_counter()-started, trials=trials)


def threshold_dct(waveform, target_nmse):
    """Keep the fewest largest DCT coefficients meeting a waveform NMSE target.

    Orthonormality makes the omitted coefficient energy exactly the waveform
    squared error. This is optimal support within this fixed basis before coding.
    """
    x = _signal(waveform)
    if not 0 < target_nmse < 1:
        raise ValueError("Require 0 < target_nmse < 1")
    dictionary = CosineDictionary(x.size, "dct")
    coefficients = dictionary.adjoint(x)
    order = np.argsort(-np.abs(coefficients), kind="stable")
    cumulative = np.cumsum(coefficients[order] ** 2)
    energy = float(np.dot(x, x))
    count = int(np.searchsorted(cumulative, (1-target_nmse) * energy) + 1) if energy else 0
    selected = np.zeros_like(coefficients)
    selected[order[:count]] = coefficients[order[:count]]
    return selected
