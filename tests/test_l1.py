"""Independent numerical references for the fixed-family L1 experiment."""
import numpy as np
import pytest
from scipy.optimize import minimize

from lrac.l1 import solve_l1, solve_lasso, threshold_dct
from lrac.linear import CosineDictionary


def _matrix(n):
    # Explicit textbook DCT synthesis is an independent small reference. The
    # production optimizer never allocates this N x 2N matrix.
    q = np.sqrt(2/n)*np.cos(np.pi*(np.arange(n)[:, None]+.5)*np.arange(n)[None, :]/n)
    q[:, 0] /= np.sqrt(2)
    return np.concatenate((np.eye(n), q), axis=1)


def test_linear_operators_match_explicit_unit_norm_tight_frame():
    rng = np.random.default_rng(11)
    bank = CosineDictionary(17)
    matrix = _matrix(17)
    x, z = rng.normal(size=17), rng.normal(size=34)
    np.testing.assert_allclose(bank.synthesis(z), matrix@z, atol=1e-13)
    np.testing.assert_allclose(bank.adjoint(x), matrix.T@x, atol=1e-13)
    np.testing.assert_allclose(np.linalg.norm(matrix, axis=0), 1, atol=1e-13)
    np.testing.assert_allclose(matrix@matrix.T, 2*np.eye(17), atol=1e-13)


def test_lasso_matches_closed_form_orthonormal_soft_threshold():
    x = np.random.default_rng(4).normal(size=31)
    bank = CosineDictionary(len(x), "dct")
    a = bank.adjoint(x)
    expected = np.sign(a)*np.maximum(np.abs(a)-.2, 0)
    result = solve_lasso(x, bank, .2, gap_tolerance=1e-10)
    assert result["converged"]
    np.testing.assert_allclose(result["coefficients"], expected, atol=1e-12)
    assert result["dual_objective"] <= result["objective"] + 1e-12


def test_constrained_l1_matches_independent_positive_negative_variable_optimizer():
    x = np.array([.9, -.2, .1, .4, -.3, .7, -.1, .05])
    matrix, eta = _matrix(len(x)), .025
    m = matrix.shape[1]
    epsilon_sq = eta*np.dot(x, x)
    def residual(v):
        return x-matrix@(v[:m]-v[m:])
    start = np.concatenate((np.maximum(np.r_[x, np.zeros(len(x))], 0),
                            np.maximum(-np.r_[x, np.zeros(len(x))], 0)))
    # This constrained smooth formulation does not use FISTA or a lambda path.
    oracle = minimize(np.sum, start, jac=lambda v: np.ones_like(v), bounds=[(0, None)]*(2*m),
        constraints={"type": "ineq", "fun": lambda v: epsilon_sq-np.dot(residual(v), residual(v)),
                     "jac": lambda v: np.r_[2*matrix.T@residual(v), -2*matrix.T@residual(v)]},
        method="SLSQP", options={"ftol": 1e-11, "maxiter": 1000})
    assert oracle.success
    result = solve_l1(x, eta, max_iterations=5000, search_steps=30,
                      gap_tolerance=1e-9, constrained_tolerance=1e-4)
    assert result["converged"] and result["nmse"] <= eta
    assert result["l1_lower_bound"] <= oracle.fun + 1e-7
    assert oracle.fun <= result["l1_norm"] + 1e-7
    assert result["l1_norm"] <= oracle.fun*1.0002


def test_iteration_limit_is_not_reported_as_an_optimum():
    x = np.random.default_rng(8).normal(size=37)
    result = solve_l1(x, .001, max_iterations=1, search_steps=1, constrained_tolerance=1e-9)
    assert not result["converged"]
    assert result["nmse"] <= .001  # Identity fallback still supplies a feasible code.
    assert result["constrained_relative_gap"] > 1e-9
    assert np.isfinite(result["coefficients"]).all()


def test_dct_threshold_support_and_zero_signal():
    bank = CosineDictionary(32, "dct")
    z = np.zeros(32)
    z[[1, 5, 7]] = [3, 2, 1]
    selected = threshold_dct(bank.synthesis(z), .1)
    assert np.count_nonzero(selected) == 2
    np.testing.assert_allclose(np.sum((z-selected)**2), 1, atol=1e-12)
    assert solve_l1(np.zeros(32), .01)["converged"]
    with pytest.raises(ValueError):
        solve_l1(np.ones(32), 0)
