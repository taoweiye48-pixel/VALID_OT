from __future__ import annotations

import numpy as np

from validot.p1 import arm_weights
from validot.postreview import (
    balanced_plan_implicit_derivative,
    cost_direction,
    row_softmax_plan_derivative,
)
from validot.solvers import log_sinkhorn, row_softmax


def test_cost_direction_matches_path_finite_difference() -> None:
    rng = np.random.default_rng(7)
    expression = rng.random((5, 4))
    spatial = rng.random((5, 4))
    h = 1e-7
    for arm in ("R", "N"):
        for intervention in ("I_EXPR", "I_SPATIAL"):
            w0 = arm_weights(arm, intervention, 0.0)
            wh = arm_weights(arm, intervention, h)
            c0 = w0[0] * expression + w0[1] * spatial
            ch = wh[0] * expression + wh[1] * spatial
            numerical = (ch - c0) / h
            analytic = cost_direction(expression, spatial, arm, intervention)
            assert np.allclose(numerical, analytic, atol=1e-9, rtol=1e-8)


def test_row_softmax_analytic_derivative_matches_small_step() -> None:
    rng = np.random.default_rng(11)
    expression = rng.random((7, 6))
    spatial = rng.random((7, 6))
    a = np.full(7, 1.0 / 7.0)
    epsilon = 0.25
    base_cost = 0.5 * expression + 0.5 * spatial
    base = row_softmax(base_cost, a, epsilon).plan
    h = 1e-6
    for arm in ("R", "N"):
        for intervention in ("I_EXPR", "I_SPATIAL"):
            weights = arm_weights(arm, intervention, h)
            perturbed_cost = weights[0] * expression + weights[1] * spatial
            perturbed = row_softmax(perturbed_cost, a, epsilon).plan
            numerical = (perturbed - base) / h
            analytic = row_softmax_plan_derivative(
                base, a, cost_direction(expression, spatial, arm, intervention), epsilon
            )
            assert np.allclose(numerical, analytic, atol=2e-7, rtol=2e-5)


def test_balanced_implicit_derivative_matches_small_step() -> None:
    rng = np.random.default_rng(19)
    expression = rng.random((6, 5))
    spatial = rng.random((6, 5))
    a = np.full(6, 1.0 / 6.0)
    b = np.full(5, 1.0 / 5.0)
    epsilon = 0.25
    base_cost = 0.5 * expression + 0.5 * spatial
    base = log_sinkhorn(base_cost, a, b, epsilon=epsilon, tol=1e-10).plan
    direction = cost_direction(expression, spatial, "N", "I_EXPR")
    implicit, diagnostics = balanced_plan_implicit_derivative(
        base, direction, epsilon, relative_tolerance=1e-12
    )
    h = 1e-5
    perturbed = log_sinkhorn(base_cost + h * direction, a, b, epsilon=epsilon, tol=1e-10).plan
    numerical = (perturbed - base) / h
    assert diagnostics["converged"]
    assert diagnostics["relative_residual_l2"] < 1e-9
    assert np.allclose(numerical, implicit, atol=2e-5, rtol=2e-3)
