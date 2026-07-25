from __future__ import annotations

import numpy as np

from validot.solvers import log_sinkhorn
from validot.uot_independent import (
    implicit_plan_derivative,
    solve_uot_dual,
)


def test_independent_uot_plan_matches_log_sinkhorn() -> None:
    rng = np.random.default_rng(20260724)
    cost = rng.uniform(0.0, 2.0, size=(8, 7))
    a = np.full(8, 1.0 / 8.0)
    b = np.full(7, 1.0 / 7.0)
    epsilon = 0.25
    tau = 2.0
    independent = solve_uot_dual(
        cost,
        a,
        b,
        epsilon,
        tau,
        gradient_tolerance=1e-8,
    )
    production = log_sinkhorn(
        cost,
        a,
        b,
        epsilon=epsilon,
        tau_a=tau,
        tau_b=tau,
        tol=1e-11,
    )
    assert independent.converged
    assert production.converged
    relative = (
        np.abs(independent.plan - production.plan).sum()
        / np.abs(independent.plan).sum()
    )
    assert relative < 1e-6


def test_independent_uot_implicit_derivative_matches_finite_difference() -> None:
    rng = np.random.default_rng(20260725)
    cost = rng.uniform(0.0, 1.5, size=(7, 6))
    direction = rng.normal(0.0, 0.2, size=cost.shape)
    a = np.full(7, 1.0 / 7.0)
    b = np.full(6, 1.0 / 6.0)
    epsilon = 0.25
    tau = 2.0
    base = solve_uot_dual(
        cost,
        a,
        b,
        epsilon,
        tau,
        gradient_tolerance=1e-8,
    )
    derivative, diagnostics = implicit_plan_derivative(
        base.plan, direction, epsilon, tau
    )
    h = 1e-2
    plus = solve_uot_dual(
        cost + h * direction,
        a,
        b,
        epsilon,
        tau,
        gradient_tolerance=1e-8,
        initial=(base.source_potential, base.target_potential),
    )
    minus = solve_uot_dual(
        cost - h * direction,
        a,
        b,
        epsilon,
        tau,
        gradient_tolerance=1e-8,
        initial=(base.source_potential, base.target_potential),
    )
    numerical = (plus.plan - minus.plan) / (2.0 * h)
    relative = np.abs(numerical - derivative).sum() / np.abs(derivative).sum()
    assert base.converged
    assert plus.converged
    assert minus.converged
    assert diagnostics["linear_relative_residual_l2"] < 1e-12
    assert relative < 5e-4
