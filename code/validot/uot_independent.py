"""Independent numerical checks for KL-unbalanced entropic OT.

This module deliberately does not call the project's log-Sinkhorn solver or
its response-metric helpers.  It solves the smooth UOT dual with a generic
trust-region optimizer and differentiates the primal first-order conditions.
The implementation is used only for cross-validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.stats import spearmanr


@dataclass(frozen=True)
class IndependentUOTResult:
    plan: np.ndarray
    source_potential: np.ndarray
    target_potential: np.ndarray
    converged: bool
    diagnostics: dict[str, Any]


def solve_uot_dual(
    cost: np.ndarray,
    source_mass: np.ndarray,
    target_mass: np.ndarray,
    epsilon: float,
    tau: float,
    gradient_tolerance: float = 1e-10,
    max_iterations: int = 500,
    initial: tuple[np.ndarray, np.ndarray] | None = None,
) -> IndependentUOTResult:
    """Solve KL-UOT through its unconstrained convex dual.

    The minimized dual objective is

      tau * sum_i a_i(exp(-f_i/tau)-1)
      + tau * sum_j b_j(exp(-g_j/tau)-1)
      + epsilon * sum_ij exp((f_i+g_j-C_ij)/epsilon).
    """

    cost = np.asarray(cost, dtype=np.float64)
    a = np.asarray(source_mass, dtype=np.float64)
    b = np.asarray(target_mass, dtype=np.float64)
    if cost.shape != (len(a), len(b)):
        raise ValueError("cost and marginal dimensions differ")
    if np.any(a <= 0) or np.any(b <= 0):
        raise ValueError("marginals must be strictly positive")
    if epsilon <= 0 or tau <= 0:
        raise ValueError("epsilon and tau must be positive")

    n_source, n_target = cost.shape
    if initial is None:
        x0 = np.zeros(n_source + n_target, dtype=np.float64)
    else:
        x0 = np.concatenate(
            [
                np.asarray(initial[0], dtype=np.float64),
                np.asarray(initial[1], dtype=np.float64),
            ]
        )

    cache: dict[str, np.ndarray | None] = {
        "x": None,
        "plan": None,
        "a_exp": None,
        "b_exp": None,
    }

    def state(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cached = cache["x"]
        if cached is None or not np.array_equal(cached, x):
            f = x[:n_source]
            g = x[n_source:]
            exponent = (f[:, None] + g[None, :] - cost) / float(epsilon)
            cache["plan"] = np.exp(np.clip(exponent, -745.0, 700.0))
            cache["a_exp"] = a * np.exp(np.clip(-f / float(tau), -745.0, 700.0))
            cache["b_exp"] = b * np.exp(np.clip(-g / float(tau), -745.0, 700.0))
            cache["x"] = x.copy()
        return (
            np.asarray(cache["plan"]),
            np.asarray(cache["a_exp"]),
            np.asarray(cache["b_exp"]),
        )

    def objective(x: np.ndarray) -> float:
        plan, a_exp, b_exp = state(x)
        return float(
            tau * np.sum(a_exp - a)
            + tau * np.sum(b_exp - b)
            + epsilon * np.sum(plan)
        )

    def gradient(x: np.ndarray) -> np.ndarray:
        plan, a_exp, b_exp = state(x)
        return np.concatenate(
            [plan.sum(axis=1) - a_exp, plan.sum(axis=0) - b_exp]
        )

    def hessian_product(x: np.ndarray, vector: np.ndarray) -> np.ndarray:
        plan, a_exp, b_exp = state(x)
        source = vector[:n_source]
        target = vector[n_source:]
        row_mass = plan.sum(axis=1)
        column_mass = plan.sum(axis=0)
        return np.concatenate(
            [
                (a_exp / tau + row_mass / epsilon) * source
                + plan @ target / epsilon,
                plan.T @ source / epsilon
                + (b_exp / tau + column_mass / epsilon) * target,
            ]
        )

    optimization = minimize(
        objective,
        x0,
        jac=gradient,
        method="L-BFGS-B",
        options={
            "gtol": float(gradient_tolerance),
            "maxiter": int(max_iterations),
            "maxls": 50,
            "ftol": 1e-15,
        },
    )
    plan, _, _ = state(optimization.x)
    plan = plan.copy()
    final_gradient = gradient(optimization.x)
    gradient_inf = float(np.max(np.abs(final_gradient)))
    row_mass = plan.sum(axis=1)
    column_mass = plan.sum(axis=0)
    first_order = (
        cost
        + epsilon * np.log(np.maximum(plan, 1e-300))
        + tau * np.log(np.maximum(row_mass, 1e-300) / a)[:, None]
        + tau * np.log(np.maximum(column_mass, 1e-300) / b)[None, :]
    )
    converged = bool(
        np.all(np.isfinite(plan))
        and np.min(plan) >= 0
        and gradient_inf <= float(gradient_tolerance)
    )
    return IndependentUOTResult(
        plan=plan,
        source_potential=optimization.x[:n_source].copy(),
        target_potential=optimization.x[n_source:].copy(),
        converged=converged,
        diagnostics={
            "optimizer_success": bool(optimization.success),
            "optimizer_message": str(optimization.message),
            "optimizer_iterations": int(optimization.nit),
            "gradient_inf": gradient_inf,
            "first_order_residual_inf": float(np.max(np.abs(first_order))),
            "transported_mass": float(plan.sum()),
            "objective": float(optimization.fun),
        },
    )


def implicit_plan_derivative(
    plan: np.ndarray,
    cost_derivative: np.ndarray,
    epsilon: float,
    tau: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Differentiate the KL-UOT primal first-order conditions."""

    plan = np.asarray(plan, dtype=np.float64)
    cost_dot = np.asarray(cost_derivative, dtype=np.float64)
    if plan.shape != cost_dot.shape:
        raise ValueError("plan and cost derivative shapes differ")
    row_mass = plan.sum(axis=1)
    column_mass = plan.sum(axis=0)
    weighted = plan * cost_dot
    system = np.block(
        [
            [
                (epsilon + tau) * np.diag(row_mass),
                tau * plan,
            ],
            [
                tau * plan.T,
                (epsilon + tau) * np.diag(column_mass),
            ],
        ]
    )
    rhs = -np.concatenate(
        [weighted.sum(axis=1), weighted.sum(axis=0)]
    )
    solution = np.linalg.solve(system, rhs)
    n_source = plan.shape[0]
    source_rate = solution[:n_source]
    target_rate = solution[n_source:]
    derivative = -(plan / epsilon) * (
        cost_dot
        + tau * source_rate[:, None]
        + tau * target_rate[None, :]
    )
    residual = system @ solution - rhs
    return derivative, {
        "linear_relative_residual_l2": float(
            np.linalg.norm(residual) / max(np.linalg.norm(rhs), 1e-300)
        ),
        "system_condition_number": float(np.linalg.cond(system)),
        "derivative_absolute_l1": float(np.abs(derivative).sum()),
    }


def row_plan_response(
    plan_derivative: np.ndarray,
    baseline_plan: np.ndarray,
    mass_minimum: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute normalized row-plan response without project metric helpers."""

    derivative = np.asarray(plan_derivative, dtype=np.float64)
    baseline = np.asarray(baseline_plan, dtype=np.float64)
    mass = baseline.sum(axis=1)
    estimable = np.isfinite(mass) & (mass > float(mass_minimum))
    values = np.full(len(mass), np.nan, dtype=np.float64)
    values[estimable] = (
        np.abs(derivative[estimable]).sum(axis=1)
        / (2.0 * mass[estimable])
    )
    return values, estimable


def comparison_metrics(
    estimate: np.ndarray,
    reference: np.ndarray,
    baseline_estimate: np.ndarray,
    baseline_reference: np.ndarray,
    top_fraction: float = 0.10,
    denominator_minimum: float = 1e-12,
) -> dict[str, Any]:
    """Compare plan derivatives and scalar row responses independently."""

    estimate = np.asarray(estimate, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    denominator = np.abs(reference).sum(axis=1)
    row_estimable = np.isfinite(denominator) & (
        denominator > float(denominator_minimum)
    )
    row_relative = np.full(len(denominator), np.nan, dtype=np.float64)
    row_relative[row_estimable] = (
        np.abs(estimate[row_estimable] - reference[row_estimable]).sum(axis=1)
        / denominator[row_estimable]
    )
    estimate_norm = np.linalg.norm(estimate, axis=1)
    reference_norm = np.linalg.norm(reference, axis=1)
    cosine_denominator = estimate_norm * reference_norm
    cosine_estimable = np.isfinite(cosine_denominator) & (
        cosine_denominator > float(denominator_minimum)
    )
    cosine = np.full(len(denominator), np.nan, dtype=np.float64)
    cosine[cosine_estimable] = (
        np.sum(
            estimate[cosine_estimable] * reference[cosine_estimable],
            axis=1,
        )
        / cosine_denominator[cosine_estimable]
    )
    estimate_score, estimate_score_ok = row_plan_response(
        estimate, baseline_estimate, denominator_minimum
    )
    reference_score, reference_score_ok = row_plan_response(
        reference, baseline_reference, denominator_minimum
    )
    scalar_ok = (
        estimate_score_ok
        & reference_score_ok
        & np.isfinite(estimate_score)
        & np.isfinite(reference_score)
    )
    x = estimate_score[scalar_ok]
    y = reference_score[scalar_ok]
    rho = (
        float(spearmanr(y, x).statistic)
        if len(x) >= 3 and np.unique(x).size > 1 and np.unique(y).size > 1
        else float("nan")
    )
    raw_mae = float(np.mean(np.abs(x - y))) if len(x) else float("nan")
    reference_scale = float(np.mean(np.abs(y))) if len(y) else float("nan")

    def finite_summary(values: np.ndarray) -> dict[str, float | int]:
        finite = values[np.isfinite(values)]
        if not len(finite):
            return {
                "n": 0,
                "median": float("nan"),
                "q90": float("nan"),
                "minimum": float("nan"),
                "maximum": float("nan"),
            }
        return {
            "n": int(len(finite)),
            "median": float(np.median(finite)),
            "q90": float(np.quantile(finite, 0.90)),
            "minimum": float(np.min(finite)),
            "maximum": float(np.max(finite)),
        }

    k = max(1, int(np.ceil(float(top_fraction) * len(x)))) if len(x) else 0
    if k:
        indices = np.arange(len(x))
        estimate_top = set(
            indices[np.lexsort((indices, -x))[:k]].tolist()
        )
        reference_top = set(
            indices[np.lexsort((indices, -y))[:k]].tolist()
        )
        top_overlap = float(len(estimate_top & reference_top) / k)
    else:
        top_overlap = float("nan")
    global_denominator = float(np.abs(reference).sum())
    return {
        "global_derivative_relative_l1": float(
            np.abs(estimate - reference).sum()
            / max(global_denominator, 1e-300)
        ),
        **{
            f"row_relative_l1_{key}": value
            for key, value in finite_summary(row_relative).items()
        },
        **{
            f"row_direction_cosine_{key}": value
            for key, value in finite_summary(cosine).items()
        },
        "scalar_n_estimable": int(np.sum(scalar_ok)),
        "scalar_nonestimable_fraction": float(1.0 - np.mean(scalar_ok)),
        "scalar_spearman": rho,
        "scalar_top_decile_overlap": top_overlap,
        "scalar_raw_mae": raw_mae,
        "scalar_rmae": float(raw_mae / max(reference_scale, 1e-300))
        if np.isfinite(raw_mae) and np.isfinite(reference_scale)
        else float("nan"),
    }
