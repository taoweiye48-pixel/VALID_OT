"""Post-review VALID-OT analysis primitives.

This module is additive.  It does not write to or alter any frozen P0/P1
artifact.  The first implementation block supports WP1: multi-step local
derivative convergence and an analytic row-softmax positive control.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.sparse.linalg import LinearOperator, cg
from scipy.stats import spearmanr

from .metrics import top_fraction_precision


CONDITIONS = (
    ("R", "I_EXPR"),
    ("R", "I_SPATIAL"),
    ("N", "I_EXPR"),
    ("N", "I_SPATIAL"),
)


def cost_direction(
    expression: np.ndarray,
    spatial: np.ndarray,
    arm: str,
    intervention: str,
) -> np.ndarray:
    """Return dC(t)/dt at t=0 for a frozen intervention path."""
    if arm == "R" and intervention == "I_EXPR":
        return -0.5 * expression
    if arm == "R" and intervention == "I_SPATIAL":
        return -0.5 * spatial
    if arm == "N" and intervention == "I_EXPR":
        return -0.5 * expression + 0.5 * spatial
    if arm == "N" and intervention == "I_SPATIAL":
        return 0.5 * expression - 0.5 * spatial
    raise ValueError(f"unsupported condition: {arm}/{intervention}")


def row_softmax_plan_derivative(
    base_plan: np.ndarray,
    row_mass: np.ndarray,
    cost_dot: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    """Analytic derivative of a row-softmax plan along a cost path."""
    base_plan = np.asarray(base_plan, dtype=np.float64)
    row_mass = np.asarray(row_mass, dtype=np.float64)
    cost_dot = np.asarray(cost_dot, dtype=np.float64)
    q = base_plan / np.maximum(row_mass[:, None], 1e-300)
    centered = cost_dot - np.sum(q * cost_dot, axis=1, keepdims=True)
    return -(base_plan / float(epsilon)) * centered


def row_response_rate(
    plan_derivative: np.ndarray,
    baseline_row_mass: np.ndarray,
    row_mass_min: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Map a plan derivative to the fixed-baseline normalized L1 row rate."""
    derivative = np.asarray(plan_derivative, dtype=np.float64)
    mass = np.asarray(baseline_row_mass, dtype=np.float64)
    estimable = np.isfinite(mass) & (mass > float(row_mass_min))
    score = np.full(len(mass), np.nan, dtype=np.float64)
    score[estimable] = np.abs(derivative[estimable]).sum(axis=1) / (2.0 * mass[estimable])
    return score, estimable


def row_relative_l1(
    estimate: np.ndarray,
    reference: np.ndarray,
    denominator_min: float,
) -> tuple[np.ndarray, np.ndarray]:
    estimate = np.asarray(estimate, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    denominator = np.abs(reference).sum(axis=1)
    estimable = np.isfinite(denominator) & (denominator > float(denominator_min))
    values = np.full(len(denominator), np.nan, dtype=np.float64)
    values[estimable] = (
        np.abs(estimate[estimable] - reference[estimable]).sum(axis=1)
        / denominator[estimable]
    )
    return values, estimable


def row_direction_cosine(
    estimate: np.ndarray,
    reference: np.ndarray,
    denominator_min: float,
) -> tuple[np.ndarray, np.ndarray]:
    estimate = np.asarray(estimate, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    norm_e = np.linalg.norm(estimate, axis=1)
    norm_r = np.linalg.norm(reference, axis=1)
    denominator = norm_e * norm_r
    estimable = np.isfinite(denominator) & (denominator > float(denominator_min))
    values = np.full(len(denominator), np.nan, dtype=np.float64)
    values[estimable] = (
        np.sum(estimate[estimable] * reference[estimable], axis=1)
        / denominator[estimable]
    )
    return values, estimable


def finite_summary(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if not finite.size:
        return {
            "n_estimable": 0,
            "median": float("nan"),
            "q90": float("nan"),
            "mean": float("nan"),
            "maximum": float("nan"),
        }
    return {
        "n_estimable": int(finite.size),
        "median": float(np.median(finite)),
        "q90": float(np.quantile(finite, 0.90)),
        "mean": float(np.mean(finite)),
        "maximum": float(np.max(finite)),
    }


def scalar_fidelity(reference: np.ndarray, estimate: np.ndarray, top_fraction: float) -> dict[str, Any]:
    reference = np.asarray(reference, dtype=np.float64)
    estimate = np.asarray(estimate, dtype=np.float64)
    keep = np.isfinite(reference) & np.isfinite(estimate)
    x = estimate[keep]
    y = reference[keep]
    if len(x) < 3:
        return {"n_estimable": int(len(x)), "spearman": float("nan"), "top_overlap": float("nan"), "raw_mae": float("nan"), "rmae": float("nan"), "intercept": float("nan"), "slope": float("nan"), "r2": float("nan")}
    rho = spearmanr(y, x).statistic if np.unique(x).size > 1 and np.unique(y).size > 1 else float("nan")
    raw_mae = float(np.mean(np.abs(x - y)))
    scale = float(np.mean(np.abs(y)))
    design = np.column_stack([np.ones(len(x)), x])
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    fitted = design @ coefficients
    residual = float(np.sum((y - fitted) ** 2))
    total = float(np.sum((y - np.mean(y)) ** 2))
    return {
        "n_estimable": int(len(x)),
        "spearman": float(rho) if np.isfinite(rho) else float("nan"),
        "top_overlap": float(top_fraction_precision(y, x, top_fraction)),
        "raw_mae": raw_mae,
        "rmae": raw_mae / max(scale, 1e-300),
        "intercept": float(coefficients[0]),
        "slope": float(coefficients[1]),
        "r2": 1.0 - residual / total if total > 1e-300 else float("nan"),
    }


def balanced_plan_implicit_derivative(
    base_plan: np.ndarray,
    cost_dot: np.ndarray,
    epsilon: float,
    relative_tolerance: float = 1e-10,
    max_iterations: int = 2000,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Matrix-free implicit derivative for balanced entropic OT.

    The final target dual is fixed to zero to remove gauge freedom.  The
    resulting linear system is the dual Hessian restricted to that gauge and
    is solved without materializing an O((n+m)^2) Jacobian.
    """
    plan = np.asarray(base_plan, dtype=np.float64)
    direction = np.asarray(cost_dot, dtype=np.float64)
    if plan.shape != direction.shape:
        raise ValueError("plan and cost derivative must have equal shape")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    n_source, n_target = plan.shape
    source_mass = plan.sum(axis=1)
    target_mass = plan.sum(axis=0)
    weighted = plan * direction
    rhs = np.concatenate(
        [weighted.sum(axis=1), weighted.sum(axis=0)[:-1]]
    )

    def matvec(vector: np.ndarray) -> np.ndarray:
        source = vector[:n_source]
        target = vector[n_source:]
        return np.concatenate(
            [
                source_mass * source + plan[:, :-1] @ target,
                plan[:, :-1].T @ source + target_mass[:-1] * target,
            ]
        )

    operator = LinearOperator(
        (n_source + n_target - 1, n_source + n_target - 1),
        matvec=matvec,
        dtype=np.float64,
    )
    iterations = 0

    def callback(_: np.ndarray) -> None:
        nonlocal iterations
        iterations += 1

    solution, info = cg(
        operator,
        rhs,
        rtol=float(relative_tolerance),
        atol=0.0,
        maxiter=int(max_iterations),
        callback=callback,
    )
    source_dual = solution[:n_source]
    target_dual = np.concatenate([solution[n_source:], np.zeros(1)])
    derivative = plan * (
        source_dual[:, None] + target_dual[None, :] - direction
    ) / float(epsilon)
    residual = matvec(solution) - rhs
    diagnostics = {
        "linear_solver_info": int(info),
        "linear_solver_iterations": int(iterations),
        "relative_residual_l2": float(
            np.linalg.norm(residual) / max(np.linalg.norm(rhs), 1e-300)
        ),
        "row_derivative_l1": float(np.abs(derivative.sum(axis=1)).sum()),
        "column_derivative_l1": float(np.abs(derivative.sum(axis=0)).sum()),
        "converged": bool(info == 0),
    }
    return derivative, diagnostics
