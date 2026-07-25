"""P1 scale/regularization sensitivity primitives.

This module is additive. It does not alter the registered v1.2/v1.3 audit
implementation or any frozen output path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.special import xlogy
from scipy.stats import spearmanr

from .metrics import (
    conditional_plan,
    exact_row_response,
    mae_diagnostics,
    top_fraction_precision,
)
from .solvers import SolverResult, log_sinkhorn, row_softmax


INTERVENTIONS = ("I_EXPR", "I_SPATIAL")
ARMS = ("R", "N")


@dataclass(frozen=True)
class P1Parameters:
    method: str
    epsilon: float
    tau: float | None = None
    max_iter: int = 4000
    tolerance: float = 1e-8


def arm_weights(arm: str, intervention: str, t: float) -> tuple[float, float]:
    """Return (expression, spatial) weights on the frozen intervention path."""
    if arm not in ARMS:
        raise ValueError(f"unknown P1 arm: {arm}")
    if intervention not in INTERVENTIONS:
        raise ValueError(f"unknown intervention: {intervention}")
    if not 0.0 <= float(t) <= 1.0:
        raise ValueError("t must lie in [0,1]")
    t = float(t)
    if arm == "R" and intervention == "I_EXPR":
        return 0.5 * (1.0 - t), 0.5
    if arm == "R" and intervention == "I_SPATIAL":
        return 0.5, 0.5 * (1.0 - t)
    if arm == "N" and intervention == "I_EXPR":
        return 0.5 * (1.0 - t), 0.5 * (1.0 + t)
    return 0.5 * (1.0 + t), 0.5 * (1.0 - t)


def mixed_cost(
    expression: np.ndarray,
    spatial: np.ndarray,
    weights: tuple[float, float],
) -> np.ndarray:
    return float(weights[0]) * np.asarray(expression) + float(weights[1]) * np.asarray(spatial)


def solve_p1(
    cost: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    parameters: P1Parameters,
) -> SolverResult:
    if parameters.method == "balanced_ot":
        return log_sinkhorn(
            cost,
            a,
            b,
            epsilon=parameters.epsilon,
            max_iter=parameters.max_iter,
            tol=parameters.tolerance,
        )
    if parameters.method == "uot":
        if parameters.tau is None:
            raise ValueError("UOT requires tau")
        return log_sinkhorn(
            cost,
            a,
            b,
            epsilon=parameters.epsilon,
            tau_a=parameters.tau,
            tau_b=parameters.tau,
            max_iter=parameters.max_iter,
            tol=parameters.tolerance,
        )
    if parameters.method == "row_softmax":
        return row_softmax(cost, a, epsilon=parameters.epsilon)
    raise ValueError(f"unsupported P1 method: {parameters.method}")


def positive_median(values: np.ndarray) -> float:
    positive = np.asarray(values, dtype=float)
    positive = positive[positive > 0]
    return float(np.median(positive)) if positive.size else float("nan")


def plan_entropy(plan: np.ndarray) -> float:
    plan = np.asarray(plan, dtype=float)
    mass = float(plan.sum())
    if mass <= 0:
        return float("nan")
    probability = plan.ravel() / mass
    return float(-np.sum(xlogy(probability, probability)))


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    return float(np.sum(xlogy(p, p / np.maximum(q, 1e-300)) - p + q))


def primal_objective(
    method: str,
    plan: np.ndarray,
    cost: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    epsilon: float,
    tau: float | None,
) -> float:
    """Evaluate the audited primal objective, up to no omitted plan terms."""
    plan = np.asarray(plan, dtype=float)
    value = float(np.sum(plan * cost) + epsilon * np.sum(xlogy(plan, plan) - plan))
    if method == "uot":
        if tau is None:
            raise ValueError("UOT objective requires tau")
        value += tau * kl_divergence(plan.sum(axis=1), a)
        value += tau * kl_divergence(plan.sum(axis=0), b)
    return value


def response_metrics(
    reference: np.ndarray,
    score: np.ndarray,
    gate: dict[str, float],
) -> dict[str, Any]:
    reference = np.asarray(reference, dtype=float)
    score = np.asarray(score, dtype=float)
    rho = spearmanr(reference, score).statistic
    rho = float(rho) if np.isfinite(rho) else 0.0
    overlap = float(top_fraction_precision(reference, score, 0.1))
    amplitude = mae_diagnostics(reference, score)
    q25, q75 = np.quantile(reference, [0.25, 0.75])
    spearman_pass = bool(rho >= float(gate["spearman_min"]))
    overlap_pass = bool(overlap >= float(gate["top_decile_overlap_min"]))
    nmae_pass = bool(
        amplitude["normalized_mae_estimable"]
        and amplitude["normalized_mae"] <= float(gate["nmae_max"])
    )
    return {
        "spearman": rho,
        "top_decile_overlap": overlap,
        "raw_mae": float(amplitude["raw_mae"]),
        "reference_mad": float(amplitude["reference_mad"]),
        "nmae": float(amplitude["normalized_mae"]),
        "reference_response_median": float(np.median(reference)),
        "reference_response_iqr": float(q75 - q25),
        "estimable": bool(amplitude["normalized_mae_estimable"]),
        "spearman_gate_pass": spearman_pass,
        "overlap_gate_pass": overlap_pass,
        "nmae_gate_pass": nmae_pass,
        "gate_pass": bool(spearman_pass and overlap_pass and nmae_pass),
    }


def plan_difference(
    plan_n: np.ndarray,
    plan_s: np.ndarray,
    tolerance: float,
) -> dict[str, Any]:
    delta = np.abs(np.asarray(plan_n) - np.asarray(plan_s))
    denominator = max(float(np.abs(plan_n).sum()), 1e-300)
    normalized = float(delta.sum() / denominator)
    return {
        "max_absolute_plan_difference": float(delta.max(initial=0.0)),
        "mean_absolute_plan_difference": float(delta.mean()),
        "normalized_l1_plan_difference": normalized,
        "equivalence_tolerance": float(tolerance),
        "equivalence_pass": bool(normalized <= tolerance),
    }


def solver_diagnostics(
    result: SolverResult,
    cost: np.ndarray,
    epsilon: float,
    process_rss_mb: float,
) -> dict[str, Any]:
    positive = np.asarray(cost, dtype=float)
    positive = positive[positive > 0]
    scaled = positive / float(epsilon) if positive.size else np.asarray([np.nan])
    q05, q50, q95 = np.nanquantile(scaled, [0.05, 0.5, 0.95])
    diagnostics = result.diagnostics or {}
    return {
        "mixed_cost_positive_median": positive_median(cost),
        "cost_over_epsilon_q05": float(q05),
        "cost_over_epsilon_median": float(q50),
        "cost_over_epsilon_q95": float(q95),
        "plan_entropy": plan_entropy(result.plan),
        "transported_mass": float(result.plan.sum()),
        "solver_iterations": int(result.iterations),
        "converged": bool(result.converged),
        "runtime_seconds": float(result.seconds),
        "peak_memory_mb": float(process_rss_mb),
        "memory_measurement": "process_rss_after_solve",
        "solver_last_error": float(diagnostics.get("last_log_scaling_error", math.nan)),
        "row_mass_l1": float(diagnostics.get("row_mass_l1", math.nan)),
        "col_mass_l1": float(diagnostics.get("col_mass_l1", math.nan)),
    }


def exact_and_fd_response(
    base: np.ndarray,
    endpoint: np.ndarray,
    finite_difference: np.ndarray,
    h: float,
) -> tuple[np.ndarray, np.ndarray]:
    return exact_row_response(base, endpoint), exact_row_response(base, finite_difference) / float(h)
