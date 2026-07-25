from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import ot
from scipy.optimize import minimize
from scipy.special import logsumexp


@dataclass
class SolverResult:
    plan: np.ndarray
    converged: bool
    iterations: int
    seconds: float
    objective: float | None = None
    diagnostics: dict[str, Any] | None = None


def _validate_problem(cost: np.ndarray, a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cost = np.asarray(cost, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if cost.shape != (len(a), len(b)):
        raise ValueError(f"cost shape {cost.shape} != {(len(a), len(b))}")
    if not np.all(np.isfinite(cost)) or np.min(cost) < 0:
        raise ValueError("cost must be finite and non-negative")
    if np.any(a <= 0) or np.any(b <= 0):
        raise ValueError("masses must be strictly positive")
    return cost, a, b


def _balanced_newton_sinkhorn(
    cost: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    epsilon: float,
    max_iter: int,
    tol: float,
    init: tuple[np.ndarray, np.ndarray] | None,
) -> SolverResult:
    """Solve balanced entropic OT, then refine its dual with Newton-CG.

    Alternating scaling can be extremely slow for near-permutation spatial
    problems.  The trust-region Newton step optimizes the identical entropic
    dual; fixing one target potential removes the additive gauge freedom.
    """
    if len(b) == 1:
        plan = a[:, None]
        return SolverResult(
            plan=plan,
            converged=bool(abs(float(b[0]) - float(a.sum())) <= tol),
            iterations=1,
            seconds=0.0,
            diagnostics={
                "last_log_scaling_error": 0.0,
                "row_mass_l1": 0.0,
                "col_mass_l1": abs(float(plan.sum()) - float(b[0])),
                "transported_mass": float(plan.sum()),
                "logu": np.log(np.maximum(a, 1e-300)),
                "logv": np.zeros(1),
                "refiner": "closed_form_single_column",
            },
        )
    kernel_log = -cost / epsilon
    loga = np.log(a)
    logb = np.log(b)
    if init is None:
        logu = np.zeros_like(a)
        logv = np.zeros_like(b)
    else:
        logu = np.asarray(init[0], dtype=np.float64).copy()
        logv = np.asarray(init[1], dtype=np.float64).copy()
    started = time.perf_counter()
    scaling_error = np.inf
    pre_iterations = min(int(max_iter), 200)
    for iteration in range(1, pre_iterations + 1):
        previous_u = logu.copy()
        previous_v = logv.copy()
        logu = loga - logsumexp(kernel_log + logv[None, :], axis=1)
        logv = logb - logsumexp(kernel_log + logu[:, None], axis=0)
        if iteration % 5 == 0 or iteration == 1:
            scaling_error = max(
                float(np.max(np.abs(logu - previous_u))),
                float(np.max(np.abs(logv - previous_v))),
            )

    f = epsilon * logu
    g = epsilon * logv
    shift = g[-1]
    initial = np.concatenate([f + shift, (g - shift)[:-1]])
    n_source = len(a)
    cache: dict[str, np.ndarray | None] = {"x": None, "plan": None}

    def current_plan(x: np.ndarray) -> np.ndarray:
        cached_x = cache["x"]
        if cached_x is None or not np.array_equal(x, cached_x):
            dual_f = x[:n_source]
            dual_g = np.concatenate([x[n_source:], np.zeros(1)])
            exponent = (dual_f[:, None] + dual_g[None, :] - cost) / epsilon
            cache["plan"] = np.exp(np.clip(exponent, -745.0, 700.0))
            cache["x"] = x.copy()
        return np.asarray(cache["plan"])

    def objective(x: np.ndarray) -> float:
        plan = current_plan(x)
        dual_f = x[:n_source]
        dual_g = np.concatenate([x[n_source:], np.zeros(1)])
        return float(epsilon * plan.sum() - a @ dual_f - b @ dual_g)

    def gradient(x: np.ndarray) -> np.ndarray:
        plan = current_plan(x)
        return np.concatenate([plan.sum(axis=1) - a, (plan.sum(axis=0) - b)[:-1]])

    def hessian_product(x: np.ndarray, vector: np.ndarray) -> np.ndarray:
        plan = current_plan(x)
        source_vector = vector[:n_source]
        target_vector = vector[n_source:]
        source_mass = plan.sum(axis=1)
        target_mass = plan.sum(axis=0)[:-1]
        return np.concatenate(
            [
                (source_mass * source_vector + plan[:, :-1] @ target_vector) / epsilon,
                (plan[:, :-1].T @ source_vector + target_mass * target_vector) / epsilon,
            ]
        )

    optimization = minimize(
        objective,
        initial,
        jac=gradient,
        hessp=hessian_product,
        method="trust-ncg",
        # SciPy's gtol is an infinity norm while the registered primal test is
        # an L1 marginal error. Scale the dual threshold by problem size so a
        # successful optimizer stop is stringent enough for the primal gate.
        options={
            "maxiter": min(300, max(50, int(max_iter))),
            "gtol": tol / max(len(a) + len(b), 1),
        },
    )
    plan = current_plan(optimization.x).copy()
    row_error = float(np.abs(plan.sum(axis=1) - a).sum())
    column_error = float(np.abs(plan.sum(axis=0) - b).sum())
    dual_f = optimization.x[:n_source]
    dual_g = np.concatenate([optimization.x[n_source:], np.zeros(1)])
    converged = bool(
        np.all(np.isfinite(plan))
        and np.min(plan) >= 0
        and row_error <= tol
        and column_error <= tol
    )
    diagnostics = {
        "last_log_scaling_error": scaling_error,
        "row_mass_l1": row_error,
        "col_mass_l1": column_error,
        "transported_mass": float(plan.sum()),
        "logu": dual_f / epsilon,
        "logv": dual_g / epsilon,
        "refiner": "trust_ncg_entropic_dual",
        "refiner_success": bool(optimization.success),
        "refiner_message": str(optimization.message),
        "refiner_gradient_inf": float(np.max(np.abs(gradient(optimization.x)))),
        "refiner_hessian_products": int(getattr(optimization, "nhev", 0)),
    }
    return SolverResult(
        plan=plan,
        converged=converged,
        iterations=pre_iterations + int(optimization.nit),
        seconds=time.perf_counter() - started,
        objective=float(optimization.fun),
        diagnostics=diagnostics,
    )


def log_sinkhorn(
    cost: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    epsilon: float = 0.05,
    tau_a: float | None = None,
    tau_b: float | None = None,
    max_iter: int = 4000,
    tol: float = 1e-9,
    init: tuple[np.ndarray, np.ndarray] | None = None,
) -> SolverResult:
    """Balanced or KL-unbalanced entropic OT using stable log-domain scaling.

    tau_a=tau_b=None gives balanced Sinkhorn. Finite positive tau values give
    unbalanced Sinkhorn with exponent tau/(tau+epsilon).
    """
    cost, a, b = _validate_problem(cost, a, b)
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    balanced = tau_a is None and tau_b is None
    if (tau_a is None) != (tau_b is None):
        raise ValueError("tau_a and tau_b must both be None or finite")
    if not balanced and (tau_a <= 0 or tau_b <= 0):
        raise ValueError("finite tau values must be positive")
    if balanced:
        return _balanced_newton_sinkhorn(cost, a, b, epsilon, max_iter, tol, init)
    power_a = 1.0 if balanced else float(tau_a / (tau_a + epsilon))
    power_b = 1.0 if balanced else float(tau_b / (tau_b + epsilon))
    loga = np.log(a)
    logb = np.log(b)
    kernel_log = -cost / epsilon
    if init is None:
        logu = np.zeros_like(a)
        logv = np.zeros_like(b)
    else:
        logu = np.asarray(init[0], dtype=np.float64).copy()
        logv = np.asarray(init[1], dtype=np.float64).copy()
    started = time.perf_counter()
    converged = False
    error = np.inf
    for iteration in range(1, max_iter + 1):
        previous_u = logu.copy()
        previous_v = logv.copy()
        logu = power_a * (loga - logsumexp(kernel_log + logv[None, :], axis=1))
        logv = power_b * (logb - logsumexp(kernel_log + logu[:, None], axis=0))
        if iteration % 5 == 0 or iteration == 1:
            error = max(float(np.max(np.abs(logu - previous_u))), float(np.max(np.abs(logv - previous_v))))
            if error <= tol:
                converged = True
                break
    plan_log = kernel_log + logu[:, None] + logv[None, :]
    plan = np.exp(np.clip(plan_log, -745.0, 700.0))
    seconds = time.perf_counter() - started
    if not np.all(np.isfinite(plan)) or np.any(plan < 0):
        raise FloatingPointError("invalid transport plan")
    diagnostics = {
        "last_log_scaling_error": error,
        "row_mass_l1": float(np.abs(plan.sum(axis=1) - a).sum()),
        "col_mass_l1": float(np.abs(plan.sum(axis=0) - b).sum()),
        "transported_mass": float(plan.sum()),
        "logu": logu,
        "logv": logv,
    }
    return SolverResult(plan, converged, iteration, seconds, diagnostics=diagnostics)


def row_softmax(cost: np.ndarray, a: np.ndarray, epsilon: float = 0.05) -> SolverResult:
    cost = np.asarray(cost, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    if cost.shape[0] != len(a):
        raise ValueError("row mass length mismatch")
    started = time.perf_counter()
    logits = -cost / epsilon
    logits -= logsumexp(logits, axis=1, keepdims=True)
    plan = a[:, None] * np.exp(logits)
    return SolverResult(
        plan=plan,
        converged=True,
        iterations=1,
        seconds=time.perf_counter() - started,
        diagnostics={"transported_mass": float(plan.sum())},
    )


def paste_fgw(
    expression_cost: np.ndarray,
    source_structure: np.ndarray,
    target_structure: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    alpha: float = 0.1,
    max_iter: int = 200,
    init: np.ndarray | None = None,
) -> SolverResult:
    from paste.PASTE import my_fused_gromov_wasserstein

    expression_cost, a, b = _validate_problem(expression_cost, a, b)
    started = time.perf_counter()
    plan, log = my_fused_gromov_wasserstein(
        expression_cost,
        np.asarray(source_structure, dtype=np.float64),
        np.asarray(target_structure, dtype=np.float64),
        a,
        b,
        G_init=init,
        alpha=float(alpha),
        log=True,
        numItermax=int(max_iter),
        tol_rel=1e-9,
        tol_abs=1e-9,
        verbose=False,
        use_gpu=False,
    )
    plan = np.asarray(plan, dtype=np.float64)
    loss = log.get("fgw_dist") if isinstance(log, dict) else None
    loss_history = np.asarray(log.get("loss", []), dtype=np.float64) if isinstance(log, dict) else np.asarray([])
    final_delta = (
        float(abs(loss_history[-1] - loss_history[-2])) if len(loss_history) >= 2 else float("nan")
    )
    stationary = bool(
        len(loss_history) >= 2
        and final_delta <= max(1e-9, 1e-9 * abs(float(loss_history[-1])))
    )
    return SolverResult(
        plan=plan,
        # For non-convex FGW, this denotes numerical stationarity of the
        # frozen solver, not a certificate of the global optimum.
        converged=bool(np.all(np.isfinite(plan)) and stationary),
        iterations=len(loss_history) if isinstance(log, dict) else max_iter,
        seconds=time.perf_counter() - started,
        objective=float(loss) if loss is not None else None,
        diagnostics={
            "transported_mass": float(plan.sum()),
            "final_objective_delta": final_delta,
            "stationarity_proxy_pass": stationary,
            "global_optimum_certified": False,
        },
    )


def paste2_partial_fgw(
    expression_cost: np.ndarray,
    source_structure: np.ndarray,
    target_structure: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    alpha: float = 0.1,
    overlap: float = 0.7,
    max_iter: int = 200,
    init: np.ndarray | None = None,
) -> SolverResult:
    from paste2.PASTE2 import partial_fused_gromov_wasserstein

    expression_cost, a, b = _validate_problem(expression_cost, a, b)
    transported_mass = float(overlap * min(a.sum(), b.sum()))
    started = time.perf_counter()
    plan, log = partial_fused_gromov_wasserstein(
        expression_cost,
        np.asarray(source_structure, dtype=np.float64),
        np.asarray(target_structure, dtype=np.float64),
        a,
        b,
        alpha=float(alpha),
        m=transported_mass,
        G0=init,
        log=True,
        verbose=False,
        numItermax=int(max_iter),
        stopThr=1e-9,
        stopThr2=1e-9,
    )
    plan = np.asarray(plan, dtype=np.float64)
    loss_history = np.asarray(log.get("loss", []), dtype=np.float64)
    final_delta = (
        float(abs(loss_history[-1] - loss_history[-2])) if len(loss_history) >= 2 else float("nan")
    )
    stationary = bool(
        len(loss_history) >= 2
        and final_delta <= max(1e-9, 1e-9 * abs(float(loss_history[-1])))
    )
    return SolverResult(
        plan=plan,
        converged=bool(np.all(np.isfinite(plan)) and stationary),
        iterations=len(loss_history),
        seconds=time.perf_counter() - started,
        objective=float(log.get("partial_fgw_cost")) if log.get("partial_fgw_cost") is not None else None,
        diagnostics={
            "transported_mass": float(plan.sum()),
            "target_mass": transported_mass,
            "final_objective_delta": final_delta,
            "stationarity_proxy_pass": stationary,
            "global_optimum_certified": False,
        },
    )


def cost_components(x_source: np.ndarray, x_target: np.ndarray, xy_source: np.ndarray, xy_target: np.ndarray) -> dict[str, np.ndarray]:
    """Frozen, scale-normalized expression and spatial cost components."""
    x_source = np.asarray(x_source, dtype=np.float64)
    x_target = np.asarray(x_target, dtype=np.float64)
    xy_source = np.asarray(xy_source, dtype=np.float64)
    xy_target = np.asarray(xy_target, dtype=np.float64)
    expression = ot.dist(x_source, x_target, metric="sqeuclidean")
    spatial_cross = ot.dist(xy_source, xy_target, metric="sqeuclidean")
    source_structure = ot.dist(xy_source, xy_source, metric="euclidean")
    target_structure = ot.dist(xy_target, xy_target, metric="euclidean")

    def scale(matrix: np.ndarray) -> np.ndarray:
        positive = matrix[matrix > 0]
        denominator = float(np.median(positive)) if positive.size else 1.0
        return matrix / max(denominator, 1e-12)

    return {
        "expression": scale(expression),
        "spatial_cross": scale(spatial_cross),
        "source_structure": scale(source_structure),
        "target_structure": scale(target_structure),
    }
