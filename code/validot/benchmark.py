from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .metrics import exact_row_response, proxy_scores, top1_flip
from .semisynthetic import PairedData
from .solvers import (
    SolverResult,
    cost_components,
    log_sinkhorn,
    paste2_partial_fgw,
    paste_fgw,
    row_softmax,
)


@dataclass
class AuditResult:
    method: str
    base: SolverResult
    deleted: dict[str, SolverResult]
    endpoint: dict[str, SolverResult]
    exact_response: dict[str, np.ndarray]
    endpoint_response: dict[str, np.ndarray]
    flips: dict[str, np.ndarray]
    proxies: dict[str, np.ndarray]
    components: dict[str, np.ndarray]


def _masses(pair: PairedData) -> tuple[np.ndarray, np.ndarray]:
    return np.full(len(pair.source_x), 1.0 / len(pair.source_x)), np.full(
        len(pair.target_x), 1.0 / len(pair.target_x)
    )


def _gw_edge_cost(
    plan: np.ndarray,
    source_structure: np.ndarray,
    target_structure: np.ndarray,
) -> np.ndarray:
    """Square-loss GW edge contribution evaluated at an incumbent plan."""
    row_mass = plan.sum(axis=1)
    col_mass = plan.sum(axis=0)
    source_term = np.square(source_structure) @ row_mass
    target_term = np.square(target_structure) @ col_mass
    cross_term = source_structure @ plan @ target_structure.T
    return np.maximum(source_term[:, None] + target_term[None, :] - 2.0 * cross_term, 0.0)


def _solve_common(
    method: str,
    components: dict[str, np.ndarray],
    a: np.ndarray,
    b: np.ndarray,
    settings: dict[str, Any],
    expression_weight: float,
    spatial_weight: float,
) -> SolverResult:
    expression = components["expression"]
    spatial_cross = components["spatial_cross"]
    total = expression_weight * expression + spatial_weight * spatial_cross
    if method == "row_softmax":
        return row_softmax(total, a, settings["epsilon"])
    if method == "balanced_ot":
        return log_sinkhorn(
            total,
            a,
            b,
            settings["epsilon"],
            max_iter=settings["sinkhorn_max_iter"],
            tol=settings["sinkhorn_tol"],
        )
    if method == "uot":
        return log_sinkhorn(
            total,
            a,
            b,
            settings["epsilon"],
            settings["uot_tau"],
            settings["uot_tau"],
            settings["sinkhorn_max_iter"],
            settings["sinkhorn_tol"],
        )
    if method == "paste_fgw":
        # With square-loss GW, multiplying both structures by sqrt(w)
        # yields a linear weight w on the structural objective.
        structure_scale = np.sqrt(max(float(spatial_weight), 0.0))
        source_structure = structure_scale * components["source_structure"]
        target_structure = structure_scale * components["target_structure"]
        return paste_fgw(
            expression_weight * expression,
            source_structure,
            target_structure,
            a,
            b,
            settings["fgw_alpha"],
            settings["fgw_max_iter"],
        )
    if method == "paste2_partial_fgw":
        structure_scale = np.sqrt(max(float(spatial_weight), 0.0))
        source_structure = structure_scale * components["source_structure"]
        target_structure = structure_scale * components["target_structure"]
        return paste2_partial_fgw(
            expression_weight * expression,
            source_structure,
            target_structure,
            a,
            b,
            settings["fgw_alpha"],
            settings["paste2_overlap"],
            settings["fgw_max_iter"],
        )
    raise KeyError(method)


def run_audit(pair: PairedData, method: str, settings: dict[str, Any], endpoint_step: float = 0.01) -> AuditResult:
    components = cost_components(pair.source_x, pair.target_x, pair.source_xy, pair.target_xy)
    a, b = _masses(pair)
    base = _solve_common(method, components, a, b, settings, 0.5, 0.5)
    deleted = {
        "I_EXPR": _solve_common(method, components, a, b, settings, 0.0, 0.5),
        "I_SPATIAL": _solve_common(method, components, a, b, settings, 0.5, 0.0),
    }
    endpoint = {
        "I_EXPR": _solve_common(method, components, a, b, settings, 0.5 * (1.0 - endpoint_step), 0.5),
        "I_SPATIAL": _solve_common(method, components, a, b, settings, 0.5, 0.5 * (1.0 - endpoint_step)),
    }
    exact = {name: exact_row_response(base.plan, result.plan) for name, result in deleted.items()}
    endpoint_response = {
        name: exact_row_response(base.plan, result.plan) / endpoint_step for name, result in endpoint.items()
    }
    flips = {name: top1_flip(base.plan, result.plan) for name, result in deleted.items()}
    if method in {"paste_fgw", "paste2_partial_fgw"}:
        structure_scale = np.sqrt(0.5)
        gw_edge = _gw_edge_cost(
            base.plan,
            structure_scale * components["source_structure"],
            structure_scale * components["target_structure"],
        )
        components["model_expression_cost"] = (
            (1.0 - settings["fgw_alpha"]) * 0.5 * components["expression"]
        )
        components["model_spatial_cost"] = settings["fgw_alpha"] * gw_edge
        # The square-loss GW derivative contributes twice its edge density.
        components["model_local_cost"] = (
            components["model_expression_cost"] + 2.0 * components["model_spatial_cost"]
        )
    else:
        components["model_expression_cost"] = 0.5 * components["expression"]
        components["model_spatial_cost"] = 0.5 * components["spatial_cross"]
        components["model_local_cost"] = (
            components["model_expression_cost"] + components["model_spatial_cost"]
        )
    local_cost = components["model_local_cost"]
    proxies = proxy_scores(base.plan, local_cost)
    return AuditResult(
        method=method,
        base=base,
        deleted=deleted,
        endpoint=endpoint,
        exact_response=exact,
        endpoint_response=endpoint_response,
        flips=flips,
        proxies=proxies,
        components=components,
    )
