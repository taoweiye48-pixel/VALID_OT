"""Frozen design primitives for the WP11 cost-composition response surface."""

from __future__ import annotations

from typing import Any

import numpy as np


BASE_ALPHA = 0.5
BASE_BETA = 0.5


def _tag(value: float) -> str:
    return f"{float(value):.6f}".replace("-", "m").replace(".", "p")


def coefficients(u: float, v: float) -> tuple[float, float]:
    """Convert total scale/composition coordinates to cost coefficients."""
    u = float(u)
    v = float(v)
    if u <= 0 or not 0.0 <= v <= 1.0:
        raise ValueError("u must be positive and v must lie in [0, 1]")
    return u * (1.0 - v), u * v


def regularization(
    method: str,
    epsilon0: float,
    tau0: float | None,
    u: float,
    regime: str,
) -> tuple[float, float | None]:
    """Return endpoint epsilon/tau under fixed or co-regularized scaling."""
    if regime not in {"fixed", "coregularized"}:
        raise ValueError(f"unknown regularization regime: {regime}")
    factor = 1.0 if regime == "fixed" else float(u)
    epsilon = float(epsilon0) * factor
    tau = None
    if method == "uot":
        if tau0 is None:
            raise ValueError("UOT requires a baseline tau")
        tau = float(tau0) * factor
    return epsilon, tau


def interpolate_from_baseline(
    alpha: float,
    beta: float,
    h: float,
) -> tuple[float, float, float]:
    """Move a fraction h from the frozen baseline to a WP11 endpoint."""
    h = float(h)
    if not 0.0 <= h <= 1.0:
        raise ValueError("h must lie in [0, 1]")
    alpha_h = BASE_ALPHA + h * (float(alpha) - BASE_ALPHA)
    beta_h = BASE_BETA + h * (float(beta) - BASE_BETA)
    return alpha_h, beta_h, alpha_h + beta_h


def build_condition_registries(design: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return logical analysis rows and deduplicated physical solve conditions.

    Fixed and co-regularized regimes are mathematically identical whenever
    u=1.  They remain separate logical rows but share a physical solve ID.
    """
    logical: list[dict[str, Any]] = []
    for u in map(float, design["u_grid"]):
        for v in map(float, design["v_grid"]):
            alpha, beta = coefficients(u, v)
            for regime in design["regularization_regimes"]:
                shared = bool(np.isclose(u, 1.0))
                physical_regime = "fixed" if shared else str(regime)
                physical_id = (
                    f"grid__u{_tag(u)}__v{_tag(v)}__"
                    f"{'shared' if shared else physical_regime}"
                )
                logical.append(
                    {
                        "condition_id": f"grid__u{_tag(u)}__v{_tag(v)}__{regime}",
                        "physical_id": physical_id,
                        "condition_family": "grid",
                        "u": u,
                        "v": v,
                        "alpha": alpha,
                        "beta": beta,
                        "regularization_regime": str(regime),
                        "physical_regularization_regime": physical_regime,
                        "shared_at_u1": shared,
                    }
                )
    for node in design["factorial_extra_nodes"]:
        alpha = float(node["alpha"])
        beta = float(node["beta"])
        u = alpha + beta
        v = beta / u
        for regime in design["regularization_regimes"]:
            physical_id = f"factor__{node['name']}__{regime}"
            logical.append(
                {
                    "condition_id": physical_id,
                    "physical_id": physical_id,
                    "condition_family": "factorial_extra",
                    "factorial_node": str(node["name"]),
                    "u": u,
                    "v": v,
                    "alpha": alpha,
                    "beta": beta,
                    "regularization_regime": str(regime),
                    "physical_regularization_regime": str(regime),
                    "shared_at_u1": False,
                }
            )
    physical_by_id: dict[str, dict[str, Any]] = {}
    for row in logical:
        physical_by_id.setdefault(
            row["physical_id"],
            {
                "physical_id": row["physical_id"],
                "condition_family": row["condition_family"],
                "u": row["u"],
                "v": row["v"],
                "alpha": row["alpha"],
                "beta": row["beta"],
                "regularization_regime": row["physical_regularization_regime"],
            },
        )
    return logical, list(physical_by_id.values())


def factor_coordinates(channel: str) -> dict[str, tuple[float, float]]:
    if channel == "expression":
        return {
            "baseline": (0.5, 0.5),
            "removal": (0.0, 0.5),
            "compensation": (0.5, 1.0),
            "joint": (0.0, 1.0),
        }
    if channel == "spatial":
        return {
            "baseline": (0.5, 0.5),
            "removal": (0.5, 0.0),
            "compensation": (1.0, 0.5),
            "joint": (1.0, 0.0),
        }
    raise ValueError(f"unknown channel: {channel}")


def factorial_effect(values: dict[str, float]) -> dict[str, float]:
    required = {"baseline", "removal", "compensation", "joint"}
    if set(values) != required:
        raise ValueError(f"factorial values must contain exactly {sorted(required)}")
    baseline = float(values["baseline"])
    removal = float(values["removal"]) - baseline
    compensation = float(values["compensation"]) - baseline
    joint = float(values["joint"]) - baseline
    return {
        "baseline": baseline,
        "delta_removal": removal,
        "delta_compensation": compensation,
        "delta_joint": joint,
        "delta_interaction": joint - removal - compensation,
    }
