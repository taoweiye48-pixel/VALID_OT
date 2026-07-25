from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from validot.wp11 import (
    build_condition_registries,
    coefficients,
    factorial_effect,
    interpolate_from_baseline,
    regularization,
)


ROOT = Path(__file__).resolve().parents[2]


def design() -> dict:
    config = json.loads((ROOT / "configs" / "postreview_wp11_v1.json").read_text(encoding="utf-8"))
    return config["wp11"]


def test_wp11_condition_counts_and_u1_deduplication() -> None:
    logical, physical = build_condition_registries(design())
    assert len(logical) == 34
    assert len(physical) == 29
    u1 = [row for row in logical if row["condition_family"] == "grid" and np.isclose(row["u"], 1.0)]
    assert len(u1) == 10
    assert len({row["physical_id"] for row in u1}) == 5


def test_wp11_coefficients_preserve_scale_and_composition() -> None:
    alpha, beta = coefficients(0.75, 0.25)
    assert np.isclose(alpha + beta, 0.75)
    assert np.isclose(beta / (alpha + beta), 0.25)


def test_wp11_coregularization_scales_all_applicable_terms() -> None:
    epsilon, tau = regularization("uot", 0.25, 2.0, 0.5, "coregularized")
    assert epsilon == 0.125
    assert tau == 1.0
    fixed_epsilon, fixed_tau = regularization("uot", 0.25, 2.0, 0.5, "fixed")
    assert fixed_epsilon == 0.25
    assert fixed_tau == 2.0


def test_wp11_local_coregularization_uses_interpolated_total_scale() -> None:
    alpha_h, beta_h, u_h = interpolate_from_baseline(0.0, 0.5, 0.01)
    assert np.isclose(alpha_h, 0.495)
    assert np.isclose(beta_h, 0.5)
    assert np.isclose(u_h, 0.995)


def test_wp11_factorial_interaction_is_nonadditive_residual() -> None:
    result = factorial_effect({"baseline": 1.0, "removal": 3.0, "compensation": 4.0, "joint": 8.0})
    assert result["delta_removal"] == 2.0
    assert result["delta_compensation"] == 3.0
    assert result["delta_joint"] == 7.0
    assert result["delta_interaction"] == 2.0
