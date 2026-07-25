import numpy as np

from validot.metrics import (
    exact_row_response,
    fixed_budget_retained_loss,
    mae_diagnostics,
    normalized_excess_aurc,
    risk_coverage_curve,
    top_fraction_precision,
)


def test_exact_response_identity_is_zero():
    plan = np.array([[0.2, 0.1], [0.1, 0.6]], dtype=float)
    assert np.allclose(exact_row_response(plan, plan), 0.0)


def test_oracle_risk_has_lower_aurc_than_reversed_risk():
    loss = np.array([0.0, 0.0, 1.0, 1.0])
    good = normalized_excess_aurc(loss, loss)["normalized_excess_aurc"]
    bad = normalized_excess_aurc(loss, -loss)["normalized_excess_aurc"]
    assert good < bad


def test_constant_risk_is_exactly_random_and_order_invariant():
    loss = np.array([0.0, 1.0, 0.0, 1.0, 1.0, 0.0])
    risk = np.zeros_like(loss)
    first = normalized_excess_aurc(loss, risk)["normalized_excess_aurc"]
    order = np.array([5, 2, 4, 0, 3, 1])
    second = normalized_excess_aurc(loss[order], risk[order])["normalized_excess_aurc"]
    assert np.isclose(first, 1.0)
    assert np.isclose(second, 1.0)


def test_partial_boundary_tie_uses_fractional_block_mean():
    loss = np.array([0.0, 1.0, 0.0, 1.0])
    risk = np.ones_like(loss)
    _, curve = risk_coverage_curve(loss, risk, np.array([0.5]))
    assert np.isclose(curve[0], 0.5)


def test_fixed_budget_tie_is_broken_by_source_index():
    loss = np.array([1.0, 0.0, 0.0, 1.0])
    risk = np.ones_like(loss)
    source_index = np.array([3, 0, 1, 2])
    _, retained = fixed_budget_retained_loss(
        loss, risk, source_index, np.array([0.5])
    )
    # The two smallest frozen source indices are rows 1 and 2.
    assert np.isclose(retained[0], 0.0)


def test_tie_aware_top_fraction_is_invariant_and_identical_is_one():
    reference = np.array([3.0, 2.0, 2.0, 1.0])
    assert np.isclose(top_fraction_precision(reference, reference, 0.5), 1.0)
    order = np.array([2, 0, 3, 1])
    assert np.isclose(
        top_fraction_precision(reference[order], reference[order], 0.5),
        1.0,
    )


def test_mae_diagnostics_marks_degenerate_reference_non_estimable():
    result = mae_diagnostics(np.ones(4), np.zeros(4))
    assert result["raw_mae"] == 1.0
    assert result["reference_mad"] == 0.0
    assert not result["normalized_mae_estimable"]
    assert np.isnan(result["normalized_mae"])
