from __future__ import annotations

import numpy as np
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score


def conditional_plan(plan: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    plan = np.asarray(plan, dtype=np.float64)
    row_mass = plan.sum(axis=1)
    conditional = plan / np.maximum(row_mass[:, None], 1e-300)
    return conditional, row_mass


def exact_row_response(base: np.ndarray, perturbed: np.ndarray) -> np.ndarray:
    numerator = np.abs(perturbed - base).sum(axis=1)
    denominator = perturbed.sum(axis=1) + base.sum(axis=1) + 1e-12
    return numerator / denominator


def top1_flip(base: np.ndarray, perturbed: np.ndarray) -> np.ndarray:
    base_cond, _ = conditional_plan(base)
    perturbed_cond, _ = conditional_plan(perturbed)
    return (np.argmax(base_cond, axis=1) != np.argmax(perturbed_cond, axis=1)).astype(int)


def barycentric_target(plan: np.ndarray, target_xy: np.ndarray) -> np.ndarray:
    conditional, _ = conditional_plan(plan)
    return conditional @ np.asarray(target_xy, dtype=np.float64)


def proxy_scores(plan: np.ndarray, total_cost: np.ndarray) -> dict[str, np.ndarray]:
    conditional, row_mass = conditional_plan(plan)
    order = np.argsort(conditional, axis=1)
    best_idx = order[:, -1]
    second_idx = order[:, -2] if plan.shape[1] > 1 else best_idx
    rows = np.arange(plan.shape[0])
    best_probability = conditional[rows, best_idx]
    second_probability = conditional[rows, second_idx]
    assigned_cost = total_cost[rows, best_idx]
    best_cost = np.partition(total_cost, 0, axis=1)[:, 0]
    second_cost = np.partition(total_cost, 1, axis=1)[:, 1] if plan.shape[1] > 1 else best_cost
    entropy = -np.sum(conditional * np.log(np.maximum(conditional, 1e-300)), axis=1)
    source_reference = np.full_like(row_mass, 1.0 / len(row_mass))
    return {
        "assigned_raw_cost": assigned_cost,
        "local_cost_margin": -(second_cost - best_cost),
        "conditional_entropy": entropy,
        "low_max_probability": 1.0 - best_probability,
        "probability_margin": -(best_probability - second_probability),
        "mass_deficit": 1.0 - row_mass / np.maximum(source_reference, 1e-300),
    }


def _fractional_top_membership(values: np.ndarray, count: int) -> np.ndarray:
    """Return deterministic fractional membership in a top-k set.

    Values strictly above the boundary have membership one.  The remaining
    capacity is divided uniformly across the boundary tie block.  This makes
    top-k comparisons invariant to input row order while retaining a value of
    one when two tied rankings are identical.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values):
        raise ValueError("values must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(values)):
        raise ValueError("values must be finite")
    count = int(np.clip(count, 1, len(values)))
    boundary = float(np.partition(values, len(values) - count)[len(values) - count])
    above = values > boundary
    tied = values == boundary
    remaining = count - int(np.sum(above))
    membership = above.astype(np.float64)
    if remaining > 0:
        membership[tied] = remaining / max(int(np.sum(tied)), 1)
    return membership


def top_fraction_precision(reference: np.ndarray, score: np.ndarray, fraction: float = 0.1) -> float:
    """Tie-aware fractional overlap of the two top-fraction sets."""
    reference = np.asarray(reference, dtype=np.float64)
    score = np.asarray(score, dtype=np.float64)
    if reference.shape != score.shape:
        raise ValueError("reference and score must have the same shape")
    count = max(1, int(np.ceil(fraction * len(reference))))
    ref_membership = _fractional_top_membership(reference, count)
    score_membership = _fractional_top_membership(score, count)
    return float(np.minimum(ref_membership, score_membership).sum() / count)


def normalized_mae(reference: np.ndarray, estimate: np.ndarray) -> float:
    median = np.median(reference)
    mad = np.median(np.abs(reference - median))
    return float(np.mean(np.abs(reference - estimate)) / max(mad, 1e-12))


def mae_diagnostics(
    reference: np.ndarray,
    estimate: np.ndarray,
    scale_tolerance: float = 1e-12,
) -> dict[str, float | bool]:
    """Report amplitude error together with the scale that normalizes it."""
    reference = np.asarray(reference, dtype=np.float64)
    estimate = np.asarray(estimate, dtype=np.float64)
    if reference.shape != estimate.shape:
        raise ValueError("reference and estimate must have the same shape")
    raw_mae = float(np.mean(np.abs(reference - estimate)))
    median = float(np.median(reference))
    reference_mad = float(np.median(np.abs(reference - median)))
    estimable = bool(reference_mad > scale_tolerance)
    return {
        "raw_mae": raw_mae,
        "reference_mad": reference_mad,
        "normalized_mae": raw_mae / reference_mad if estimable else float("nan"),
        "normalized_mae_estimable": estimable,
    }


def fidelity_metrics(reference: np.ndarray, score: np.ndarray, flips: np.ndarray | None = None) -> dict[str, float]:
    reference = np.asarray(reference, dtype=np.float64)
    score = np.asarray(score, dtype=np.float64)
    rho = (
        spearmanr(reference, score).statistic
        if np.unique(reference).size > 1 and np.unique(score).size > 1
        else float("nan")
    )
    result = {
        "spearman": float(rho) if np.isfinite(rho) else 0.0,
        "top_decile_precision": float(top_fraction_precision(reference, score)),
    }
    if flips is not None and len(np.unique(flips)) == 2:
        result["flip_auroc"] = float(roc_auc_score(flips, score))
        result["flip_auprc"] = float(average_precision_score(flips, score))
    else:
        result["flip_auroc"] = float("nan")
        result["flip_auprc"] = float("nan")
    return result


def risk_coverage_curve(loss: np.ndarray, risk: np.ndarray, coverages: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    loss = np.asarray(loss, dtype=np.float64)
    risk = np.asarray(risk, dtype=np.float64)
    if coverages is None:
        coverages = np.linspace(0.5, 1.0, 51)
    if loss.ndim != 1 or risk.ndim != 1 or loss.shape != risk.shape or not len(loss):
        raise ValueError("loss and risk must be non-empty one-dimensional arrays of equal length")
    if not np.all(np.isfinite(loss)) or not np.all(np.isfinite(risk)):
        raise ValueError("loss and risk must be finite")
    order = np.argsort(risk, kind="mergesort")
    sorted_risk = risk[order]
    sorted_loss = loss[order]
    block_starts = np.r_[0, np.flatnonzero(np.diff(sorted_risk) != 0) + 1]
    block_counts = np.diff(np.r_[block_starts, len(loss)])
    block_sums = np.add.reduceat(sorted_loss, block_starts)
    cumulative_counts = np.cumsum(block_counts)
    cumulative_sums = np.cumsum(block_sums)
    values = []
    for coverage in coverages:
        keep = max(1, int(np.ceil(coverage * len(loss))))
        block = int(np.searchsorted(cumulative_counts, keep, side="left"))
        previous_count = int(cumulative_counts[block - 1]) if block else 0
        previous_sum = float(cumulative_sums[block - 1]) if block else 0.0
        fraction = (keep - previous_count) / int(block_counts[block])
        retained_sum = previous_sum + fraction * float(block_sums[block])
        values.append(retained_sum / keep)
    return np.asarray(coverages), np.asarray(values)


def fixed_budget_retained_loss(
    loss: np.ndarray,
    risk: np.ndarray,
    source_index: np.ndarray,
    coverages: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return retained loss under an exact, deterministic row budget.

    Unlike :func:`risk_coverage_curve`, which fractionally averages a tied
    boundary block for order-invariant AURC estimation, this function selects
    an actual set of rows.  Rows are ordered lexicographically by ascending
    ``(risk, source_index)`` so that the selected count is exactly
    ``ceil(coverage * n)`` and boundary ties are reproducible.
    """
    loss = np.asarray(loss, dtype=np.float64)
    risk = np.asarray(risk, dtype=np.float64)
    source_index = np.asarray(source_index)
    if coverages is None:
        coverages = np.asarray([0.8, 0.9], dtype=np.float64)
    else:
        coverages = np.asarray(coverages, dtype=np.float64)
    if (
        loss.ndim != 1
        or risk.ndim != 1
        or source_index.ndim != 1
        or loss.shape != risk.shape
        or loss.shape != source_index.shape
        or not len(loss)
    ):
        raise ValueError("loss, risk and source_index must be non-empty aligned one-dimensional arrays")
    if not np.all(np.isfinite(loss)) or not np.all(np.isfinite(risk)):
        raise ValueError("loss and risk must be finite")
    if len(np.unique(source_index)) != len(source_index):
        raise ValueError("source_index must be unique")
    if np.any((coverages <= 0.0) | (coverages > 1.0)):
        raise ValueError("coverages must lie in (0, 1]")

    order = np.lexsort((source_index, risk))
    cumulative_loss = np.cumsum(loss[order])
    values = []
    for coverage in coverages:
        keep = max(1, int(np.ceil(float(coverage) * len(loss))))
        values.append(float(cumulative_loss[keep - 1] / keep))
    return coverages, np.asarray(values, dtype=np.float64)


def normalized_excess_aurc(loss: np.ndarray, risk: np.ndarray) -> dict[str, float]:
    coverages, curve = risk_coverage_curve(loss, risk)
    _, oracle = risk_coverage_curve(loss, loss)
    random_value = float(np.mean(loss))
    random_curve = np.full_like(curve, random_value)
    aurc = float(np.trapz(curve, coverages) / (coverages[-1] - coverages[0]))
    oracle_aurc = float(np.trapz(oracle, coverages) / (coverages[-1] - coverages[0]))
    random_aurc = float(np.trapz(random_curve, coverages) / (coverages[-1] - coverages[0]))
    normalization_range = random_aurc - oracle_aurc
    normalized = (
        (aurc - oracle_aurc) / normalization_range
        if normalization_range > 1e-12
        else float("nan")
    )
    coverage_80_index = int(np.argmin(np.abs(coverages - 0.80)))
    coverage_90_index = int(np.argmin(np.abs(coverages - 0.90)))
    return {
        "aurc": aurc,
        "oracle_aurc": oracle_aurc,
        "random_aurc": random_aurc,
        "normalized_excess_aurc": float(normalized),
        "retained_loss_at_80pct_coverage": float(curve[coverage_80_index]),
        "retained_loss_at_90pct_coverage": float(curve[coverage_90_index]),
    }


def rank_uniform(score: np.ndarray) -> np.ndarray:
    return (rankdata(score, method="average") - 0.5) / len(score)
