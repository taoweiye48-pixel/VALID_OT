from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import statsmodels.api as sm
from scipy.stats import rankdata

from .benchmark import AuditResult
from .evaluation import risk_scores
from .metrics import normalized_excess_aurc, risk_coverage_curve
from .semisynthetic import PairedData


def _stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("::".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little")


def spatial_blocks(xy: np.ndarray, bins: int = 4) -> np.ndarray:
    xy = np.asarray(xy, dtype=float)
    labels = np.zeros(len(xy), dtype=int)
    for axis in range(2):
        order = np.argsort(xy[:, axis], kind="mergesort")
        axis_bin = np.empty(len(xy), dtype=int)
        axis_bin[order] = np.minimum((np.arange(len(xy)) * bins) // max(len(xy), 1), bins - 1)
        labels += axis_bin * (bins**axis)
    return labels


def permutation_controls(
    loss: np.ndarray,
    risk: np.ndarray,
    xy: np.ndarray,
    repeats: int,
    seed: int,
) -> dict[str, float | int]:
    loss = np.asarray(loss, dtype=float)
    risk = np.asarray(risk, dtype=float)
    valid = np.isfinite(loss) & np.isfinite(risk)
    loss = loss[valid]
    risk = risk[valid]
    xy = np.asarray(xy, dtype=float)[valid]
    if len(loss) < 20 or np.unique(loss).size < 2 or np.unique(risk).size < 2:
        return {"control_status": "INSUFFICIENT_VARIATION", "control_n": int(len(loss))}
    rng = np.random.default_rng(seed)
    coverages, oracle_curve = risk_coverage_curve(loss, loss)
    _, primary_curve = risk_coverage_curve(loss, risk, coverages)
    oracle_aurc = float(np.trapz(oracle_curve, coverages) / (coverages[-1] - coverages[0]))
    random_aurc = float(np.mean(loss))
    normalization_range = random_aurc - oracle_aurc

    def normalized_from_curve(candidate_loss: np.ndarray, candidate_risk: np.ndarray) -> float:
        _, curve = risk_coverage_curve(candidate_loss, candidate_risk, coverages)
        aurc = float(np.trapz(curve, coverages) / (coverages[-1] - coverages[0]))
        return (
            float((aurc - oracle_aurc) / normalization_range)
            if normalization_range > 1e-12
            else float("nan")
        )

    primary_aurc = float(np.trapz(primary_curve, coverages) / (coverages[-1] - coverages[0]))
    primary = (
        float((primary_aurc - oracle_aurc) / normalization_range)
        if normalization_range > 1e-12
        else float("nan")
    )
    blocks = spatial_blocks(xy)
    shuffled_loss = []
    within_block = []
    for _ in range(repeats):
        shuffled_loss.append(normalized_from_curve(rng.permutation(loss), risk))
        permuted_risk = risk.copy()
        for block in np.unique(blocks):
            index = np.flatnonzero(blocks == block)
            permuted_risk[index] = rng.permutation(permuted_risk[index])
        within_block.append(normalized_from_curve(loss, permuted_risk))
    shift = max(1, len(risk) // 3)
    circular = normalized_from_curve(loss, np.roll(risk, shift))
    leakage = 0.0 if normalization_range > 1e-12 else float("nan")
    shuffled_loss_array = np.asarray(shuffled_loss)
    within_block_array = np.asarray(within_block)
    return {
        "control_status": "COMPLETED",
        "control_n": int(len(loss)),
        "primary_normalized_excess_aurc": float(primary),
        "label_shuffle_median": float(np.median(shuffled_loss_array)),
        "label_shuffle_p_lower": float((1 + np.sum(shuffled_loss_array <= primary)) / (repeats + 1)),
        "within_block_permutation_median": float(np.median(within_block_array)),
        "within_block_p_lower": float((1 + np.sum(within_block_array <= primary)) / (repeats + 1)),
        "circular_shift_normalized_excess_aurc": float(circular),
        "leakage_positive_control_normalized_excess_aurc": float(leakage),
        "permutation_repeats": int(repeats),
    }


def adjusted_association(
    loss: np.ndarray,
    risk: np.ndarray,
    confounds: dict[str, np.ndarray],
    cluster_groups: np.ndarray | None = None,
) -> dict[str, Any]:
    arrays = [np.asarray(loss, dtype=float), np.asarray(risk, dtype=float)] + [
        np.asarray(value, dtype=float) for value in confounds.values()
    ]
    valid = np.logical_and.reduce([np.isfinite(value) for value in arrays])
    y = arrays[0][valid]
    if len(y) < 30 or np.unique(y).size < 2:
        return {"adjusted_status": "INSUFFICIENT_VARIATION", "adjusted_n": int(len(y))}
    columns = [rankdata(value[valid], method="average") for value in arrays[1:]]
    x = np.column_stack(columns)
    scale = np.std(x, axis=0)
    keep = scale > 1e-12
    x = x[:, keep]
    if x.shape[1] == 0:
        return {"adjusted_status": "INSUFFICIENT_VARIATION", "adjusted_n": int(len(y))}
    x = (x - np.mean(x, axis=0)) / np.std(x, axis=0)
    x = sm.add_constant(x, has_constant="add")
    groups = None if cluster_groups is None else np.asarray(cluster_groups)[valid]
    covariance = (
        {"cov_type": "cluster", "cov_kwds": {"groups": groups}}
        if groups is not None and np.unique(groups).size >= 4
        else {"cov_type": "HC3"}
    )
    try:
        if np.array_equal(np.unique(y), np.array([0.0, 1.0])):
            fit = sm.GLM(y, x, family=sm.families.Binomial()).fit(maxiter=200, **covariance)
            model = "binomial_glm_spatial_cluster" if groups is not None else "binomial_glm_hc3"
        else:
            fit = sm.OLS(y, x).fit(**covariance)
            model = "ols_spatial_cluster" if groups is not None else "ols_hc3"
        return {
            "adjusted_status": "COMPLETED",
            "adjusted_n": int(len(y)),
            "adjusted_model": model,
            "risk_rank_coefficient": float(fit.params[1]),
            "risk_rank_se": float(fit.bse[1]),
            "risk_rank_pvalue": float(fit.pvalues[1]),
            "risk_positive_after_adjustment": bool(fit.params[1] > 0),
        }
    except Exception as exc:
        return {
            "adjusted_status": "MODEL_FAILURE",
            "adjusted_n": int(len(y)),
            "adjusted_error": f"{type(exc).__name__}: {exc}",
        }


def real_control_records(
    pair: PairedData,
    audit: AuditResult,
    losses: dict[str, np.ndarray],
    extras: dict[str, np.ndarray],
    repeats: int = 100,
) -> list[dict[str, Any]]:
    scores = risk_scores(pair, audit)
    boundary = scores["source_boundary_proximity"]
    sparsity = scores["source_sparsity"]
    target_sparsity = scores["matched_target_sparsity"]
    library = np.log1p(extras.get("source_library_size", np.zeros(len(pair.source_x))))
    region_size = np.log1p(extras.get("source_region_size", np.ones(len(pair.source_x))))
    block_groups = spatial_blocks(pair.source_xy)
    records = []
    for witness, loss in losses.items():
        for score_name, score in scores.items():
            confounds = {
                "boundary": boundary,
                "sparsity": sparsity,
                "matched_target_sparsity": target_sparsity,
                "log_library_size": library,
                "log_region_size": region_size,
            }
            if score_name == "source_boundary_proximity":
                confounds.pop("boundary")
            if score_name == "source_sparsity":
                confounds.pop("sparsity")
            if score_name == "matched_target_sparsity":
                confounds.pop("matched_target_sparsity")
            seed = _stable_seed(pair.pair_id, audit.method, witness, score_name)
            records.append(
                {
                    "dataset": pair.dataset,
                    "pair_type": pair.metadata.get("pair_type", ""),
                    "biological_pair_id": pair.metadata.get("biological_pair_id", pair.pair_id),
                    "pair_id": pair.pair_id,
                    "direction": pair.metadata.get("direction", "forward"),
                    "method": audit.method,
                    "witness": witness,
                    "score": score_name,
                    **permutation_controls(loss, score, pair.source_xy, repeats, seed),
                    **adjusted_association(loss, score, confounds, cluster_groups=block_groups),
                }
            )
    return records
