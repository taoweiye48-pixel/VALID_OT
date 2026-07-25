from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial import ConvexHull, cKDTree
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

from .benchmark import AuditResult
from .metrics import (
    barycentric_target,
    conditional_plan,
    fidelity_metrics,
    mae_diagnostics,
    normalized_excess_aurc,
    normalized_mae,
    rank_uniform,
)
from .semisynthetic import PairedData


def boundary_proximity(xy: np.ndarray) -> np.ndarray:
    xy = np.asarray(xy, dtype=np.float64)
    hull = ConvexHull(xy)
    normals = hull.equations[:, :-1]
    offsets = hull.equations[:, -1]
    distances = -(xy @ normals.T + offsets) / np.maximum(
        np.linalg.norm(normals, axis=1)[None, :], 1e-12
    )
    depth = np.maximum(np.min(distances, axis=1), 0.0)
    return -depth


def local_sparsity(xy: np.ndarray, k: int = 8) -> np.ndarray:
    xy = np.asarray(xy, dtype=np.float64)
    neighbors = min(k + 1, len(xy))
    distances = cKDTree(xy).query(xy, k=neighbors)[0]
    return distances[:, -1]


def group_contribution(audit: AuditResult, group: str) -> np.ndarray:
    conditional, _ = conditional_plan(audit.base.plan)
    component = audit.components[
        "model_expression_cost" if group == "I_EXPR" else "model_spatial_cost"
    ]
    return np.sum(conditional * component, axis=1)


def spatial_block_consistency(reference: np.ndarray, score: np.ndarray, xy: np.ndarray) -> dict[str, float]:
    bins = 4
    block = np.zeros(len(xy), dtype=int)
    for axis in range(2):
        order = np.argsort(xy[:, axis], kind="mergesort")
        axis_bin = np.empty(len(xy), dtype=int)
        axis_bin[order] = np.minimum((np.arange(len(xy)) * bins) // max(len(xy), 1), bins - 1)
        block += axis_bin * (bins**axis)
    global_rho = (
        spearmanr(reference, score).statistic
        if np.unique(reference).size > 1 and np.unique(score).size > 1
        else float("nan")
    )
    block_rhos = []
    for label in np.unique(block):
        index = np.flatnonzero(block == label)
        if len(index) < 10 or np.unique(reference[index]).size < 2 or np.unique(score[index]).size < 2:
            continue
        rho = spearmanr(reference[index], score[index]).statistic
        if np.isfinite(rho):
            block_rhos.append(float(rho))
    if not block_rhos:
        return {"spatial_block_count": 0, "spatial_block_same_sign_fraction": float("nan")}
    if not np.isfinite(global_rho) or global_rho == 0:
        same_sign = np.mean(np.asarray(block_rhos) == 0)
    else:
        same_sign = np.mean(np.sign(block_rhos) == np.sign(global_rho))
    return {
        "spatial_block_count": len(block_rhos),
        "spatial_block_same_sign_fraction": float(same_sign),
    }


def internal_fidelity_records(pair: PairedData, audit: AuditResult) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for intervention, reference in audit.exact_response.items():
        scores = dict(audit.proxies)
        scores["raw_group_contribution"] = group_contribution(audit, intervention)
        scores["finite_difference_sensitivity_h001"] = audit.endpoint_response[intervention]
        for name, score in scores.items():
            metrics = fidelity_metrics(reference, score, audit.flips[intervention])
            metrics.update(spatial_block_consistency(reference, score, pair.source_xy))
            # Endpoint response and exact response share the same response units.
            # Other proxies require a frozen development calibration, so their
            # amplitude error is deliberately not fabricated here.
            if name == "finite_difference_sensitivity_h001":
                metrics.update(mae_diagnostics(reference, score))
            else:
                metrics["normalized_mae"] = float("nan")
                metrics["raw_mae"] = float("nan")
                metrics["reference_mad"] = float("nan")
                metrics["normalized_mae_estimable"] = False
            records.append(
                {
                    "dataset": pair.dataset,
                    "pair_type": pair.metadata.get("pair_type", ""),
                    "biological_pair_id": pair.metadata.get("biological_pair_id", pair.pair_id),
                    "direction": pair.metadata.get("direction", "synthetic"),
                    "pair_id": pair.pair_id,
                    "method": audit.method,
                    "intervention": intervention,
                    "proxy": name,
                    **metrics,
                }
            )
    return records


def exact_combined_risk(audit: AuditResult) -> np.ndarray:
    return np.maximum(audit.exact_response["I_EXPR"], audit.exact_response["I_SPATIAL"])


def semisynthetic_losses(pair: PairedData, audit: AuditResult) -> dict[str, np.ndarray]:
    conditional, _ = conditional_plan(audit.base.plan)
    predicted = np.argmax(conditional, axis=1)
    top1_error = np.zeros(len(predicted), dtype=float)
    for index, target in enumerate(predicted):
        if pair.truth_missing[index]:
            top1_error[index] = 1.0
        else:
            top1_error[index] = float(int(target) not in pair.equivalent_targets[index])
    endpoint = barycentric_target(audit.base.plan, pair.target_xy)
    endpoint_error = np.full(len(predicted), np.nan, dtype=float)
    valid = ~pair.truth_missing
    endpoint_error[valid] = np.linalg.norm(
        endpoint[valid] - pair.target_xy[pair.truth_target[valid]], axis=1
    )
    return {"top1_error": top1_error, "endpoint_error": endpoint_error}


def real_losses(
    pair: PairedData,
    audit: AuditResult,
    source_heldout: np.ndarray,
    target_heldout: np.ndarray,
) -> dict[str, np.ndarray]:
    conditional, _ = conditional_plan(audit.base.plan)
    predicted = np.argmax(conditional, axis=1)
    label_error = (pair.source_labels != pair.target_labels[predicted]).astype(float)
    transported = conditional @ target_heldout
    transported /= np.maximum(np.linalg.norm(transported, axis=1, keepdims=True), 1e-12)
    source = source_heldout / np.maximum(np.linalg.norm(source_heldout, axis=1, keepdims=True), 1e-12)
    heldout_loss = 1.0 - np.sum(source * transported, axis=1)
    return {"label_error": label_error, "heldout_loss": heldout_loss}


def risk_scores(pair: PairedData, audit: AuditResult) -> dict[str, np.ndarray]:
    scores = dict(audit.proxies)
    conditional, _ = conditional_plan(audit.base.plan)
    predicted = np.argmax(conditional, axis=1)
    scores["exact_combined"] = exact_combined_risk(audit)
    scores["exact_I_EXPR"] = audit.exact_response["I_EXPR"]
    scores["exact_I_SPATIAL"] = audit.exact_response["I_SPATIAL"]
    scores["endpoint_combined"] = np.maximum(
        audit.endpoint_response["I_EXPR"], audit.endpoint_response["I_SPATIAL"]
    )
    scores["finite_difference_I_EXPR_h001"] = audit.endpoint_response["I_EXPR"]
    scores["finite_difference_I_SPATIAL_h001"] = audit.endpoint_response["I_SPATIAL"]
    scores["source_boundary_proximity"] = boundary_proximity(pair.source_xy)
    scores["source_sparsity"] = local_sparsity(pair.source_xy)
    scores["matched_target_sparsity"] = local_sparsity(pair.target_xy)[predicted]
    return scores


def external_utility_records(
    pair: PairedData,
    audit: AuditResult,
    losses: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    scores = risk_scores(pair, audit)
    for witness, loss in losses.items():
        valid = np.isfinite(loss)
        for score_name, score in scores.items():
            row = {
                "dataset": pair.dataset,
                "pair_type": pair.metadata.get("pair_type", ""),
                "biological_pair_id": pair.metadata.get("biological_pair_id", pair.pair_id),
                "direction": pair.metadata.get("direction", "synthetic"),
                "pair_id": pair.pair_id,
                "method": audit.method,
                "witness": witness,
                "score": score_name,
                "n": int(valid.sum()),
                **normalized_excess_aurc(loss[valid], score[valid]),
            }
            unique_loss = np.unique(loss[valid])
            if np.array_equal(unique_loss, np.asarray([0.0, 1.0])):
                row["auroc"] = float(roc_auc_score(loss[valid], score[valid]))
                row["auprc"] = float(average_precision_score(loss[valid], score[valid]))
            else:
                row["auroc"] = float("nan")
                row["auprc"] = float("nan")
            records.append(row)
    return records


def missing_records(pair: PairedData, audit: AuditResult) -> list[dict[str, Any]]:
    if len(np.unique(pair.truth_missing)) < 2:
        return []
    records = []
    for name, score in risk_scores(pair, audit).items():
        records.append(
            {
                "dataset": pair.dataset,
                "pair_id": pair.pair_id,
                "method": audit.method,
                "score": name,
                "missing_fraction": float(pair.truth_missing.mean()),
                "auroc": float(roc_auc_score(pair.truth_missing, score)),
                "auprc": float(average_precision_score(pair.truth_missing, score)),
            }
        )
    return records


def rank_calibrated_mae(reference: np.ndarray, score: np.ndarray) -> float:
    """Exploratory scale-free amplitude diagnostic; not the registered NMAE."""
    return normalized_mae(rank_uniform(reference), rank_uniform(score))
