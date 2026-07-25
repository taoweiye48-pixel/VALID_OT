from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class PairedData:
    dataset: str
    pair_id: str
    source_x: np.ndarray
    target_x: np.ndarray
    source_xy: np.ndarray
    target_xy: np.ndarray
    source_labels: np.ndarray
    target_labels: np.ndarray
    truth_target: np.ndarray
    truth_missing: np.ndarray
    equivalent_targets: list[list[int]]
    metadata: dict[str, Any]


def _standardize_xy(xy: np.ndarray) -> np.ndarray:
    xy = np.asarray(xy, dtype=np.float64)
    centered = xy - xy.mean(axis=0)
    scale = np.sqrt(np.mean(np.sum(centered**2, axis=1)))
    return centered / max(float(scale), 1e-12)


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def deterministic_stratified_indices(labels: np.ndarray, n: int, seed: int) -> np.ndarray:
    labels = np.asarray(labels).astype(str)
    if n >= len(labels):
        return np.arange(len(labels))
    rng = np.random.default_rng(seed)
    unique, counts = np.unique(labels, return_counts=True)
    quotas = np.floor(n * counts / counts.sum()).astype(int)
    quotas = np.maximum(quotas, 1)
    while quotas.sum() > n:
        eligible = np.where(quotas > 1)[0]
        quotas[eligible[np.argmax(quotas[eligible])]] -= 1
    remainders = n - quotas.sum()
    if remainders > 0:
        fractions = n * counts / counts.sum() - np.floor(n * counts / counts.sum())
        for index in np.argsort(fractions)[::-1][:remainders]:
            quotas[index] += 1
    selected: list[int] = []
    for label, quota in zip(unique, quotas):
        candidates = np.flatnonzero(labels == label)
        selected.extend(rng.choice(candidates, size=min(quota, len(candidates)), replace=False).tolist())
    if len(selected) < n:
        remaining = np.setdiff1d(np.arange(len(labels)), np.asarray(selected), assume_unique=False)
        selected.extend(rng.choice(remaining, size=n - len(selected), replace=False).tolist())
    return np.asarray(sorted(selected), dtype=int)


def generate_pair(
    x: np.ndarray,
    xy: np.ndarray,
    labels: np.ndarray,
    scenario: str,
    seed: int,
    n: int = 800,
    crop_fraction: float = 0.25,
    dataset: str = "synthetic",
) -> PairedData:
    """Generate a paired spatial dataset with an explicit correspondence table."""
    x = _normalize_rows(x)
    xy = _standardize_xy(xy)
    labels = np.asarray(labels).astype(str)
    indices = deterministic_stratified_indices(labels, min(n, len(labels)), seed)
    source_x = x[indices].copy()
    source_xy = xy[indices].copy()
    source_labels = labels[indices].copy()
    target_x = source_x.copy()
    target_xy = source_xy.copy()
    target_labels = source_labels.copy()
    truth_target = np.arange(len(source_x), dtype=int)
    truth_missing = np.zeros(len(source_x), dtype=bool)
    equivalent_targets = [[int(i)] for i in truth_target]
    rng = np.random.default_rng(seed)
    metadata: dict[str, Any] = {"scenario": scenario, "seed": seed, "n_source": len(source_x)}

    if scenario in {"rigid", "combined"}:
        angle = rng.uniform(-0.45, 0.45)
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        scale = rng.uniform(0.90, 1.10)
        shift = rng.uniform(-0.20, 0.20, size=2)
        target_xy = scale * (target_xy @ rotation.T) + shift
        metadata.update({"angle_radians": float(angle), "scale": float(scale), "shift": shift.tolist()})

    if scenario in {"nonrigid", "combined"}:
        amplitude = 0.12
        original = target_xy.copy()
        displacement = amplitude * np.column_stack(
            [np.sin(np.pi * original[:, 1]), np.sin(np.pi * original[:, 0])]
        )
        target_xy += displacement
        metadata["nonrigid_amplitude"] = amplitude

    if scenario in {"batch_noise", "combined"}:
        feature_shift = rng.normal(0.0, 0.08, size=(1, target_x.shape[1]))
        target_x = target_x + feature_shift + rng.normal(0.0, 0.05, size=target_x.shape)
        dropout = rng.random(target_x.shape) < 0.08
        target_x[dropout] = 0.0
        target_x = _normalize_rows(target_x)
        metadata.update({"feature_noise_sd": 0.05, "dropout": 0.08})

    if scenario in {"crop_missing", "combined"}:
        fraction = 0.20 if scenario == "combined" else float(crop_fraction)
        direction = rng.normal(size=2)
        direction /= np.linalg.norm(direction)
        projection = target_xy @ direction
        remove_count = max(1, int(round(fraction * len(target_xy))))
        removed_old = np.argsort(projection)[-remove_count:]
        keep_mask = np.ones(len(target_xy), dtype=bool)
        keep_mask[removed_old] = False
        old_to_new = np.full(len(target_xy), -1, dtype=int)
        old_to_new[np.flatnonzero(keep_mask)] = np.arange(keep_mask.sum())
        truth_target = old_to_new[truth_target]
        truth_missing = truth_target < 0
        equivalent_targets = [[] if target < 0 else [int(target)] for target in truth_target]
        target_x = target_x[keep_mask]
        target_xy = target_xy[keep_mask]
        target_labels = target_labels[keep_mask]
        metadata.update({"crop_fraction": fraction, "removed_count": int(remove_count)})

    if scenario == "duplicate_motif":
        # Co-located duplicates make both expression and spatial evidence exactly
        # symmetric, preventing absolute coordinates from breaking the designed tie.
        duplicate_count = max(4, int(round(0.15 * len(target_x))))
        duplicate_old = np.sort(rng.choice(len(target_x), size=duplicate_count, replace=False))
        start = len(target_x)
        target_x = np.vstack([target_x, target_x[duplicate_old]])
        target_xy = np.vstack([target_xy, target_xy[duplicate_old]])
        target_labels = np.concatenate([target_labels, target_labels[duplicate_old]])
        for offset, old_index in enumerate(duplicate_old):
            source_index = int(old_index)
            equivalent_targets[source_index] = [source_index, start + offset]
        metadata.update({"duplicate_count": int(duplicate_count), "duplicate_old_indices": duplicate_old.tolist()})

    metadata.update({"n_target": int(len(target_x)), "missing_count": int(truth_missing.sum())})
    return PairedData(
        dataset=dataset,
        pair_id=f"{dataset}_{scenario}_seed{seed}",
        source_x=source_x,
        target_x=target_x,
        source_xy=source_xy,
        target_xy=target_xy,
        source_labels=source_labels,
        target_labels=target_labels,
        truth_target=truth_target,
        truth_missing=truth_missing,
        equivalent_targets=equivalent_targets,
        metadata=metadata,
    )


def validate_truth(pair: PairedData) -> dict[str, Any]:
    valid = ~pair.truth_missing
    in_range = bool(
        np.all(pair.truth_target[pair.truth_target >= 0] < len(pair.target_x))
        and np.all(pair.truth_target[pair.truth_target >= 0] >= 0)
    )
    label_agreement = float(
        np.mean(pair.source_labels[valid] == pair.target_labels[pair.truth_target[valid]])
    ) if np.any(valid) else 1.0
    equivalent_valid = all(
        all(0 <= target < len(pair.target_x) for target in targets)
        for targets in pair.equivalent_targets
    )
    return {
        "pair_id": pair.pair_id,
        "n_source": len(pair.source_x),
        "n_target": len(pair.target_x),
        "missing_fraction": float(pair.truth_missing.mean()),
        "truth_indices_in_range": in_range,
        "truth_label_agreement": label_agreement,
        "equivalent_targets_valid": equivalent_valid,
        "passed": bool(in_range and label_agreement == 1.0 and equivalent_valid),
    }
