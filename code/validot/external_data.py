from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
from scipy import sparse

from .semisynthetic import PairedData, deterministic_stratified_indices


@dataclass
class PreparedSlice:
    dataset: str
    slice_id: str
    x: np.ndarray
    heldout_x: np.ndarray
    xy: np.ndarray
    labels: np.ndarray
    unit_ids: np.ndarray
    cost_genes: np.ndarray
    heldout_genes: np.ndarray
    source_path: str


def _matrix(data: ad.AnnData):
    if "counts" in data.layers:
        return data.layers["counts"]
    return data.X


def _normalize(matrix):
    if sparse.issparse(matrix):
        values = matrix.tocsr().astype(np.float64)
        minimum = float(values.data.min()) if values.nnz else 0.0
        maximum = float(values.data.max()) if values.nnz else 0.0
        integer_like = bool(values.nnz == 0 or np.mean(np.isclose(values.data, np.round(values.data))) > 0.95)
        if minimum >= 0 and (maximum > 20 or integer_like):
            library = np.asarray(values.sum(axis=1)).ravel()
            values = sparse.diags(1e4 / np.maximum(library, 1.0)) @ values
            values.data = np.log1p(values.data)
        return values.tocsr()
    values = np.asarray(matrix, dtype=np.float64)
    minimum = float(np.nanmin(values))
    maximum = float(np.nanmax(values))
    integer_like = bool(np.mean(np.isclose(values, np.round(values))) > 0.95)
    if minimum >= 0 and (maximum > 20 or integer_like):
        library = values.sum(axis=1)
        values = np.log1p(values * (1e4 / np.maximum(library, 1.0))[:, None])
    return values


def _row_sums(matrix) -> np.ndarray:
    if sparse.issparse(matrix):
        return np.asarray(matrix.sum(axis=1)).ravel().astype(np.float64)
    return np.asarray(matrix, dtype=np.float64).sum(axis=1)


def _feature_moments(matrix) -> tuple[np.ndarray, np.ndarray]:
    if sparse.issparse(matrix):
        mean = np.asarray(matrix.mean(axis=0)).ravel()
        second = np.asarray(matrix.power(2).mean(axis=0)).ravel()
    else:
        mean = np.mean(matrix, axis=0)
        second = np.mean(matrix**2, axis=0)
    return mean, np.maximum(second - mean**2, 0.0)


def _dense_columns(matrix, indices: np.ndarray) -> np.ndarray:
    selected = matrix[:, indices]
    return selected.toarray() if sparse.issparse(selected) else np.asarray(selected)


def _row_normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def _standardize_xy(xy: np.ndarray) -> np.ndarray:
    xy = np.asarray(xy, dtype=np.float64)
    centered = xy - xy.mean(axis=0)
    scale = np.sqrt(np.mean(np.sum(centered**2, axis=1)))
    return centered / max(float(scale), 1e-12)


def _spatial_key(data: ad.AnnData) -> str:
    priorities = ["spatial", "X_spatial", "coordinates", "coord", "position"]
    for key in priorities:
        if key in data.obsm and data.obsm[key].shape[1] >= 2:
            return key
    for key in data.obsm.keys():
        if data.obsm[key].ndim == 2 and data.obsm[key].shape[1] in {2, 3}:
            return str(key)
    raise KeyError(f"no spatial obsm key in {list(data.obsm.keys())}")


def prepare_pair_from_h5ad(
    source_path: Path,
    target_path: Path,
    source_label_column: str,
    target_label_column: str,
    dataset: str,
    pair_id: str,
    n_hvg: int = 500,
    heldout_n: int = 100,
    max_n: int = 1500,
    seed: int = 20260716,
    sampling_mode: str = "label_stratified",
) -> tuple[PairedData, dict[str, Any]]:
    source = ad.read_h5ad(source_path)
    target = ad.read_h5ad(target_path)
    common = source.var_names.intersection(target.var_names)
    if len(common) < n_hvg + heldout_n:
        raise RuntimeError(f"{pair_id}: only {len(common)} common features")
    source = source[:, common]
    target = target[:, common]
    source_labels_all = source.obs[source_label_column].astype(str).to_numpy()
    target_labels_all = target.obs[target_label_column].astype(str).to_numpy()
    source_valid = ~np.isin(source_labels_all, ["nan", "None", "NA", "Unknown", "UNKNOWN"])
    target_valid = ~np.isin(target_labels_all, ["nan", "None", "NA", "Unknown", "UNKNOWN"])
    source_index_pool = np.flatnonzero(source_valid)
    target_index_pool = np.flatnonzero(target_valid)
    if sampling_mode == "label_stratified":
        source_local = deterministic_stratified_indices(
            source_labels_all[source_index_pool], min(max_n, len(source_index_pool)), seed
        )
        target_local = deterministic_stratified_indices(
            target_labels_all[target_index_pool], min(max_n, len(target_index_pool)), seed + 1
        )
    elif sampling_mode == "label_agnostic":
        source_rng = np.random.default_rng(seed)
        target_rng = np.random.default_rng(seed + 1)
        source_local = np.sort(
            source_rng.choice(
                len(source_index_pool), size=min(max_n, len(source_index_pool)), replace=False
            )
        )
        target_local = np.sort(
            target_rng.choice(
                len(target_index_pool), size=min(max_n, len(target_index_pool)), replace=False
            )
        )
    else:
        raise ValueError(f"unknown sampling_mode: {sampling_mode}")
    source_indices = source_index_pool[source_local]
    target_indices = target_index_pool[target_local]

    source_raw = _matrix(source)[source_indices]
    target_raw = _matrix(target)[target_indices]
    source_library_size = _row_sums(source_raw)
    target_library_size = _row_sums(target_raw)
    source_matrix = _normalize(source_raw)
    target_matrix = _normalize(target_raw)
    source_mean, source_variance = _feature_moments(source_matrix)
    target_mean, target_variance = _feature_moments(target_matrix)
    pooled_variance = 0.5 * (source_variance + target_variance) + 0.25 * (source_mean - target_mean) ** 2
    feature_order = np.argsort(pooled_variance)[::-1]
    cost_indices = feature_order[:n_hvg]
    heldout_indices = feature_order[n_hvg : n_hvg + heldout_n]
    source_x = _row_normalize(_dense_columns(source_matrix, cost_indices))
    target_x = _row_normalize(_dense_columns(target_matrix, cost_indices))
    source_heldout = _row_normalize(_dense_columns(source_matrix, heldout_indices))
    target_heldout = _row_normalize(_dense_columns(target_matrix, heldout_indices))
    source_xy = _standardize_xy(np.asarray(source.obsm[_spatial_key(source)])[source_indices, :2])
    target_xy = _standardize_xy(np.asarray(target.obsm[_spatial_key(target)])[target_indices, :2])
    source_labels = source_labels_all[source_indices]
    target_labels = target_labels_all[target_indices]
    source_counts = dict(zip(*np.unique(source_labels, return_counts=True)))
    target_counts = dict(zip(*np.unique(target_labels, return_counts=True)))
    pair = PairedData(
        dataset=dataset,
        pair_id=pair_id,
        source_x=source_x,
        target_x=target_x,
        source_xy=source_xy,
        target_xy=target_xy,
        source_labels=source_labels,
        target_labels=target_labels,
        truth_target=np.full(len(source_x), -1, dtype=int),
        truth_missing=np.zeros(len(source_x), dtype=bool),
        equivalent_targets=[[] for _ in range(len(source_x))],
        metadata={
            "source_path": str(source_path),
            "target_path": str(target_path),
            "source_label_column": source_label_column,
            "target_label_column": target_label_column,
            "n_source": len(source_x),
            "n_target": len(target_x),
            "common_features": len(common),
            "cost_features": n_hvg,
            "heldout_features": heldout_n,
            "sampling_mode": sampling_mode,
        },
    )
    extras = {
        "source_heldout": source_heldout,
        "target_heldout": target_heldout,
        "source_library_size": source_library_size,
        "target_library_size": target_library_size,
        "source_region_size": np.asarray([source_counts[label] for label in source_labels], dtype=float),
        "target_region_size": np.asarray([target_counts[label] for label in target_labels], dtype=float),
        "cost_genes": np.asarray(common)[cost_indices],
        "heldout_genes": np.asarray(common)[heldout_indices],
        "source_ids": source.obs_names.to_numpy()[source_indices],
        "target_ids": target.obs_names.to_numpy()[target_indices],
    }
    return pair, extras
