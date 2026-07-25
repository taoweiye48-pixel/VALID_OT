"""Prepare the frozen P1 v2 expanded real-data pairs.

The script converts the official spatialDLPFC SpatialExperiment RDS directly
in Python, selects one predeclared anterior--middle pair per donor, and copies
the legacy frozen pair files into a new output root. It never edits the source
RDS or any v1.2/v1.3/P1-v1 artifact.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rdata
from scipy import sparse

from validot.io import load_pair, save_pair
from validot.semisynthetic import PairedData, deterministic_stratified_indices


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "p1_scale_regularization_v2.yaml"
RAW_ROOT = Path(
    os.environ.get(
        "VALIDOT_SPATIALDLPFC_ROOT",
        str(ROOT / "data" / "raw" / "p1_v2_spatialDLPFC"),
    )
)
ORIGINAL_RDS_PATH = RAW_ROOT / "spe_filtered_final_with_clusters_and_deconvolution_results.rds"
RDS_PATH = RAW_ROOT / "spatialDLPFC_p1_v2_compact.rds"
LEGACY_ROOT = Path(
    os.environ.get(
        "VALIDOT_LEGACY_PAIRS_ROOT",
        str(ROOT / "data" / "processed" / "external_pairs"),
    )
)
OUTPUT_ROOT = Path(
    os.environ.get(
        "VALIDOT_EXPANDED_PAIRS_ROOT",
        str(ROOT / "data" / "processed" / "p1_v2_expanded_pairs"),
    )
)
REPORT_ROOT = ROOT / "analysis" / "p1_scale_regularization_v2" / "data_preparation"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def as_array(value: Any) -> np.ndarray:
    """Convert rdata/xarray/pandas-backed columns without changing order."""
    if hasattr(value, "to_numpy"):
        return np.asarray(value.to_numpy())
    if hasattr(value, "values"):
        return np.asarray(value.values)
    return np.asarray(value)


def row_sums(matrix: sparse.spmatrix) -> np.ndarray:
    return np.asarray(matrix.sum(axis=1)).ravel().astype(np.float64)


def normalize_counts(matrix: sparse.spmatrix) -> sparse.csr_matrix:
    values = matrix.tocsr().astype(np.float64)
    library = row_sums(values)
    values = sparse.diags(1e4 / np.maximum(library, 1.0)) @ values
    values.data = np.log1p(values.data)
    return values.tocsr()


def feature_moments(matrix: sparse.spmatrix) -> tuple[np.ndarray, np.ndarray]:
    mean = np.asarray(matrix.mean(axis=0)).ravel()
    second = np.asarray(matrix.power(2).mean(axis=0)).ravel()
    return mean, np.maximum(second - mean**2, 0.0)


def row_normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def standardize_xy(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    centered = values - values.mean(axis=0)
    scale = np.sqrt(np.mean(np.sum(centered**2, axis=1)))
    return centered / max(float(scale), 1e-12)


def reverse_pair(pair: PairedData, extras: dict[str, np.ndarray]) -> tuple[PairedData, dict[str, np.ndarray]]:
    metadata = dict(pair.metadata)
    metadata["direction"] = "reverse"
    reverse = PairedData(
        dataset=pair.dataset,
        pair_id=f"{pair.pair_id}__reverse",
        source_x=pair.target_x,
        target_x=pair.source_x,
        source_xy=pair.target_xy,
        target_xy=pair.source_xy,
        source_labels=pair.target_labels,
        target_labels=pair.source_labels,
        truth_target=np.full(len(pair.target_x), -1, dtype=int),
        truth_missing=np.zeros(len(pair.target_x), dtype=bool),
        equivalent_targets=[[] for _ in range(len(pair.target_x))],
        metadata=metadata,
    )
    reverse_extras = {
        "source_heldout": extras["target_heldout"],
        "target_heldout": extras["source_heldout"],
        "source_library_size": extras["target_library_size"],
        "target_library_size": extras["source_library_size"],
        "source_region_size": extras["target_region_size"],
        "target_region_size": extras["source_region_size"],
        "source_ids": extras["target_ids"],
        "target_ids": extras["source_ids"],
        "cost_genes": extras["cost_genes"],
        "heldout_genes": extras["heldout_genes"],
    }
    return reverse, reverse_extras


def build_pair(
    counts_gene_by_spot: sparse.csc_matrix,
    genes: np.ndarray,
    barcodes: np.ndarray,
    coordinates: np.ndarray,
    subjects: np.ndarray,
    positions: np.ndarray,
    sample_ids: np.ndarray,
    labels: np.ndarray,
    donor: str,
    config: dict[str, Any],
) -> tuple[PairedData, dict[str, np.ndarray], dict[str, Any]]:
    invalid = {"", "nan", "none", "na", "unknown"}
    valid_label = np.asarray([str(label).strip().lower() not in invalid for label in labels])
    source_pool = np.flatnonzero((subjects == donor) & (positions == "anterior") & valid_label)
    target_pool = np.flatnonzero((subjects == donor) & (positions == "middle") & valid_label)
    if not len(source_pool) or not len(target_pool):
        raise RuntimeError(f"{donor}: fixed anterior/middle positions are not both available")
    source_samples = np.unique(sample_ids[source_pool])
    target_samples = np.unique(sample_ids[target_pool])
    if len(source_samples) != 1 or len(target_samples) != 1:
        raise RuntimeError(
            f"{donor}: expected one section per fixed position; got {source_samples.tolist()} and {target_samples.tolist()}"
        )
    seed = int(config["preprocessing"]["seed"])
    max_n = int(config["preprocessing"]["max_units"])
    source_local = deterministic_stratified_indices(labels[source_pool], min(max_n, len(source_pool)), seed)
    target_local = deterministic_stratified_indices(labels[target_pool], min(max_n, len(target_pool)), seed + 1)
    source_index = source_pool[source_local]
    target_index = target_pool[target_local]

    source_raw = counts_gene_by_spot[:, source_index].T.tocsr()
    target_raw = counts_gene_by_spot[:, target_index].T.tocsr()
    source_library = row_sums(source_raw)
    target_library = row_sums(target_raw)
    source_matrix = normalize_counts(source_raw)
    target_matrix = normalize_counts(target_raw)
    source_mean, source_variance = feature_moments(source_matrix)
    target_mean, target_variance = feature_moments(target_matrix)
    pooled_variance = (
        0.5 * (source_variance + target_variance)
        + 0.25 * (source_mean - target_mean) ** 2
    )
    order = np.argsort(pooled_variance)[::-1]
    n_hvg = int(config["preprocessing"]["n_hvg"])
    heldout_n = int(config["preprocessing"]["heldout_n"])
    if len(order) < n_hvg + heldout_n:
        raise RuntimeError(f"{donor}: insufficient features ({len(order)})")
    cost_index = order[:n_hvg]
    heldout_index = order[n_hvg : n_hvg + heldout_n]
    source_labels = labels[source_index].astype(str)
    target_labels = labels[target_index].astype(str)
    source_counts = dict(zip(*np.unique(source_labels, return_counts=True)))
    target_counts = dict(zip(*np.unique(target_labels, return_counts=True)))
    pair_id = f"SDLPFC_{donor}_ANT_MID"
    pair = PairedData(
        dataset="spatialDLPFC",
        pair_id=pair_id,
        source_x=row_normalize(source_matrix[:, cost_index].toarray()),
        target_x=row_normalize(target_matrix[:, cost_index].toarray()),
        source_xy=standardize_xy(coordinates[source_index, :2]),
        target_xy=standardize_xy(coordinates[target_index, :2]),
        source_labels=source_labels,
        target_labels=target_labels,
        truth_target=np.full(len(source_index), -1, dtype=int),
        truth_missing=np.zeros(len(source_index), dtype=bool),
        equivalent_targets=[[] for _ in range(len(source_index))],
        metadata={
            "source_path": str(ORIGINAL_RDS_PATH),
            "compact_extraction_path": str(RDS_PATH),
            "source_sample_id": str(source_samples[0]),
            "target_sample_id": str(target_samples[0]),
            "source_position": "anterior",
            "target_position": "middle",
            "source_label_column": config["preprocessing"]["label_column"],
            "target_label_column": config["preprocessing"]["label_column"],
            "n_source": int(len(source_index)),
            "n_target": int(len(target_index)),
            "common_features": int(len(genes)),
            "cost_features": n_hvg,
            "heldout_features": heldout_n,
            "sampling_mode": config["preprocessing"]["sampling"],
            "sampling_seed": seed,
            "pair_type": "within_donor_adjacent_position",
            "biological_pair_id": pair_id,
            "direction": "forward",
            "independent_unit_id": f"spatialDLPFC::{donor}",
            "cohort_role": "primary_expansion",
        },
    )
    extras = {
        "source_heldout": row_normalize(source_matrix[:, heldout_index].toarray()),
        "target_heldout": row_normalize(target_matrix[:, heldout_index].toarray()),
        "source_library_size": source_library,
        "target_library_size": target_library,
        "source_region_size": np.asarray([source_counts[label] for label in source_labels], dtype=float),
        "target_region_size": np.asarray([target_counts[label] for label in target_labels], dtype=float),
        "cost_genes": genes[cost_index].astype(str),
        "heldout_genes": genes[heldout_index].astype(str),
        "source_ids": np.asarray([f"{source_samples[0]}::{barcodes[i]}" for i in source_index]),
        "target_ids": np.asarray([f"{target_samples[0]}::{barcodes[i]}" for i in target_index]),
    }
    record = {
        "donor": donor,
        "independent_unit_id": f"spatialDLPFC::{donor}",
        "pair_id": pair_id,
        "source_sample_id": str(source_samples[0]),
        "target_sample_id": str(target_samples[0]),
        "source_spots_available": int(len(source_pool)),
        "target_spots_available": int(len(target_pool)),
        "source_spots_used": int(len(source_index)),
        "target_spots_used": int(len(target_index)),
        "source_label_count": int(len(np.unique(source_labels))),
        "target_label_count": int(len(np.unique(target_labels))),
        "shared_label_count": int(len(np.intersect1d(np.unique(source_labels), np.unique(target_labels)))),
        "label_semantics": config["preprocessing"]["label_semantics"],
    }
    return pair, extras, record


def main() -> int:
    started = time.time()
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    if not ORIGINAL_RDS_PATH.is_file() or ORIGINAL_RDS_PATH.stat().st_size < 1_000_000_000:
        raise FileNotFoundError(f"full spatialDLPFC RDS is missing or incomplete: {ORIGINAL_RDS_PATH}")
    if not RDS_PATH.is_file() or RDS_PATH.stat().st_size < 1_000_000:
        raise FileNotFoundError(f"native-R compact extraction is missing or incomplete: {RDS_PATH}")

    planned_donors = [
        pair["independent_unit_id"].split("::", 1)[1]
        for pair in config["pairs"]
        if pair["dataset"] == "spatialDLPFC"
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        compact = rdata.read_rds(RDS_PATH)
    subjects = as_array(compact["subjects"]).astype(str)
    positions = as_array(compact["positions"]).astype(str)
    sample_ids = as_array(compact["sample_ids"]).astype(str)
    labels = as_array(compact["labels"]).astype(str)
    barcodes = as_array(compact["barcodes"]).astype(str)
    coordinates = as_array(compact["coordinates"]).astype(float)
    counts = sparse.csc_matrix(
        (
            np.asarray(compact["counts_x"]),
            np.asarray(compact["counts_i"]),
            np.asarray(compact["counts_p"]),
        ),
        shape=tuple(int(value) for value in compact["counts_dim"]),
    )
    genes = np.asarray(compact["genes"]).astype(str)
    observed = sorted(
        subject
        for subject in np.unique(subjects)
        if {"anterior", "middle"}.issubset(set(positions[subjects == subject]))
    )
    if sorted(planned_donors) != observed:
        raise RuntimeError(f"predeclared donors differ from eligible donors: planned={planned_donors}, observed={observed}")
    selection_table = pd.DataFrame(compact["selection_table"])

    records: list[dict[str, Any]] = []
    for donor in planned_donors:
        pair, extras, record = build_pair(
            counts,
            genes,
            barcodes,
            coordinates,
            subjects,
            positions,
            sample_ids,
            labels,
            donor,
            config,
        )
        donor_selection = selection_table[selection_table["donor"].astype(str) == donor]
        source_selection = donor_selection[donor_selection["position"].astype(str) == "anterior"]
        target_selection = donor_selection[donor_selection["position"].astype(str) == "middle"]
        available_column = "available_spots" if "available_spots" in selection_table else "available_labelled_spots"
        record["source_spots_available"] = int(source_selection.iloc[0][available_column])
        record["target_spots_available"] = int(target_selection.iloc[0][available_column])
        forward_path = OUTPUT_ROOT / f"{pair.pair_id}.npz"
        reverse, reverse_extras = reverse_pair(pair, extras)
        reverse_path = OUTPUT_ROOT / f"{reverse.pair_id}.npz"
        save_pair(forward_path, pair, extras)
        save_pair(reverse_path, reverse, reverse_extras)
        record.update(
            forward_path=str(forward_path),
            forward_sha256=sha256(forward_path),
            reverse_path=str(reverse_path),
            reverse_sha256=sha256(reverse_path),
        )
        records.append(record)
        print(f"prepared {pair.pair_id}: {len(pair.source_x)} x {len(pair.target_x)}", flush=True)

    for pair in config["pairs"]:
        if pair["cohort_role"] != "legacy_replication":
            continue
        for suffix in ("", "__reverse"):
            name = f"{pair['pair_id']}{suffix}.npz"
            source = LEGACY_ROOT / name
            target = OUTPUT_ROOT / name
            if not source.is_file():
                raise FileNotFoundError(source)
            shutil.copy2(source, target)
            loaded, _ = load_pair(target)
            records.append(
                {
                    "donor": "",
                    "independent_unit_id": pair["independent_unit_id"],
                    "pair_id": loaded.pair_id,
                    "source_sample_id": "legacy frozen pair",
                    "target_sample_id": "legacy frozen pair",
                    "source_spots_available": int(len(loaded.source_x)),
                    "target_spots_available": int(len(loaded.target_x)),
                    "source_spots_used": int(len(loaded.source_x)),
                    "target_spots_used": int(len(loaded.target_x)),
                    "source_label_count": int(len(np.unique(loaded.source_labels))),
                    "target_label_count": int(len(np.unique(loaded.target_labels))),
                    "shared_label_count": int(len(np.intersect1d(np.unique(loaded.source_labels), np.unique(loaded.target_labels)))),
                    "label_semantics": "legacy dataset label; see frozen pair metadata",
                    "forward_path": str(target),
                    "forward_sha256": sha256(target),
                    "reverse_path": "",
                    "reverse_sha256": "",
                }
            )

    table = pd.DataFrame(records)
    table.to_csv(REPORT_ROOT / "p1_v2_pair_preparation_manifest.csv", index=False)
    all_files = sorted(OUTPUT_ROOT.glob("*.npz"))
    (REPORT_ROOT / "p1_v2_processed_pairs.sha256").write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in all_files), encoding="utf-8"
    )
    provenance = {
        "analysis_version": config["analysis_version"],
        "source_url": config["preprocessing"]["new_data_url"],
        "source_path": str(ORIGINAL_RDS_PATH),
        "source_bytes": ORIGINAL_RDS_PATH.stat().st_size,
        "source_sha256": sha256(ORIGINAL_RDS_PATH),
        "compact_extraction_path": str(RDS_PATH),
        "compact_extraction_bytes": RDS_PATH.stat().st_size,
        "compact_extraction_sha256": sha256(RDS_PATH),
        "eligible_and_included_donors": observed,
        "primary_independent_units": len(observed),
        "planned_primary_independent_units": config["aggregation"]["real_cross_slice_independent_units"],
        "processed_pair_files": len(all_files),
        "python": sys.version,
        "platform": platform.platform(),
        "rdata_version": getattr(rdata, "__version__", "unknown"),
        "numpy_version": np.__version__,
        "scipy_version": __import__("scipy").__version__,
        "elapsed_seconds": time.time() - started,
        "selection_rule": "all ten donors; fixed anterior--middle; both directions; no outcome inspection",
    }
    write_json(REPORT_ROOT / "p1_v2_data_provenance.json", provenance)
    print(json.dumps(provenance, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
