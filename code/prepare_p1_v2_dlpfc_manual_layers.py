"""Prepare the spatialLIBD manual cortical-layer extension for P1 v2.

The 12 sections form six adjacent-section pairs but only three independent
donor units.  Manual L1--L6/WM labels are used as label-level ground truth;
no spot-to-spot correspondence truth is claimed.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rdata
from scipy import sparse

from prepare_p1_v2_spatialdlpfc import (
    as_array,
    feature_moments,
    normalize_counts,
    reverse_pair,
    row_normalize,
    row_sums,
    standardize_xy,
)
from validot.io import save_pair
from validot.semisynthetic import PairedData


ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = Path(
    os.environ.get("VALIDOT_MANUAL_LAYER_RAW_ROOT", str(ROOT / "data" / "raw" / "p1_v2_dlpfc_manual_layers"))
)
RAW_PATH = RAW_ROOT / "Human_DLPFC_Visium_processedData_sce_scran_spatialLIBD.Rdata"
COMPACT_PATH = RAW_ROOT / "spatialLIBD_manual_layers_p1_v2_compact.rds"
OUTPUT_ROOT = Path(
    os.environ.get(
        "VALIDOT_MANUAL_LAYER_PAIRS_ROOT",
        str(ROOT / "data" / "processed" / "p1_v2_manual_layer_pairs"),
    )
)
REPORT_ROOT = ROOT / "analysis" / "p1_scale_regularization_v2_manual_layers" / "data_preparation"

PAIR_SPECS = (
    ("Br5292", "151507", "151508"),
    ("Br5292", "151509", "151510"),
    ("Br5595", "151669", "151670"),
    ("Br5595", "151671", "151672"),
    ("Br8100", "151673", "151674"),
    ("Br8100", "151675", "151676"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_pair(
    counts: sparse.csc_matrix,
    genes: np.ndarray,
    barcodes: np.ndarray,
    coordinates: np.ndarray,
    subjects: np.ndarray,
    sample_ids: np.ndarray,
    labels: np.ndarray,
    donor: str,
    source_sample: str,
    target_sample: str,
) -> tuple[PairedData, dict[str, np.ndarray], dict[str, Any]]:
    source_index = np.flatnonzero(sample_ids == source_sample)
    target_index = np.flatnonzero(sample_ids == target_sample)
    if len(source_index) != 1500 or len(target_index) != 1500:
        raise RuntimeError(
            f"{source_sample}/{target_sample}: expected 1500 x 1500 spots, "
            f"found {len(source_index)} x {len(target_index)}"
        )
    if set(np.unique(subjects[np.r_[source_index, target_index]])) != {donor}:
        raise RuntimeError(f"{source_sample}/{target_sample}: donor mapping mismatch")

    source_raw = counts[:, source_index].T.tocsr()
    target_raw = counts[:, target_index].T.tocsr()
    source_library = row_sums(source_raw)
    target_library = row_sums(target_raw)
    source_matrix = normalize_counts(source_raw)
    target_matrix = normalize_counts(target_raw)
    source_mean, source_variance = feature_moments(source_matrix)
    target_mean, target_variance = feature_moments(target_matrix)
    pooled_variance = 0.5 * (source_variance + target_variance) + 0.25 * (source_mean - target_mean) ** 2
    order = np.argsort(pooled_variance)[::-1]
    cost_index = order[:500]
    heldout_index = order[500:600]

    source_labels = labels[source_index].astype(str)
    target_labels = labels[target_index].astype(str)
    allowed = {"Layer1", "Layer2", "Layer3", "Layer4", "Layer5", "Layer6", "WM"}
    if not set(source_labels).issubset(allowed) or not set(target_labels).issubset(allowed):
        raise RuntimeError(f"{source_sample}/{target_sample}: invalid manual layer label")
    source_counts = dict(zip(*np.unique(source_labels, return_counts=True)))
    target_counts = dict(zip(*np.unique(target_labels, return_counts=True)))

    pair_id = f"DLPFCML_{source_sample}_{target_sample}"
    pair = PairedData(
        dataset="spatialLIBD_manual_layers",
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
            "source_path": str(RAW_PATH),
            "compact_extraction_path": str(COMPACT_PATH),
            "source_sample_id": source_sample,
            "target_sample_id": target_sample,
            "source_label_column": "layer_guess_reordered",
            "target_label_column": "layer_guess_reordered",
            "label_semantics": "manual L1-L6/WM spot annotation from spatialLIBD",
            "n_source": int(len(source_index)),
            "n_target": int(len(target_index)),
            "common_features": int(len(genes)),
            "cost_features": 500,
            "heldout_features": 100,
            "sampling_mode": "deterministic label-agnostic compact selection",
            "sampling_seed": 20260718,
            "pair_type": "within_donor_adjacent_replicates",
            "biological_pair_id": pair_id,
            "direction": "forward",
            "independent_unit_id": f"spatialLIBD_manual::{donor}",
            "cohort_role": "manual_layer_truth",
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
        "source_ids": np.asarray([f"{source_sample}::{barcodes[i]}" for i in source_index]),
        "target_ids": np.asarray([f"{target_sample}::{barcodes[i]}" for i in target_index]),
    }
    record = {
        "donor": donor,
        "independent_unit_id": f"spatialLIBD_manual::{donor}",
        "pair_id": pair_id,
        "source_sample_id": source_sample,
        "target_sample_id": target_sample,
        "source_spots_used": len(source_index),
        "target_spots_used": len(target_index),
        "source_manual_classes": len(np.unique(source_labels)),
        "target_manual_classes": len(np.unique(target_labels)),
        "shared_manual_classes": len(np.intersect1d(np.unique(source_labels), np.unique(target_labels))),
    }
    return pair, extras, record


def main() -> int:
    started = time.time()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        compact = rdata.read_rds(COMPACT_PATH)
    counts = sparse.csc_matrix(
        (
            np.asarray(compact["counts_x"]),
            np.asarray(compact["counts_i"]),
            np.asarray(compact["counts_p"]),
        ),
        shape=tuple(int(value) for value in compact["counts_dim"]),
    )
    genes = as_array(compact["genes"]).astype(str)
    barcodes = as_array(compact["barcodes"]).astype(str)
    coordinates = as_array(compact["coordinates"]).astype(float)
    subjects = as_array(compact["subjects"]).astype(str)
    sample_ids = as_array(compact["sample_ids"]).astype(str)
    labels = as_array(compact["labels"]).astype(str)

    records: list[dict[str, Any]] = []
    for donor, source_sample, target_sample in PAIR_SPECS:
        pair, extras, record = build_pair(
            counts, genes, barcodes, coordinates, subjects, sample_ids, labels,
            donor, source_sample, target_sample,
        )
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

    pd.DataFrame(records).to_csv(REPORT_ROOT / "manual_layer_pair_manifest.csv", index=False)
    files = sorted(OUTPUT_ROOT.glob("*.npz"))
    (REPORT_ROOT / "manual_layer_pairs.sha256").write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in files), encoding="utf-8"
    )
    provenance = {
        "source_path": str(RAW_PATH),
        "source_bytes": RAW_PATH.stat().st_size,
        "source_sha256": sha256(RAW_PATH),
        "compact_path": str(COMPACT_PATH),
        "compact_bytes": COMPACT_PATH.stat().st_size,
        "compact_sha256": sha256(COMPACT_PATH),
        "sections": 12,
        "biological_pairs": 6,
        "directional_pair_files": len(files),
        "independent_donors": 3,
        "manual_label_semantics": "manual L1-L6/WM spot annotation",
        "exact_correspondence_truth": False,
        "python": sys.version,
        "platform": platform.platform(),
        "elapsed_seconds": time.time() - started,
    }
    (REPORT_ROOT / "manual_layer_data_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(provenance, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
