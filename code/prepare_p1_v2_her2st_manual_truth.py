"""Prepare eight independent HER2ST pathologist-labelled control pairs."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

from validot.io import save_pair
from validot.semisynthetic import PairedData, generate_pair, validate_truth


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "p1_scale_regularization_v2.yaml"
RAW_ROOT = Path(
    os.environ.get(
        "VALIDOT_HER2ST_ROOT",
        str(ROOT / "data" / "raw" / "p1_v2_her2st" / "github_master"),
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


def json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def normalize_counts(matrix: sparse.spmatrix) -> sparse.csr_matrix:
    values = matrix.tocsr().astype(np.float64)
    library = np.asarray(values.sum(axis=1)).ravel()
    values = sparse.diags(1e4 / np.maximum(library, 1.0)) @ values
    values.data = np.log1p(values.data)
    return values.tocsr()


def row_normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def reverse_controlled_pair(
    pair: PairedData, extras: dict[str, np.ndarray]
) -> tuple[PairedData, dict[str, np.ndarray]]:
    kept_old = np.flatnonzero(~pair.truth_missing)
    if not np.array_equal(pair.truth_target[kept_old], np.arange(len(pair.target_x))):
        raise RuntimeError(f"{pair.pair_id}: unexpected controlled truth ordering")
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
        truth_target=kept_old.astype(int),
        truth_missing=np.zeros(len(pair.target_x), dtype=bool),
        equivalent_targets=[[int(index)] for index in kept_old],
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


def prepare_section(
    section: str,
    seed: int,
    config: dict[str, Any],
) -> tuple[PairedData, dict[str, np.ndarray], dict[str, Any]]:
    counts_path = RAW_ROOT / "counts" / f"{section}.tsv.gz"
    labels_path = RAW_ROOT / "labels" / f"{section}_labeled_coordinates.tsv"
    spots_path = RAW_ROOT / "spots" / f"{section}_selection.tsv"
    for path in (counts_path, labels_path, spots_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    counts = pd.read_csv(counts_path, sep="\t", index_col=0)
    labels = pd.read_csv(labels_path, sep="\t").dropna(subset=["x", "y", "label"]).copy()
    labels["spot_id"] = (
        np.rint(labels["x"]).astype(int).astype(str)
        + "x"
        + np.rint(labels["y"]).astype(int).astype(str)
    )
    labels = labels.drop_duplicates("spot_id").set_index("spot_id")
    common = counts.index.intersection(labels.index, sort=False)
    exclude = set(config["manual_truth_preprocessing"]["exclude_labels"])
    common = common[~labels.loc[common, "label"].astype(str).isin(exclude)]
    if len(common) < 100:
        raise RuntimeError(f"{section}: only {len(common)} determined pathologist-labelled spots")
    counts = counts.loc[common]
    labels = labels.loc[common]
    raw = sparse.csr_matrix(counts.to_numpy(dtype=np.float64))
    library = np.asarray(raw.sum(axis=1)).ravel()
    normalized = normalize_counts(raw)
    mean = np.asarray(normalized.mean(axis=0)).ravel()
    second = np.asarray(normalized.power(2).mean(axis=0)).ravel()
    variance = np.maximum(second - mean**2, 0.0)
    order = np.argsort(variance)[::-1]
    n_hvg = int(config["manual_truth_preprocessing"]["n_hvg"])
    heldout_n = int(config["manual_truth_preprocessing"]["heldout_n"])
    if len(order) < n_hvg + heldout_n:
        raise RuntimeError(f"{section}: insufficient genes")
    cost_index = order[:n_hvg]
    heldout_index = order[n_hvg : n_hvg + heldout_n]
    source_x = row_normalize(normalized[:, cost_index].toarray())
    source_heldout = row_normalize(normalized[:, heldout_index].toarray())
    source_xy = labels[["x", "y"]].to_numpy(dtype=float)
    source_labels = labels["label"].astype(str).to_numpy()

    pair = generate_pair(
        source_x,
        source_xy,
        source_labels,
        scenario="combined",
        seed=seed,
        n=len(source_x),
        dataset="HER2ST_manual_controlled",
    )
    pair_id = f"HER2ST_{section}_CONTROLLED"
    pair.pair_id = pair_id
    patient = section[0]
    pair.metadata.update(
        {
            "source_path": str(counts_path),
            "label_path": str(labels_path),
            "coordinate_path": str(spots_path),
            "source_section": section,
            "source_patient": patient,
            "target_status": "controlled target generated from the same measured section",
            "label_column": "pathologist region label",
            "label_semantics": "manual pathology ground truth",
            "pair_type": "within_section_controlled_combined",
            "biological_pair_id": pair_id,
            "direction": "forward",
            "independent_unit_id": f"HER2ST::{patient}",
            "cohort_role": "manual_truth_controlled",
            "cost_features": n_hvg,
            "heldout_features": heldout_n,
        }
    )
    kept_old = np.flatnonzero(~pair.truth_missing)
    source_counts = dict(zip(*np.unique(pair.source_labels, return_counts=True)))
    target_counts = dict(zip(*np.unique(pair.target_labels, return_counts=True)))
    source_ids = counts.index.to_numpy().astype(str)
    extras = {
        "source_heldout": source_heldout,
        "target_heldout": source_heldout[kept_old],
        "source_library_size": library,
        "target_library_size": library[kept_old],
        "source_region_size": np.asarray([source_counts[label] for label in pair.source_labels], dtype=float),
        "target_region_size": np.asarray([target_counts[label] for label in pair.target_labels], dtype=float),
        "cost_genes": counts.columns.to_numpy().astype(str)[cost_index],
        "heldout_genes": counts.columns.to_numpy().astype(str)[heldout_index],
        "source_ids": np.asarray([f"{section}::{spot}" for spot in source_ids]),
        "target_ids": np.asarray([f"{section}::{source_ids[index]}" for index in kept_old]),
    }
    truth = validate_truth(pair)
    record = {
        "section": section,
        "patient": patient,
        "independent_unit_id": f"HER2ST::{patient}",
        "pair_id": pair_id,
        "seed": seed,
        "count_spots": int(len(counts.index)),
        "determined_manual_spots": int(len(source_x)),
        "source_spots": int(len(pair.source_x)),
        "target_spots": int(len(pair.target_x)),
        "missing_source_spots": int(pair.truth_missing.sum()),
        "manual_label_classes": int(len(np.unique(pair.source_labels))),
        "truth_label_agreement": truth["truth_label_agreement"],
        "truth_valid": bool(truth["passed"]),
        "counts_sha256": sha256(counts_path),
        "labels_sha256": sha256(labels_path),
        "spots_sha256": sha256(spots_path),
    }
    return pair, extras, record


def main() -> int:
    started = time.time()
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    sections = config["manual_truth_preprocessing"]["sections"]
    seeds = config["manual_truth_preprocessing"]["seeds"]
    if len(sections) != 8 or len(seeds) != 8:
        raise RuntimeError("HER2ST manual-truth cohort must contain exactly eight frozen patient sections")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for section, seed in zip(sections, seeds):
        pair, extras, record = prepare_section(section, int(seed), config)
        reverse, reverse_extras = reverse_controlled_pair(pair, extras)
        forward_path = OUTPUT_ROOT / f"{pair.pair_id}.npz"
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
        print(
            f"prepared {pair.pair_id}: {len(pair.source_x)} x {len(pair.target_x)}, "
            f"manual classes={record['manual_label_classes']}",
            flush=True,
        )
    table = pd.DataFrame(records)
    table.to_csv(REPORT_ROOT / "p1_v2_her2st_manual_truth_manifest.csv", index=False)
    provenance = {
        "analysis_version": config["analysis_version"],
        "official_repository": config["manual_truth_preprocessing"]["official_repository"],
        "source_commit": config["manual_truth_preprocessing"]["source_commit"],
        "sections": sections,
        "independent_patients": int(table["patient"].nunique()),
        "manual_spots_total": int(table["source_spots"].sum()),
        "controlled_target_spots_total": int(table["target_spots"].sum()),
        "truth_failures": int((~table["truth_valid"]).sum()),
        "target_status": "real measured source; deterministic controlled target; not a measured cross-section",
        "python": sys.version,
        "platform": platform.platform(),
        "elapsed_seconds": time.time() - started,
    }
    json_dump(REPORT_ROOT / "p1_v2_her2st_data_provenance.json", provenance)
    print(json.dumps(provenance, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
