from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from validot.benchmark import run_audit
from validot.evaluation import external_utility_records, internal_fidelity_records, real_losses
from validot.io import load_pair
from validot.semisynthetic import PairedData
from validot.utils import read_json, status_payload, write_json


ROOT = Path(__file__).resolve().parents[1]
CONFIG = read_json(ROOT / "00_protocol" / "frozen_config.json")
OUTPUT = ROOT / "15_v1_3_correction" / "04_p1_sensitivity" / "coordinate_frame"
PAIR_ROOT = ROOT / "03_data_processed" / "external_pairs"
PAIR_IDS = ["STAR_8M_D1_D2", "ST_E15_5_S1_S2"]
METHODS = (
    CONFIG["v1_3_correction"]["confirmatory_ot_methods"]
    + CONFIG["v1_3_correction"]["non_ot_stress_test"]
)
VARIANTS = [
    "baseline",
    "rotate90",
    "reflect_x",
    "canonical_baseline",
    "canonical_rotate90",
    "canonical_reflect_x",
]
N = 800


def pca_chamfer_align(source_xy: np.ndarray, target_xy: np.ndarray) -> np.ndarray:
    """Label-free orthogonal alignment selected by symmetric Chamfer distance."""
    source = source_xy - np.mean(source_xy, axis=0)
    target = target_xy - np.mean(target_xy, axis=0)
    source_values, source_vectors = np.linalg.eigh(np.cov(source.T))
    target_values, target_vectors = np.linalg.eigh(np.cov(target.T))
    source_vectors = source_vectors[:, np.argsort(source_values)[::-1]]
    target_vectors = target_vectors[:, np.argsort(target_values)[::-1]]
    permutations = [np.eye(2), np.array([[0.0, 1.0], [1.0, 0.0]])]
    best = None
    best_score = float("inf")
    for permutation in permutations:
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                transform = (
                    target_vectors
                    @ permutation
                    @ np.diag([sx, sy])
                    @ source_vectors.T
                )
                candidate = target @ transform
                forward = cKDTree(source).query(candidate, k=1)[0]
                reverse = cKDTree(candidate).query(source, k=1)[0]
                score = float(np.mean(forward) + np.mean(reverse))
                if score < best_score:
                    best_score = score
                    best = candidate
    if best is None:
        raise RuntimeError("orthogonal alignment failed")
    return best


def transformed_target(source_xy: np.ndarray, target_xy: np.ndarray, variant: str) -> np.ndarray:
    rotate = np.array([[0.0, -1.0], [1.0, 0.0]])
    reflect = np.array([[-1.0, 0.0], [0.0, 1.0]])
    if "rotate90" in variant:
        target = target_xy @ rotate.T
    elif "reflect_x" in variant:
        target = target_xy @ reflect.T
    else:
        target = target_xy.copy()
    if variant.startswith("canonical_"):
        target = pca_chamfer_align(source_xy, target)
    return target


def subset_pair(pair: PairedData, extras: dict[str, np.ndarray], variant: str) -> tuple[PairedData, dict[str, np.ndarray]]:
    rng = np.random.default_rng(CONFIG["preprocessing"]["seed"])
    source_index = np.sort(rng.choice(len(pair.source_x), size=min(N, len(pair.source_x)), replace=False))
    target_index = np.sort(rng.choice(len(pair.target_x), size=min(N, len(pair.target_x)), replace=False))
    source_xy = pair.source_xy[source_index]
    target_xy = transformed_target(source_xy, pair.target_xy[target_index], variant)
    subset = PairedData(
        dataset=pair.dataset,
        pair_id=pair.pair_id,
        source_x=pair.source_x[source_index],
        target_x=pair.target_x[target_index],
        source_xy=source_xy,
        target_xy=target_xy,
        source_labels=pair.source_labels[source_index],
        target_labels=pair.target_labels[target_index],
        truth_target=np.full(len(source_index), -1, dtype=int),
        truth_missing=np.zeros(len(source_index), dtype=bool),
        equivalent_targets=[[] for _ in range(len(source_index))],
        metadata={**pair.metadata, "coordinate_variant": variant, "sensitivity_n": N},
    )
    subset_extras = {
        "source_heldout": extras["source_heldout"][source_index],
        "target_heldout": extras["target_heldout"][target_index],
        "source_library_size": extras["source_library_size"][source_index],
        "target_library_size": extras["target_library_size"][target_index],
        "source_region_size": extras["source_region_size"][source_index],
        "target_region_size": extras["target_region_size"][target_index],
    }
    return subset, subset_extras


def execute(pair_id: str, method: str, variant: str) -> dict[str, object]:
    task_dir = OUTPUT / "tasks" / pair_id / variant / method
    status_path = task_dir / "status.json"
    if status_path.exists() and read_json(status_path).get("status") == "COMPLETED":
        return read_json(status_path)
    task_dir.mkdir(parents=True, exist_ok=True)
    pair, extras = load_pair(PAIR_ROOT / f"{pair_id}.npz")
    pair, extras = subset_pair(pair, extras, variant)
    started = time.perf_counter()
    audit = run_audit(pair, method, CONFIG["solver"])
    full_losses = real_losses(pair, audit, extras["source_heldout"], extras["target_heldout"])
    shared_labels = np.intersect1d(np.unique(pair.source_labels), np.unique(pair.target_labels))
    shared = np.isin(pair.source_labels, shared_labels)
    losses = {
        "label_error_shared_closed_set": np.where(shared, full_losses["label_error"], np.nan),
        "heldout_loss": full_losses["heldout_loss"],
    }
    fidelity = pd.DataFrame(internal_fidelity_records(pair, audit))
    utility = pd.DataFrame(external_utility_records(pair, audit, losses))
    fidelity["coordinate_variant"] = variant
    utility["coordinate_variant"] = variant
    fidelity.to_csv(task_dir / "fidelity.tsv", sep="\t", index=False)
    utility.to_csv(task_dir / "utility.tsv", sep="\t", index=False)
    converged = bool(
        audit.base.converged
        and all(result.converged for result in audit.deleted.values())
        and all(result.converged for result in audit.endpoint.values())
    )
    status = status_payload(
        "V1_3_COORDINATE_TASK",
        "COMPLETED" if converged else "FAILED_NUMERIC",
        pair_id=pair_id,
        method=method,
        coordinate_variant=variant,
        n=N,
        seconds=time.perf_counter() - started,
    )
    write_json(status_path, status)
    return status


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    specs = [
        (pair_id, method, variant)
        for pair_id in PAIR_IDS
        for method in METHODS
        for variant in VARIANTS
    ]
    statuses = []
    with ProcessPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(execute, *spec): spec for spec in specs}
        for completed, future in enumerate(as_completed(futures), start=1):
            status = future.result()
            statuses.append(status)
            if status["status"] != "COMPLETED":
                raise RuntimeError(status)
            print(f"coordinate sensitivity {completed}/{len(specs)}", flush=True)
    utility = pd.concat(
        [pd.read_csv(path, sep="\t") for path in (OUTPUT / "tasks").glob("*/*/*/utility.tsv")],
        ignore_index=True,
    )
    fidelity = pd.concat(
        [pd.read_csv(path, sep="\t") for path in (OUTPUT / "tasks").glob("*/*/*/fidelity.tsv")],
        ignore_index=True,
    )
    utility.to_csv(OUTPUT / "coordinate_utility.tsv", sep="\t", index=False)
    fidelity.to_csv(OUTPUT / "coordinate_fidelity.tsv", sep="\t", index=False)
    baseline = utility[utility.coordinate_variant == "baseline"][
        ["pair_id", "method", "witness", "score", "normalized_excess_aurc"]
    ].rename(columns={"normalized_excess_aurc": "baseline_nex"})
    comparison = utility.merge(
        baseline,
        on=["pair_id", "method", "witness", "score"],
        how="left",
    )
    comparison["delta_nex_from_baseline"] = comparison.normalized_excess_aurc - comparison.baseline_nex
    comparison.to_csv(OUTPUT / "coordinate_comparison.tsv", sep="\t", index=False)
    summary = (
        comparison.groupby(["method", "coordinate_variant", "witness", "score"], dropna=False)
        .agg(
            pairs=("pair_id", "nunique"),
            median_delta_nex=("delta_nex_from_baseline", "median"),
            max_absolute_delta_nex=(
                "delta_nex_from_baseline", lambda x: float(np.nanmax(np.abs(x)))
            ),
        )
        .reset_index()
    )
    summary.to_csv(OUTPUT / "coordinate_summary.tsv", sep="\t", index=False)
    decision = status_payload(
        "V1_3_COORDINATE_SENSITIVITY",
        "COMPLETED",
        pairs=PAIR_IDS,
        methods=METHODS,
        variants=VARIANTS,
        tasks=len(statuses),
        n=N,
        canonicalization="label-free PCA frame with symmetric Chamfer selection",
        interpretation="coordinate-frame sensitivity, not independent biological validation",
    )
    write_json(OUTPUT / "COORDINATE_SENSITIVITY_DECISION.json", decision)
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
