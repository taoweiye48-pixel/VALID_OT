from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from validot.benchmark import run_audit
from validot.evaluation import external_utility_records, internal_fidelity_records, real_losses
from validot.external_data import prepare_pair_from_h5ad
from validot.io import load_pair, save_pair
from validot.semisynthetic import PairedData
from validot.utils import read_json, status_payload, write_json


ROOT = Path(__file__).resolve().parents[1]
CONFIG = read_json(ROOT / "00_protocol" / "frozen_config.json")
OUTPUT = ROOT / "15_v1_3_correction" / "04_p1_sensitivity" / "label_agnostic_sampling"
PAIR_OUTPUT = OUTPUT / "pairs"
TASK_OUTPUT = OUTPUT / "tasks"
PAIR_IDS = ["STAR_8M_D1_D2", "ST_E15_5_S1_S2", "ST_DEV_E13_5_E14_5"]
METHODS = (
    CONFIG["v1_3_correction"]["confirmatory_ot_methods"]
    + CONFIG["v1_3_correction"]["non_ot_stress_test"]
)


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


def prepare_pairs() -> list[Path]:
    PAIR_OUTPUT.mkdir(parents=True, exist_ok=True)
    selection = pd.read_csv(ROOT / "01_manifest" / "pair_selection.tsv", sep="\t")
    paths = []
    for row in selection[selection.pair_id.isin(PAIR_IDS)].to_dict(orient="records"):
        forward_path = PAIR_OUTPUT / f"{row['pair_id']}.npz"
        reverse_path = PAIR_OUTPUT / f"{row['pair_id']}__reverse.npz"
        if not forward_path.exists():
            pair, extras = prepare_pair_from_h5ad(
                Path(row["source_path"]),
                Path(row["target_path"]),
                row["source_label_column"],
                row["target_label_column"],
                row["dataset"],
                row["pair_id"],
                n_hvg=CONFIG["preprocessing"]["n_hvg"],
                heldout_n=100,
                max_n=CONFIG["preprocessing"]["main_n_max"],
                seed=CONFIG["preprocessing"]["seed"],
                sampling_mode="label_agnostic",
            )
            pair.metadata.update(
                pair_type=row["pair_type"],
                biological_pair_id=row["pair_id"],
                direction="forward",
            )
            save_pair(forward_path, pair, extras)
            reverse, reverse_extras = reverse_pair(pair, extras)
            save_pair(reverse_path, reverse, reverse_extras)
        paths.extend([forward_path, reverse_path])
    if len(paths) != 2 * len(PAIR_IDS):
        raise RuntimeError(f"expected {2 * len(PAIR_IDS)} direction files, found {len(paths)}")
    return paths


def execute(pair_path_string: str, method: str) -> dict[str, object]:
    pair_path = Path(pair_path_string)
    pair, extras = load_pair(pair_path)
    task_dir = TASK_OUTPUT / pair.pair_id / method
    status_path = task_dir / "status.json"
    if status_path.exists() and read_json(status_path).get("status") == "COMPLETED":
        return read_json(status_path)
    task_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    audit = run_audit(pair, method, CONFIG["solver"])
    full_losses = real_losses(pair, audit, extras["source_heldout"], extras["target_heldout"])
    shared_labels = np.intersect1d(np.unique(pair.source_labels), np.unique(pair.target_labels))
    shared_mask = np.isin(pair.source_labels, shared_labels)
    losses = {
        "label_error_shared_closed_set": np.where(
            shared_mask, full_losses["label_error"], np.nan
        ),
        "source_only_open_set": (~shared_mask).astype(float),
        "heldout_loss": full_losses["heldout_loss"],
    }
    fidelity = pd.DataFrame(internal_fidelity_records(pair, audit))
    utility = pd.DataFrame(external_utility_records(pair, audit, losses))
    utility["shared_label_coverage"] = float(np.mean(shared_mask))
    utility["source_only_fraction"] = float(np.mean(~shared_mask))
    fidelity.to_csv(task_dir / "fidelity.tsv", sep="\t", index=False)
    utility.to_csv(task_dir / "utility.tsv", sep="\t", index=False)
    converged = bool(
        audit.base.converged
        and all(result.converged for result in audit.deleted.values())
        and all(result.converged for result in audit.endpoint.values())
    )
    status = status_payload(
        "V1_3_LABEL_AGNOSTIC_TASK",
        "COMPLETED" if converged else "FAILED_NUMERIC",
        pair_id=pair.pair_id,
        method=method,
        seconds=time.perf_counter() - started,
        sampling_mode="label_agnostic",
        shared_label_coverage=float(np.mean(shared_mask)),
    )
    write_json(status_path, status)
    return status


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    TASK_OUTPUT.mkdir(parents=True, exist_ok=True)
    pair_paths = prepare_pairs()
    specs = [(str(path), method) for path in pair_paths for method in METHODS]
    statuses = []
    with ProcessPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(execute, *spec): spec for spec in specs}
        for completed, future in enumerate(as_completed(futures), start=1):
            status = future.result()
            statuses.append(status)
            if status["status"] != "COMPLETED":
                raise RuntimeError(status)
            print(f"label-agnostic sensitivity {completed}/{len(specs)}", flush=True)
    utility = pd.concat(
        [pd.read_csv(path, sep="\t") for path in TASK_OUTPUT.glob("*/*/utility.tsv")],
        ignore_index=True,
    )
    fidelity = pd.concat(
        [pd.read_csv(path, sep="\t") for path in TASK_OUTPUT.glob("*/*/fidelity.tsv")],
        ignore_index=True,
    )
    utility.to_csv(OUTPUT / "label_agnostic_utility.tsv", sep="\t", index=False)
    fidelity.to_csv(OUTPUT / "label_agnostic_fidelity.tsv", sep="\t", index=False)
    averaged = (
        utility.groupby(
            ["dataset", "biological_pair_id", "method", "witness", "score"],
            dropna=False,
        )
        .agg(
            label_agnostic_nex=("normalized_excess_aurc", "mean"),
            shared_label_coverage=("shared_label_coverage", "mean"),
            source_only_fraction=("source_only_fraction", "mean"),
            directions=("direction", "nunique"),
        )
        .reset_index()
    )
    primary = pd.read_csv(
        ROOT / "15_v1_3_correction" / "03_statistics" / "utility_direction_averaged.tsv",
        sep="\t",
    )
    primary = primary[
        (primary.source == "real") & primary.biological_pair_id.isin(PAIR_IDS)
    ][["biological_pair_id", "method", "witness", "score", "normalized_excess_aurc"]]
    primary = primary.rename(columns={"normalized_excess_aurc": "label_stratified_nex"})
    comparison = averaged.merge(
        primary,
        on=["biological_pair_id", "method", "witness", "score"],
        how="inner",
    )
    comparison["delta_nex_label_agnostic_minus_stratified"] = (
        comparison.label_agnostic_nex - comparison.label_stratified_nex
    )
    estimable = np.isfinite(comparison.label_agnostic_nex) & np.isfinite(
        comparison.label_stratified_nex
    )
    comparison["same_direction_relative_to_random"] = np.where(
        estimable,
        np.sign(1.0 - comparison.label_agnostic_nex)
        == np.sign(1.0 - comparison.label_stratified_nex),
        np.nan,
    )
    comparison.to_csv(OUTPUT / "sampling_mode_comparison.tsv", sep="\t", index=False)
    summary = (
        comparison.groupby(["method", "witness", "score"], dropna=False)
        .agg(
            pairs=("biological_pair_id", "nunique"),
            median_delta_nex=("delta_nex_label_agnostic_minus_stratified", "median"),
            max_absolute_delta_nex=(
                "delta_nex_label_agnostic_minus_stratified",
                lambda x: float(np.max(np.abs(x))),
            ),
            direction_stability=("same_direction_relative_to_random", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(OUTPUT / "sampling_mode_summary.tsv", sep="\t", index=False)
    decision = status_payload(
        "V1_3_LABEL_AGNOSTIC_SENSITIVITY",
        "COMPLETED",
        pairs=PAIR_IDS,
        directions=len(pair_paths),
        methods=METHODS,
        tasks=len(statuses),
        sampling_mode="label_agnostic",
        purpose="selection-bias sensitivity; not an independent biological replication",
    )
    write_json(OUTPUT / "LABEL_AGNOSTIC_DECISION.json", decision)
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
