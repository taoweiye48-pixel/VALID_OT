from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from validot.benchmark import _masses, _solve_common
from validot.evaluation import spatial_block_consistency
from validot.io import load_pair
from validot.metrics import conditional_plan, fidelity_metrics, normalized_excess_aurc
from validot.solvers import cost_components
from validot.utils import read_json, status_payload, write_json


ROOT = Path(__file__).resolve().parents[1]
CONFIG = read_json(ROOT / "00_protocol" / "frozen_config.json")
OUTPUT = ROOT / "15_v1_3_correction" / "04_p1_sensitivity" / "frozen_coupling_baseline"
PAIR_ROOT = ROOT / "03_data_processed" / "external_pairs"
RESPONSE_ROOT = ROOT / "10_E6_real_external"
PAIR_IDS = ["STAR_8M_D1_D2", "ST_E15_5_S1_S2", "ST_DEV_E13_5_E14_5"]
METHODS = CONFIG["v1_3_correction"]["confirmatory_ot_methods"]


def utility_row(
    pair,
    method: str,
    intervention: str,
    witness: str,
    loss: np.ndarray,
    score: np.ndarray,
) -> dict[str, object]:
    valid = np.isfinite(loss) & np.isfinite(score)
    row = {
        "dataset": pair.dataset,
        "pair_type": pair.metadata.get("pair_type", ""),
        "biological_pair_id": pair.metadata.get("biological_pair_id", pair.pair_id),
        "direction": pair.metadata.get("direction", "forward"),
        "pair_id": pair.pair_id,
        "method": method,
        "intervention": intervention,
        "proxy": "frozen_coupling_group_attribution",
        "witness": witness,
        "n": int(np.sum(valid)),
        **normalized_excess_aurc(loss[valid], score[valid]),
    }
    if np.array_equal(np.unique(loss[valid]), np.asarray([0.0, 1.0])):
        row["auroc"] = float(roc_auc_score(loss[valid], score[valid]))
        row["auprc"] = float(average_precision_score(loss[valid], score[valid]))
    else:
        row["auroc"] = float("nan")
        row["auprc"] = float("nan")
    return row


def execute(pair_id: str, method: str) -> dict[str, object]:
    task_dir = OUTPUT / "tasks" / pair_id / method
    status_path = task_dir / "status.json"
    if status_path.exists() and read_json(status_path).get("status") == "COMPLETED":
        return read_json(status_path)
    task_dir.mkdir(parents=True, exist_ok=True)
    pair, _ = load_pair(PAIR_ROOT / f"{pair_id}.npz")
    components = cost_components(pair.source_x, pair.target_x, pair.source_xy, pair.target_xy)
    a, b = _masses(pair)
    started = time.perf_counter()
    base = _solve_common(method, components, a, b, CONFIG["solver"], 0.5, 0.5)
    conditional, _ = conditional_plan(base.plan)
    group_scores = {
        "I_EXPR": np.sum(conditional * (0.5 * components["expression"]), axis=1),
        "I_SPATIAL": np.sum(conditional * (0.5 * components["spatial_cross"]), axis=1),
    }
    with np.load(RESPONSE_ROOT / pair_id / method / "row_responses.npz") as response:
        exact = {
            "I_EXPR": response["exact_I_EXPR"],
            "I_SPATIAL": response["exact_I_SPATIAL"],
        }
        source_labels = response["source_labels"].astype(str)
        target_labels = pair.target_labels.astype(str)
        shared_labels = np.intersect1d(np.unique(source_labels), np.unique(target_labels))
        shared = np.isin(source_labels, shared_labels)
        losses = {
            "label_error_shared_closed_set": np.where(
                shared, response["label_error"], np.nan
            ),
            "source_only_open_set": (~shared).astype(float),
            "heldout_loss": response["heldout_loss"],
        }
        fidelity_rows = []
        utility_rows = []
        for intervention, score in group_scores.items():
            metrics = fidelity_metrics(exact[intervention], score)
            metrics.update(spatial_block_consistency(exact[intervention], score, pair.source_xy))
            fidelity_rows.append(
                {
                    "dataset": pair.dataset,
                    "pair_type": pair.metadata.get("pair_type", ""),
                    "biological_pair_id": pair.metadata.get("biological_pair_id", pair.pair_id),
                    "direction": pair.metadata.get("direction", "forward"),
                    "pair_id": pair.pair_id,
                    "method": method,
                    "intervention": intervention,
                    "proxy": "frozen_coupling_group_attribution",
                    "baseline_scope": "WaX-inspired fixed-coupling group cost; not exact WaX reproduction",
                    **metrics,
                }
            )
            for witness, loss in losses.items():
                if np.isfinite(loss).sum() >= 2 and np.unique(loss[np.isfinite(loss)]).size >= 2:
                    utility_rows.append(
                        utility_row(pair, method, intervention, witness, loss, score)
                    )
    pd.DataFrame(fidelity_rows).to_csv(task_dir / "fidelity.tsv", sep="\t", index=False)
    pd.DataFrame(utility_rows).to_csv(task_dir / "utility.tsv", sep="\t", index=False)
    status = status_payload(
        "V1_3_FROZEN_COUPLING_BASELINE_TASK",
        "COMPLETED" if base.converged else "FAILED_NUMERIC",
        pair_id=pair_id,
        method=method,
        base_solver_seconds=base.seconds,
        wall_seconds=time.perf_counter() - started,
        baseline_scope="WaX-inspired fixed-coupling group cost; not exact WaX reproduction",
    )
    write_json(status_path, status)
    return status


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    pair_directions = [
        direction
        for pair_id in PAIR_IDS
        for direction in (pair_id, f"{pair_id}__reverse")
    ]
    specs = [(pair_id, method) for pair_id in pair_directions for method in METHODS]
    statuses = []
    with ProcessPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(execute, *spec): spec for spec in specs}
        for completed, future in enumerate(as_completed(futures), start=1):
            status = future.result()
            statuses.append(status)
            if status["status"] != "COMPLETED":
                raise RuntimeError(status)
            print(f"frozen-coupling baseline {completed}/{len(specs)}", flush=True)
    fidelity = pd.concat(
        [pd.read_csv(path, sep="\t") for path in (OUTPUT / "tasks").glob("*/*/fidelity.tsv")],
        ignore_index=True,
    )
    utility = pd.concat(
        [pd.read_csv(path, sep="\t") for path in (OUTPUT / "tasks").glob("*/*/utility.tsv")],
        ignore_index=True,
    )
    fidelity.to_csv(OUTPUT / "frozen_coupling_fidelity.tsv", sep="\t", index=False)
    utility.to_csv(OUTPUT / "frozen_coupling_utility.tsv", sep="\t", index=False)
    fidelity_summary = (
        fidelity.groupby(["method", "intervention"], dropna=False)
        .agg(
            pairs=("biological_pair_id", "nunique"),
            median_spearman=("spearman", "median"),
            median_top_decile_precision=("top_decile_precision", "median"),
        )
        .reset_index()
    )
    utility_summary = (
        utility.groupby(["method", "intervention", "witness"], dropna=False)
        .agg(
            pairs=("biological_pair_id", "nunique"),
            median_nex=("normalized_excess_aurc", "median"),
        )
        .reset_index()
    )
    fidelity_summary.to_csv(OUTPUT / "frozen_coupling_fidelity_summary.tsv", sep="\t", index=False)
    utility_summary.to_csv(OUTPUT / "frozen_coupling_utility_summary.tsv", sep="\t", index=False)
    decision = status_payload(
        "V1_3_FROZEN_COUPLING_BASELINE",
        "COMPLETED",
        biological_pairs=PAIR_IDS,
        directions=len(pair_directions),
        methods=METHODS,
        tasks=len(statuses),
        baseline_scope="WaX-inspired fixed-coupling group cost; not exact WaX reproduction",
    )
    write_json(OUTPUT / "FROZEN_COUPLING_BASELINE_DECISION.json", decision)
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
