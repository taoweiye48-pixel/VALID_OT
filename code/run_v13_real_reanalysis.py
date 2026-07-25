from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

from validot.controls import adjusted_association, permutation_controls, spatial_blocks
from validot.evaluation import spatial_block_consistency
from validot.io import load_pair
from validot.metrics import fidelity_metrics, mae_diagnostics, normalized_excess_aurc
from validot.utils import read_json, status_payload, write_json


ROOT = Path(__file__).resolve().parents[1]
CONFIG = read_json(ROOT / "00_protocol" / "frozen_config.json")
CORRECTION = CONFIG["v1_3_correction"]
INPUT = ROOT / "10_E6_real_external"
PAIR_ROOT = ROOT / "03_data_processed" / "external_pairs"
OUTPUT = ROOT / "15_v1_3_correction" / "02_real_reanalysis"
METHODS = CORRECTION["confirmatory_ot_methods"] + CORRECTION["non_ot_stress_test"]
QC_SCORES = {
    "source_boundary_proximity",
    "source_sparsity",
    "matched_target_sparsity",
}
CONTROL_SCORES = QC_SCORES | {
    "exact_combined",
    "endpoint_combined",
    "exact_I_EXPR",
    "exact_I_SPATIAL",
    "finite_difference_I_EXPR_h001",
    "finite_difference_I_SPATIAL_h001",
    "finite_difference_combined_h001",
}


def stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("::".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little")


def task_metadata(pair_id: str) -> tuple[object, dict[str, np.ndarray], dict[str, object]]:
    pair, extras = load_pair(PAIR_ROOT / f"{pair_id}.npz")
    return pair, extras, {
        "dataset": pair.dataset,
        "pair_type": pair.metadata.get("pair_type", ""),
        "biological_pair_id": pair.metadata.get("biological_pair_id", pair.pair_id),
        "direction": pair.metadata.get("direction", "forward"),
        "pair_id": pair.pair_id,
    }


def risk_arrays(row: np.lib.npyio.NpzFile) -> dict[str, np.ndarray]:
    risks = {key.removeprefix("risk__"): row[key] for key in row.files if key.startswith("risk__")}
    risks["exact_I_EXPR"] = row["exact_I_EXPR"]
    risks["exact_I_SPATIAL"] = row["exact_I_SPATIAL"]
    risks["finite_difference_I_EXPR_h001"] = row["endpoint_I_EXPR"]
    risks["finite_difference_I_SPATIAL_h001"] = row["endpoint_I_SPATIAL"]
    risks["finite_difference_combined_h001"] = np.maximum(
        row["endpoint_I_EXPR"], row["endpoint_I_SPATIAL"]
    )
    return risks


def witness_arrays(
    row: np.lib.npyio.NpzFile,
    target_labels: np.ndarray,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, float]]:
    source_labels = row["source_labels"].astype(str)
    shared_labels = np.intersect1d(np.unique(source_labels), np.unique(target_labels.astype(str)))
    shared = np.isin(source_labels, shared_labels)
    source_only = ~shared
    witnesses = {
        "label_error_shared_closed_set": (row["label_error"], shared),
        "source_only_open_set": (source_only.astype(float), np.ones(len(shared), dtype=bool)),
        "heldout_loss": (row["heldout_loss"], np.ones(len(shared), dtype=bool)),
    }
    coverage = {
        "shared_label_coverage": float(np.mean(shared)),
        "source_only_fraction": float(np.mean(source_only)),
        "shared_label_count": int(len(shared_labels)),
    }
    return witnesses, coverage


def utility_record(
    metadata: dict[str, object],
    method: str,
    witness: str,
    score_name: str,
    loss: np.ndarray,
    score: np.ndarray,
    mask: np.ndarray,
    coverage: dict[str, float],
) -> dict[str, object]:
    valid = mask & np.isfinite(loss) & np.isfinite(score)
    result = {
        **metadata,
        "method": method,
        "witness": witness,
        "score": score_name,
        "n": int(np.sum(valid)),
        "witness_coverage": float(np.mean(valid)),
        **coverage,
    }
    if np.sum(valid) < 2 or np.unique(loss[valid]).size < 2:
        return {
            **result,
            "aurc": float("nan"),
            "oracle_aurc": float("nan"),
            "random_aurc": float("nan"),
            "normalized_excess_aurc": float("nan"),
            "retained_loss_at_80pct_coverage": float("nan"),
            "retained_loss_at_90pct_coverage": float("nan"),
            "auroc": float("nan"),
            "auprc": float("nan"),
            "estimable": False,
        }
    result.update(normalized_excess_aurc(loss[valid], score[valid]))
    binary = np.array_equal(np.unique(loss[valid]), np.asarray([0.0, 1.0]))
    result["auroc"] = float(roc_auc_score(loss[valid], score[valid])) if binary else float("nan")
    result["auprc"] = (
        float(average_precision_score(loss[valid], score[valid])) if binary else float("nan")
    )
    result["estimable"] = bool(np.isfinite(result["normalized_excess_aurc"]))
    return result


def fidelity_records(
    metadata: dict[str, object],
    method: str,
    row: np.lib.npyio.NpzFile,
) -> list[dict[str, object]]:
    records = []
    static_scores = {
        key.removeprefix("risk__"): row[key]
        for key in row.files
        if key.startswith("risk__")
        and key.removeprefix("risk__")
        in {
            "assigned_raw_cost",
            "local_cost_margin",
            "conditional_entropy",
            "low_max_probability",
            "probability_margin",
            "mass_deficit",
        }
    }
    for intervention in ("I_EXPR", "I_SPATIAL"):
        reference = row[f"exact_{intervention}"]
        scores = dict(static_scores)
        scores["finite_difference_sensitivity_h001"] = row[f"endpoint_{intervention}"]
        for proxy, score in scores.items():
            metrics = fidelity_metrics(reference, score)
            metrics.update(spatial_block_consistency(reference, score, row["source_xy"]))
            if proxy == "finite_difference_sensitivity_h001":
                metrics.update(mae_diagnostics(reference, score))
            else:
                metrics.update(
                    raw_mae=float("nan"),
                    reference_mad=float("nan"),
                    normalized_mae=float("nan"),
                    normalized_mae_estimable=False,
                )
            records.append(
                {
                    **metadata,
                    "method": method,
                    "intervention": intervention,
                    "proxy": proxy,
                    "proxy_scope": "model_local_v1.2" if method not in {"paste_fgw", "paste2_partial_fgw"} else "invalid_for_fgw",
                    **metrics,
                }
            )
    return records


def control_record(
    metadata: dict[str, object],
    method: str,
    witness: str,
    score_name: str,
    loss: np.ndarray,
    score: np.ndarray,
    mask: np.ndarray,
    row: np.lib.npyio.NpzFile,
) -> dict[str, object]:
    valid = mask & np.isfinite(loss) & np.isfinite(score)
    confounds = {
        "boundary": row["risk__source_boundary_proximity"][valid],
        "sparsity": row["risk__source_sparsity"][valid],
        "matched_target_sparsity": row["risk__matched_target_sparsity"][valid],
        "log_library_size": np.log1p(row["source_library_size"][valid]),
        "log_region_size": np.log1p(row["source_region_size"][valid]),
    }
    if score_name == "source_boundary_proximity":
        confounds.pop("boundary")
    if score_name == "source_sparsity":
        confounds.pop("sparsity")
    if score_name == "matched_target_sparsity":
        confounds.pop("matched_target_sparsity")
    return {
        **metadata,
        "method": method,
        "witness": witness,
        "score": score_name,
        **permutation_controls(
            loss[valid],
            score[valid],
            row["source_xy"][valid],
            repeats=100,
            seed=stable_seed("v1.3", metadata["pair_id"], method, witness, score_name),
        ),
        **adjusted_association(
            loss[valid],
            score[valid],
            confounds,
            cluster_groups=spatial_blocks(row["source_xy"][valid]),
        ),
    }


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    utility_rows = []
    fidelity_rows = []
    control_rows = []
    coverage_rows = []
    task_count = 0
    pair_dirs = sorted(
        path for path in INPUT.iterdir() if path.is_dir() and path.name != "optional_3dot"
    )
    total_tasks = len(pair_dirs) * len(METHODS)
    for pair_dir in pair_dirs:
        pair, _, metadata = task_metadata(pair_dir.name)
        for method in METHODS:
            response_path = pair_dir / method / "row_responses.npz"
            if not response_path.exists():
                raise FileNotFoundError(response_path)
            with np.load(response_path, allow_pickle=False) as row:
                risks = risk_arrays(row)
                witnesses, coverage = witness_arrays(row, pair.target_labels)
                coverage_rows.append({**metadata, "method": method, **coverage})
                fidelity_rows.extend(fidelity_records(metadata, method, row))
                for witness, (loss, mask) in witnesses.items():
                    for score_name, score in risks.items():
                        utility_rows.append(
                            utility_record(
                                metadata,
                                method,
                                witness,
                                score_name,
                                loss,
                                score,
                                mask,
                                coverage,
                            )
                        )
                        if score_name in CONTROL_SCORES:
                            control_rows.append(
                                control_record(
                                    metadata,
                                    method,
                                    witness,
                                    score_name,
                                    loss,
                                    score,
                                    mask,
                                    row,
                                )
                            )
            task_count += 1
            print(f"v1.3 real reanalysis {task_count}/{total_tasks}", flush=True)
    fidelity = pd.DataFrame(fidelity_rows)
    pair_keys = [
        "dataset",
        "pair_type",
        "biological_pair_id",
        "method",
        "intervention",
        "proxy",
        "proxy_scope",
    ]
    pair_fidelity = (
        fidelity.groupby(pair_keys, dropna=False)
        .agg(
            directions=("direction", "nunique"),
            spearman=("spearman", "mean"),
            top_decile_precision=("top_decile_precision", "mean"),
            raw_mae=("raw_mae", "mean"),
            reference_mad=("reference_mad", "mean"),
            normalized_mae=("normalized_mae", "mean"),
            normalized_mae_estimable=("normalized_mae_estimable", "all"),
            spatial_block_same_sign_fraction=("spatial_block_same_sign_fraction", "mean"),
        )
        .reset_index()
    )
    fidelity.to_csv(OUTPUT / "internal_fidelity_direction_level.tsv", sep="\t", index=False)
    pair_fidelity.to_csv(OUTPUT / "internal_fidelity_pair_averaged.tsv", sep="\t", index=False)
    pd.DataFrame(utility_rows).to_csv(OUTPUT / "external_utility_tie_aware.tsv", sep="\t", index=False)
    pd.DataFrame(control_rows).to_csv(OUTPUT / "controls_tie_aware_spatial_cluster.tsv", sep="\t", index=False)
    pd.DataFrame(coverage_rows).to_csv(OUTPUT / "label_witness_coverage.tsv", sep="\t", index=False)
    decision = status_payload(
        "V1_3_REAL_REANALYSIS",
        "COMPLETED",
        pair_directions=len({row["pair_id"] for row in utility_rows}),
        method_tasks=task_count,
        utility_rows=len(utility_rows),
        control_rows=len(control_rows),
        tie_handling="fractional",
        label_witness="closed_set_plus_open_set",
        covariance="spatial_block_cluster",
        fgw_status="excluded_pending_objective_consistent_rerun",
    )
    write_json(OUTPUT / "V1_3_REAL_REANALYSIS_DECISION.json", decision)
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
