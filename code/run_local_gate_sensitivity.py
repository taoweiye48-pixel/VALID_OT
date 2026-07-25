"""Recompute local-qualification robustness from frozen WP1/WP3 outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from validot.metrics import top_fraction_precision


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results" / "local_gate_sensitivity_v1"
WP1_ROOT = ROOT / "analysis" / "postreview_wp1_wp10_v1" / "wp1"
WP3_UNIT = (
    ROOT
    / "results"
    / "postreview_wp1_wp10_v1"
    / "wp3"
    / "wp3_local_fidelity_unit.tsv"
)
WP3_FAMILY = (
    ROOT
    / "results"
    / "postreview_wp1_wp10_v1"
    / "wp3"
    / "wp3_local_fidelity_family.tsv"
)
PASTE_GATE = (
    ROOT
    / "paste_fgw_extension"
    / "full"
    / "validation"
    / "descriptive_gate_summary.tsv"
)


THRESHOLD_SETS = [
    {
        "threshold_set": "prespecified",
        "status": "primary prespecified qualification tolerance",
        "spearman_min": 0.95,
        "top_overlap_min": 0.90,
        "rmae_max": 0.10,
        "relative_l1_max": 0.10,
        "cosine_min": 0.99,
        "neighborhood_max": 0.15,
    },
    {
        "threshold_set": "moderately_stricter",
        "status": "sensitivity analysis",
        "spearman_min": 0.975,
        "top_overlap_min": 0.95,
        "rmae_max": 0.05,
        "relative_l1_max": 0.05,
        "cosine_min": 0.995,
        "neighborhood_max": 0.075,
    },
    {
        "threshold_set": "stringent",
        "status": "sensitivity analysis",
        "spearman_min": 0.99,
        "top_overlap_min": 0.975,
        "rmae_max": 0.02,
        "relative_l1_max": 0.02,
        "cosine_min": 0.999,
        "neighborhood_max": 0.05,
    },
    {
        "threshold_set": "stress",
        "status": "descriptive stress tolerance",
        "spearman_min": 0.999,
        "top_overlap_min": 0.99,
        "rmae_max": 0.01,
        "relative_l1_max": 0.01,
        "cosine_min": 0.9999,
        "neighborhood_max": 0.02,
    },
]


def dump_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def apply_gate(frame: pd.DataFrame, threshold: dict[str, Any]) -> pd.Series:
    return (
        (frame["h001_spearman"] >= threshold["spearman_min"])
        & (frame["h001_top_overlap"] >= threshold["top_overlap_min"])
        & (frame["h001_rmae"] <= threshold["rmae_max"])
        & (
            frame["vector_relative_l1_median"]
            <= threshold["relative_l1_max"]
        )
        & (
            frame["direction_cosine_median"]
            >= threshold["cosine_min"]
        )
        & (
            frame["neighborhood_error_median"]
            <= threshold["neighborhood_max"]
        )
    )


def threshold_sensitivity() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    unit = pd.read_csv(WP3_UNIT, sep="\t")
    family = pd.read_csv(WP3_FAMILY, sep="\t")
    summaries = []
    condition_rows = []
    reason_rows = []
    metric_rules = [
        ("spearman", "h001_spearman", "min", "spearman_min"),
        ("top_overlap", "h001_top_overlap", "min", "top_overlap_min"),
        ("rmae", "h001_rmae", "max", "rmae_max"),
        (
            "relative_l1",
            "vector_relative_l1_median",
            "max",
            "relative_l1_max",
        ),
        (
            "direction_cosine",
            "direction_cosine_median",
            "min",
            "cosine_min",
        ),
        (
            "neighborhood_error",
            "neighborhood_error_median",
            "max",
            "neighborhood_max",
        ),
    ]
    for threshold in THRESHOLD_SETS:
        unit_pass = apply_gate(unit, threshold)
        family_pass = apply_gate(family, threshold)
        summaries.append(
            {
                **threshold,
                "unit_family_records": int(len(unit)),
                "unit_family_passes": int(unit_pass.sum()),
                "unit_family_pass_fraction": float(unit_pass.mean()),
                "families": int(len(family)),
                "family_passes": int(family_pass.sum()),
                "family_pass_fraction": float(family_pass.mean()),
            }
        )
        selected = unit.copy()
        selected.insert(0, "threshold_set", threshold["threshold_set"])
        selected["sensitivity_gate_pass"] = unit_pass
        condition_rows.append(selected)
        for label, column, direction, threshold_key in metric_rules:
            if direction == "min":
                failures = unit[column] < float(threshold[threshold_key])
            else:
                failures = unit[column] > float(threshold[threshold_key])
            reason_rows.append(
                {
                    "threshold_set": threshold["threshold_set"],
                    "metric": label,
                    "failed_records": int(failures.sum()),
                    "threshold": float(threshold[threshold_key]),
                    "observed_min": float(unit[column].min()),
                    "observed_median": float(unit[column].median()),
                    "observed_max": float(unit[column].max()),
                }
            )
    return (
        pd.DataFrame(summaries),
        pd.concat(condition_rows, ignore_index=True),
        pd.DataFrame(reason_rows),
    )


def scalar_metrics(reference: np.ndarray, estimate: np.ndarray) -> dict[str, float]:
    keep = np.isfinite(reference) & np.isfinite(estimate)
    y = np.asarray(reference[keep], dtype=np.float64)
    x = np.asarray(estimate[keep], dtype=np.float64)
    rho = (
        float(spearmanr(y, x).statistic)
        if len(x) >= 3 and np.unique(x).size > 1 and np.unique(y).size > 1
        else float("nan")
    )
    raw_mae = float(np.mean(np.abs(x - y)))
    return {
        "h_spearman": rho,
        "h_top_overlap": float(top_fraction_precision(y, x, 0.10)),
        "h_rmae": float(raw_mae / max(np.mean(np.abs(y)), 1e-300)),
    }


def step_size_sensitivity() -> tuple[pd.DataFrame, pd.DataFrame]:
    registry = pd.read_csv(WP1_ROOT / "TASK_REGISTRY_full.tsv", sep="\t")
    direction_rows = []
    for _, task in registry.iterrows():
        path = (
            WP1_ROOT
            / "arrays"
            / f"{task['pair_file_id']}__{task['method']}.npz"
        )
        with np.load(path, allow_pickle=False) as data:
            h_grid = data["h_grid"]
            condition_names = data["condition_names"].astype(str)
            scores = data["scores"]
            references = data["reference_scores"]
            estimable = data["reference_estimable"]
        metadata = task.to_dict()
        for condition_index, condition_name in enumerate(condition_names):
            arm, intervention = condition_name.split("__", 1)
            for h_index, h in enumerate(h_grid):
                keep = estimable[condition_index]
                metrics = scalar_metrics(
                    references[condition_index, keep],
                    scores[condition_index, h_index, keep],
                )
                direction_rows.append(
                    {
                        **metadata,
                        "arm": arm,
                        "intervention": intervention,
                        "h": float(h),
                        **metrics,
                    }
                )
    direction = pd.DataFrame(direction_rows)
    pair_keys = [
        "dataset",
        "pair_id",
        "pair_type",
        "independent_unit_id",
        "cohort_role",
        "method",
        "epsilon",
        "tau",
        "arm",
        "intervention",
        "h",
    ]
    metrics = ["h_spearman", "h_top_overlap", "h_rmae"]
    pair = (
        direction.groupby(pair_keys, dropna=False)[metrics]
        .mean()
        .reset_index()
    )
    unit_keys = [
        "independent_unit_id",
        "cohort_role",
        "method",
        "epsilon",
        "tau",
        "arm",
        "intervention",
        "h",
    ]
    unit = (
        pair.groupby(unit_keys, dropna=False)[metrics]
        .median()
        .reset_index()
    )
    unit["scalar_gate_pass"] = (
        (unit["h_spearman"] >= 0.95)
        & (unit["h_top_overlap"] >= 0.90)
        & (unit["h_rmae"] <= 0.10)
    )
    summary = (
        unit.groupby("h", dropna=False)
        .agg(
            unit_family_records=("scalar_gate_pass", "size"),
            scalar_gate_passes=("scalar_gate_pass", "sum"),
            scalar_gate_pass_fraction=("scalar_gate_pass", "mean"),
            minimum_spearman=("h_spearman", "min"),
            minimum_top_overlap=("h_top_overlap", "min"),
            maximum_rmae=("h_rmae", "max"),
        )
        .reset_index()
    )
    return unit, summary


def failure_envelope() -> pd.DataFrame:
    paste = pd.read_csv(PASTE_GATE, sep="\t")
    paste.insert(0, "method_family", "PASTE pairwise FGW")
    paste.insert(1, "evidence_status", "descriptive method-family envelope")
    return paste


def main() -> int:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    threshold_summary, threshold_conditions, reasons = threshold_sensitivity()
    step_units, step_summary = step_size_sensitivity()
    envelope = failure_envelope()
    threshold_summary.to_csv(
        RESULT_ROOT / "threshold_sensitivity_summary.tsv",
        sep="\t",
        index=False,
    )
    threshold_conditions.to_csv(
        RESULT_ROOT / "threshold_sensitivity_unit_records.tsv",
        sep="\t",
        index=False,
    )
    reasons.to_csv(
        RESULT_ROOT / "threshold_failure_reasons.tsv",
        sep="\t",
        index=False,
    )
    step_units.to_csv(
        RESULT_ROOT / "step_size_sensitivity_unit_records.tsv",
        sep="\t",
        index=False,
    )
    step_summary.to_csv(
        RESULT_ROOT / "step_size_sensitivity_summary.tsv",
        sep="\t",
        index=False,
    )
    envelope.to_csv(
        RESULT_ROOT / "method_family_failure_envelope.tsv",
        sep="\t",
        index=False,
    )
    summary = {
        "analysis_version": "valid-ot-local-gate-sensitivity-v1",
        "unit_family_records": int(
            threshold_summary.iloc[0]["unit_family_records"]
        ),
        "threshold_sets": threshold_summary[
            [
                "threshold_set",
                "unit_family_passes",
                "family_passes",
            ]
        ].to_dict(orient="records"),
        "step_size_range": [
            float(step_summary["h"].min()),
            float(step_summary["h"].max()),
        ],
        "step_size_scalar_passes": step_summary[
            ["h", "scalar_gate_passes", "unit_family_records"]
        ].to_dict(orient="records"),
        "paste_conditions": int(len(envelope)),
        "paste_descriptive_gate_passes": int(
            envelope["descriptive_joint_gate"].astype(bool).sum()
        ),
    }
    dump_json(RESULT_ROOT / "LOCAL_GATE_SENSITIVITY_SUMMARY.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
