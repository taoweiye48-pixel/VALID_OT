"""Execute the amended P1 v2 expanded-sample analysis.

The computational definitions are deliberately inherited from the frozen P1
v1 implementation. This wrapper changes only versioned input/output paths and
adds cohort-stratified summaries required by Amendment 001.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

import run_p1_scale_regularization as engine


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "p1_scale_regularization_v2.yaml"
ANALYSIS = ROOT / "analysis" / "p1_scale_regularization_v2"
RESULTS = ROOT / "results" / "p1_scale_regularization_v2"
BUILD = ROOT / "build" / "p1_scale_regularization_v2"
EXPECTED_CONFIG_HASH = "fcd957f47e2f6077ea2d6fdc41d4e0b679b4fa96bb878aad3b7fbf7f9fda4ac0"
ORIGINAL_COMPUTE_TASK = engine.compute_task


def configure_engine() -> None:
    engine.CONFIG_PATH = CONFIG_PATH
    engine.ANALYSIS = ANALYSIS
    engine.RESULTS = RESULTS
    engine.BUILD = BUILD
    engine.CHECKPOINTS = BUILD / "checkpoints"
    engine.LOGS = ANALYSIS / "logs"
    engine.EXPECTED_CONFIG_HASH = EXPECTED_CONFIG_HASH


def compute_task_v2(task: dict, config: dict) -> dict:
    """Reapply v2 globals inside each Windows-spawned worker."""
    configure_engine()
    return ORIGINAL_COMPUTE_TASK(task, config)


def append_v2_freeze_materials() -> None:
    additions = [
        ROOT / "code" / "run_p1_scale_regularization_v2.py",
        ROOT / "code" / "prepare_p1_v2_spatialdlpfc.py",
        ROOT / "code" / "prepare_p1_v2_her2st_manual_truth.py",
        ROOT / "code" / "export_spatialdlpfc_p1_v2_compact.R",
        ROOT / "docs" / "P1_AMENDMENT_001_SAMPLE_EXPANSION.md",
        ROOT / "docs" / "P1_AMENDMENT_002_LABEL_COVERAGE.md",
        ROOT / "docs" / "P1_AMENDMENT_003_MANUAL_TRUTH_COHORT.md",
        ROOT / "docs" / "P1_IMPLEMENTATION_CORRECTION_001_GATE_SUMMARY_FIELD_MAPPING.md",
        ROOT / "docs" / "P1_IMPLEMENTATION_CORRECTION_002_COHORT_STRATIFIED_GATES.md",
    ]
    manifest = ANALYSIS / "code_manifest.sha256"
    existing = manifest.read_text(encoding="utf-8") if manifest.is_file() else ""
    present = {line.split("  ", 1)[1] for line in existing.splitlines() if "  " in line}
    with manifest.open("a", encoding="utf-8") as handle:
        for path in additions:
            relative = path.relative_to(ROOT).as_posix()
            if relative not in present:
                handle.write(f"{engine.sha256(path)}  {relative}\n")
    for index, name in enumerate(
        [
            "P1_AMENDMENT_001_SAMPLE_EXPANSION.md",
            "P1_AMENDMENT_002_LABEL_COVERAGE.md",
            "P1_AMENDMENT_003_MANUAL_TRUTH_COHORT.md",
        ], 1
    ):
        amendment = ROOT / "docs" / name
        snapshot = ANALYSIS / f"amendment_{index:03d}_snapshot.md"
        snapshot.write_text(amendment.read_text(encoding="utf-8"), encoding="utf-8")
        (ANALYSIS / f"amendment_{index:03d}_snapshot.sha256").write_text(
            f"{engine.sha256(amendment)}  {snapshot.name}\n", encoding="utf-8"
        )
    correction = ROOT / "docs" / "P1_IMPLEMENTATION_CORRECTION_001_GATE_SUMMARY_FIELD_MAPPING.md"
    correction_snapshot = ANALYSIS / "implementation_correction_001_snapshot.md"
    correction_snapshot.write_text(correction.read_text(encoding="utf-8"), encoding="utf-8")
    (ANALYSIS / "implementation_correction_001_snapshot.sha256").write_text(
        f"{engine.sha256(correction)}  {correction_snapshot.name}\n", encoding="utf-8"
    )
    correction_002 = ROOT / "docs" / "P1_IMPLEMENTATION_CORRECTION_002_COHORT_STRATIFIED_GATES.md"
    correction_002_snapshot = ANALYSIS / "implementation_correction_002_snapshot.md"
    correction_002_snapshot.write_text(correction_002.read_text(encoding="utf-8"), encoding="utf-8")
    (ANALYSIS / "implementation_correction_002_snapshot.sha256").write_text(
        f"{engine.sha256(correction_002)}  {correction_002_snapshot.name}\n", encoding="utf-8"
    )


def add_cohort_summaries(config: dict) -> None:
    unit_to_cohort = {
        pair["independent_unit_id"]: pair["cohort_role"] for pair in config["pairs"]
    }
    internal_path = RESULTS / "p1_internal_unit_level.csv"
    external_path = RESULTS / "p1_external_unit_level.csv"
    internal = pd.read_csv(internal_path)
    external = pd.read_csv(external_path)
    internal["cohort_role"] = internal["independent_unit_id"].map(unit_to_cohort)
    external["cohort_role"] = external["independent_unit_id"].map(unit_to_cohort)
    if internal["cohort_role"].isna().any() or external["cohort_role"].isna().any():
        raise RuntimeError("at least one independent unit lacks a frozen cohort assignment")
    internal.to_csv(internal_path, index=False)
    external.to_csv(external_path, index=False)

    ikeys = [
        "cohort_role", "method", "epsilon", "tau", "grid_role", "arm",
        "intervention", "score", "claim_status",
    ]
    internal_summary = (
        internal.groupby(ikeys, dropna=False)
        .agg(
            independent_units=("independent_unit_id", "nunique"),
            median_spearman=("spearman", "median"),
            median_top_decile_overlap=("top_decile_overlap", "median"),
            median_nmae=("nmae", "median"),
            unit_gate_pass_fraction=("gate_pass", "mean"),
            estimable_units=("estimable", "sum"),
        )
        .reset_index()
    )
    internal_summary.to_csv(RESULTS / "p1_internal_cohort_summary.csv", index=False)

    ekeys = [
        "cohort_role", "method", "epsilon", "tau", "grid_role", "arm",
        "witness", "score", "claim_status",
    ]
    external_summary = (
        external.groupby(ekeys, dropna=False)
        .agg(
            independent_units=("independent_unit_id", "nunique"),
            median_nex_aurc=("normalized_excess_aurc", "median"),
            median_relative_fixed_qc_gain=("relative_fixed_qc_gain", "median"),
            positive_unit_fraction=("fixed_qc_gain", lambda value: float((value > 0).mean())),
            negative_control_pass_fraction=("negative_control_pass", "mean"),
            leakage_control_pass_fraction=("leakage_control_pass", "mean"),
            confound_direction_pass_fraction=("confound_direction_pass", "mean"),
        )
        .reset_index()
    )
    external_summary.to_csv(RESULTS / "p1_external_cohort_summary.csv", index=False)

    # The expanded data contain three deliberately different evidence cohorts.
    # A pooled gate is retained for backward-compatible descriptive auditing,
    # but it must not be interpreted as one biological-effect estimate.  Apply
    # the unchanged frozen gates within each cohort and make this table the
    # inferentially valid gate summary for the amended design.
    cohort_gates = []
    for cohort_role in sorted(internal["cohort_role"].unique()):
        cohort_internal = internal.loc[internal["cohort_role"] == cohort_role].copy()
        cohort_external = external.loc[external["cohort_role"] == cohort_role].copy()
        cohort_gate = engine.gate_summary(cohort_internal, cohort_external, config)
        cohort_gate.insert(0, "cohort_role", cohort_role)
        cohort_gates.append(cohort_gate)
    cohort_gate_summary = pd.concat(cohort_gates, ignore_index=True)
    cohort_gate_summary.to_csv(RESULTS / "p1_cohort_gate_summary.csv", index=False)

    summary_path = RESULTS / "p1_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    observed = internal.groupby("cohort_role").independent_unit_id.nunique().to_dict()
    pooled_passes = int(
        pd.read_csv(RESULTS / "p1_gate_summary.csv")
        .get("sensitivity_external_gate_with_controls", pd.Series(dtype=bool))
        .eq(True)
        .sum()
    )
    cohort_passes = int(
        cohort_gate_summary
        .get("sensitivity_external_gate_with_controls", pd.Series(dtype=bool))
        .eq(True)
        .sum()
    )
    summary.update(
        amendments=config["amendments"],
        primary_independent_units=int(observed.get("primary_expansion", 0)),
        real_cross_slice_independent_units=int(observed.get("primary_expansion", 0)),
        manual_truth_independent_units=int(observed.get("manual_truth_controlled", 0)),
        legacy_replication_independent_units=int(observed.get("legacy_replication", 0)),
        pooled_independent_units=int(internal.independent_unit_id.nunique()),
        independence_unit="donor for spatialDLPFC; pre-existing biological unit for legacy cohorts",
        cohort_stratified_outputs=True,
        primary_gate_basis="cohort-stratified; unchanged thresholds",
        cohort_stratified_external_gate_with_controls_passes=cohort_passes,
        pooled_sensitivity_external_gate_with_controls_passes_descriptive=pooled_passes,
        pooled_gate_inference_allowed=False,
    )
    engine.json_dump(summary_path, summary)
    engine.write_result_manifest()


def main() -> int:
    configure_engine()
    engine.compute_task = compute_task_v2
    preparing = "--prepare-freeze" in sys.argv
    status = engine.main()
    if status != 0:
        return status
    if preparing:
        append_v2_freeze_materials()
        return 0
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    add_cohort_summaries(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
