import json
from pathlib import Path

import pandas as pd

import run_p1_scale_regularization as engine


ROOT = Path(__file__).resolve().parents[2]


def test_cohort_stratified_gate_does_not_borrow_technologies_across_cohorts():
    config = json.loads((ROOT / "configs" / "p1_scale_regularization_v2.yaml").read_text(encoding="utf-8"))
    result_dir = ROOT / "results" / "p1_scale_regularization_v2"
    internal = pd.read_csv(result_dir / "p1_internal_unit_level.csv")
    external = pd.read_csv(result_dir / "p1_external_unit_level.csv")

    for cohort_role in sorted(internal["cohort_role"].unique()):
        observed = engine.gate_summary(
            internal.loc[internal["cohort_role"] == cohort_role].copy(),
            external.loc[external["cohort_role"] == cohort_role].copy(),
            config,
        )
        ext = observed.loc[observed["gate_family"] == "external_utility"]
        technologies = external.loc[external["cohort_role"] == cohort_role, "technology"].nunique()
        if technologies < config["external_gate"]["minimum_positive_technology_count"]:
            assert not ext["sensitivity_external_gate"].eq(True).any()
