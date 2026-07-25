from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
STATS = ROOT / "15_v1_3_correction" / "03_statistics"
OUTPUT = ROOT / "15_v1_3_correction" / "08_review_followup"


def summarize(group: pd.DataFrame) -> pd.Series:
    nex = pd.to_numeric(group.normalized_excess_aurc, errors="coerce").dropna()
    return pd.Series(
        {
            "biological_pairs": group.biological_pair_id.nunique(),
            "independent_units": group.independent_unit_id.nunique(),
            "median_nex_aurc": nex.median() if len(nex) else np.nan,
            "fraction_worse_than_random": (nex > 1).mean() if len(nex) else np.nan,
            "fraction_better_than_random": (nex < 1).mean() if len(nex) else np.nan,
            "median_witness_coverage": pd.to_numeric(
                group.witness_coverage, errors="coerce"
            ).median(),
        }
    )


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    utility = pd.read_csv(STATS / "utility_direction_averaged.tsv", sep="\t")
    real = utility[utility.source == "real"].copy()

    exact = real[real.score.str.startswith("exact_", na=False)].copy()
    exact_summary = (
        exact.groupby(["method", "witness", "score"], dropna=False)
        .apply(summarize, include_groups=False)
        .reset_index()
    )
    exact_summary["signed_utility_median"] = 1 - exact_summary.median_nex_aurc
    exact_summary["interpretation"] = np.where(
        exact_summary.median_nex_aurc > 1,
        "model-response hotspots rank this witness worse than random in the current sample",
        "model-response hotspots rank this witness better than random in the current sample",
    )
    exact_summary.to_csv(OUTPUT / "exact_response_vs_random.tsv", sep="\t", index=False)

    score_subset = {
        "exact_combined",
        "exact_I_EXPR",
        "exact_I_SPATIAL",
        "finite_difference_I_EXPR_h001",
        "source_boundary_proximity",
    }
    stratified = real[real.score.isin(score_subset)].copy()
    stratified_summary = (
        stratified.groupby(["pair_type", "method", "witness", "score"], dropna=False)
        .apply(summarize, include_groups=False)
        .reset_index()
    )
    stratified_summary.to_csv(
        OUTPUT / "cross_stage_witness_stratified.tsv", sep="\t", index=False
    )

    dlpfc_path = ROOT / "07_E3_DLPFC_retrospective" / "E3_DECISION.json"
    dlpfc = json.loads(dlpfc_path.read_text(encoding="utf-8"))
    dlpfc_summary = pd.DataFrame([dlpfc["summary"]])
    dlpfc_summary.to_csv(OUTPUT / "dlpfc_retrospective_context.tsv", sep="\t", index=False)

    decision = {
        "stage": "V1_3_REVIEW_FOLLOWUP",
        "status": "COMPLETED",
        "exact_summary_rows": len(exact_summary),
        "exact_conditions_median_worse_than_random": int(
            (exact_summary.median_nex_aurc > 1).sum()
        ),
        "stratified_rows": len(stratified_summary),
        "dlpfc_role": "retrospective negative context only; never pooled as prospective replication",
        "causal_claim_boundary": (
            "NEX-AURC above one establishes anti-informative ranking for a witness; "
            "it does not identify the biological mechanism"
        ),
    }
    (OUTPUT / "REVIEW_FOLLOWUP_DECISION.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
