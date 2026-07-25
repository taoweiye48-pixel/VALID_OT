from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
from statsmodels.stats.multitest import multipletests

from validot.utils import read_json, status_payload, write_json


ROOT = Path(__file__).resolve().parents[1]
CONFIG = read_json(ROOT / "00_protocol" / "frozen_config.json")
CORRECTION = CONFIG["v1_3_correction"]
SEMI = ROOT / "15_v1_3_correction" / "01_semisynthetic_rerun"
REAL = ROOT / "15_v1_3_correction" / "02_real_reanalysis"
OUTPUT = ROOT / "15_v1_3_correction" / "03_statistics"
METHODS = CORRECTION["confirmatory_ot_methods"] + CORRECTION["non_ot_stress_test"]
QC_SCORES = ["source_boundary_proximity", "source_sparsity", "matched_target_sparsity"]
FIXED_QC = CORRECTION["fixed_qc_primary"]


def safe_median(values: pd.Series | np.ndarray) -> float:
    clean = pd.Series(values, dtype=float).dropna()
    return float(clean.median()) if len(clean) else float("nan")


def technology_name(dataset: str) -> str:
    return re.sub(r"_base\d+$", "", str(dataset))


def scenario_name(pair_id: str) -> str:
    for scenario in CONFIG["semisynthetic"]["scenarios"]:
        if f"_{scenario}_seed" in str(pair_id):
            return scenario
    return "real"


def add_design_columns(table: pd.DataFrame, source: str) -> pd.DataFrame:
    result = table.copy()
    if "pair_id" not in result:
        result["pair_id"] = (
            result["biological_pair_id"]
            if "biological_pair_id" in result
            else np.arange(len(result)).astype(str)
        )
    result["source"] = source
    result["technology"] = result.dataset.astype(str).map(technology_name)
    if "pair_type" not in result:
        result["pair_type"] = "semisynthetic" if source == "semisynthetic" else "unknown"
    result["pair_type"] = result["pair_type"].fillna(
        "semisynthetic" if source == "semisynthetic" else "unknown"
    )
    if "biological_pair_id" not in result:
        result["biological_pair_id"] = result.pair_id
    result["biological_pair_id"] = result.biological_pair_id.fillna(result.pair_id)
    result["scenario"] = result.biological_pair_id.astype(str).map(scenario_name)
    result["independent_unit_id"] = result.biological_pair_id.astype(str)
    if source == "semisynthetic":
        result["independent_unit_id"] = result.dataset.astype(str)
    else:
        cross_stage = result.pair_type.astype(str) == "cross_stage"
        result.loc[cross_stage, "independent_unit_id"] = (
            result.loc[cross_stage, "technology"].astype(str)
            + "::shared_developmental_series"
        )
    return result


def summarize_fidelity(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    group_keys = ["source", "method", "intervention", "proxy", "independent_unit_id"]
    unit = (
        table.groupby(group_keys, dropna=False)
        .agg(
            technology=("technology", "first"),
            biological_pairs=("biological_pair_id", "nunique"),
            spearman=("spearman", "median"),
            top_decile_precision=("top_decile_precision", "median"),
            raw_mae=("raw_mae", "median"),
            reference_mad=("reference_mad", "median"),
            normalized_mae=("normalized_mae", "median"),
            normalized_mae_estimable=("normalized_mae_estimable", "any"),
        )
        .reset_index()
    )
    rows = []
    for keys, group in unit.groupby(["source", "method", "intervention", "proxy"], dropna=False):
        values = group.spearman.dropna().to_numpy(float)
        pvalue = 1.0
        if len(values) >= 2 and not np.allclose(values, 0):
            try:
                pvalue = float(wilcoxon(values).pvalue)
            except ValueError:
                pvalue = 1.0
        median_spearman = float(group.spearman.median())
        median_top = float(group.top_decile_precision.median())
        nmae = group.loc[group.normalized_mae_estimable, "normalized_mae"].dropna()
        median_nmae = float(nmae.median()) if len(nmae) else float("nan")
        rank_pass = bool(
            median_spearman >= CONFIG["fidelity_gate"]["spearman_min"]
            and median_top >= CONFIG["fidelity_gate"]["top_decile_precision_min"]
        )
        full_pass = bool(
            rank_pass
            and len(nmae)
            and median_nmae <= CONFIG["fidelity_gate"]["normalized_mae_max"]
        )
        rows.append(
            {
                "source": keys[0],
                "method": keys[1],
                "intervention": keys[2],
                "proxy": keys[3],
                "independent_units": group.independent_unit_id.nunique(),
                "technologies": group.technology.nunique(),
                "median_spearman": median_spearman,
                "median_top_decile_precision": median_top,
                "median_raw_mae": safe_median(group.raw_mae),
                "median_reference_mad": safe_median(group.reference_mad),
                "median_normalized_mae": median_nmae,
                "nmae_estimable_units": int(len(nmae)),
                "spearman_gate_margin": median_spearman - CONFIG["fidelity_gate"]["spearman_min"],
                "top_decile_gate_margin": median_top
                - CONFIG["fidelity_gate"]["top_decile_precision_min"],
                "nmae_gate_margin": (
                    CONFIG["fidelity_gate"]["normalized_mae_max"] - median_nmae
                    if np.isfinite(median_nmae)
                    else float("nan")
                ),
                "rank_fidelity_pass": rank_pass,
                "full_fidelity_pass": full_pass,
                "wilcoxon_p_descriptive": pvalue,
                "inference_note": "descriptive; independent-unit count is small",
            }
        )
    summary = pd.DataFrame(rows)
    summary["wilcoxon_fdr_bh_descriptive"] = multipletests(
        summary.wilcoxon_p_descriptive, method="fdr_bh"
    )[1]
    return unit, summary


def summarize_fidelity_pair_gate(table: pd.DataFrame) -> pd.DataFrame:
    """Apply the frozen Gate after direction averaging, before unit sensitivity."""
    rows = []
    for keys, group in table.groupby(
        ["source", "method", "intervention", "proxy"], dropna=False
    ):
        nmae = group.loc[group.normalized_mae_estimable, "normalized_mae"].dropna()
        median_spearman = float(group.spearman.median())
        median_top = float(group.top_decile_precision.median())
        median_nmae = float(nmae.median()) if len(nmae) else float("nan")
        rank_pass = bool(
            median_spearman >= CONFIG["fidelity_gate"]["spearman_min"]
            and median_top >= CONFIG["fidelity_gate"]["top_decile_precision_min"]
        )
        full_pass = bool(
            rank_pass
            and len(nmae)
            and median_nmae <= CONFIG["fidelity_gate"]["normalized_mae_max"]
        )
        rows.append(
            {
                "source": keys[0],
                "method": keys[1],
                "intervention": keys[2],
                "proxy": keys[3],
                "biological_or_seed_pairs": group.biological_pair_id.nunique(),
                "independent_units": group.independent_unit_id.nunique(),
                "technologies": group.technology.nunique(),
                "median_spearman": median_spearman,
                "median_top_decile_precision": median_top,
                "median_raw_mae": safe_median(group.raw_mae),
                "median_reference_mad": safe_median(group.reference_mad),
                "median_normalized_mae": median_nmae,
                "nmae_estimable_pairs": int(len(nmae)),
                "spearman_gate_margin": median_spearman
                - CONFIG["fidelity_gate"]["spearman_min"],
                "top_decile_gate_margin": median_top
                - CONFIG["fidelity_gate"]["top_decile_precision_min"],
                "nmae_gate_margin": (
                    CONFIG["fidelity_gate"]["normalized_mae_max"] - median_nmae
                    if np.isfinite(median_nmae)
                    else float("nan")
                ),
                "rank_fidelity_pass": rank_pass,
                "full_fidelity_pass": full_pass,
                "gate_unit": "direction-averaged biological pair"
                if keys[0] == "real"
                else "semisynthetic pair-seed task",
                "inference_note": "Gate decision only; independent-unit sensitivity reported separately",
            }
        )
    return pd.DataFrame(rows)


def direction_average_utility(table: pd.DataFrame) -> pd.DataFrame:
    if "direction" not in table:
        table = table.assign(direction="synthetic")
    keys = [
        "source",
        "technology",
        "dataset",
        "pair_type",
        "biological_pair_id",
        "independent_unit_id",
        "method",
        "witness",
        "score",
    ]
    numeric = [
        column
        for column in [
            "normalized_excess_aurc",
            "aurc",
            "oracle_aurc",
            "random_aurc",
            "witness_coverage",
            "shared_label_coverage",
            "source_only_fraction",
        ]
        if column in table
    ]
    averaged = table.groupby(keys, dropna=False)[numeric].mean().reset_index()
    directions = table.groupby(keys, dropna=False).direction.nunique().rename("directions").reset_index()
    return averaged.merge(directions, on=keys, how="left")


def baseline_comparison(direction_averaged: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "source",
        "technology",
        "dataset",
        "pair_type",
        "biological_pair_id",
        "independent_unit_id",
        "method",
        "witness",
    ]
    rows = []
    for group_keys, group in direction_averaged.groupby(keys, dropna=False):
        score_values = group.set_index("score").normalized_excess_aurc.to_dict()
        qc_values = {name: score_values.get(name, np.nan) for name in QC_SCORES}
        finite_qc = [value for value in qc_values.values() if np.isfinite(value)]
        best_qc = min(finite_qc) if finite_qc else float("nan")
        for score, nex in score_values.items():
            if score in QC_SCORES or not np.isfinite(nex):
                continue
            record = dict(zip(keys, group_keys))
            record.update(
                score=score,
                normalized_excess_aurc=float(nex),
                gain_over_random=float(1.0 - nex),
                better_than_random=bool(nex < 1.0),
                best_qc_envelope_nex=float(best_qc),
                gain_over_best_qc_envelope=float(best_qc - nex),
                best_qc_role="hindsight_oracle_envelope",
            )
            for qc_name, qc_value in qc_values.items():
                record[f"{qc_name}_nex"] = float(qc_value)
                record[f"gain_over_{qc_name}"] = float(qc_value - nex)
            rows.append(record)
    return pd.DataFrame(rows)


def leave_one_unit_out(values: pd.Series, units: pd.Series) -> tuple[float, float]:
    estimates = []
    unique = pd.unique(units)
    for omitted in unique:
        retained = values[units != omitted].dropna()
        if len(retained):
            estimates.append(float(retained.median()))
    if not estimates:
        return float("nan"), float("nan")
    return float(min(estimates)), float(max(estimates))


def summarize_external(comparison: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    unit = (
        comparison.groupby(
            ["source", "method", "witness", "score", "independent_unit_id"], dropna=False
        )
        .agg(
            technology=("technology", "first"),
            biological_pairs=("biological_pair_id", "nunique"),
            normalized_excess_aurc=("normalized_excess_aurc", "mean"),
            gain_over_random=("gain_over_random", "mean"),
            gain_over_fixed_qc=(f"gain_over_{FIXED_QC}", "mean"),
            gain_over_best_qc_envelope=("gain_over_best_qc_envelope", "mean"),
        )
        .reset_index()
    )
    rows = []
    for keys, group in unit.groupby(["source", "method", "witness", "score"], dropna=False):
        loo_low, loo_high = leave_one_unit_out(
            group.gain_over_fixed_qc, group.independent_unit_id
        )
        tech_direction = group.groupby("technology").gain_over_fixed_qc.median()
        relative = group.gain_over_fixed_qc / np.maximum(
            np.abs(group.normalized_excess_aurc + group.gain_over_fixed_qc), 1e-12
        )
        rows.append(
            {
                "source": keys[0],
                "method": keys[1],
                "witness": keys[2],
                "score": keys[3],
                "independent_units": group.independent_unit_id.nunique(),
                "technologies": group.technology.nunique(),
                "median_nex": float(group.normalized_excess_aurc.median()),
                "median_gain_over_random": float(group.gain_over_random.median()),
                "better_than_random_unit_fraction": float(np.mean(group.gain_over_random > 0)),
                "median_gain_over_fixed_qc": float(group.gain_over_fixed_qc.median()),
                "median_relative_gain_over_fixed_qc": float(relative.median()),
                "positive_fixed_qc_unit_fraction": float(np.mean(group.gain_over_fixed_qc > 0)),
                "positive_fixed_qc_technology_count": int(np.sum(tech_direction > 0)),
                "median_gain_over_best_qc_envelope": float(
                    group.gain_over_best_qc_envelope.median()
                ),
                "leave_one_unit_out_fixed_qc_low": loo_low,
                "leave_one_unit_out_fixed_qc_high": loo_high,
                "corrected_external_gate": bool(
                    relative.median()
                    >= CONFIG["external_gate"]["median_relative_aurc_improvement_min"]
                    and np.sum(tech_direction > 0)
                    >= CONFIG["external_gate"]["minimum_positive_technology_count"]
                    and np.mean(group.gain_over_fixed_qc > 0)
                    >= CONFIG["external_gate"]["minimum_positive_pair_fraction"]
                    and loo_low >= 0
                ),
                "gate_status": "corrective_sensitivity_not_v1.2_registered",
            }
        )
    return unit, pd.DataFrame(rows)


def summarize_controls(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    controls = add_design_columns(pd.read_csv(path, sep="\t"), "real")
    pair = (
        controls.groupby(
            [
                "technology",
                "pair_type",
                "biological_pair_id",
                "independent_unit_id",
                "method",
                "witness",
                "score",
            ],
            dropna=False,
        )
        .agg(
            directions=("direction", "nunique"),
            primary=("primary_normalized_excess_aurc", "mean"),
            label_shuffle=("label_shuffle_median", "mean"),
            label_shuffle_p=("label_shuffle_p_lower", "max"),
            within_block=("within_block_permutation_median", "mean"),
            within_block_p=("within_block_p_lower", "max"),
            leakage_positive=("leakage_positive_control_normalized_excess_aurc", "mean"),
            adjusted_positive=("risk_positive_after_adjustment", "mean"),
            adjusted_p=("risk_rank_pvalue", "max"),
        )
        .reset_index()
    )
    pair["negative_controls_pass"] = (
        (pair.primary < pair.label_shuffle)
        & (pair.primary < pair.within_block)
        & (pair.label_shuffle_p <= 0.05)
        & (pair.within_block_p <= 0.05)
    )
    pair["leakage_control_pass"] = np.abs(pair.leakage_positive) <= 1e-8
    unit = (
        pair.groupby(["method", "witness", "score", "independent_unit_id"], dropna=False)
        .agg(
            technology=("technology", "first"),
            negative_controls_pass=("negative_controls_pass", "mean"),
            leakage_control_pass=("leakage_control_pass", "mean"),
            adjusted_positive=("adjusted_positive", "mean"),
            adjusted_p=("adjusted_p", "max"),
        )
        .reset_index()
    )
    summary = (
        unit.groupby(["method", "witness", "score"], dropna=False)
        .agg(
            independent_units=("independent_unit_id", "nunique"),
            negative_control_pass_fraction=("negative_controls_pass", "mean"),
            leakage_control_pass_fraction=("leakage_control_pass", "mean"),
            adjusted_positive_fraction=("adjusted_positive", "mean"),
            adjusted_significant_fraction=("adjusted_p", lambda x: float(np.mean(np.asarray(x) <= 0.05))),
        )
        .reset_index()
    )
    return pair, summary


def matched_score(proxy: str, intervention: str) -> str:
    if proxy == "finite_difference_sensitivity_h001":
        return f"finite_difference_{intervention}_h001"
    return proxy


def build_relation(
    fidelity: pd.DataFrame,
    utility: pd.DataFrame,
    fidelity_summary: pd.DataFrame,
    external_summary: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fidelity = fidelity.copy()
    fidelity["score"] = [
        matched_score(proxy, intervention)
        for proxy, intervention in zip(fidelity.proxy, fidelity.intervention)
    ]
    relation = fidelity.merge(
        utility,
        on=[
            "source",
            "technology",
            "dataset",
            "pair_type",
            "biological_pair_id",
            "independent_unit_id",
            "method",
            "score",
        ],
        how="inner",
        suffixes=("_fidelity", "_utility"),
    )
    relation = relation[
        np.isfinite(relation.spearman) & np.isfinite(relation.normalized_excess_aurc)
    ].copy()
    relation["external_utility"] = -relation.normalized_excess_aurc
    unit = (
        relation.groupby(
            [
                "source",
                "independent_unit_id",
                "technology",
                "method",
                "intervention",
                "proxy",
                "witness",
                "score",
            ],
            dropna=False,
        )
        .agg(spearman=("spearman", "median"), external_utility=("external_utility", "median"))
        .reset_index()
    )
    associations = []
    for source, group in unit.groupby("source"):
        per_unit = []
        for unit_id, unit_group in group.groupby("independent_unit_id"):
            rho = (
                spearmanr(unit_group.spearman, unit_group.external_utility).statistic
                if len(unit_group) >= 3
                and unit_group.spearman.nunique() > 1
                and unit_group.external_utility.nunique() > 1
                else float("nan")
            )
            per_unit.append(rho)
        associations.append(
            {
                "source": source,
                "independent_units": group.independent_unit_id.nunique(),
                "median_within_unit_spearman": float(np.nanmedian(per_unit)),
                "min_within_unit_spearman": float(np.nanmin(per_unit)),
                "max_within_unit_spearman": float(np.nanmax(per_unit)),
                "interpretation": "descriptive; does not establish orthogonality",
            }
        )
    cards = []
    for keys, group in unit.groupby(
        ["source", "method", "intervention", "proxy", "witness", "score"], dropna=False
    ):
        fid = fidelity_summary[
            (fidelity_summary.source == keys[0])
            & (fidelity_summary.method == keys[1])
            & (fidelity_summary.intervention == keys[2])
            & (fidelity_summary.proxy == keys[3])
        ]
        ext = external_summary[
            (external_summary.source == keys[0])
            & (external_summary.method == keys[1])
            & (external_summary.witness == keys[4])
            & (external_summary.score == keys[5])
        ]
        internal_pass = bool(len(fid) and fid.full_fidelity_pass.iloc[0])
        external_pass = bool(len(ext) and ext.corrected_external_gate.iloc[0])
        quadrant = ("A" if external_pass else "B") if internal_pass else ("C" if external_pass else "D")
        cards.append(
            {
                "source": keys[0],
                "method": keys[1],
                "intervention": keys[2],
                "proxy": keys[3],
                "witness": keys[4],
                "matched_score": keys[5],
                "median_spearman": float(group.spearman.median()),
                "median_external_utility": float(group.external_utility.median()),
                "utility_reference_oracle": 0.0,
                "utility_reference_random": -1.0,
                "internal_pass": internal_pass,
                "external_pass": external_pass,
                "quadrant": quadrant,
                "independent_units": group.independent_unit_id.nunique(),
            }
        )
    return unit, pd.DataFrame(associations), pd.DataFrame(cards)


def correct_robustness() -> tuple[pd.DataFrame, pd.DataFrame]:
    robustness = pd.read_csv(ROOT / "11_E7_robustness" / "robustness_comparison.tsv", sep="\t")
    robustness = robustness[robustness.method.isin(METHODS)].copy()
    current = robustness.exact_top1_nex_aurc
    primary = robustness.primary_exact_top1_nex_aurc
    robustness["same_direction_relative_to_random"] = np.sign(1.0 - current) == np.sign(
        1.0 - primary
    )
    robustness["delta_nex_from_primary"] = current - primary
    robustness["absolute_delta_nex_from_primary"] = np.abs(
        robustness.delta_nex_from_primary
    )
    summary = (
        robustness.groupby(["method", "variant"], dropna=False)
        .agg(
            units=("dataset", "size"),
            median_nex=("exact_top1_nex_aurc", "median"),
            median_delta_nex=("delta_nex_from_primary", "median"),
            median_absolute_delta_nex=("absolute_delta_nex_from_primary", "median"),
            direction_stability_relative_to_random=("same_direction_relative_to_random", "mean"),
        )
        .reset_index()
    )
    return robustness, summary


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    semi_fidelity = add_design_columns(
        pd.read_csv(SEMI / "internal_fidelity_all.tsv", sep="\t"), "semisynthetic"
    )
    real_fidelity = add_design_columns(
        pd.read_csv(REAL / "internal_fidelity_pair_averaged.tsv", sep="\t"), "real"
    )
    fidelity = pd.concat([semi_fidelity, real_fidelity], ignore_index=True, sort=False)
    for column, default in [
        ("raw_mae", np.nan),
        ("reference_mad", np.nan),
        ("normalized_mae_estimable", False),
    ]:
        if column not in fidelity:
            fidelity[column] = default
    fidelity_units, fidelity_unit_sensitivity = summarize_fidelity(fidelity)
    fidelity_summary = summarize_fidelity_pair_gate(fidelity)
    fidelity.to_csv(OUTPUT / "fidelity_analysis_rows.tsv", sep="\t", index=False)
    fidelity_units.to_csv(OUTPUT / "fidelity_independent_units.tsv", sep="\t", index=False)
    fidelity_summary.to_csv(OUTPUT / "fidelity_summary_corrected.tsv", sep="\t", index=False)
    fidelity_summary.to_csv(OUTPUT / "fidelity_gate_pair_level_corrected.tsv", sep="\t", index=False)
    fidelity_unit_sensitivity.to_csv(
        OUTPUT / "fidelity_independent_unit_sensitivity.tsv", sep="\t", index=False
    )

    semi_utility = add_design_columns(
        pd.read_csv(SEMI / "external_utility_all.tsv", sep="\t"), "semisynthetic"
    )
    real_utility = add_design_columns(
        pd.read_csv(REAL / "external_utility_tie_aware.tsv", sep="\t"), "real"
    )
    utility = pd.concat([semi_utility, real_utility], ignore_index=True, sort=False)
    direction_averaged = direction_average_utility(utility)
    comparison = baseline_comparison(direction_averaged)
    external_units, external_summary = summarize_external(comparison)
    direction_averaged.to_csv(OUTPUT / "utility_direction_averaged.tsv", sep="\t", index=False)
    comparison.to_csv(OUTPUT / "utility_baseline_comparison.tsv", sep="\t", index=False)
    external_units.to_csv(OUTPUT / "external_independent_units.tsv", sep="\t", index=False)
    external_summary.to_csv(OUTPUT / "external_summary_corrected.tsv", sep="\t", index=False)

    control_pair, control_summary = summarize_controls(
        REAL / "controls_tie_aware_spatial_cluster.tsv"
    )
    control_pair.to_csv(OUTPUT / "real_control_pair_averaged.tsv", sep="\t", index=False)
    control_summary.to_csv(OUTPUT / "real_control_summary_corrected.tsv", sep="\t", index=False)
    real_external_with_controls = external_summary[external_summary.source == "real"].merge(
        control_summary,
        on=["method", "witness", "score"],
        how="left",
    )
    real_external_with_controls["corrected_external_gate_with_controls"] = (
        real_external_with_controls.corrected_external_gate
        & (real_external_with_controls.negative_control_pass_fraction >= 2 / 3)
        & (real_external_with_controls.leakage_control_pass_fraction == 1)
        & (real_external_with_controls.adjusted_positive_fraction >= 2 / 3)
    )
    real_external_with_controls.to_csv(
        OUTPUT / "real_external_gate_with_controls.tsv", sep="\t", index=False
    )

    relation_units, associations, cards = build_relation(
        fidelity, direction_averaged, fidelity_summary, external_summary
    )
    relation_units.to_csv(OUTPUT / "fidelity_utility_matched_units.tsv", sep="\t", index=False)
    associations.to_csv(OUTPUT / "fidelity_utility_association_descriptive.tsv", sep="\t", index=False)
    cards.to_csv(OUTPUT / "validity_cards_intervention_witness.tsv", sep="\t", index=False)

    robustness, robustness_summary = correct_robustness()
    robustness.to_csv(OUTPUT / "robustness_corrected.tsv", sep="\t", index=False)
    robustness_summary.to_csv(OUTPUT / "robustness_summary_corrected.tsv", sep="\t", index=False)

    legacy_gate = pd.read_csv(
        ROOT / "12_E8_statistics" / "registered_real_external_gate.tsv", sep="\t"
    )
    legacy_gate.to_csv(OUTPUT / "registered_v1_2_gate_immutable.tsv", sep="\t", index=False)
    exploratory = external_summary[
        (external_summary.source == "real")
        & (external_summary.method == "uot")
        & external_summary.score.isin(
            ["exact_I_EXPR", "finite_difference_I_EXPR_h001"]
        )
    ].copy()
    exploratory["claim_status"] = "post_hoc_exploratory_requires_independent_replication"
    exploratory.to_csv(OUTPUT / "uot_expression_hypothesis_exploratory.tsv", sep="\t", index=False)

    decision = status_payload(
        "V1_3_STATISTICS",
        "COMPLETED",
        confirmatory_ot_methods=CORRECTION["confirmatory_ot_methods"],
        non_ot_stress_test=CORRECTION["non_ot_stress_test"],
        excluded_pending_rerun=CORRECTION["exploratory_excluded_pending_rerun"],
        real_independent_units=int(
            real_fidelity.independent_unit_id.nunique()
        ),
        semisynthetic_independent_units=int(
            semi_fidelity.independent_unit_id.nunique()
        ),
        corrected_external_gate_passes=int(external_summary.corrected_external_gate.sum()),
        corrected_external_gate_passes_scope="all sources, witnesses, and candidate scores; sensitivity only",
        corrected_real_external_gate_passes=int(
            external_summary[
                (external_summary.source == "real")
                & external_summary.corrected_external_gate
            ].shape[0]
        ),
        corrected_real_external_gate_passes_scope="all real witnesses and candidate scores before controls; sensitivity only",
        corrected_real_external_gate_with_controls_passes=int(
            real_external_with_controls.corrected_external_gate_with_controls.sum()
        ),
        corrected_real_exact_combined_gate_with_controls_passes=int(
            real_external_with_controls[
                (real_external_with_controls.score == "exact_combined")
                & real_external_with_controls.corrected_external_gate_with_controls
            ].shape[0]
        ),
        registered_v1_2_gate_passes=int(legacy_gate.registered_real_external_gate.sum()),
        primary_interpretation=(
            "registered combined Gate remains negative; three post-review, intervention-specific "
            "real-data signals pass corrective controls and remain exploratory"
        ),
        tie_handling="fractional",
        best_qc_role="hindsight_oracle_envelope_only",
        inference="independent-unit equal with leave-one-unit-out sensitivity",
    )
    write_json(OUTPUT / "V1_3_STATISTICS_DECISION.json", decision)
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
