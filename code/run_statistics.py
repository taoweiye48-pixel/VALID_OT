from __future__ import annotations

import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import spearmanr, wilcoxon
from statsmodels.stats.multitest import multipletests

from validot.utils import read_json, status_payload, write_json


ROOT = Path(__file__).resolve().parents[1]
CONFIG = read_json(ROOT / "00_protocol" / "frozen_config.json")
OUTPUT = ROOT / "12_E8_statistics"
SCENARIOS = CONFIG["semisynthetic"]["scenarios"]
QC_SCORES = {"source_boundary_proximity", "source_sparsity", "matched_target_sparsity"}
PROXY_TO_UTILITY = {
    "assigned_raw_cost": "assigned_raw_cost",
    "local_cost_margin": "local_cost_margin",
    "conditional_entropy": "conditional_entropy",
    "low_max_probability": "low_max_probability",
    "probability_margin": "probability_margin",
    "mass_deficit": "mass_deficit",
    "endpoint_gradient": "endpoint_combined",
}


def technology_name(dataset: str) -> str:
    return re.sub(r"_base\d+$", "", str(dataset))


def scenario_name(pair_id: str) -> str:
    for scenario in SCENARIOS:
        if f"_{scenario}_seed" in str(pair_id):
            return scenario
    return "real"


def normalize_table(table: pd.DataFrame, source: str) -> pd.DataFrame:
    result = table.copy()
    result["source"] = source
    result["technology"] = result.dataset.astype(str).map(technology_name)
    if "pair_type" not in result:
        result["pair_type"] = "semisynthetic" if source == "semisynthetic" else "unknown"
    result["pair_type"] = result["pair_type"].fillna(
        "semisynthetic" if source == "semisynthetic" else "unknown"
    )
    result.loc[result.pair_type.astype(str).str.len() == 0, "pair_type"] = (
        "semisynthetic" if source == "semisynthetic" else "unknown"
    )
    if "biological_pair_id" not in result:
        result["biological_pair_id"] = result.pair_id
    result["biological_pair_id"] = result["biological_pair_id"].fillna(result.pair_id)
    if "direction" not in result:
        result["direction"] = "synthetic" if source == "semisynthetic" else "forward"
    result["direction"] = result["direction"].fillna(
        "synthetic" if source == "semisynthetic" else "forward"
    )
    result["scenario"] = result.pair_id.astype(str).map(scenario_name)
    result["independent_unit_id"] = result.biological_pair_id.astype(str)
    if source == "semisynthetic":
        result["independent_unit_id"] = result.dataset.astype(str)
    else:
        cross_stage = result.pair_type.astype(str) == "cross_stage"
        result.loc[cross_stage, "independent_unit_id"] = (
            result.loc[cross_stage, "technology"].astype(str) + "::shared_developmental_series"
        )
    return result


def fix_degenerate_aurc(table: pd.DataFrame) -> pd.DataFrame:
    result = table.copy()
    degenerate = np.abs(result.random_aurc - result.oracle_aurc) <= 1e-12
    result.loc[degenerate, "normalized_excess_aurc"] = np.nan
    result["aurc_estimable"] = ~degenerate
    return result


def paired_hierarchical_bootstrap(
    table: pd.DataFrame,
    value: str,
    repeats: int,
    seed: int,
    statistic=np.median,
) -> dict[str, float | int]:
    clean = table.dropna(subset=[value]).copy()
    if clean.empty:
        return {"estimate": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "bootstrap_valid": 0}
    rng = np.random.default_rng(seed)
    technologies = clean.technology.unique()
    estimates = []
    for _ in range(repeats):
        sampled_values = []
        for technology in rng.choice(technologies, size=len(technologies), replace=True):
            tech = clean[clean.technology == technology]
            units = tech.independent_unit_id.unique()
            for unit_id in rng.choice(units, size=len(units), replace=True):
                sampled_values.extend(tech.loc[tech.independent_unit_id == unit_id, value].tolist())
        if sampled_values:
            estimates.append(float(statistic(np.asarray(sampled_values, dtype=float))))
    return {
        "estimate": float(statistic(clean[value].to_numpy(float))),
        "ci_low": float(np.quantile(estimates, 0.025)),
        "ci_high": float(np.quantile(estimates, 0.975)),
        "bootstrap_valid": len(estimates),
    }


def association_bootstrap(table: pd.DataFrame, repeats: int, seed: int) -> dict[str, float | int]:
    clean = table.dropna(subset=["spearman", "external_utility"]).copy()
    observed = spearmanr(clean.spearman, clean.external_utility).statistic if len(clean) > 2 else np.nan
    rng = np.random.default_rng(seed)
    estimates = []
    technologies = clean.technology.unique()
    for _ in range(repeats):
        pieces = []
        for technology in rng.choice(technologies, size=len(technologies), replace=True):
            tech = clean[clean.technology == technology]
            units = tech.independent_unit_id.unique()
            for draw, unit_id in enumerate(rng.choice(units, size=len(units), replace=True)):
                piece = tech[tech.independent_unit_id == unit_id].copy()
                piece["bootstrap_pair"] = f"{technology}::{draw}"
                pieces.append(piece)
        if pieces:
            boot = pd.concat(pieces, ignore_index=True)
            rho = spearmanr(boot.spearman, boot.external_utility).statistic
            if np.isfinite(rho):
                estimates.append(float(rho))
    return {
        "estimate": float(observed),
        "ci_low": float(np.quantile(estimates, 0.025)) if estimates else float("nan"),
        "ci_high": float(np.quantile(estimates, 0.975)) if estimates else float("nan"),
        "bootstrap_valid": len(estimates),
    }


def fidelity_summary(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["source", "method", "intervention", "proxy"]
    for group_keys, group in table.groupby(keys, dropna=False):
        nmae = group.normalized_mae.dropna()
        values = group.spearman.dropna().to_numpy(float)
        if len(values) and not np.allclose(values, 0):
            try:
                pvalue = float(wilcoxon(values).pvalue)
            except ValueError:
                pvalue = 1.0
        else:
            pvalue = 1.0
        median_spearman = float(np.nanmedian(group.spearman))
        median_top = float(np.nanmedian(group.top_decile_precision))
        median_nmae = float(np.nanmedian(nmae)) if len(nmae) else float("nan")
        block_values = (
            group.spatial_block_same_sign_fraction.dropna()
            if "spatial_block_same_sign_fraction" in group
            else pd.Series(dtype=float)
        )
        rank_pass = bool(
            median_spearman >= CONFIG["fidelity_gate"]["spearman_min"]
            and median_top >= CONFIG["fidelity_gate"]["top_decile_precision_min"]
        )
        full_pass = bool(
            rank_pass
            and len(nmae) > 0
            and median_nmae <= CONFIG["fidelity_gate"]["normalized_mae_max"]
        )
        rows.append(
            {
                "source": group_keys[0],
                "method": group_keys[1],
                "intervention": group_keys[2],
                "proxy": group_keys[3],
                "units": len(group),
                "biological_or_seed_units": group.biological_pair_id.nunique(),
                "technologies": group.technology.nunique(),
                "median_spearman": median_spearman,
                "median_top_decile_precision": median_top,
                "median_normalized_mae": median_nmae,
                "normalized_mae_available_units": len(nmae),
                "median_spatial_block_same_sign_fraction": (
                    float(block_values.median()) if len(block_values) else float("nan")
                ),
                "positive_spearman_fraction": float(np.mean(values > 0)) if len(values) else float("nan"),
                "spearman_wilcoxon_p": pvalue,
                "rank_fidelity_pass": rank_pass,
                "full_registered_fidelity_pass": full_pass,
            }
        )
    summary = pd.DataFrame(rows)
    summary["spearman_fdr_bh"] = multipletests(summary.spearman_wilcoxon_p, method="fdr_bh")[1]
    return summary


def direction_average_utility(table: pd.DataFrame) -> pd.DataFrame:
    return (
        table.groupby(
            [
                "source",
                "technology",
                "dataset",
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
            normalized_excess_aurc=("normalized_excess_aurc", "mean"),
            directions=("direction", "nunique"),
            aurc_estimable=("aurc_estimable", "all"),
        )
        .reset_index()
    )


def utility_gain_units(direction_averaged: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_keys = [
        "source",
        "technology",
        "dataset",
        "pair_type",
        "biological_pair_id",
        "independent_unit_id",
        "method",
        "witness",
    ]
    for keys, group in direction_averaged.groupby(group_keys, dropna=False):
        qc = group[group.score.isin(QC_SCORES)].dropna(subset=["normalized_excess_aurc"])
        if qc.empty:
            continue
        best_qc = float(qc.normalized_excess_aurc.min())
        for row in group.itertuples(index=False):
            if row.score in QC_SCORES or not np.isfinite(row.normalized_excess_aurc):
                continue
            absolute = best_qc - float(row.normalized_excess_aurc)
            rows.append(
                {
                    "source": keys[0],
                    "technology": keys[1],
                    "dataset": keys[2],
                    "pair_type": keys[3],
                    "biological_pair_id": keys[4],
                    "independent_unit_id": keys[5],
                    "method": keys[6],
                    "witness": keys[7],
                    "score": row.score,
                    "directions": row.directions,
                    "score_normalized_excess_aurc": row.normalized_excess_aurc,
                    "best_qc_normalized_excess_aurc": best_qc,
                    "absolute_gain_over_best_qc": absolute,
                    "relative_gain_over_best_qc": absolute / max(abs(best_qc), 1e-12),
                    "positive_direction": bool(absolute > 0),
                }
            )
    return pd.DataFrame(rows)


def external_summary(gains: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in gains.groupby(["source", "method", "witness", "score"], dropna=False):
        bootstrap = paired_hierarchical_bootstrap(
            group,
            "absolute_gain_over_best_qc",
            CONFIG["external_gate"]["bootstrap_replicates"],
            CONFIG["preprocessing"]["seed"],
        )
        technology_direction = group.groupby("technology").absolute_gain_over_best_qc.median()
        independent_direction = group.groupby("independent_unit_id").absolute_gain_over_best_qc.median()
        rows.append(
            {
                "source": keys[0],
                "method": keys[1],
                "witness": keys[2],
                "score": keys[3],
                "pairs": group.biological_pair_id.nunique(),
                "independent_units": group.independent_unit_id.nunique(),
                "technologies": group.technology.nunique(),
                "median_absolute_gain": float(group.absolute_gain_over_best_qc.median()),
                "median_relative_gain": float(group.relative_gain_over_best_qc.median()),
                "positive_pair_fraction": float(group.positive_direction.mean()),
                "positive_independent_unit_fraction": float((independent_direction > 0).mean()),
                "positive_technology_count": int((technology_direction > 0).sum()),
                "bootstrap_ci_low": bootstrap["ci_low"],
                "bootstrap_ci_high": bootstrap["ci_high"],
                "generic_external_gate": bool(
                    group.relative_gain_over_best_qc.median()
                    >= CONFIG["external_gate"]["median_relative_aurc_improvement_min"]
                    and (technology_direction > 0).sum()
                    >= CONFIG["external_gate"]["minimum_positive_technology_count"]
                    and group.positive_direction.mean()
                    >= CONFIG["external_gate"]["minimum_positive_pair_fraction"]
                    and (independent_direction > 0).mean()
                    >= CONFIG["external_gate"]["minimum_positive_pair_fraction"]
                    and bootstrap["ci_low"] >= 0
                ),
            }
        )
    return pd.DataFrame(rows)


def summarize_controls(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not path.exists():
        return pd.DataFrame(), pd.DataFrame()
    controls = pd.read_csv(path, sep="\t")
    averaged = (
        controls.groupby(
            ["dataset", "pair_type", "biological_pair_id", "method", "witness", "score"],
            dropna=False,
        )
        .agg(
            directions=("direction", "nunique"),
            primary=("primary_normalized_excess_aurc", "mean"),
            label_shuffle=("label_shuffle_median", "mean"),
            label_shuffle_p=("label_shuffle_p_lower", "max"),
            within_block=("within_block_permutation_median", "mean"),
            within_block_p=("within_block_p_lower", "max"),
            circular_shift=("circular_shift_normalized_excess_aurc", "mean"),
            leakage_positive=("leakage_positive_control_normalized_excess_aurc", "mean"),
            adjusted_positive=("risk_positive_after_adjustment", "mean"),
            adjusted_p=("risk_rank_pvalue", "max"),
        )
        .reset_index()
    )
    averaged["negative_controls_pass"] = (
        (averaged.primary < averaged.label_shuffle)
        & (averaged.primary < averaged.within_block)
        & (averaged.label_shuffle_p <= 0.05)
        & (averaged.within_block_p <= 0.05)
    )
    averaged["leakage_control_pass"] = np.abs(averaged.leakage_positive) <= 1e-8
    averaged["independent_unit_id"] = averaged.biological_pair_id.astype(str)
    cross_stage = averaged.pair_type.astype(str) == "cross_stage"
    averaged.loc[cross_stage, "independent_unit_id"] = "Stereo-seq::shared_developmental_series"
    unit_controls = (
        averaged.groupby(["method", "witness", "score", "independent_unit_id"], dropna=False)
        .agg(
            negative_controls_pass=("negative_controls_pass", "mean"),
            leakage_control_pass=("leakage_control_pass", "mean"),
            adjusted_positive=("adjusted_positive", "mean"),
            adjusted_p=("adjusted_p", "max"),
        )
        .reset_index()
    )
    summary = (
        unit_controls.groupby(["method", "witness", "score"], dropna=False)
        .agg(
            independent_units=("independent_unit_id", "nunique"),
            negative_control_pass_fraction=("negative_controls_pass", "mean"),
            leakage_control_pass_fraction=("leakage_control_pass", "mean"),
            adjusted_positive_fraction=("adjusted_positive", "mean"),
            adjusted_significant_fraction=("adjusted_p", lambda x: float(np.mean(np.asarray(x) <= 0.05))),
        )
        .reset_index()
    )
    return averaged, summary


def fit_clustered_model(semi_fidelity: pd.DataFrame) -> pd.DataFrame:
    model_data = semi_fidelity.dropna(subset=["spearman"]).copy()
    formula = "spearman ~ C(method) + C(intervention) + C(proxy) + C(scenario)"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = smf.ols(formula, data=model_data).fit(
            cov_type="cluster", cov_kwds={"groups": model_data.pair_id}
        )
    interval = fit.conf_int()
    return pd.DataFrame(
        {
            "term": fit.params.index,
            "coefficient": fit.params.values,
            "cluster_robust_se": fit.bse.values,
            "pvalue": fit.pvalues.values,
            "ci_low": interval.iloc[:, 0].values,
            "ci_high": interval.iloc[:, 1].values,
            "model": formula,
            "cluster": "semisynthetic_pair_seed",
        }
    )


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    semi_fidelity = normalize_table(
        pd.read_csv(ROOT / "08_E4_internal_fidelity" / "internal_fidelity_all.tsv", sep="\t"),
        "semisynthetic",
    )
    semi_utility = normalize_table(
        fix_degenerate_aurc(
            pd.read_csv(ROOT / "09_E5_semisynthetic_external" / "external_utility_all.tsv", sep="\t")
        ),
        "semisynthetic",
    )
    real_fidelity = normalize_table(
        pd.read_csv(ROOT / "10_E6_real_external" / "all_internal_fidelity.tsv", sep="\t"),
        "real",
    )
    real_utility = normalize_table(
        fix_degenerate_aurc(
            pd.read_csv(ROOT / "10_E6_real_external" / "all_external_utility.tsv", sep="\t")
        ),
        "real",
    )

    fidelity = pd.concat([semi_fidelity, real_fidelity], ignore_index=True)
    utility = pd.concat([semi_utility, real_utility], ignore_index=True)
    fidelity.to_csv(OUTPUT / "fidelity_units_all.tsv", sep="\t", index=False)
    utility.to_csv(OUTPUT / "utility_units_all.tsv", sep="\t", index=False)

    fidelity_results = fidelity_summary(fidelity)
    fidelity_results.to_csv(OUTPUT / "fidelity_summary.tsv", sep="\t", index=False)
    direction_averaged = direction_average_utility(utility)
    direction_averaged.to_csv(OUTPUT / "utility_direction_averaged.tsv", sep="\t", index=False)
    gains = utility_gain_units(direction_averaged)
    gains.to_csv(OUTPUT / "utility_gain_units.tsv", sep="\t", index=False)
    external_results = external_summary(gains)
    external_results.to_csv(OUTPUT / "external_summary.tsv", sep="\t", index=False)

    quality_path = ROOT / "10_E6_real_external" / "all_alignment_quality.tsv"
    quality_sensitivity = pd.DataFrame()
    if quality_path.exists():
        quality = pd.read_csv(quality_path, sep="\t")
        quality = (
            quality.groupby(["dataset", "pair_type", "biological_pair_id", "method"], dropna=False)
            .agg(
                region_matching_index=("region_matching_index", "mean"),
                mean_heldout_expression_loss=("mean_heldout_expression_loss", "mean"),
                directions=("direction", "nunique"),
            )
            .reset_index()
        )
        exact_gains = gains[(gains.source == "real") & (gains.score == "exact_combined")]
        quality_units = exact_gains.merge(
            quality,
            on=["dataset", "pair_type", "biological_pair_id", "method"],
            how="left",
        )
        quality_units["poor_base_alignment"] = np.where(
            quality_units.witness == "label_error",
            1.0 - quality_units.region_matching_index,
            quality_units.mean_heldout_expression_loss,
        )
        rows = []
        for keys, group in quality_units.groupby(["method", "witness"], dropna=False):
            valid = group.dropna(subset=["poor_base_alignment", "absolute_gain_over_best_qc"])
            rho = (
                spearmanr(valid.poor_base_alignment, valid.absolute_gain_over_best_qc).statistic
                if len(valid) >= 3 and valid.poor_base_alignment.nunique() > 1
                else float("nan")
            )
            rows.append(
                {
                    "method": keys[0],
                    "witness": keys[1],
                    "pairs": valid.biological_pair_id.nunique(),
                    "spearman_gain_vs_poor_base_alignment": float(rho),
                }
            )
        quality_sensitivity = pd.DataFrame(rows)
        quality_units.to_csv(OUTPUT / "base_quality_units.tsv", sep="\t", index=False)
        quality_sensitivity.to_csv(OUTPUT / "base_quality_sensitivity.tsv", sep="\t", index=False)

    control_units, control_summary = summarize_controls(ROOT / "10_E6_real_external" / "all_controls.tsv")
    control_units.to_csv(OUTPUT / "real_control_units.tsv", sep="\t", index=False)
    control_summary.to_csv(OUTPUT / "real_control_summary.tsv", sep="\t", index=False)

    real_exact_gate = external_results[
        (external_results.source == "real") & (external_results.score == "exact_combined")
    ].copy()
    if len(control_summary):
        exact_controls = control_summary[control_summary.score == "exact_combined"]
        real_exact_gate = real_exact_gate.merge(
            exact_controls,
            on=["method", "witness", "score"],
            how="left",
        )
        real_exact_gate["registered_real_external_gate"] = (
            real_exact_gate.generic_external_gate
            & (real_exact_gate.negative_control_pass_fraction >= 2 / 3)
            & (real_exact_gate.leakage_control_pass_fraction == 1)
            & (real_exact_gate.adjusted_positive_fraction >= 2 / 3)
        )
    else:
        real_exact_gate["registered_real_external_gate"] = False
    real_exact_gate.to_csv(OUTPUT / "registered_real_external_gate.tsv", sep="\t", index=False)

    fidelity_pair = (
        fidelity.groupby(
            ["source", "technology", "biological_pair_id", "independent_unit_id", "method", "proxy"], dropna=False
        )
        .agg(spearman=("spearman", "median"), top_decile_precision=("top_decile_precision", "median"))
        .reset_index()
    )
    fidelity_pair["score"] = fidelity_pair.proxy.map(PROXY_TO_UTILITY)
    utility_pair = (
        direction_averaged.groupby(
            ["source", "technology", "biological_pair_id", "independent_unit_id", "method", "score"], dropna=False
        )
        .normalized_excess_aurc.median()
        .reset_index()
    )
    relation = fidelity_pair.dropna(subset=["score"]).merge(
        utility_pair,
        on=["source", "technology", "biological_pair_id", "independent_unit_id", "method", "score"],
        how="inner",
    )
    relation["external_utility"] = -relation.normalized_excess_aurc
    relation.to_csv(OUTPUT / "fidelity_utility_units.tsv", sep="\t", index=False)
    association_rows = []
    for source, group in relation.groupby("source"):
        association_rows.append(
            {
                "source": source,
                "units": len(group),
                "pairs": group.biological_pair_id.nunique(),
                "independent_units": group.independent_unit_id.nunique(),
                "technologies": group.technology.nunique(),
                **association_bootstrap(
                    group,
                    CONFIG["external_gate"]["bootstrap_replicates"],
                    CONFIG["preprocessing"]["seed"],
                ),
            }
        )
    association = pd.DataFrame(association_rows)
    association.to_csv(OUTPUT / "fidelity_utility_association.tsv", sep="\t", index=False)

    cards = []
    for keys, group in relation.groupby(["source", "method", "proxy"], dropna=False):
        fid = fidelity_results[
            (fidelity_results.source == keys[0])
            & (fidelity_results.method == keys[1])
            & (fidelity_results.proxy == keys[2])
        ]
        external = external_results[
            (external_results.source == keys[0])
            & (external_results.method == keys[1])
            & (external_results.score == PROXY_TO_UTILITY.get(keys[2]))
        ]
        internal_pass = bool(len(fid) and fid.full_registered_fidelity_pass.all())
        rank_pass = bool(len(fid) and fid.rank_fidelity_pass.all())
        external_pass = bool(len(external) and external.generic_external_gate.all())
        quadrant = ("A" if external_pass else "B") if internal_pass else ("C" if external_pass else "D")
        cards.append(
            {
                "source": keys[0],
                "method": keys[1],
                "proxy": keys[2],
                "median_spearman": float(group.spearman.median()),
                "median_external_utility": float(group.external_utility.median()),
                "rank_fidelity_pass": rank_pass,
                "full_fidelity_pass": internal_pass,
                "external_gate_pass": external_pass,
                "quadrant": quadrant,
                "units": len(group),
            }
        )
    cards = pd.DataFrame(cards)
    cards.to_csv(OUTPUT / "validity_cards.tsv", sep="\t", index=False)

    model = fit_clustered_model(semi_fidelity)
    model.to_csv(OUTPUT / "clustered_fidelity_model.tsv", sep="\t", index=False)

    robustness_path = ROOT / "11_E7_robustness" / "robustness_comparison.tsv"
    if robustness_path.exists():
        robustness = pd.read_csv(robustness_path, sep="\t")
        robustness_summary = (
            robustness.groupby(["method", "variant"], dropna=False)
            .agg(
                units=("dataset", "size"),
                median_spearman=("median_spearman", "median"),
                median_external=("exact_top1_nex_aurc", "median"),
                same_direction_fraction=("same_external_direction_as_primary", "mean"),
            )
            .reset_index()
        )
        robustness_summary.to_csv(OUTPUT / "robustness_summary.tsv", sep="\t", index=False)

    decision = status_payload(
        "E8",
        "COMPLETED",
        fidelity_units=len(fidelity),
        utility_units=len(utility),
        direction_averaged_units=len(direction_averaged),
        relation_units=len(relation),
        cards=len(cards),
        registered_real_external_passes=int(real_exact_gate.registered_real_external_gate.sum()),
        bootstrap_replicates=CONFIG["external_gate"]["bootstrap_replicates"],
    )
    write_json(OUTPUT / "E8_DECISION.json", decision)
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
