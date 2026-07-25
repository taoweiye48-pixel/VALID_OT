from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t")


def _clean(value):
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if pd.isna(value):
        return None
    return value


def grouped_summary(
    frame: pd.DataFrame,
    keys: list[str],
    metrics: list[str],
    unit_col: str = "independent_unit_id",
) -> list[dict]:
    rows: list[dict] = []
    for group_key, group in frame.groupby(keys, dropna=False, sort=True):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        row = {key: value for key, value in zip(keys, group_key)}
        row["rows"] = int(len(group))
        if unit_col in group.columns:
            row["independent_units"] = int(group[unit_col].nunique())
        for metric in metrics:
            if metric not in group.columns:
                continue
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            if values.dtype == bool:
                values = values.astype(float)
            prefix = metric
            row[f"{prefix}__n"] = int(len(values))
            if len(values):
                row[f"{prefix}__median"] = float(values.median())
                row[f"{prefix}__q25"] = float(values.quantile(0.25))
                row[f"{prefix}__q75"] = float(values.quantile(0.75))
                row[f"{prefix}__min"] = float(values.min())
                row[f"{prefix}__max"] = float(values.max())
        rows.append(row)
    return rows


def direction_to_unit(
    frame: pd.DataFrame,
    keys: list[str],
    metrics: list[str],
) -> pd.DataFrame:
    group_keys = ["independent_unit_id", *keys]
    existing_metrics = [metric for metric in metrics if metric in frame.columns]
    return (
        frame.groupby(group_keys, dropna=False, sort=True)[existing_metrics]
        .mean(numeric_only=True)
        .reset_index()
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    workspace = args.workspace.resolve()
    results = workspace / "results"
    wp = results / "postreview_wp1_wp10_v1"
    wp11 = results / "postreview_wp11_v1" / "wp11"

    evidence: dict[str, object] = {
        "workspace": str(workspace),
        "aggregation_rule": (
            "direction mean within biological pair; biological-pair median within "
            "independent unit where supplied by frozen result tables; cohort summaries "
            "remain descriptive"
        ),
    }

    # WP1 and WP2: numerical reference validation.
    evidence["wp1_full_gate"] = json.loads(
        (wp / "wp1" / "WP1_FULL_GATE.json").read_text(encoding="utf-8")
    )
    evidence["wp2_gate"] = json.loads(
        (wp / "wp2" / "WP2_GATE.json").read_text(encoding="utf-8")
    )
    wp2 = read_tsv(wp / "wp2" / "wp2_derivative_cross_validation.tsv")
    evidence["wp2_cross_validation"] = grouped_summary(
        wp2,
        ["method", "arm", "intervention"],
        [
            "global_plan_relative_l1",
            "row_relative_l1_median",
            "row_relative_l1_q90",
            "row_direction_cosine_median",
            "scalar_spearman",
            "scalar_rmae",
        ],
    )

    # WP3: actual h=0.01 score against the high-accuracy local reference.
    wp3_family = read_tsv(wp / "wp3" / "wp3_local_fidelity_family.tsv")
    evidence["wp3_local_fidelity_family"] = wp3_family.to_dict(orient="records")
    wp3_unit = read_tsv(wp / "wp3" / "wp3_local_fidelity_unit.tsv")
    evidence["wp3_local_fidelity_units"] = grouped_summary(
        wp3_unit,
        ["method", "arm", "intervention"],
        [
            "h001_spearman",
            "h001_top_overlap",
            "h001_rmae",
            "vector_relative_l1_median",
            "direction_cosine_median",
            "neighborhood_error_median",
            "neighborhood_error_q90",
            "gate_pass",
        ],
    )

    # WP4: local-to-endpoint transportability and path geometry.
    wp4 = read_tsv(wp / "wp4" / "wp4_path_geometry_unit.tsv")
    evidence["wp4_endpoint_transportability"] = grouped_summary(
        wp4,
        ["method", "arm", "intervention"],
        [
            "local_to_endpoint_spearman",
            "local_to_endpoint_top_overlap",
            "local_to_endpoint_rmae",
            "local_to_endpoint_slope",
            "h001_to_endpoint_spearman",
            "h001_to_endpoint_top_overlap",
            "h001_to_endpoint_rmae",
            "h001_to_endpoint_slope",
            "path_eta_median",
            "path_kappa_median",
        ],
    )

    # WP6: UOT mass/shape decomposition.
    wp6 = read_tsv(wp / "wp6" / "wp6_uot_mass_shape_unit.tsv")
    evidence["wp6_uot_mass_shape"] = grouped_summary(
        wp6,
        ["arm", "intervention"],
        [
            "mass_local_to_endpoint_spearman",
            "mass_local_to_endpoint_rmae",
            "shape_local_to_endpoint_spearman",
            "shape_local_to_endpoint_rmae",
            "local_mass_median",
            "local_shape_median",
            "endpoint_mass_median",
            "endpoint_shape_median",
        ],
    )

    # WP7: coordinate-frame sensitivity.
    wp7 = read_tsv(wp / "wp7" / "wp7_coordinate_frame_sensitivity_unit.tsv")
    evidence["wp7_coordinate_sensitivity"] = grouped_summary(
        wp7,
        ["method", "variant", "arm", "score"],
        [
            "spatial_cost_pearson_vs_baseline",
            "spatial_cost_scale_ratio",
            "local_fidelity_spearman",
            "local_fidelity_rmae",
            "endpoint_transfer_spearman",
            "endpoint_transfer_rmae",
            "heldout_normalized_excess_aurc",
        ],
    )

    # WP8: feature-split robustness.
    wp8 = read_tsv(wp / "wp8" / "wp8_heldout_split_robustness_unit.tsv")
    evidence["wp8_gene_split_robustness"] = grouped_summary(
        wp8,
        ["method", "split", "score"],
        [
            "local_fidelity_spearman",
            "local_fidelity_rmae",
            "endpoint_transfer_spearman",
            "endpoint_transfer_rmae",
            "heldout_normalized_excess_aurc",
        ],
    )

    # WP5: same-score external utility. Absolute NEX difference versus fixed QC
    # is computed within each independent unit before cohort-level summaries.
    wp5 = read_tsv(wp / "wp5" / "wp5_same_score_external_utility_unit.tsv")
    fixed = (
        wp5.loc[wp5["score"].eq("source_boundary_proximity")]
        .set_index(["independent_unit_id", "cohort_role", "method", "witness"])[
            "normalized_excess_aurc"
        ]
        .rename("fixed_qc_nex")
    )
    wp5 = wp5.join(
        fixed,
        on=["independent_unit_id", "cohort_role", "method", "witness"],
    )
    wp5["delta_nex_vs_fixed_qc"] = (
        wp5["fixed_qc_nex"] - wp5["normalized_excess_aurc"]
    )
    evidence["wp5_same_score_external_utility"] = grouped_summary(
        wp5,
        ["cohort_role", "method", "witness", "score"],
        ["normalized_excess_aurc", "delta_nex_vs_fixed_qc", "spearman"],
    )

    # WP9: audit realistic scores against internal references and external witnesses.
    wp9i = read_tsv(wp / "wp9" / "wp9_real_score_internal_audit_unit.tsv")
    evidence["wp9_real_score_internal"] = grouped_summary(
        wp9i,
        ["branch", "method", "reference", "score", "amplitude_comparable"],
        ["spearman", "top_overlap", "rmae", "slope", "r2"],
    )
    wp9e = read_tsv(wp / "wp9" / "wp9_real_score_external_audit_unit.tsv")
    evidence["wp9_real_score_external"] = grouped_summary(
        wp9e,
        ["branch", "method", "witness", "score"],
        ["spearman", "normalized_excess_aurc"],
    )

    # WP10: controlled correspondence and crop-missingness truth.
    wp10c = read_tsv(wp / "wp10" / "wp10_her2st_correspondence_truth_direction.tsv")
    baseline_plan = wp10c.loc[wp10c["metric"].isna()].copy()
    baseline_unit = direction_to_unit(
        baseline_plan,
        ["method"],
        [
            "top1",
            "top5",
            "top10",
            "median_true_probability",
            "median_reciprocal_rank",
            "median_normalized_barycentric_error",
        ],
    )
    evidence["wp10_baseline_correspondence"] = grouped_summary(
        baseline_unit,
        ["method"],
        [
            "top1",
            "top5",
            "top10",
            "median_true_probability",
            "median_reciprocal_rank",
            "median_normalized_barycentric_error",
        ],
    )
    scored_correspondence = wp10c.loc[wp10c["metric"].notna()].copy()
    scored_corr_unit = direction_to_unit(
        scored_correspondence,
        ["method", "metric", "score"],
        ["normalized_excess_aurc", "auroc", "auprc", "spearman"],
    )
    evidence["wp10_correspondence_score_utility"] = grouped_summary(
        scored_corr_unit,
        ["method", "metric", "score"],
        ["normalized_excess_aurc", "auroc", "auprc", "spearman"],
    )
    wp10m = read_tsv(wp / "wp10" / "wp10_crop_missingness_utility_direction.tsv")
    wp10m_unit = direction_to_unit(
        wp10m,
        ["method", "score"],
        [
            "normalized_excess_aurc",
            "auroc",
            "auprc",
            "precision_at_prevalence",
            "recall_at_prevalence",
        ],
    )
    evidence["wp10_crop_missingness"] = grouped_summary(
        wp10m_unit,
        ["method", "score"],
        [
            "normalized_excess_aurc",
            "auroc",
            "auprc",
            "precision_at_prevalence",
            "recall_at_prevalence",
        ],
    )

    # Downstream score-selection case: fixed-budget pseudo-positive quality and
    # explicit validity gates for any stronger representation-learning claim.
    downstream_quality_path = (
        results / "postreview_downstream_v1" / "positive_pair_quality_summary.tsv"
    )
    downstream_decision_path = (
        results / "postreview_downstream_decision_v1" / "downstream_decision.json"
    )
    if downstream_quality_path.is_file():
        downstream_quality = read_tsv(downstream_quality_path)
        evidence["downstream_fixed_budget_pair_quality"] = downstream_quality.to_dict(
            orient="records"
        )
    if downstream_decision_path.is_file():
        evidence["downstream_claim_gate"] = json.loads(
            downstream_decision_path.read_text(encoding="utf-8")
        )

    # WP11: sparse response surface and objective decomposition.
    wp11_gate = json.loads((wp11 / "WP11_GATE.json").read_text(encoding="utf-8"))
    evidence["wp11_gate"] = wp11_gate
    wp11_surface = read_tsv(wp11 / "wp11_alpha_beta_surface_unit.tsv")
    evidence["wp11_surface_selected"] = grouped_summary(
        wp11_surface.loc[
            wp11_surface["condition_family"].isin(["uv_grid", "factorial_extra"])
        ],
        ["method", "condition_family", "u", "v", "regularization_regime"],
        [
            "endpoint_response_median",
            "h001_response_median",
            "h001_to_endpoint_spearman",
            "h001_to_endpoint_rmae",
            "heldout_endpoint_normalized_excess_aurc",
            "heldout_h001_normalized_excess_aurc",
            "truth_endpoint_normalized_excess_aurc",
            "truth_h001_normalized_excess_aurc",
            "crop_endpoint_auroc",
            "crop_h001_auroc",
        ],
    )
    wp11_factor = read_tsv(wp11 / "wp11_factorial_contrasts_unit.tsv")
    evidence["wp11_factorial_contrasts"] = grouped_summary(
        wp11_factor,
        ["method", "channel", "regularization_regime", "metric"],
        [
            "baseline",
            "delta_removal",
            "delta_compensation",
            "delta_joint",
            "delta_interaction",
        ],
    )
    wp11_scale = read_tsv(wp11 / "wp11_scale_regularization_contrasts_unit.tsv")
    evidence["wp11_scale_regularization_contrasts"] = grouped_summary(
        wp11_scale,
        [
            "method",
            "contrast_type",
            "regularization_regime",
            "u",
            "v",
            "metric",
        ],
        ["reference_value", "comparison_value", "contrast"],
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(_clean(evidence), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
