# -*- coding: utf-8 -*-
"""VALID-OT paper-analysis assembly.

This project is an assets->paper pipeline. The authoritative numerical values
are FROZEN in ../results.json and ../RESULTS.md (protocol
valid-ot-v1.3-post-review-correction-2026-07-16, 26/26 verification checks).

This script does NOT recompute or simulate anything. It reads the frozen
result sheet and reshapes the verbatim values into per-figure JSON blocks
that the downstream paper-figure step consumes. Every number written here is
traceable to results.json / RESULTS.md; nothing is invented.
"""

# bootstrap module path (sibling imports independent of caller cwd)
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import json

ROOT = os.path.dirname(_HERE)                 # workspace root
FIG_DIR = os.path.join(ROOT, "figures")
FROZEN = os.path.join(ROOT, "results.json")


def load_frozen():
    with open(FROZEN, "r", encoding="utf-8") as fh:
        return json.load(fh)


def build_fig_fidelity_intervention(R):
    """Fig 2: grouped bar of internal fidelity by method x intervention."""
    gate = R["fidelity_gate"]
    expr = R["real_expression_fidelity"]
    methods = [r["method"] for r in expr]
    return {
        "figure_id": "fig_fidelity_intervention",
        "gate_thresholds": {
            "spearman_min": gate["spearman_min"],
            "top_decile_overlap_min": gate["top_decile_overlap_min"],
            "normalized_mae_max": gate["normalized_mae_max"],
        },
        "methods": methods,
        "expression_intervention": expr,
        "spatial_intervention": {
            "full_fidelity_passes": R["spatial_finite_difference_full_fidelity_passes"],
            "median_spatial_nmae_lower_bound_ot": 2.45,
            "note": "Every real spatial finite-difference condition FAILS the fidelity Gate.",
        },
        "panel_A_metrics": ["median_spearman", "median_top_decile_overlap"],
        "panel_B_metric": "median_normalized_mae",
    }


def build_fig_external_utility(R):
    """Fig 3: registered negative first, then 3 exploratory NEX-AURC signals."""
    return {
        "figure_id": "fig_external_utility",
        "registered_negative": {
            "registered_v1_2_external_gate_passes": R["registered_v1_2_external_gate_passes"],
            "corrected_real_exact_combined_gate_with_controls_passes":
                R["corrected_real_exact_combined_gate_with_controls_passes"],
            "note": "Report FIRST: registered combined external Gate = 0.",
        },
        "exploratory_signals": R["exploratory_real_external_signals"],
        "nex_aurc_reference": {"oracle": 0.0, "random": 1.0, "direction": "lower_is_better"},
        "n_independent_units": R["study_scale"]["real_independent_units"],
        "caveat": "Post-review exploratory; 3 units only; independent prospective replication required.",
    }


def build_fig_fidelity_vs_utility(R):
    """Fig 4: scatter of internal fidelity vs external ranking + anti-informative panel."""
    # Non-equivalence counterexample values are recorded verbatim in RESULTS.md.
    non_equivalence = {
        "comparator": "fixed_coupling_group_cost_WaX_inspired",
        "balanced_ot_median_spearman": 0.102,
        "uot_median_spearman": 0.462,
        "median_nex_aurc_heldout_expression": 0.25,
        "interpretation": "Weak internal fidelity yet strong external ranking: counterexample to equivalence. "
                          "No orthogonality/statistical-independence claimed.",
    }
    anti = R["anti_informative_exact_response"]
    anti_detail = {
        "summary_conditions": anti["summary_conditions"],
        "conditions_median_worse_than_random": anti["conditions_median_worse_than_random"],
        "exact_expr_vs_heldout_expr_nex": {"balanced_ot": 0.759, "uot": 0.511},
        "exact_expr_vs_shared_label_mismatch_nex": {
            "balanced_ot": 1.379, "uot": 1.358, "row_softmax": 1.305},
        "exact_spatial_combined_vs_source_only_open_set_nex": {
            "balanced_ot": 1.17, "uot": 1.17, "row_softmax": 1.17},
        "row_softmax_exact_expr_vs_heldout_nex": 1.407,
    }
    return {
        "figure_id": "fig_fidelity_vs_utility",
        "quadrant_guides": {"fidelity_axis": "median_spearman", "utility_axis": "median_nex_aurc",
                            "random_line_nex": 1.0},
        "non_equivalence_counterexample": non_equivalence,
        "anti_informative": anti_detail,
    }


def build_fig_sensitivity_boundary(R):
    """Fig 5: FD step sweep + coordinate-frame sensitivity (verbatim from RESULTS.md)."""
    return {
        "figure_id": "fig_sensitivity_boundary",
        "finite_difference_step": {
            "steps": R["parameters"]["endpoint_steps_sensitivity"],
            "primary_step": R["parameters"]["endpoint_step_primary"],
            "n_rows": 72,
            "expression_ranking_stable": True,
            "spatial_magnitude_rescued": False,
        },
        "label_agnostic_sampling": {"n_tasks": 18, "type": "selection_bias_sensitivity_only"},
        "coordinate_frame": {
            "n_tasks": 36, "n_pairs": 2, "n_methods": 3, "n_variants": 6, "n": 800,
            "raw_rotation_reflection_delta_nex": 0.10,
            "canonicalization": "PCA/Chamfer label-free canonicalization collapses variants to baseline",
        },
        "retrospective_dlpfc": {
            "exact_reference_stable": True,
            "real_error_gate": "NO_GO",
            "missing_data_gate": "NO_GO",
            "note": "Retrospective context only; not prospective replication.",
        },
    }


def build_fig_validity_cards(R):
    """Fig 6: per-condition validity cards (Gate/claim-status grid)."""
    cards = []
    for r in R["real_expression_fidelity"]:
        cards.append({
            "intervention": "I_EXPR",
            "method": r["method"],
            "score": "finite_difference_h001",
            "internal_gate": "PASS" if r["full_fidelity_pass"] else "FAIL",
            "median_spearman": r["median_spearman"],
            "median_normalized_mae": r["median_normalized_mae"],
        })
    cards.append({
        "intervention": "I_SPATIAL", "method": "balanced_ot/uot",
        "score": "finite_difference_h001", "internal_gate": "FAIL",
        "note": "median spatial NMAE >= ~2.45",
    })
    return {
        "figure_id": "fig_validity_cards",
        "columns": ["estimability", "internal_gate", "external_gate", "controls", "claim_status"],
        "cards": cards,
        "registered_external_gate_passes": R["registered_v1_2_external_gate_passes"],
        "rule": "A single green axis is NOT an overall endorsement.",
    }


def build_fig_robustness_supp(R):
    """Supp Fig S1: corrected robustness sign-relative-to-random deltas."""
    return {
        "figure_id": "fig_robustness_supp",
        "exploratory_signals_vs_random": [
            {"label": f"{s['method']}/{s['score']}",
             "median_nex_aurc": s["median_nex_aurc"],
             "delta_vs_random": round(1.0 - s["median_nex_aurc"], 6),
             "fixed_qc_gain": s["fixed_qc_gain"]}
            for s in R["exploratory_real_external_signals"]
        ],
        "stable_failure_conditions": R["anti_informative_exact_response"]["conditions_median_worse_than_random"],
        "total_conditions": R["anti_informative_exact_response"]["summary_conditions"],
    }


def build_tables(R):
    """Table 1 (params + fidelity Gate) and Table 2 (external Gate)."""
    return {
        "table_1_parameters_and_fidelity": {
            "parameters": R["parameters"],
            "fidelity_gate": R["fidelity_gate"],
            "expression_fidelity_rows": R["real_expression_fidelity"],
        },
        "table_2_external_gate": {
            "registered_v1_2_external_gate_passes": R["registered_v1_2_external_gate_passes"],
            "corrected_real_exact_combined_gate_with_controls_passes":
                R["corrected_real_exact_combined_gate_with_controls_passes"],
            "exploratory_signals": R["exploratory_real_external_signals"],
        },
    }


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    R = load_frozen()
    print("[main] loaded frozen results.json protocol=%s" % R["protocol_id"])

    blocks = {
        "fig_fidelity_intervention": build_fig_fidelity_intervention(R),
        "fig_external_utility": build_fig_external_utility(R),
        "fig_fidelity_vs_utility": build_fig_fidelity_vs_utility(R),
        "fig_sensitivity_boundary": build_fig_sensitivity_boundary(R),
        "fig_validity_cards": build_fig_validity_cards(R),
        "fig_robustness_supp": build_fig_robustness_supp(R),
        "tables": build_tables(R),
    }

    all_results = {
        "schema": "valid-ot-figures-v1.3",
        "protocol_id": R["protocol_id"],
        "source": "results.json (frozen); RESULTS.md verbatim",
        "verification": R["verification"],
        "study_scale": R["study_scale"],
        "parameters": R["parameters"],
        "fidelity_gate": R["fidelity_gate"],
        "claim_boundaries": R["claim_boundaries"],
        "figures": blocks,
    }

    out = os.path.join(FIG_DIR, "all_results.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(all_results, fh, ensure_ascii=False, indent=2)
    print("[main] wrote %s (%d bytes)" % (out, os.path.getsize(out)))

    # also emit per-figure JSON so paper-figure can pick either granularity
    for fid, block in blocks.items():
        if fid == "tables":
            continue
        p = os.path.join(FIG_DIR, "%s_results.json" % fid)
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(block, fh, ensure_ascii=False, indent=2)
        print("[main] wrote %s" % p)

    # descriptive stats block for study scale
    desc = os.path.join(FIG_DIR, "descriptive_stats.json")
    with open(desc, "w", encoding="utf-8") as fh:
        json.dump({"study_scale": R["study_scale"], "parameters": R["parameters"],
                   "fidelity_gate": R["fidelity_gate"]}, fh, ensure_ascii=False, indent=2)
    print("[main] wrote %s" % desc)


if __name__ == "__main__":
    main()
