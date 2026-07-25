import pandas as pd

from run_p1_scale_regularization import gate_summary


def test_internal_gate_summary_maps_median_fields_to_canonical_gate_inputs():
    internal = pd.DataFrame([
        {
            "method": "balanced_ot",
            "epsilon": 0.25,
            "tau": None,
            "grid_role": "baseline",
            "arm": "R",
            "intervention": "I_EXPR",
            "score": "matched_finite_difference",
            "claim_status": "sensitivity",
            "independent_unit_id": "unit-1",
            "spearman": 0.80,
            "top_decile_overlap": 0.70,
            "nmae": 0.50,
            "gate_pass": True,
        }
    ])
    external = pd.DataFrame(columns=[
        "method", "epsilon", "tau", "grid_role", "arm", "witness", "score",
        "claim_status", "independent_unit_id", "technology", "fixed_qc_gain",
        "relative_fixed_qc_gain", "normalized_excess_aurc", "negative_control_pass",
        "leakage_control_pass", "confound_direction_pass",
    ])
    config = {
        "fidelity_gate": {
            "spearman_min": 0.70,
            "top_decile_overlap_min": 0.60,
            "nmae_max": 0.75,
        },
        "external_gate": {
            "median_relative_aurc_improvement_min": 0.05,
            "minimum_positive_technology_count": 2,
            "minimum_positive_unit_fraction": 2 / 3,
        },
    }

    result = gate_summary(internal, external, config)

    assert len(result) == 1
    assert bool(result.loc[0, "gate_pass"])
    assert result.loc[0, "median_nmae"] == 0.50
