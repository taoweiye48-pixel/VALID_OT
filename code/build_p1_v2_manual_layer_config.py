"""Create the frozen derived config for the spatialLIBD manual-layer extension."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "configs" / "p1_scale_regularization_v2.yaml"
TARGET = ROOT / "configs" / "p1_scale_regularization_v2_manual_layers.yaml"

PAIR_SPECS = (
    ("Br5292", "151507", "151508"),
    ("Br5292", "151509", "151510"),
    ("Br5595", "151669", "151670"),
    ("Br5595", "151671", "151672"),
    ("Br8100", "151673", "151674"),
    ("Br8100", "151675", "151676"),
)


def main() -> int:
    config = json.loads(SOURCE.read_text(encoding="utf-8"))
    config.update(
        analysis_version="valid-ot-p1-scale-regularization-sensitivity-v2-manual-layers",
        claim_status="post-review sensitivity analysis; manual cortical-layer extension",
        created_local="2026-07-18T14:10:00+08:00",
        amendments=["docs/P1_AMENDMENT_004_MANUAL_LAYER_EXTENSION.md"],
        supersedes_data_scope_only="valid-ot-p1-scale-regularization-sensitivity-v2-expanded",
        processed_pair_root="data/processed/p1_v2_manual_layer_pairs",
        datasets=["spatialLIBD_manual_layers"],
        pairs=[
            {
                "dataset": "spatialLIBD_manual_layers",
                "pair_id": f"DLPFCML_{source}_{target}",
                "pair_type": "within_donor_adjacent_replicates",
                "independent_unit_id": f"spatialLIBD_manual::{donor}",
                "cohort_role": "manual_layer_truth",
            }
            for donor, source, target in PAIR_SPECS
        ],
        aggregation={
            "order": ["direction", "biological_pair", "independent_unit"],
            "pair_first": True,
            "unit_weighting": "equal",
            "independent_units": 3,
            "sections": 12,
            "biological_pairs": 6,
            "interpretation": "manual layer truth; three donor units",
        },
        preprocessing={
            "cohort": "spatialLIBD HumanPilot DLPFC",
            "official_source": "https://research.libd.org/spatialLIBD/",
            "source_object": "Human_DLPFC_Visium_processedData_sce_scran_spatialLIBD.Rdata",
            "label_column": "layer_guess_reordered",
            "label_semantics": "manual L1-L6/WM spot annotation",
            "n_hvg": 500,
            "heldout_n": 100,
            "max_units": 1500,
            "seed": 20260718,
            "sampling": "deterministic label-agnostic",
        },
        manual_layer_witness_status={
            "heldout_loss": "primary held-out expression witness",
            "label_error_shared_closed_set": "manual cortical-layer witness",
            "source_only_open_set": "manual cortical-layer witness",
        },
        outputs={
            "analysis": "analysis/p1_scale_regularization_v2_manual_layers",
            "results": "results/p1_scale_regularization_v2_manual_layers",
            "build": "build/p1_scale_regularization_v2_manual_layers",
        },
    )
    config.pop("manual_truth_preprocessing", None)
    config.pop("expanded_cohort_witness_status", None)
    config.pop("manual_truth_cohort_witness_status", None)
    config["prohibitions"] = list(dict.fromkeys(config["prohibitions"] + [
        "count twelve sections or six directions-pairs as independent donors",
        "claim exact spot correspondence truth",
        "pool manual-layer extension with unlike cohorts as one biological effect",
    ]))
    TARGET.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(TARGET)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
