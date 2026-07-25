from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "p1_scale_regularization_v2.yaml"
EXPECTED_HASH = "fcd957f47e2f6077ea2d6fdc41d4e0b679b4fa96bb878aad3b7fbf7f9fda4ac0"


def load_config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_v2_config_is_frozen() -> None:
    assert hashlib.sha256(CONFIG.read_bytes()).hexdigest() == EXPECTED_HASH


def test_primary_expansion_is_ten_unique_donors() -> None:
    config = load_config()
    primary = [pair for pair in config["pairs"] if pair["cohort_role"] == "primary_expansion"]
    assert len(primary) == 10
    assert len({pair["independent_unit_id"] for pair in primary}) == 10
    assert {pair["pair_type"] for pair in primary} == {"within_donor_adjacent_position"}
    assert config["preprocessing"]["position_pair"] == ["anterior", "middle"]
    assert config["preprocessing"]["sampling"] == "deterministic label-agnostic"
    assert "pseudo-label" in config["preprocessing"]["label_semantics"]


def test_manual_truth_cohort_has_eight_independent_patients() -> None:
    config = load_config()
    manual = [pair for pair in config["pairs"] if pair["cohort_role"] == "manual_truth_controlled"]
    assert len(manual) == 8
    assert len({pair["independent_unit_id"] for pair in manual}) == 8
    assert {pair["pair_type"] for pair in manual} == {"within_section_controlled_combined"}


def test_pooled_design_has_twenty_one_independent_units() -> None:
    config = load_config()
    units = {pair["independent_unit_id"] for pair in config["pairs"]}
    assert len(units) == 21
    assert config["aggregation"]["real_cross_slice_independent_units"] == 10
    assert config["aggregation"]["manual_truth_independent_units"] == 8
    assert config["aggregation"]["pooled_independent_units"] == 21


def test_v2_outputs_do_not_overlap_v1() -> None:
    config = load_config()
    assert config["outputs"]["analysis"] != "analysis/p1_scale_regularization"
    assert config["outputs"]["results"] != "results/p1_scale_regularization"
    assert config["outputs"]["build"] != "build/p1_scale_regularization"
    assert "modify P1 v1 artifacts" in config["prohibitions"]
