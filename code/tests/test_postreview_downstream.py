from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def load_script(name: str):
    path = ROOT / "code" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_fixed_row_distance_uses_frozen_baseline_mass() -> None:
    module = load_script("run_postreview_wp4_wp6")
    first = np.asarray([[0.25, 0.25], [0.20, 0.30]])
    second = np.asarray([[0.15, 0.35], [0.10, 0.40]])
    values, estimable = module.fixed_row_distance(first, second, np.asarray([0.5, 0.5]), 1e-12)
    assert np.all(estimable)
    assert np.allclose(values, [0.2, 0.2])


def test_binary_utility_prevalence_selection_is_deterministic() -> None:
    module = load_script("run_postreview_wp5_wp9_wp10")
    label = np.asarray([0, 1, 0, 1, 0], dtype=float)
    score = np.asarray([0.1, 0.9, 0.2, 0.8, 0.3])
    result = module.binary_utility(label, score)
    assert result["auroc"] == 1.0
    assert result["auprc"] == 1.0
    assert result["precision_at_prevalence"] == 1.0
    assert result["recall_at_prevalence"] == 1.0


def test_label_free_rigid_recovers_a_rotated_point_cloud() -> None:
    module = load_script("run_postreview_wp7")
    source = np.asarray([[-2.0, -1.0], [-1.0, 2.0], [0.5, -1.5], [2.0, 0.5], [1.0, 2.5]])
    rotation = np.asarray([[0.0, -1.0], [1.0, 0.0]])
    target = 3.0 * (source @ rotation.T) + np.asarray([7.0, -4.0])
    aligned = module.label_free_rigid(source, target)
    normalized = source - np.mean(source, axis=0)
    normalized /= np.sqrt(np.mean(np.sum(normalized**2, axis=1)))
    forward = module.cKDTree(normalized).query(aligned, k=1)[0]
    reverse = module.cKDTree(aligned).query(normalized, k=1)[0]
    assert float(np.mean(forward) + np.mean(reverse)) < 1e-10


def test_wp8_random_panel_has_no_overlap_and_is_repeatable() -> None:
    module = load_script("run_postreview_wp8")

    class Pair:
        source_x = np.arange(20.0).reshape(4, 5)
        target_x = np.arange(20.0, 40.0).reshape(4, 5)

    extras = {
        "source_heldout": np.arange(40.0, 60.0).reshape(4, 5),
        "target_heldout": np.arange(60.0, 80.0).reshape(4, 5),
        "cost_genes": np.asarray([f"g{i}" for i in range(5)]),
        "heldout_genes": np.asarray([f"g{i}" for i in range(5, 10)]),
    }
    original = module.eligible
    module.eligible = lambda pair, extras: (
        np.column_stack([pair.source_x, extras["source_heldout"]]),
        np.column_stack([pair.target_x, extras["target_heldout"]]),
        np.asarray([f"g{i}" for i in range(10)]),
    )
    config = {"wp8": {"cost_features": 7, "heldout_features": 3}}
    try:
        one = module.panel(Pair(), extras, "random_20260720", config)
        two = module.panel(Pair(), extras, "random_20260720", config)
    finally:
        module.eligible = original
    assert np.array_equal(one["cost_genes"], two["cost_genes"])
    assert not set(one["cost_genes"]) & set(one["heldout_genes"])

