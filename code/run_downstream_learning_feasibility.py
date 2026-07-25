"""Oracle-only feasibility gate for a controlled downstream learning task.

This script deliberately does not compute maximum-probability versus top-two-
margin results. It first asks whether a two-view representation learner trained
with known correct pairs can outperform the same learner trained with permuted
pairs. Only a passing gate authorizes a separately frozen score-comparison run.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.cross_decomposition import PLSSVD

from validot.io import load_pair


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "downstream_learning_feasibility_v1.json"


def stable_seed(*parts: object) -> int:
    payload = "|".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**32 - 1)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_tsv(path: Path, table: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, sep="\t", index=False)


def normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def spatial_folds(xy: np.ndarray, valid: np.ndarray, n_splits: int, seed: int) -> np.ndarray:
    fold = np.full(len(xy), -1, dtype=int)
    model = KMeans(n_clusters=n_splits, random_state=seed, n_init=20)
    fold[valid] = model.fit_predict(np.asarray(xy, dtype=float)[valid])
    return fold


def fit_evaluate(
    source_view: np.ndarray,
    target_view: np.ndarray,
    source_labels: np.ndarray,
    target_labels: np.ndarray,
    truth: np.ndarray,
    train_source: np.ndarray,
    train_target: np.ndarray,
    test_source: np.ndarray,
    n_components: int,
) -> dict[str, float]:
    components = min(n_components, source_view.shape[1], target_view.shape[1], len(train_source) - 1)
    if components < 2:
        raise RuntimeError("insufficient observations for the frozen latent dimension")
    model = PLSSVD(n_components=components, scale=True, copy=True)
    model.fit(source_view[train_source], target_view[train_target])
    source_embedding, target_embedding = model.transform(source_view[test_source], target_view)
    source_embedding = normalize_rows(source_embedding)
    target_embedding = normalize_rows(target_embedding)
    similarity = source_embedding @ target_embedding.T
    true_target = truth[test_source].astype(int)
    true_score = similarity[np.arange(len(test_source)), true_target]
    ranks = 1 + np.sum(similarity > true_score[:, None], axis=1)
    prediction = np.argmax(similarity, axis=1)
    return {
        "top1": float(np.mean(ranks <= 1)),
        "top5": float(np.mean(ranks <= 5)),
        "top10": float(np.mean(ranks <= 10)),
        "mean_reciprocal_rank": float(np.mean(1.0 / ranks)),
        "median_true_rank": float(np.median(ranks)),
        "region_label_accuracy": float(
            np.mean(np.asarray(source_labels)[test_source] == np.asarray(target_labels)[prediction])
        ),
        "true_pair_cosine": float(np.mean(true_score)),
    }


def main() -> int:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    config_hash = sha256(CONFIG_PATH)
    analysis = ROOT / config["analysis_root"]
    results = ROOT / config["results_root"]
    analysis.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    status_path = analysis / "STATUS.json"
    started = time.time()
    write_json(status_path, {
        "state": "RUNNING",
        "config": str(CONFIG_PATH),
        "config_sha256": config_hash,
        "started_epoch": started,
        "python": sys.version,
        "platform": platform.platform(),
    })

    pair_root = Path(config["processed_pair_root"])
    forward_paths = sorted(path for path in pair_root.glob("HER2ST_*_CONTROLLED.npz") if "__reverse" not in path.name)
    folds_n = int(config["validation"]["spatial_folds"])
    repeats = int(config["validation"]["permuted_pair_repeats"])
    coverage = float(config["learner"]["training_coverage"])
    components = int(config["learner"]["n_components"])
    base_seed = int(config["validation"]["oracle_subset_seed"])
    rows: list[dict[str, Any]] = []
    source_manifest: list[dict[str, Any]] = []

    for forward in forward_paths:
        for direction in config["directions"]:
            path = forward if direction == "forward" else forward.with_name(forward.stem + "__reverse.npz")
            pair, extras = load_pair(path)
            held_source = np.asarray(extras["source_heldout"], dtype=np.float64)
            held_target = np.asarray(extras["target_heldout"], dtype=np.float64)
            if held_source.shape[1] != 100 or held_target.shape[1] != 100:
                raise RuntimeError(f"expected 100 held-out genes: {path}")
            source_view = held_source[:, 0::2]
            target_view = held_target[:, 1::2]
            valid = np.flatnonzero((~pair.truth_missing) & (pair.truth_target >= 0))
            fold_id = spatial_folds(
                pair.source_xy,
                valid,
                folds_n,
                stable_seed(base_seed, pair.pair_id, direction, "spatial-folds"),
            )
            patient = str(pair.metadata.get("independent_unit_id", pair.pair_id))
            source_manifest.append({
                "pair_path": str(path),
                "pair_sha256": sha256(path),
                "patient": patient,
                "direction": direction,
                "n_source": int(len(pair.source_x)),
                "n_target": int(len(pair.target_x)),
                "n_valid": int(len(valid)),
            })
            for fold in range(folds_n):
                test = valid[fold_id[valid] == fold]
                available = valid[fold_id[valid] != fold]
                keep = max(2, int(math.ceil(coverage * len(available))))
                rng = np.random.default_rng(stable_seed(base_seed, pair.pair_id, direction, fold, "subset"))
                selected = np.sort(rng.choice(available, size=keep, replace=False))
                oracle_targets = pair.truth_target[selected].astype(int)
                common = {
                    "dataset": pair.dataset,
                    "pair_id": pair.pair_id,
                    "independent_unit_id": patient,
                    "direction": direction,
                    "fold": fold,
                    "n_train_available": int(len(available)),
                    "n_train_selected": int(len(selected)),
                    "n_test": int(len(test)),
                    "n_components": components,
                    "coverage": coverage,
                }
                oracle_metrics = fit_evaluate(
                    source_view,
                    target_view,
                    pair.source_labels,
                    pair.target_labels,
                    pair.truth_target,
                    selected,
                    oracle_targets,
                    test,
                    components,
                )
                rows.append({**common, "strategy": "oracle_truth", "repeat": 0, **oracle_metrics})
                for repeat in range(repeats):
                    perm_rng = np.random.default_rng(
                        stable_seed(base_seed, pair.pair_id, direction, fold, "permuted", repeat)
                    )
                    permuted = oracle_targets[perm_rng.permutation(len(oracle_targets))]
                    permuted_metrics = fit_evaluate(
                        source_view,
                        target_view,
                        pair.source_labels,
                        pair.target_labels,
                        pair.truth_target,
                        selected,
                        permuted,
                        test,
                        components,
                    )
                    rows.append({
                        **common,
                        "strategy": "permuted_truth_control",
                        "repeat": repeat,
                        **permuted_metrics,
                    })

    fold_table = pd.DataFrame(rows)
    metrics = [
        "top1", "top5", "top10", "mean_reciprocal_rank", "median_true_rank",
        "region_label_accuracy", "true_pair_cosine",
    ]
    repeat_keys = [
        "dataset", "pair_id", "independent_unit_id", "direction", "fold", "strategy",
        "n_components", "coverage",
    ]
    repeat_mean = fold_table.groupby(repeat_keys, as_index=False)[metrics].mean()
    direction = repeat_mean.groupby(
        ["dataset", "pair_id", "independent_unit_id", "direction", "strategy", "n_components", "coverage"],
        as_index=False,
    )[metrics].mean()
    unit = direction.groupby(
        ["independent_unit_id", "strategy", "n_components", "coverage"], as_index=False
    )[metrics].mean()

    oracle = unit[unit["strategy"] == "oracle_truth"].set_index("independent_unit_id")
    permuted = unit[unit["strategy"] == "permuted_truth_control"].set_index("independent_unit_id")
    effects = []
    for patient in sorted(set(oracle.index) & set(permuted.index)):
        row: dict[str, Any] = {"independent_unit_id": patient}
        for metric in metrics:
            sign = -1.0 if metric == "median_true_rank" else 1.0
            row[f"oracle__{metric}"] = float(oracle.loc[patient, metric])
            row[f"permuted__{metric}"] = float(permuted.loc[patient, metric])
            row[f"beneficial_delta__{metric}"] = sign * (
                float(oracle.loc[patient, metric]) - float(permuted.loc[patient, metric])
            )
        effects.append(row)
    effect_table = pd.DataFrame(effects)
    gate_spec = config["go_no_go_gate"]
    mrr = effect_table["beneficial_delta__mean_reciprocal_rank"].to_numpy(float)
    top10 = effect_table["beneficial_delta__top10"].to_numpy(float)
    gate_checks = {
        "minimum_patients": int(len(effect_table)) >= int(gate_spec["minimum_patients"]),
        "mrr_median_effect": float(np.median(mrr)) >= float(gate_spec["mrr_min_patient_median_oracle_minus_permuted"]),
        "mrr_patients_improved": int(np.sum(mrr > 0)) >= int(gate_spec["mrr_min_patients_improved"]),
        "top10_median_effect": float(np.median(top10)) >= float(gate_spec["top10_min_patient_median_oracle_minus_permuted"]),
        "top10_patients_improved": int(np.sum(top10 > 0)) >= int(gate_spec["top10_min_patients_improved"]),
    }
    passed = bool(all(gate_checks.values()))
    gate = {
        "decision": "GO" if passed else "NO_GO",
        "checks": gate_checks,
        "n_independent_patients": int(len(effect_table)),
        "mrr_patient_median_oracle_minus_permuted": float(np.median(mrr)),
        "mrr_patients_improved": int(np.sum(mrr > 0)),
        "top10_patient_median_oracle_minus_permuted": float(np.median(top10)),
        "top10_patients_improved": int(np.sum(top10 > 0)),
        "rule": gate_spec["rule"],
    }
    write_tsv(results / "feasibility_fold.tsv", fold_table)
    write_tsv(results / "feasibility_direction.tsv", direction)
    write_tsv(results / "feasibility_unit.tsv", unit)
    write_tsv(results / "feasibility_oracle_minus_permuted.tsv", effect_table)
    write_tsv(results / "SOURCE_MANIFEST.tsv", pd.DataFrame(source_manifest))
    write_json(results / "FEASIBILITY_GATE.json", gate)
    completed = time.time()
    status = {
        "state": "COMPLETED",
        "decision": gate["decision"],
        "config": str(CONFIG_PATH),
        "config_sha256": config_hash,
        "n_fold_strategy_rows": int(len(fold_table)),
        "n_independent_patients": int(len(effect_table)),
        "elapsed_seconds": completed - started,
        "completed_epoch": completed,
    }
    write_json(status_path, status)
    print(json.dumps({"status": status, "gate": gate}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
