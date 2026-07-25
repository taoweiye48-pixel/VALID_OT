"""One-shot controlled HER2ST score comparison for downstream label transfer."""

from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon

from validot.io import load_pair
from run_downstream_contrastive_calibration import (
    fit_model,
    sha256,
    spatial_folds,
    stable_seed,
    standardize,
    write_json,
    write_tsv,
)
from run_downstream_label_transfer_calibration import probe_metrics


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "downstream_label_transfer_score_comparison_v1.json"


def exact_two_sided(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[values != 0]
    if not len(values):
        return 1.0
    return float(wilcoxon(values, alternative="two-sided", method="exact").pvalue)


def holm_adjust(p_values: list[float]) -> list[float]:
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p, kind="mergesort")
    adjusted = np.empty(len(p), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(p) - rank) * p[index])
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def bootstrap_median(values: np.ndarray, repeats: int, seed: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    samples = np.median(values[rng.integers(0, len(values), size=(repeats, len(values)))], axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return float(low), float(high)


def load_scores(root: Path, pair_path: Path, method: str) -> dict[str, np.ndarray]:
    identifier = f"{pair_path.stem}__{method}"
    array_path = root / f"{identifier}.npz"
    metadata_path = root / f"{identifier}.json"
    if not array_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(identifier)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("status") != "COMPLETED" or not metadata.get("converged", False):
        raise RuntimeError(f"invalid score checkpoint: {identifier}")
    if metadata.get("pair_sha256") != sha256(pair_path):
        raise RuntimeError(f"pair checksum mismatch: {identifier}")
    with np.load(array_path, allow_pickle=False) as stored:
        return {key: stored[key].copy() for key in stored.files}


def main() -> int:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    config_hash = sha256(CONFIG_PATH)
    analysis = ROOT / config["analysis_root"]
    results = ROOT / config["results_root"]
    analysis.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    status_path = analysis / "STATUS.json"
    started = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.use_deterministic_algorithms(True)
    write_json(status_path, {
        "state": "RUNNING", "config": str(CONFIG_PATH), "config_sha256": config_hash,
        "device": str(device), "torch": torch.__version__, "python": sys.version,
        "platform": platform.platform(), "started_epoch": started,
    })

    representation = config["representation"]
    design = config["comparison_design"]
    model_spec = {"hidden": [], "latent_dim": int(representation["latent_dim"])}
    optimizer_spec = {
        "epochs": int(representation["epochs"]),
        "batch_size": int(representation["batch_size"]),
        "negative_targets": int(representation["negative_targets"]),
        "temperature": float(representation["temperature"]),
        "learning_rate": float(representation["learning_rate"]),
        "weight_decay": float(representation["weight_decay"]),
    }
    n_folds = int(design["spatial_folds"])
    coverage = float(design["training_coverage"])
    repeats = int(design["training_seeds"])
    base_seed = int(design["base_seed"])
    pair_root = Path(config["processed_pair_root"])
    checkpoint_root = ROOT / config["plan_score_checkpoint_root"]
    forward_paths = sorted(path for path in pair_root.glob("HER2ST_*_CONTROLLED.npz") if "__reverse" not in path.stem)
    rows: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []

    for forward in forward_paths:
        for direction in config["directions"]:
            path = forward if direction == "forward" else forward.with_name(forward.stem + "__reverse.npz")
            pair, extras = load_pair(path)
            source_raw = np.asarray(extras["source_heldout"], dtype=np.float32)[:, 0::2]
            target_raw = np.asarray(extras["target_heldout"], dtype=np.float32)[:, 1::2]
            valid = np.flatnonzero((~pair.truth_missing) & (pair.truth_target >= 0))
            truth_labels = np.asarray(pair.target_labels)[pair.truth_target[valid].astype(int)]
            if not np.array_equal(np.asarray(pair.source_labels)[valid], truth_labels):
                raise RuntimeError(f"source and truth labels differ: {path}")
            patient = str(pair.metadata.get("independent_unit_id", pair.pair_id))
            folds = spatial_folds(
                pair.source_xy, valid, n_folds, stable_seed(base_seed, patient, direction, "score-fold")
            )
            manifest.append({
                "path": str(path), "sha256": sha256(path), "patient": patient,
                "direction": direction, "n_valid": len(valid),
            })
            for method in config["methods"]:
                scores = load_scores(checkpoint_root, path, method)
                predicted = scores["predicted_target"].astype(int)
                maximum = scores["max_probability"].astype(float)
                margin = scores["top2_margin"].astype(float)
                for fold in range(n_folds):
                    test = valid[folds[valid] == fold]
                    train = valid[folds[valid] != fold]
                    keep = max(16, int(np.ceil(coverage * len(train))))
                    max_selected = train[np.argsort(-maximum[train], kind="mergesort")[:keep]]
                    margin_selected = train[np.argsort(-margin[train], kind="mergesort")[:keep]]
                    source = standardize(source_raw[train], source_raw)
                    target = standardize(target_raw, target_raw)
                    for repeat in range(repeats):
                        rng = np.random.default_rng(
                            stable_seed(base_seed, patient, direction, method, fold, repeat, "controls")
                        )
                        control_selected = np.sort(rng.choice(train, size=keep, replace=False))
                        designs = [
                            ("max_probability_gate", max_selected, predicted[max_selected]),
                            ("top2_margin_gate", margin_selected, predicted[margin_selected]),
                            ("random_gate", control_selected, predicted[control_selected]),
                            ("oracle_truth", control_selected, pair.truth_target[control_selected].astype(int)),
                        ]
                        shared_seed = stable_seed(
                            base_seed, patient, direction, method, fold, repeat, "shared-initialization"
                        )
                        for strategy, selected, targets in designs:
                            model = fit_model(
                                source, target, selected, targets, model_spec, optimizer_spec,
                                shared_seed, device,
                            )
                            metrics = probe_metrics(
                                model, source, target, test, pair.source_labels, pair.target_labels,
                                shared_seed, device,
                            )
                            rows.append({
                                "independent_unit_id": patient,
                                "pair_id": pair.pair_id,
                                "direction": direction,
                                "method": method,
                                "fold": fold,
                                "repeat": repeat,
                                "strategy": strategy,
                                "coverage": coverage,
                                "n_train_available": len(train),
                                "n_train_selected": len(selected),
                                "n_test": len(test),
                                "training_positive_precision": float(
                                    np.mean(np.asarray(targets, dtype=int) == pair.truth_target[selected].astype(int))
                                ),
                                **metrics,
                            })
                            del model

    fold_table = pd.DataFrame(rows)
    metrics = ["macro_f1", "balanced_accuracy", "accuracy", "training_positive_precision"]
    direction = fold_table.groupby(
        ["independent_unit_id", "pair_id", "direction", "method", "strategy", "coverage"],
        as_index=False,
    )[metrics].mean()
    unit = direction.groupby(
        ["independent_unit_id", "method", "strategy", "coverage"], as_index=False
    )[metrics].mean()

    effects: list[dict[str, Any]] = []
    for (patient, method), group in unit.groupby(["independent_unit_id", "method"], sort=True):
        indexed = group.set_index("strategy")
        maximum_row = indexed.loc["max_probability_gate"]
        margin_row = indexed.loc["top2_margin_gate"]
        row: dict[str, Any] = {"independent_unit_id": patient, "method": method}
        for metric in metrics:
            row[f"max_probability__{metric}"] = float(maximum_row[metric])
            row[f"top2_margin__{metric}"] = float(margin_row[metric])
            row[f"margin_minus_max__{metric}"] = float(margin_row[metric] - maximum_row[metric])
        effects.append(row)
    effect_table = pd.DataFrame(effects)

    statistics = config["statistics"]
    summaries: list[dict[str, Any]] = []
    raw_p: list[float] = []
    for method in config["methods"]:
        group = effect_table[effect_table["method"] == method]
        macro = group["margin_minus_max__macro_f1"].to_numpy(float)
        balanced = group["margin_minus_max__balanced_accuracy"].to_numpy(float)
        precision = group["margin_minus_max__training_positive_precision"].to_numpy(float)
        p_value = exact_two_sided(macro)
        raw_p.append(p_value)
        ci_low, ci_high = bootstrap_median(
            macro,
            int(statistics["bootstrap_replicates"]),
            stable_seed(statistics["bootstrap_seed"], method, "macro-f1-bootstrap"),
        )
        summaries.append({
            "method": method,
            "n_patients": len(group),
            "median_macro_f1_gain": float(np.median(macro)),
            "macro_f1_bootstrap_ci_low": ci_low,
            "macro_f1_bootstrap_ci_high": ci_high,
            "macro_f1_positive_patients": int(np.sum(macro > 0)),
            "macro_f1_tied_patients": int(np.sum(macro == 0)),
            "macro_f1_exact_wilcoxon_p": p_value,
            "median_balanced_accuracy_gain": float(np.median(balanced)),
            "balanced_accuracy_positive_patients": int(np.sum(balanced > 0)),
            "median_training_precision_gain": float(np.median(precision)),
            "training_precision_positive_patients": int(np.sum(precision > 0)),
        })
    adjusted = holm_adjust(raw_p)
    for row, value in zip(summaries, adjusted):
        row["macro_f1_holm_p"] = value
    summary_table = pd.DataFrame(summaries)

    requirement = config["support_gate"]["per_method_requirements"]
    method_decisions: list[dict[str, Any]] = []
    for row in summaries:
        checks = {
            "holm_p": row["macro_f1_holm_p"] <= float(requirement["holm_p_at_most"]),
            "macro_f1_effect": row["median_macro_f1_gain"] >= float(requirement["minimum_median_macro_f1_gain"]),
            "macro_f1_consistency": row["macro_f1_positive_patients"] >= int(requirement["minimum_patients_improved_macro_f1"]),
            "balanced_accuracy_direction": row["median_balanced_accuracy_gain"] >= float(requirement["minimum_median_balanced_accuracy_gain"]),
        }
        method_decisions.append({"method": row["method"], "pass": bool(all(checks.values())), "checks": checks})
    n_pass = int(sum(item["pass"] for item in method_decisions))
    supported = n_pass >= int(config["support_gate"]["minimum_methods_passing"])
    decision = {
        "decision": "SUPPORTED_CONTROLLED_TASK" if supported else "NOT_SUPPORTED",
        "n_methods_passing": n_pass,
        "minimum_methods_required": int(config["support_gate"]["minimum_methods_passing"]),
        "method_decisions": method_decisions,
        "rule": config["support_gate"]["rule"],
        "claim_boundary": config["claim_boundary"],
        "calibration_disclosure": "The preceding label-transfer calibration was NO-GO because median macro-F1 improvement was 0.0408 versus a frozen 0.05 threshold; this score comparison is exploratory.",
    }
    write_tsv(results / "score_comparison_fold.tsv", fold_table)
    write_tsv(results / "score_comparison_direction.tsv", direction)
    write_tsv(results / "score_comparison_unit.tsv", unit)
    write_tsv(results / "score_comparison_margin_vs_max_effects.tsv", effect_table)
    write_tsv(results / "score_comparison_summary.tsv", summary_table)
    write_tsv(results / "SOURCE_MANIFEST.tsv", pd.DataFrame(manifest))
    write_json(results / "SCORE_COMPARISON_DECISION.json", decision)
    status = {
        "state": "COMPLETED", "decision": decision["decision"], "device": str(device),
        "n_fold_rows": len(fold_table), "n_patients": effect_table["independent_unit_id"].nunique(),
        "n_methods": effect_table["method"].nunique(), "elapsed_seconds": time.time() - started,
        "config_sha256": config_hash,
    }
    write_json(status_path, status)
    print(json.dumps({"status": status, "decision": decision}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
