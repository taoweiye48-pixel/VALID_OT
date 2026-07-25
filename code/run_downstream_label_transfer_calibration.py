"""Score-blind calibration of controlled HER2ST label transfer."""

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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from validot.io import load_pair
from run_downstream_contrastive_calibration import (
    assigned_targets,
    fit_model,
    sha256,
    spatial_folds,
    stable_seed,
    standardize,
    write_json,
    write_tsv,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "downstream_label_transfer_calibration_v1.json"


@torch.no_grad()
def probe_metrics(
    model: torch.nn.Module,
    source: np.ndarray,
    target: np.ndarray,
    test_source: np.ndarray,
    source_labels: np.ndarray,
    target_labels: np.ndarray,
    seed: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    target_embedding = model.target(torch.as_tensor(target, dtype=torch.float32, device=device)).cpu().numpy()
    source_embedding = model.source(
        torch.as_tensor(source[test_source], dtype=torch.float32, device=device)
    ).cpu().numpy()
    classifier = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        solver="lbfgs",
        max_iter=2000,
        random_state=seed,
    )
    classifier.fit(target_embedding, np.asarray(target_labels))
    predicted = classifier.predict(source_embedding)
    observed = np.asarray(source_labels)[test_source]
    return {
        "macro_f1": float(f1_score(observed, predicted, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(observed, predicted)),
        "accuracy": float(accuracy_score(observed, predicted)),
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.use_deterministic_algorithms(True)
    write_json(status_path, {
        "state": "RUNNING", "config": str(CONFIG_PATH), "config_sha256": config_hash,
        "device": str(device), "torch": torch.__version__, "python": sys.version,
        "platform": platform.platform(), "started_epoch": started,
    })

    representation = config["representation"]
    design = config["calibration_design"]
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
    purities = [float(x) for x in design["pair_purities"]]
    repeats = int(design["repeats"])
    base_seed = int(design["base_seed"])
    root = Path(config["processed_pair_root"])
    forward_paths = sorted(path for path in root.glob("HER2ST_*_CONTROLLED.npz") if "__reverse" not in path.stem)
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
                raise RuntimeError(f"source and controlled-truth labels differ: {path}")
            patient = str(pair.metadata.get("independent_unit_id", pair.pair_id))
            folds = spatial_folds(
                pair.source_xy, valid, n_folds, stable_seed(base_seed, patient, direction, "probe-fold")
            )
            manifest.append({
                "path": str(path), "sha256": sha256(path), "patient": patient,
                "direction": direction, "n_valid": len(valid),
                "n_label_classes": len(np.unique(np.asarray(pair.source_labels)[valid])),
            })
            for fold in range(n_folds):
                test = valid[folds[valid] == fold]
                available = valid[folds[valid] != fold]
                keep = max(16, int(np.ceil(coverage * len(available))))
                subset_rng = np.random.default_rng(stable_seed(base_seed, patient, direction, fold, "probe-subset"))
                selected = np.sort(subset_rng.choice(available, size=keep, replace=False))
                source = standardize(source_raw[available], source_raw)
                target = standardize(target_raw, target_raw)
                truth_selected = np.asarray(pair.truth_target[selected], dtype=int)
                for repeat in range(repeats):
                    dose_rng = np.random.default_rng(stable_seed(base_seed, patient, direction, fold, repeat, "probe-dose"))
                    clean_order = dose_rng.permutation(len(selected))
                    wrong_seed = stable_seed(base_seed, patient, direction, fold, repeat, "probe-wrong")
                    for purity in purities:
                        targets = assigned_targets(
                            truth_selected, purity, len(target), clean_order, np.random.default_rng(wrong_seed)
                        )
                        seed = stable_seed(base_seed, patient, direction, fold, repeat, purity, "linear-probe")
                        model = fit_model(
                            source, target, selected, targets, model_spec, optimizer_spec, seed, device
                        )
                        metrics = probe_metrics(
                            model, source, target, test, pair.source_labels, pair.target_labels, seed, device
                        )
                        rows.append({
                            "independent_unit_id": patient,
                            "pair_id": pair.pair_id,
                            "direction": direction,
                            "fold": fold,
                            "repeat": repeat,
                            "pair_purity": purity,
                            "n_train_available": len(available),
                            "n_train_selected": len(selected),
                            "n_test": len(test),
                            **metrics,
                        })
                        del model

    fold_table = pd.DataFrame(rows)
    metrics = ["macro_f1", "balanced_accuracy", "accuracy"]
    direction = fold_table.groupby(
        ["independent_unit_id", "pair_id", "direction", "pair_purity"], as_index=False
    )[metrics].mean()
    unit = direction.groupby(["independent_unit_id", "pair_purity"], as_index=False)[metrics].mean()
    effects: list[dict[str, Any]] = []
    for patient, group in unit.groupby("independent_unit_id", sort=True):
        indexed = group.set_index("pair_purity")
        q0, q05, q1 = indexed.loc[0.0], indexed.loc[0.5], indexed.loc[1.0]
        effects.append({
            "independent_unit_id": patient,
            "macro_f1_q0": q0["macro_f1"], "macro_f1_q05": q05["macro_f1"], "macro_f1_q1": q1["macro_f1"],
            "macro_f1_q1_minus_q0": q1["macro_f1"] - q0["macro_f1"],
            "balanced_accuracy_q0": q0["balanced_accuracy"],
            "balanced_accuracy_q05": q05["balanced_accuracy"],
            "balanced_accuracy_q1": q1["balanced_accuracy"],
            "balanced_accuracy_q1_minus_q0": q1["balanced_accuracy"] - q0["balanced_accuracy"],
            "accuracy_q0": q0["accuracy"], "accuracy_q05": q05["accuracy"], "accuracy_q1": q1["accuracy"],
            "accuracy_q1_minus_q0": q1["accuracy"] - q0["accuracy"],
            "macro_f1_non_decreasing": bool(
                q05["macro_f1"] >= q0["macro_f1"] - 1e-12
                and q1["macro_f1"] >= q05["macro_f1"] - 1e-12
            ),
        })
    effect_table = pd.DataFrame(effects)
    gate_spec = config["go_no_go_gate"]
    macro = effect_table["macro_f1_q1_minus_q0"].to_numpy(float)
    balanced = effect_table["balanced_accuracy_q1_minus_q0"].to_numpy(float)
    checks = {
        "minimum_patients": len(effect_table) >= int(gate_spec["minimum_patients"]),
        "macro_f1_positive_patients": int(np.sum(macro > 0)) >= int(gate_spec["minimum_positive_patients_macro_f1"]),
        "macro_f1_median_effect": float(np.median(macro)) >= float(gate_spec["minimum_median_macro_f1_q1_minus_q0"]),
        "balanced_accuracy_positive_patients": int(np.sum(balanced > 0)) >= int(gate_spec["minimum_positive_patients_balanced_accuracy"]),
        "balanced_accuracy_median_effect": float(np.median(balanced)) >= float(gate_spec["minimum_median_balanced_accuracy_q1_minus_q0"]),
        "macro_f1_dose_monotonic_patients": int(effect_table["macro_f1_non_decreasing"].sum()) >= int(gate_spec["minimum_patients_with_non_decreasing_macro_f1_dose_curve"]),
    }
    gate = {
        "decision": "GO" if all(checks.values()) else "NO_GO",
        "checks": checks,
        "n_patients": len(effect_table),
        "median_macro_f1_q1_minus_q0": float(np.median(macro)),
        "macro_f1_positive_patients": int(np.sum(macro > 0)),
        "median_balanced_accuracy_q1_minus_q0": float(np.median(balanced)),
        "balanced_accuracy_positive_patients": int(np.sum(balanced > 0)),
        "macro_f1_dose_monotonic_patients": int(effect_table["macro_f1_non_decreasing"].sum()),
        "rule": gate_spec["rule"],
    }
    write_tsv(results / "label_transfer_fold.tsv", fold_table)
    write_tsv(results / "label_transfer_direction.tsv", direction)
    write_tsv(results / "label_transfer_unit.tsv", unit)
    write_tsv(results / "label_transfer_effects.tsv", effect_table)
    write_tsv(results / "SOURCE_MANIFEST.tsv", pd.DataFrame(manifest))
    write_json(results / "LABEL_TRANSFER_GATE.json", gate)
    status = {
        "state": "COMPLETED", "decision": gate["decision"], "device": str(device),
        "n_fold_rows": len(fold_table), "n_patients": len(effect_table),
        "elapsed_seconds": time.time() - started, "config_sha256": config_hash,
    }
    write_json(status_path, status)
    print(json.dumps({"status": status, "gate": gate}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
