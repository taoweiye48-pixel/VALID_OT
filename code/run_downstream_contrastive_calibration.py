"""Frozen pair-purity calibration for downstream contrastive learning.

No OT confidence score is loaded in this stage. Candidate learners are judged
only by whether known training-pair correctness produces a dose-dependent gain
on spatially held-out source spots. The output is a leave-one-patient-out model
selection lock for the later, one-shot score comparison.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon
from sklearn.cluster import KMeans
from torch import nn
from torch.nn import functional as F

from validot.io import load_pair


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "downstream_contrastive_calibration_v1.json"


def stable_seed(*parts: object) -> int:
    payload = "|".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**31 - 1)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_tsv(path: Path, table: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, sep="\t", index=False)


def normalize_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def standardize(train: np.ndarray, values: np.ndarray) -> np.ndarray:
    mean = np.asarray(train, dtype=np.float64).mean(axis=0, keepdims=True)
    scale = np.asarray(train, dtype=np.float64).std(axis=0, keepdims=True)
    return ((np.asarray(values, dtype=np.float64) - mean) / np.maximum(scale, 1e-6)).astype(np.float32)


def spatial_folds(xy: np.ndarray, valid: np.ndarray, n_folds: int, seed: int) -> np.ndarray:
    result = np.full(len(xy), -1, dtype=int)
    result[valid] = KMeans(n_clusters=n_folds, n_init=20, random_state=seed).fit_predict(
        np.asarray(xy, dtype=float)[valid]
    )
    return result


class Encoder(nn.Module):
    def __init__(self, input_dim: int, hidden: list[int], latent_dim: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current = input_dim
        for width in hidden:
            layers.extend([nn.Linear(current, int(width)), nn.ReLU()])
            current = int(width)
        layers.append(nn.Linear(current, latent_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.network(x), dim=1)


class TwoTower(nn.Module):
    def __init__(self, source_dim: int, target_dim: int, hidden: list[int], latent_dim: int) -> None:
        super().__init__()
        self.source = Encoder(source_dim, hidden, latent_dim)
        self.target = Encoder(target_dim, hidden, latent_dim)


def make_wrong_targets(truth: np.ndarray, n_target: int, rng: np.random.Generator) -> np.ndarray:
    draw = rng.integers(0, n_target - 1, size=len(truth), endpoint=False)
    return draw + (draw >= truth)


def assigned_targets(
    truth: np.ndarray,
    purity: float,
    n_target: int,
    clean_order: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    truth = np.asarray(truth, dtype=int)
    assigned = make_wrong_targets(truth, n_target, rng)
    n_clean = int(round(float(purity) * len(truth)))
    if n_clean:
        assigned[clean_order[:n_clean]] = truth[clean_order[:n_clean]]
    return assigned.astype(np.int64)


def fit_model(
    source: np.ndarray,
    target: np.ndarray,
    train_source: np.ndarray,
    train_target: np.ndarray,
    model_spec: dict[str, Any],
    optimizer_spec: dict[str, Any],
    seed: int,
    device: torch.device,
) -> TwoTower:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = TwoTower(
        source.shape[1], target.shape[1], list(model_spec["hidden"]), int(model_spec["latent_dim"])
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(optimizer_spec["learning_rate"]),
        weight_decay=float(optimizer_spec["weight_decay"]),
    )
    source_tensor = torch.as_tensor(source, dtype=torch.float32, device=device)
    target_tensor = torch.as_tensor(target, dtype=torch.float32, device=device)
    source_ids = torch.as_tensor(train_source, dtype=torch.long, device=device)
    target_ids = torch.as_tensor(train_target, dtype=torch.long, device=device)
    batch_size = int(optimizer_spec["batch_size"])
    negatives = int(optimizer_spec["negative_targets"])
    temperature = float(optimizer_spec["temperature"])
    generator = torch.Generator(device=device.type)
    generator.manual_seed(seed)
    n_target = len(target)
    model.train()
    for _ in range(int(optimizer_spec["epochs"])):
        order = torch.randperm(len(source_ids), generator=generator, device=device)
        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size]
            source_batch = source_ids[batch]
            positive_ids = target_ids[batch]
            raw = torch.randint(
                0, n_target - 1, (len(batch), negatives), generator=generator, device=device
            )
            negative_ids = raw + (raw >= positive_ids[:, None]).to(torch.long)
            z_source = model.source(source_tensor[source_batch])
            z_positive = model.target(target_tensor[positive_ids])
            z_negative = model.target(target_tensor[negative_ids.reshape(-1)]).reshape(
                len(batch), negatives, -1
            )
            positive_similarity = torch.sum(z_source * z_positive, dim=1)
            negative_similarity = torch.einsum("bd,bnd->bn", z_source, z_negative)
            loss = F.softplus((negative_similarity - positive_similarity[:, None]) / temperature).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return model


@torch.no_grad()
def evaluate(
    model: TwoTower,
    source: np.ndarray,
    target: np.ndarray,
    test_source: np.ndarray,
    truth: np.ndarray,
    source_labels: np.ndarray,
    target_labels: np.ndarray,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    source_tensor = torch.as_tensor(source[test_source], dtype=torch.float32, device=device)
    target_tensor = torch.as_tensor(target, dtype=torch.float32, device=device)
    z_source = model.source(source_tensor).cpu().numpy()
    z_target = model.target(target_tensor).cpu().numpy()
    similarity = z_source @ z_target.T
    true_target = np.asarray(truth[test_source], dtype=int)
    true_score = similarity[np.arange(len(test_source)), true_target]
    ranks = 1 + np.sum(similarity > true_score[:, None], axis=1)
    predicted = np.argmax(similarity, axis=1)
    return {
        "top1": float(np.mean(ranks <= 1)),
        "top5": float(np.mean(ranks <= 5)),
        "top10": float(np.mean(ranks <= 10)),
        "mean_reciprocal_rank": float(np.mean(1.0 / ranks)),
        "median_true_rank": float(np.median(ranks)),
        "region_label_accuracy": float(
            np.mean(np.asarray(source_labels)[test_source] == np.asarray(target_labels)[predicted])
        ),
        "true_pair_cosine": float(np.mean(true_score)),
    }


def exact_one_sided(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[values != 0]
    if not len(values):
        return 1.0
    return float(wilcoxon(values, alternative="greater", method="exact").pvalue)


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
        "state": "RUNNING",
        "config": str(CONFIG_PATH),
        "config_sha256": config_hash,
        "device": str(device),
        "torch": torch.__version__,
        "python": sys.version,
        "platform": platform.platform(),
        "started_epoch": started,
    })

    design = config["calibration_design"]
    optimizer_spec = config["optimizer"]
    n_folds = int(design["spatial_folds"])
    coverage = float(design["training_coverage"])
    purities = [float(value) for value in design["pair_purities"]]
    repeats = int(design["repeats"])
    base_seed = int(design["base_seed"])
    pair_root = Path(config["processed_pair_root"])
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
            folds = spatial_folds(
                pair.source_xy, valid, n_folds, stable_seed(base_seed, pair.pair_id, direction, "fold")
            )
            patient = str(pair.metadata.get("independent_unit_id", pair.pair_id))
            manifest.append({
                "path": str(path), "sha256": sha256(path), "patient": patient,
                "direction": direction, "n_source": len(source_raw), "n_target": len(target_raw),
                "n_valid": len(valid),
            })
            for fold in range(n_folds):
                test = valid[folds[valid] == fold]
                available = valid[folds[valid] != fold]
                keep = max(16, int(np.ceil(coverage * len(available))))
                subset_rng = np.random.default_rng(stable_seed(base_seed, patient, direction, fold, "subset"))
                selected = np.sort(subset_rng.choice(available, size=keep, replace=False))
                source = standardize(source_raw[available], source_raw)
                target = standardize(target_raw, target_raw)
                truth_selected = np.asarray(pair.truth_target[selected], dtype=int)
                for repeat in range(repeats):
                    dose_rng = np.random.default_rng(stable_seed(base_seed, patient, direction, fold, repeat, "dose"))
                    clean_order = dose_rng.permutation(len(selected))
                    wrong_seed = stable_seed(base_seed, patient, direction, fold, repeat, "wrong")
                    for purity in purities:
                        assigned = assigned_targets(
                            truth_selected,
                            purity,
                            len(target),
                            clean_order,
                            np.random.default_rng(wrong_seed),
                        )
                        for model_name, model_spec in config["candidate_learners"].items():
                            seed = stable_seed(base_seed, patient, direction, fold, repeat, purity, model_name)
                            model = fit_model(
                                source, target, selected, assigned, model_spec, optimizer_spec, seed, device
                            )
                            metrics = evaluate(
                                model, source, target, test, pair.truth_target,
                                pair.source_labels, pair.target_labels, device,
                            )
                            rows.append({
                                "independent_unit_id": patient,
                                "pair_id": pair.pair_id,
                                "direction": direction,
                                "fold": fold,
                                "repeat": repeat,
                                "model": model_name,
                                "pair_purity": purity,
                                "n_train_available": len(available),
                                "n_train_selected": len(selected),
                                "n_test": len(test),
                                **metrics,
                            })
                            del model

    fold_table = pd.DataFrame(rows)
    metrics = [
        "top1", "top5", "top10", "mean_reciprocal_rank", "median_true_rank",
        "region_label_accuracy", "true_pair_cosine",
    ]
    direction_table = fold_table.groupby(
        ["independent_unit_id", "pair_id", "direction", "model", "pair_purity"], as_index=False
    )[metrics].mean()
    unit_table = direction_table.groupby(
        ["independent_unit_id", "model", "pair_purity"], as_index=False
    )[metrics].mean()

    effects: list[dict[str, Any]] = []
    for (patient, model_name), group in unit_table.groupby(["independent_unit_id", "model"], sort=True):
        indexed = group.set_index("pair_purity")
        q0 = indexed.loc[0.0]
        q05 = indexed.loc[0.5]
        q1 = indexed.loc[1.0]
        effects.append({
            "independent_unit_id": patient,
            "model": model_name,
            "mrr_q0": q0["mean_reciprocal_rank"],
            "mrr_q05": q05["mean_reciprocal_rank"],
            "mrr_q1": q1["mean_reciprocal_rank"],
            "mrr_q1_minus_q0": q1["mean_reciprocal_rank"] - q0["mean_reciprocal_rank"],
            "top10_q0": q0["top10"],
            "top10_q05": q05["top10"],
            "top10_q1": q1["top10"],
            "top10_q1_minus_q0": q1["top10"] - q0["top10"],
            "mrr_non_decreasing": bool(
                q05["mean_reciprocal_rank"] >= q0["mean_reciprocal_rank"] - 1e-12
                and q1["mean_reciprocal_rank"] >= q05["mean_reciprocal_rank"] - 1e-12
            ),
        })
    effect_table = pd.DataFrame(effects)

    summaries: list[dict[str, Any]] = []
    for model_name, group in effect_table.groupby("model", sort=False):
        mrr = group["mrr_q1_minus_q0"].to_numpy(float)
        top10 = group["top10_q1_minus_q0"].to_numpy(float)
        summaries.append({
            "model": model_name,
            "n_patients": len(group),
            "median_mrr_q1_minus_q0": float(np.median(mrr)),
            "mrr_positive_patients": int(np.sum(mrr > 0)),
            "mrr_one_sided_exact_wilcoxon_p": exact_one_sided(mrr),
            "median_top10_q1_minus_q0": float(np.median(top10)),
            "top10_positive_patients": int(np.sum(top10 > 0)),
            "top10_one_sided_exact_wilcoxon_p": exact_one_sided(top10),
            "mrr_non_decreasing_patients": int(group["mrr_non_decreasing"].sum()),
        })
    summary_table = pd.DataFrame(summaries)

    gate_spec = config["leave_one_patient_out_selection"]["eligibility_on_other_seven_patients"]
    model_order = list(config["candidate_learners"])
    selections: list[dict[str, Any]] = []
    patients = sorted(effect_table["independent_unit_id"].unique())
    for heldout in patients:
        training = effect_table[effect_table["independent_unit_id"] != heldout]
        candidates: list[dict[str, Any]] = []
        for model_name in model_order:
            group = training[training["model"] == model_name]
            mrr = group["mrr_q1_minus_q0"].to_numpy(float)
            top10 = group["top10_q1_minus_q0"].to_numpy(float)
            checks = {
                "mrr_positive": int(np.sum(mrr > 0)) >= int(gate_spec["minimum_positive_patients_for_mrr_q1_minus_q0"]),
                "mrr_effect": float(np.median(mrr)) >= float(gate_spec["minimum_median_mrr_q1_minus_q0"]),
                "top10_positive": int(np.sum(top10 > 0)) >= int(gate_spec["minimum_positive_patients_for_top10_q1_minus_q0"]),
                "top10_effect": float(np.median(top10)) >= float(gate_spec["minimum_median_top10_q1_minus_q0"]),
                "dose_monotonic": int(group["mrr_non_decreasing"].sum()) >= int(gate_spec["minimum_patients_with_non_decreasing_median_mrr_dose_curve"]),
            }
            candidates.append({
                "model": model_name,
                "eligible": bool(all(checks.values())),
                "median_mrr_q1_minus_q0": float(np.median(mrr)),
                "median_top10_q1_minus_q0": float(np.median(top10)),
                "mrr_positive_patients": int(np.sum(mrr > 0)),
                "top10_positive_patients": int(np.sum(top10 > 0)),
                "mrr_non_decreasing_patients": int(group["mrr_non_decreasing"].sum()),
                "checks": checks,
            })
        eligible = [item for item in candidates if item["eligible"]]
        selected_model = None
        if eligible:
            selected_model = sorted(
                eligible,
                key=lambda item: (-item["median_mrr_q1_minus_q0"], model_order.index(item["model"])),
            )[0]["model"]
        selections.append({
            "heldout_patient": heldout,
            "selected_model": selected_model,
            "eligible_models": [item["model"] for item in eligible],
            "candidate_details": candidates,
        })

    passed = all(item["selected_model"] is not None for item in selections)
    lock = {
        "decision": "GO" if passed else "NO_GO",
        "calibration_config": str(CONFIG_PATH),
        "calibration_config_sha256": config_hash,
        "selection_rule": config["leave_one_patient_out_selection"]["selection_rule"],
        "patient_model_selection": selections,
        "failure_rule": config["leave_one_patient_out_selection"]["failure_rule"],
    }
    write_tsv(results / "calibration_fold.tsv", fold_table)
    write_tsv(results / "calibration_direction.tsv", direction_table)
    write_tsv(results / "calibration_unit.tsv", unit_table)
    write_tsv(results / "calibration_effects.tsv", effect_table)
    write_tsv(results / "calibration_model_summary.tsv", summary_table)
    write_tsv(results / "SOURCE_MANIFEST.tsv", pd.DataFrame(manifest))
    write_json(results / "LEAVE_ONE_PATIENT_OUT_MODEL_LOCK.json", lock)
    status = {
        "state": "COMPLETED",
        "decision": lock["decision"],
        "device": str(device),
        "n_fold_rows": len(fold_table),
        "n_patients": len(patients),
        "n_models": len(config["candidate_learners"]),
        "elapsed_seconds": time.time() - started,
        "config_sha256": config_hash,
    }
    write_json(status_path, status)
    print(json.dumps({"status": status, "lock": lock}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
