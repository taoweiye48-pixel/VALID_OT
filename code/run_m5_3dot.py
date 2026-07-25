from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from lib_3d_OT.ottools.ot import sinkhorn as official_sinkhorn
from validot.controls import adjusted_association, permutation_controls
from validot.evaluation import boundary_proximity, local_sparsity
from validot.io import load_pair
from validot.metrics import conditional_plan, exact_row_response, normalized_excess_aurc
from validot.utils import file_hash, read_json, status_payload, write_json


ROOT = Path(__file__).resolve().parents[1]
SELECTION = ROOT / "01_manifest" / "pair_selection.tsv"
PROCESSED = ROOT / "03_data_processed" / "external_pairs"
OUTPUT = ROOT / "10_E6_real_external" / "optional_3dot"
CHECKPOINTS = OUTPUT / "checkpoints"
METHOD = "3d_ot_transport_head"
EPSILON = 1.0
GAMMA = 1.0
MAX_ITER = 100
SIM_K = 5
DIST_K = 200


def stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("::".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little")


def normalize_coordinates(value: torch.Tensor) -> torch.Tensor:
    minimum = value.amin(dim=(0, 1), keepdim=True)
    maximum = value.amax(dim=(0, 1), keepdim=True)
    return (value - minimum) / (maximum - minimum + 1e-8)


def solve_arrays(
    source_x: np.ndarray,
    target_x: np.ndarray,
    source_xy: np.ndarray,
    target_xy: np.ndarray,
    mode: str,
    device: str,
    max_iter: int = MAX_ITER,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    started = time.perf_counter()
    x_source = torch.as_tensor(source_x, dtype=torch.float32, device=device).unsqueeze(0)
    x_target = torch.as_tensor(target_x, dtype=torch.float32, device=device).unsqueeze(0)
    xy_source = normalize_coordinates(
        torch.as_tensor(source_xy, dtype=torch.float32, device=device).unsqueeze(0)
    )
    xy_target = normalize_coordinates(
        torch.as_tensor(target_xy, dtype=torch.float32, device=device).unsqueeze(0)
    )
    n_target = int(x_target.shape[1])
    spatial_candidates = min(DIST_K, n_target)
    if mode == "base":
        sim_k = min(SIM_K, spatial_candidates)
        dist_k = spatial_candidates
    elif mode == "I_EXPR":
        x_source = torch.ones_like(x_source)
        x_target = torch.ones_like(x_target)
        sim_k = spatial_candidates
        dist_k = spatial_candidates
    elif mode == "I_SPATIAL":
        sim_k = min(SIM_K, n_target)
        dist_k = n_target
    else:
        raise KeyError(mode)
    with torch.no_grad():
        plan, similarity = official_sinkhorn(
            x_source,
            x_target,
            xy_source,
            xy_target,
            epsilon=torch.tensor(EPSILON, dtype=torch.float32, device=device),
            gamma=torch.tensor(GAMMA, dtype=torch.float32, device=device),
            max_iter=int(max_iter),
            sim_k=int(sim_k),
            dist_k=int(dist_k),
        )
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    plan_np = plan.squeeze(0).detach().cpu().numpy().astype(np.float64)
    similarity_np = similarity.squeeze(0).detach().cpu().numpy().astype(np.float64)
    row_mass = plan_np.sum(axis=1)
    diagnostics = {
        "mode": mode,
        "sim_k": int(sim_k),
        "dist_k": int(dist_k),
        "iterations": int(max_iter),
        "seconds": time.perf_counter() - started,
        "transported_mass": float(plan_np.sum()),
        "minimum_plan_value": float(plan_np.min()),
        "maximum_plan_value": float(plan_np.max()),
        "zero_mass_rows": int(np.sum(row_mass <= 0)),
        "finite": bool(np.all(np.isfinite(plan_np))),
    }
    if not diagnostics["finite"] or diagnostics["minimum_plan_value"] < 0 or diagnostics["zero_mass_rows"]:
        raise FloatingPointError(f"invalid official 3d-OT plan: {diagnostics}")
    return plan_np, similarity_np, diagnostics


def numerical_validation(device: str) -> dict[str, Any]:
    path = PROCESSED / "STAR_8M_D1_D2.npz"
    pair, _ = load_pair(path)
    n = min(256, len(pair.source_x), len(pair.target_x))
    arrays = (pair.source_x[:n], pair.target_x[:n], pair.source_xy[:n], pair.target_xy[:n])
    base_100, _, base_diagnostics = solve_arrays(*arrays, "base", device, 100)
    base_repeat, _, _ = solve_arrays(*arrays, "base", device, 100)
    base_200, _, _ = solve_arrays(*arrays, "base", device, 200)
    base_cpu, _, _ = solve_arrays(*arrays, "base", "cpu", 100)
    expression_deleted, _, expression_diagnostics = solve_arrays(*arrays, "I_EXPR", device, 100)
    spatial_deleted, _, spatial_diagnostics = solve_arrays(*arrays, "I_SPATIAL", device, 100)
    repeat_error = float(np.max(np.abs(base_100 - base_repeat)))
    iteration_error = float(np.max(np.abs(base_100 - base_200)))
    cpu_gpu_error = float(np.max(np.abs(base_100 - base_cpu)))
    expression_response = float(np.mean(exact_row_response(base_100, expression_deleted)))
    spatial_response = float(np.mean(exact_row_response(base_100, spatial_deleted)))
    passed = bool(
        repeat_error <= 1e-8
        and iteration_error <= 1e-8
        and cpu_gpu_error <= 1e-7
        and expression_response > 0
        and spatial_response > 0
    )
    payload = status_payload(
        "M5_3D_OT_NUMERIC_VALIDATION",
        "COMPLETED_GO" if passed else "FAILED_NUMERIC",
        device=device,
        n=n,
        repeat_max_abs=repeat_error,
        iteration_100_vs_200_max_abs=iteration_error,
        cpu_gpu_max_abs=cpu_gpu_error,
        mean_expression_deletion_response=expression_response,
        mean_spatial_deletion_response=spatial_response,
        diagnostics=[base_diagnostics, expression_diagnostics, spatial_diagnostics],
    )
    write_json(OUTPUT / "M5_NUMERIC_VALIDATION.json", payload)
    if not passed:
        raise RuntimeError(f"M5 numerical validation failed: {payload}")
    return payload


def real_losses(pair, extras: dict[str, np.ndarray], plan: np.ndarray) -> dict[str, np.ndarray]:
    conditional, _ = conditional_plan(plan)
    predicted = np.argmax(conditional, axis=1)
    label_error = (pair.source_labels != pair.target_labels[predicted]).astype(float)
    transported = conditional @ extras["target_heldout"]
    transported /= np.maximum(np.linalg.norm(transported, axis=1, keepdims=True), 1e-12)
    source = extras["source_heldout"] / np.maximum(
        np.linalg.norm(extras["source_heldout"], axis=1, keepdims=True), 1e-12
    )
    heldout_loss = 1.0 - np.sum(source * transported, axis=1)
    return {"label_error": label_error, "heldout_loss": heldout_loss}


def risk_scores(pair, base: np.ndarray, responses: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    conditional, _ = conditional_plan(base)
    predicted = np.argmax(conditional, axis=1)
    return {
        "exact_I_EXPR": responses["I_EXPR"],
        "exact_I_SPATIAL": responses["I_SPATIAL"],
        "exact_combined": np.maximum(responses["I_EXPR"], responses["I_SPATIAL"]),
        "source_boundary_proximity": boundary_proximity(pair.source_xy),
        "source_sparsity": local_sparsity(pair.source_xy),
        "matched_target_sparsity": local_sparsity(pair.target_xy)[predicted],
    }


def external_records(pair, losses: dict[str, np.ndarray], scores: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    records = []
    for witness, loss in losses.items():
        valid = np.isfinite(loss)
        for score_name, score in scores.items():
            row = {
                "dataset": pair.dataset,
                "pair_type": pair.metadata.get("pair_type", ""),
                "biological_pair_id": pair.metadata.get("biological_pair_id", pair.pair_id),
                "direction": pair.metadata.get("direction", "forward"),
                "pair_id": pair.pair_id,
                "method": METHOD,
                "witness": witness,
                "score": score_name,
                "n": int(valid.sum()),
                **normalized_excess_aurc(loss[valid], score[valid]),
            }
            unique = np.unique(loss[valid])
            if np.array_equal(unique, np.asarray([0.0, 1.0])):
                row["auroc"] = float(roc_auc_score(loss[valid], score[valid]))
                row["auprc"] = float(average_precision_score(loss[valid], score[valid]))
            else:
                row["auroc"] = float("nan")
                row["auprc"] = float("nan")
            records.append(row)
    return records


def control_records(
    pair,
    extras: dict[str, np.ndarray],
    losses: dict[str, np.ndarray],
    scores: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    boundary = scores["source_boundary_proximity"]
    sparsity = scores["source_sparsity"]
    matched_sparsity = scores["matched_target_sparsity"]
    library = np.log1p(extras.get("source_library_size", np.zeros(len(pair.source_x))))
    region = np.log1p(extras.get("source_region_size", np.ones(len(pair.source_x))))
    rows = []
    for witness, loss in losses.items():
        for score_name, score in scores.items():
            confounds = {
                "boundary": boundary,
                "sparsity": sparsity,
                "matched_target_sparsity": matched_sparsity,
                "log_library_size": library,
                "log_region_size": region,
            }
            if score_name == "source_boundary_proximity":
                confounds.pop("boundary")
            if score_name == "source_sparsity":
                confounds.pop("sparsity")
            if score_name == "matched_target_sparsity":
                confounds.pop("matched_target_sparsity")
            rows.append(
                {
                    "dataset": pair.dataset,
                    "pair_type": pair.metadata.get("pair_type", ""),
                    "biological_pair_id": pair.metadata.get("biological_pair_id", pair.pair_id),
                    "pair_id": pair.pair_id,
                    "direction": pair.metadata.get("direction", "forward"),
                    "method": METHOD,
                    "witness": witness,
                    "score": score_name,
                    **permutation_controls(
                        loss,
                        score,
                        pair.source_xy,
                        repeats=100,
                        seed=stable_seed(pair.pair_id, METHOD, witness, score_name),
                    ),
                    **adjusted_association(loss, score, confounds),
                }
            )
    return rows


def run_one(pair_path: Path, device: str) -> dict[str, Any]:
    pair, extras = load_pair(pair_path)
    task_dir = CHECKPOINTS / pair.pair_id
    status_path = task_dir / "status.json"
    if status_path.exists() and read_json(status_path).get("status") == "COMPLETED":
        return read_json(status_path)
    task_dir.mkdir(parents=True, exist_ok=True)
    write_json(status_path, status_payload("M5_3D_OT_PAIR", "RUNNING", pair_id=pair.pair_id))
    started = time.perf_counter()
    try:
        base, similarity, base_diag = solve_arrays(
            pair.source_x, pair.target_x, pair.source_xy, pair.target_xy, "base", device
        )
        deleted_expression, _, expression_diag = solve_arrays(
            pair.source_x, pair.target_x, pair.source_xy, pair.target_xy, "I_EXPR", device
        )
        deleted_spatial, _, spatial_diag = solve_arrays(
            pair.source_x, pair.target_x, pair.source_xy, pair.target_xy, "I_SPATIAL", device
        )
        responses = {
            "I_EXPR": exact_row_response(base, deleted_expression),
            "I_SPATIAL": exact_row_response(base, deleted_spatial),
        }
        losses = real_losses(pair, extras, base)
        scores = risk_scores(pair, base, responses)
        external = pd.DataFrame(external_records(pair, losses, scores))
        controls = pd.DataFrame(control_records(pair, extras, losses, scores))
        conditional, row_mass = conditional_plan(base)
        predicted = np.argmax(conditional, axis=1)
        quality = pd.DataFrame(
            [
                {
                    "dataset": pair.dataset,
                    "pair_type": pair.metadata.get("pair_type", ""),
                    "biological_pair_id": pair.metadata.get("biological_pair_id", pair.pair_id),
                    "pair_id": pair.pair_id,
                    "direction": pair.metadata.get("direction", "forward"),
                    "method": METHOD,
                    "region_matching_index": float(1.0 - np.mean(losses["label_error"])),
                    "mean_heldout_expression_loss": float(np.mean(losses["heldout_loss"])),
                    "transported_mass": float(base.sum()),
                    "zero_mass_rows": int(np.sum(row_mass <= 0)),
                }
            ]
        )
        external_path = task_dir / "external_utility.tsv"
        controls_path = task_dir / "controls.tsv"
        quality_path = task_dir / "alignment_quality.tsv"
        external.to_csv(external_path, sep="\t", index=False)
        controls.to_csv(controls_path, sep="\t", index=False)
        quality.to_csv(quality_path, sep="\t", index=False)
        np.savez_compressed(
            task_dir / "row_responses.npz",
            exact_I_EXPR=responses["I_EXPR"],
            exact_I_SPATIAL=responses["I_SPATIAL"],
            exact_combined=scores["exact_combined"],
            label_error=losses["label_error"],
            heldout_loss=losses["heldout_loss"],
            source_xy=pair.source_xy,
            source_labels=pair.source_labels.astype(str),
            predicted_target_labels=pair.target_labels[predicted].astype(str),
            source_library_size=extras["source_library_size"],
            source_region_size=extras["source_region_size"],
            base_row_mass=row_mass,
            base_similarity=similarity,
        )
        status = status_payload(
            "M5_3D_OT_PAIR",
            "COMPLETED",
            pair_id=pair.pair_id,
            biological_pair_id=pair.metadata.get("biological_pair_id", pair.pair_id),
            dataset=pair.dataset,
            direction=pair.metadata.get("direction", "forward"),
            method=METHOD,
            n_source=len(pair.source_x),
            n_target=len(pair.target_x),
            seconds=time.perf_counter() - started,
            plan_diagnostics=[base_diag, expression_diag, spatial_diag],
            external_sha256=file_hash(external_path),
            controls_sha256=file_hash(controls_path),
            quality_sha256=file_hash(quality_path),
        )
    except Exception as exc:
        status = status_payload(
            "M5_3D_OT_PAIR",
            "FAILED_NUMERIC",
            pair_id=pair.pair_id,
            seconds=time.perf_counter() - started,
            error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(),
        )
    write_json(status_path, status)
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return status


def pair_gain_summary(external: pd.DataFrame) -> pd.DataFrame:
    qc = {"source_boundary_proximity", "source_sparsity", "matched_target_sparsity"}
    averaged = (
        external.groupby(
            ["dataset", "pair_type", "biological_pair_id", "method", "witness", "score"],
            dropna=False,
        )
        .agg(normalized_excess_aurc=("normalized_excess_aurc", "mean"), directions=("direction", "nunique"))
        .reset_index()
    )
    rows = []
    for keys, group in averaged.groupby(
        ["dataset", "pair_type", "biological_pair_id", "method", "witness"], dropna=False
    ):
        exact = group[group.score == "exact_combined"]
        baselines = group[group.score.isin(qc)]
        if exact.empty or baselines.empty:
            continue
        exact_value = float(exact.iloc[0].normalized_excess_aurc)
        baseline_value = float(baselines.normalized_excess_aurc.min())
        rows.append(
            {
                "dataset": keys[0],
                "pair_type": keys[1],
                "biological_pair_id": keys[2],
                "method": keys[3],
                "witness": keys[4],
                "directions": int(group.directions.min()),
                "exact_normalized_excess_aurc": exact_value,
                "best_qc_normalized_excess_aurc": baseline_value,
                "absolute_gain": baseline_value - exact_value,
                "relative_gain": 1.0 - exact_value / max(abs(baseline_value), 1e-12),
                "positive": bool(exact_value < baseline_value),
            }
        )
    return pd.DataFrame(rows)


def aggregate_gain(pair_gain: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for witness, group in pair_gain.groupby("witness"):
        technology_positive = (
            group.groupby("dataset").absolute_gain.median().gt(0).sum()
        )
        rows.append(
            {
                "method": METHOD,
                "witness": witness,
                "pairs": int(group.biological_pair_id.nunique()),
                "technologies": int(group.dataset.nunique()),
                "median_absolute_gain": float(group.absolute_gain.median()),
                "median_relative_gain": float(group.relative_gain.median()),
                "positive_pair_fraction": float(group.positive.mean()),
                "positive_technology_count": int(technology_positive),
                "registered_gate_eligible": False,
                "interpretation": "post-hoc exploratory M5",
            }
        )
    return pd.DataFrame(rows)


def environment_record(device: str) -> dict[str, Any]:
    freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True).splitlines()
    return status_payload(
        "M5_3D_OT_ENVIRONMENT",
        "COMPLETED",
        python=sys.version,
        executable=sys.executable,
        torch=torch.__version__,
        torch_cuda=torch.version.cuda,
        cuda_available=torch.cuda.is_available(),
        device=device,
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        official_commit="39a7cb02748d83299cd471f172f3b972896e61d8",
        packages=freeze,
    )


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the frozen M5 run")
    device = "cuda:0"
    write_json(OUTPUT / "M5_ENVIRONMENT.json", environment_record(device))
    validation = numerical_validation(device)
    selection = pd.read_csv(SELECTION, sep="\t")
    paths = []
    for pair_id in selection.pair_id:
        paths.extend(
            [
                PROCESSED / f"{pair_id}.npz",
                PROCESSED / f"{pair_id}__reverse.npz",
            ]
        )
    statuses = []
    for index, path in enumerate(paths, start=1):
        status = run_one(path, device)
        statuses.append(status)
        print(f"M5 progress {index}/{len(paths)}: {path.stem} -> {status['status']}", flush=True)
        if status["status"] != "COMPLETED":
            write_json(
                OUTPUT / "M5_DECISION.json",
                status_payload(
                    "M5_3D_OT_OPTIONAL",
                    "BLOCKED_NUMERIC",
                    completed=sum(item["status"] == "COMPLETED" for item in statuses),
                    failed_task=status,
                ),
            )
            return 2
    external_files = list(CHECKPOINTS.glob("*/external_utility.tsv"))
    controls_files = list(CHECKPOINTS.glob("*/controls.tsv"))
    quality_files = list(CHECKPOINTS.glob("*/alignment_quality.tsv"))
    external = pd.concat([pd.read_csv(path, sep="\t") for path in external_files], ignore_index=True)
    controls = pd.concat([pd.read_csv(path, sep="\t") for path in controls_files], ignore_index=True)
    quality = pd.concat([pd.read_csv(path, sep="\t") for path in quality_files], ignore_index=True)
    external.to_csv(OUTPUT / "all_external_utility.tsv", sep="\t", index=False)
    controls.to_csv(OUTPUT / "all_controls.tsv", sep="\t", index=False)
    quality.to_csv(OUTPUT / "all_alignment_quality.tsv", sep="\t", index=False)
    pair_gain = pair_gain_summary(external)
    aggregate = aggregate_gain(pair_gain)
    pair_gain.to_csv(OUTPUT / "pair_direction_averaged_gain.tsv", sep="\t", index=False)
    aggregate.to_csv(OUTPUT / "M5_EXPLORATORY_SUMMARY.tsv", sep="\t", index=False)
    decision = status_payload(
        "M5_3D_OT_OPTIONAL",
        "COMPLETED_EXPLORATORY",
        scope="official transport head only; not full deep 3d-OT retraining",
        validation_status=validation["status"],
        pairs=int(selection.pair_id.nunique()),
        technologies=int(selection.dataset.nunique()),
        directions_per_pair=2,
        runs=len(statuses),
        numerical_failures=0,
        registered_gate_eligible=False,
        summary=aggregate.to_dict(orient="records"),
    )
    write_json(OUTPUT / "M5_DECISION.json", decision)
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
