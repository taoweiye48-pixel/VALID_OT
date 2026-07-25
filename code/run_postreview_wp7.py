"""WP7 coordinate-frame sensitivity for spatial-channel responses."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from validot.io import load_pair
from validot.metrics import normalized_excess_aurc
from validot.p1 import P1Parameters, arm_weights, mixed_cost, solve_p1
from validot.postreview import row_response_rate, scalar_fidelity
from validot.solvers import cost_components


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "postreview_wp2_wp10_v1.json"
WP1_CONFIG_PATH = ROOT / "configs" / "postreview_wp1_wp10_v1.json"
PARENT_CONFIG_PATH = ROOT / "configs" / "p1_scale_regularization_v2.yaml"
SPATIAL_CONDITIONS = (("R", "I_SPATIAL"), ("N", "I_SPATIAL"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def paths(config: dict[str, Any]) -> dict[str, Path]:
    analysis = ROOT / config["output_root"]
    return {
        "wp1_checkpoints": analysis / "wp1" / "checkpoints",
        "wp1_arrays": analysis / "wp1" / "arrays",
        "wp46_checkpoints": analysis / "wp4_wp6" / "checkpoints",
        "wp46_arrays": analysis / "wp4_wp6" / "arrays",
        "checkpoints": analysis / "wp7" / "checkpoints",
        "arrays": analysis / "wp7" / "arrays",
        "logs": analysis / "logs",
        "results": ROOT / config["results_root"] / "wp7",
    }


def parameters(task: dict[str, Any], config: dict[str, Any]) -> P1Parameters:
    return P1Parameters(task["method"], float(task["epsilon"]), task.get("tau"), int(config["solver"]["max_iter"]), float(config["solver"]["tolerance"]))


def label_free_rigid(source_xy: np.ndarray, target_xy: np.ndarray) -> np.ndarray:
    """PCA candidates with Chamfer selection, using coordinates only."""
    source = source_xy - np.mean(source_xy, axis=0)
    target = target_xy - np.mean(target_xy, axis=0)
    source_rms = np.sqrt(np.mean(np.sum(source**2, axis=1)))
    target_rms = np.sqrt(np.mean(np.sum(target**2, axis=1)))
    source = source / max(source_rms, 1e-12)
    target = target / max(target_rms, 1e-12)
    _, source_vectors = np.linalg.eigh(np.cov(source.T))
    _, target_vectors = np.linalg.eigh(np.cov(target.T))
    source_vectors = source_vectors[:, ::-1]
    target_vectors = target_vectors[:, ::-1]
    best: np.ndarray | None = None
    best_score = float("inf")
    for swap in (np.eye(2), np.array([[0.0, 1.0], [1.0, 0.0]])):
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                transform = target_vectors @ swap @ np.diag([sx, sy]) @ source_vectors.T
                candidate = target @ transform
                score = float(np.mean(cKDTree(source).query(candidate, k=1)[0]) + np.mean(cKDTree(candidate).query(source, k=1)[0]))
                if score < best_score:
                    best_score, best = score, candidate
    if best is None:
        raise RuntimeError("label-free rigid alignment failed")
    return best


def oracle_coordinates(pair: Any) -> np.ndarray:
    target = pair.target_xy.copy()
    assigned = np.zeros(len(target), dtype=bool)
    for source_index, target_index in enumerate(np.asarray(pair.truth_target, dtype=int)):
        if 0 <= target_index < len(target):
            target[target_index] = pair.source_xy[source_index]
            assigned[target_index] = True
    if not np.any(assigned):
        raise RuntimeError("HER2 oracle variant has no usable truth correspondences")
    # Unmatched target locations (possible only in a reverse crop direction) are
    # retained; they are not assigned using a downstream witness.
    return target


def transformed(pair: Any, variant: str) -> np.ndarray:
    if variant == "baseline":
        return pair.target_xy.copy()
    if variant == "label_free_rigid":
        return label_free_rigid(pair.source_xy, pair.target_xy)
    if variant == "her2_oracle":
        return oracle_coordinates(pair)
    raise ValueError(variant)


def squared_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.maximum(np.sum(x*x, axis=1)[:, None] + np.sum(y*y, axis=1)[None, :] - 2.0*x@y.T, 0.0)


def heldout(pair: Any, extras: dict[str, np.ndarray], plan: np.ndarray) -> np.ndarray:
    mass = plan.sum(axis=1)
    q = plan / np.maximum(mass[:, None], 1e-300)
    predicted = q @ extras["target_heldout"]
    predicted /= np.maximum(np.linalg.norm(predicted, axis=1, keepdims=True), 1e-12)
    source = extras["source_heldout"]
    source = source / np.maximum(np.linalg.norm(source, axis=1, keepdims=True), 1e-12)
    return 1.0 - np.sum(source * predicted, axis=1)


def registry(config: dict[str, Any], parent: dict[str, Any]) -> list[dict[str, Any]]:
    tasks = []
    for pair in parent["pairs"]:
        for direction in parent["directions"]:
            pair_file_id = pair["pair_id"] + ("__reverse" if direction == "reverse" else "")
            for method, values in config["methods"].items():
                for variant in config["wp7"]["primary_variants"]:
                    if variant == "her2_oracle" and not str(pair.get("dataset", "")).startswith("HER2ST"):
                        continue
                    tasks.append({**pair, "direction": direction, "pair_file_id": pair_file_id, "method": method, "epsilon": values["epsilon"], "tau": values["tau"], "variant": variant})
    return tasks


def dependency(task: dict[str, Any], output: dict[str, Path], config_hash: str, wp1_hash: str) -> dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    identifier = f"{task['pair_file_id']}__{task['method']}"
    wp1_status = json.loads((output["wp1_checkpoints"] / f"{identifier}.json").read_text(encoding="utf-8"))
    wp46_status = json.loads((output["wp46_checkpoints"] / f"{identifier}.json").read_text(encoding="utf-8"))
    if wp1_status.get("status") != "COMPLETED" or wp1_status.get("config_sha256") != wp1_hash:
        raise RuntimeError("invalid WP1 dependency")
    if wp46_status.get("status") != "COMPLETED" or wp46_status.get("config_sha256") != config_hash:
        raise RuntimeError("invalid WP4 dependency")
    with np.load(output["wp1_arrays"] / f"{identifier}.npz", allow_pickle=False) as wp1, np.load(output["wp46_arrays"] / f"{identifier}.npz", allow_pickle=False) as wp46:
        names = wp1["condition_names"].astype(str)
        h_index = int(np.flatnonzero(np.isclose(wp1["h_grid"], 0.01))[0])
        values = {}
        for condition in SPATIAL_CONDITIONS:
            name = f"{condition[0]}__{condition[1]}"
            index = int(np.flatnonzero(names == name)[0])
            values[condition] = (wp1["scores"][index, h_index].copy(), wp1["reference_scores"][index].copy(), wp46["endpoint_scores"][index].copy())
    return values


def compute(task: dict[str, Any], config: dict[str, Any], config_hash: str, wp1_hash: str) -> dict[str, Any]:
    output = paths(config)
    identifier = f"{task['pair_file_id']}__{task['method']}__{task['variant']}"
    checkpoint = output["checkpoints"] / f"{identifier}.json"
    array_path = output["arrays"] / f"{identifier}.npz"
    if checkpoint.is_file() and array_path.is_file():
        prior = json.loads(checkpoint.read_text(encoding="utf-8"))
        if prior.get("status") == "COMPLETED" and prior.get("config_sha256") == config_hash:
            return prior
    started = time.perf_counter()
    try:
        pair_path = Path(config["processed_pair_root"]) / f"{task['pair_file_id']}.npz"
        pair, extras = load_pair(pair_path)
        target_xy = transformed(pair, task["variant"])
        baseline_raw = squared_distance(pair.source_xy, pair.target_xy)
        variant_raw = squared_distance(pair.source_xy, target_xy)
        positive_baseline = baseline_raw[baseline_raw > 0]
        positive_variant = variant_raw[variant_raw > 0]
        baseline_scale = float(np.median(positive_baseline)) if len(positive_baseline) else 1.0
        variant_scale = float(np.median(positive_variant)) if len(positive_variant) else 1.0
        cost_rho = float(np.corrcoef(baseline_raw.ravel(), variant_raw.ravel())[0, 1])
        components = cost_components(pair.source_x, pair.target_x, pair.source_xy, target_xy)
        expression, spatial = components["expression"], components["spatial_cross"]
        a = np.full(len(pair.source_x), 1.0/len(pair.source_x))
        b = np.full(len(pair.target_x), 1.0/len(pair.target_x))
        param = parameters(task, config)
        base = solve_p1(mixed_cost(expression, spatial, (0.5, 0.5)), a, b, param)
        solver = [{"stage": "base", "converged": base.converged, "iterations": base.iterations, "seconds": base.seconds}]
        if not base.converged:
            raise RuntimeError("WP7 baseline failed")
        baseline_mass = base.plan.sum(axis=1)
        if task["variant"] == "baseline":
            responses = dependency(task, output, config_hash, wp1_hash)
        else:
            responses = {}
            h_large, h_small = map(float, config["wp7"]["local_steps"])
            observed_h = float(config["wp7"]["observed_step"])
            for arm, intervention in SPATIAL_CONDITIONS:
                derivatives = []
                observed = None
                endpoint_plan = None
                for t in (h_large, h_small, observed_h, 1.0):
                    result = solve_p1(mixed_cost(expression, spatial, arm_weights(arm, intervention, t)), a, b, param)
                    solver.append({"stage": "coordinate_response", "arm": arm, "intervention": intervention, "t": t, "converged": result.converged, "iterations": result.iterations, "seconds": result.seconds})
                    if not result.converged:
                        raise RuntimeError(f"WP7 solve failed {arm}/{intervention}/t={t}")
                    if t in (h_large, h_small):
                        derivatives.append((result.plan-base.plan)/t)
                    elif t == observed_h:
                        observed = result.plan
                    else:
                        endpoint_plan = result.plan
                derivative = 2.0*derivatives[1]-derivatives[0]
                local, _ = row_response_rate(derivative, baseline_mass, 1e-12)
                finite, _ = row_response_rate((observed-base.plan)/observed_h, baseline_mass, 1e-12)
                endpoint, _ = row_response_rate(endpoint_plan-base.plan, baseline_mass, 1e-12)
                responses[(arm, intervention)] = (finite, local, endpoint)
        loss = heldout(pair, extras, base.plan)
        rows = []
        arrays: dict[str, np.ndarray] = {}
        for (arm, intervention), (finite, local, endpoint) in responses.items():
            local_fidelity = scalar_fidelity(local, finite, 0.10)
            endpoint_transfer = scalar_fidelity(endpoint, local, 0.10)
            for score_name, score in (("finite_response_h001", finite), ("local_reference", local), ("endpoint_response", endpoint)):
                external = normalized_excess_aurc(loss, score)
                row = {
                    **task,
                    "arm": arm,
                    "intervention": intervention,
                    "score": score_name,
                    "spatial_cost_pearson_vs_baseline": cost_rho,
                    "spatial_cost_scale_ratio": variant_scale/max(baseline_scale, 1e-300),
                    **{f"local_fidelity_{key}": value for key, value in local_fidelity.items()},
                    **{f"endpoint_transfer_{key}": value for key, value in endpoint_transfer.items()},
                    **{f"heldout_{key}": value for key, value in external.items()},
                }
                rows.append(row)
                arrays[f"{arm}__{score_name}"] = score
        array_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = array_path.with_suffix(".npz.tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, heldout_loss=loss, **arrays)
        temporary.replace(array_path)
        payload = {"status": "COMPLETED", "config_sha256": config_hash, "task_id": identifier, "seconds": time.perf_counter()-started, "rows": rows, "solver": [{**task, **value} for value in solver], "failures": []}
    except Exception as exc:
        payload = {"status": "FAILED", "config_sha256": config_hash, "task_id": identifier, "seconds": time.perf_counter()-started, "rows": [], "solver": [], "failures": [{**task, "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc()}]}
    dump_json(checkpoint, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not args.run:
        raise SystemExit("use --run after WP4/WP6")
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    parent = json.loads(PARENT_CONFIG_PATH.read_text(encoding="utf-8"))
    config_hash, wp1_hash = sha256(CONFIG_PATH), sha256(WP1_CONFIG_PATH)
    output = paths(config)
    for key in ("checkpoints", "arrays", "logs", "results"):
        output[key].mkdir(parents=True, exist_ok=True)
    tasks = registry(config, parent)
    pd.DataFrame(tasks).to_csv(output["results"] / "wp7_coordinate_variant_registry.tsv", sep="\t", index=False)
    payloads = []
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=int(config["execution"]["workers"])) as executor:
        futures = {executor.submit(compute, task, config, config_hash, wp1_hash): task for task in tasks}
        for index, future in enumerate(as_completed(futures), 1):
            payloads.append(future.result())
            progress = {"completed": index, "planned": len(tasks), "successful": sum(value["status"] == "COMPLETED" for value in payloads), "failed": sum(value["status"] == "FAILED" for value in payloads), "wall_seconds": time.perf_counter()-started}
            dump_json(output["logs"] / "wp7_progress.json", progress)
            print(json.dumps(progress), flush=True)
    completed = [value for value in payloads if value["status"] == "COMPLETED"]
    rows = pd.DataFrame([row for value in completed for row in value["rows"]])
    solver = pd.DataFrame([row for value in completed for row in value["solver"]])
    failures = pd.DataFrame([row for value in payloads for row in value["failures"]])
    rows.to_csv(output["results"] / "wp7_coordinate_frame_sensitivity_direction.tsv", sep="\t", index=False)
    solver.to_csv(output["results"] / "wp7_solver_diagnostics.tsv", sep="\t", index=False)
    failures.to_csv(output["results"] / "wp7_failures.tsv", sep="\t", index=False)
    metric_columns = [column for column in rows.columns if column.startswith(("local_fidelity_", "endpoint_transfer_", "heldout_", "spatial_cost_"))]
    if "biological_pair_id" not in rows.columns:
        rows = rows.assign(biological_pair_id=rows["pair_id"])
    pair_keys = ["dataset", "pair_type", "biological_pair_id", "independent_unit_id", "cohort_role", "method", "epsilon", "tau", "variant", "arm", "intervention", "score"]
    pair = rows.groupby(pair_keys, dropna=False)[metric_columns].mean().reset_index()
    unit_keys = ["independent_unit_id", "cohort_role", "method", "epsilon", "tau", "variant", "arm", "intervention", "score"]
    unit = pair.groupby(unit_keys, dropna=False)[metric_columns].median().reset_index()
    pair.to_csv(output["results"] / "wp7_coordinate_frame_sensitivity_pair.tsv", sep="\t", index=False)
    unit.to_csv(output["results"] / "wp7_coordinate_frame_sensitivity_unit.tsv", sep="\t", index=False)
    gate = {"planned_tasks": len(tasks), "completed_tasks": len(completed), "failed_tasks": len(tasks)-len(completed), "solver_calls": len(solver), "solver_nonconvergence": int((~solver.converged.astype(bool)).sum()) if len(solver) else 0, "computational_pass": bool(len(completed) == len(tasks) and not len(failures))}
    dump_json(output["results"] / "WP7_GATE_C_COORDINATE.json", gate)
    print(json.dumps(gate, indent=2), flush=True)
    return 0 if gate["computational_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
