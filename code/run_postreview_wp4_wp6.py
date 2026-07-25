"""Run WP4 endpoint/path analysis and WP6 UOT mass-shape decomposition."""

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

from validot.io import load_pair
from validot.p1 import P1Parameters, arm_weights, mixed_cost, solve_p1
from validot.postreview import CONDITIONS, finite_summary, row_response_rate, scalar_fidelity
from validot.solvers import cost_components


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "postreview_wp2_wp10_v1.json"
PARENT_CONFIG_PATH = ROOT / "configs" / "p1_scale_regularization_v2.yaml"
WP1_CONFIG_PATH = ROOT / "configs" / "postreview_wp1_wp10_v1.json"


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


def load_configuration() -> tuple[dict[str, Any], dict[str, Any], str, str]:
    return (
        json.loads(CONFIG_PATH.read_text(encoding="utf-8")),
        json.loads(PARENT_CONFIG_PATH.read_text(encoding="utf-8")),
        sha256(CONFIG_PATH),
        sha256(WP1_CONFIG_PATH),
    )


def output_paths(config: dict[str, Any]) -> dict[str, Path]:
    analysis = ROOT / config["output_root"]
    return {
        "analysis": analysis,
        "wp1_checkpoints": analysis / "wp1" / "checkpoints",
        "wp1_arrays": analysis / "wp1" / "arrays",
        "checkpoints": analysis / "wp4_wp6" / "checkpoints",
        "arrays": analysis / "wp4_wp6" / "arrays",
        "logs": analysis / "logs",
        "wp4_results": ROOT / config["results_root"] / "wp4",
        "wp6_results": ROOT / config["results_root"] / "wp6",
    }


def registry(config: dict[str, Any], parent: dict[str, Any]) -> list[dict[str, Any]]:
    tasks = []
    for pair in parent["pairs"]:
        for direction in parent["directions"]:
            pair_file_id = pair["pair_id"] + ("__reverse" if direction == "reverse" else "")
            for method, parameters in config["methods"].items():
                tasks.append({**pair, "direction": direction, "pair_file_id": pair_file_id, "method": method, "epsilon": parameters["epsilon"], "tau": parameters["tau"]})
    return tasks


def task_id(task: dict[str, Any]) -> str:
    return f"{task['pair_file_id']}__{task['method']}"


def fixed_row_distance(plan_a: np.ndarray, plan_b: np.ndarray, baseline_mass: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    mass = np.asarray(baseline_mass, dtype=float)
    estimable = np.isfinite(mass) & (mass > threshold)
    result = np.full(len(mass), np.nan)
    result[estimable] = np.abs(plan_a[estimable]-plan_b[estimable]).sum(axis=1)/(2.0*mass[estimable])
    return result, estimable


def conditional(plan: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mass = plan.sum(axis=1)
    estimable = np.isfinite(mass) & (mass > threshold)
    q = np.full_like(plan, np.nan)
    q[estimable] = plan[estimable]/mass[estimable, None]
    return q, mass, estimable


def compute(task: dict[str, Any], config: dict[str, Any], config_hash: str, wp1_hash: str) -> dict[str, Any]:
    output = output_paths(config)
    identifier = task_id(task)
    checkpoint = output["checkpoints"] / f"{identifier}.json"
    array_path = output["arrays"] / f"{identifier}.npz"
    if checkpoint.is_file() and array_path.is_file():
        prior = json.loads(checkpoint.read_text(encoding="utf-8"))
        if prior.get("status") == "COMPLETED" and prior.get("config_sha256") == config_hash:
            return prior
    started = time.perf_counter()
    try:
        wp1_checkpoint = output["wp1_checkpoints"] / f"{identifier}.json"
        wp1_array = output["wp1_arrays"] / f"{identifier}.npz"
        wp1_payload = json.loads(wp1_checkpoint.read_text(encoding="utf-8"))
        if wp1_payload.get("status") != "COMPLETED" or wp1_payload.get("config_sha256") != wp1_hash:
            raise RuntimeError("WP1 dependency invalid")
        with np.load(wp1_array, allow_pickle=False) as stored:
            condition_names = stored["condition_names"].astype(str)
            h_grid = stored["h_grid"].copy()
            wp1_scores = stored["scores"].copy()
            local_reference = stored["reference_scores"].copy()
        pair, _ = load_pair(Path(config["processed_pair_root"]) / f"{task['pair_file_id']}.npz")
        components = cost_components(pair.source_x, pair.target_x, pair.source_xy, pair.target_xy)
        expression, spatial = components["expression"], components["spatial_cross"]
        n_source, n_target = len(pair.source_x), len(pair.target_x)
        a = np.full(n_source, 1.0/n_source)
        b = np.full(n_target, 1.0/n_target)
        parameters = P1Parameters(task["method"], float(task["epsilon"]), task.get("tau"), int(config["solver"]["max_iter"]), float(config["solver"]["tolerance"]))
        base = solve_p1(mixed_cost(expression, spatial, (0.5, 0.5)), a, b, parameters)
        if not base.converged:
            raise RuntimeError("WP4 baseline did not converge")
        baseline_mass = base.plan.sum(axis=1)
        t_grid = np.asarray(config["wp4"]["path_t"], dtype=float)
        segment_starts = t_grid[:-1]
        endpoint_scores = np.full((len(CONDITIONS), n_source), np.nan)
        eta_rows = np.full_like(endpoint_scores, np.nan)
        kappa_rows = np.full_like(endpoint_scores, np.nan)
        length_rows = np.full_like(endpoint_scores, np.nan)
        speed_cube = np.full((len(CONDITIONS), len(t_grid)-1, n_source), np.nan)
        wp4_rows = []
        wp6_rows = []
        solver_rows = [{**task, "stage": "base", "arm": "BASE", "intervention": "NONE", "t": 0.0, "converged": base.converged, "iterations": base.iterations, "seconds": base.seconds}]
        uot_arrays: dict[str, np.ndarray] = {}
        for condition_index, (arm, intervention) in enumerate(CONDITIONS):
            expected = f"{arm}__{intervention}"
            if condition_names[condition_index] != expected:
                raise RuntimeError("condition order mismatch")
            previous = base.plan
            segment_distances = []
            endpoint = None
            for t_left, t_right in zip(t_grid[:-1], t_grid[1:]):
                weights = arm_weights(arm, intervention, float(t_right))
                result = solve_p1(mixed_cost(expression, spatial, weights), a, b, parameters)
                solver_rows.append({**task, "stage": "path", "arm": arm, "intervention": intervention, "t": float(t_right), "converged": result.converged, "iterations": result.iterations, "seconds": result.seconds})
                if not result.converged:
                    raise RuntimeError(f"WP4 path solve failed for {arm}/{intervention}/t={t_right}")
                distance, _ = fixed_row_distance(previous, result.plan, baseline_mass, float(config["wp4"]["path_estimable_min"]))
                segment_distances.append(distance)
                speed_cube[condition_index, len(segment_distances)-1] = distance/float(t_right-t_left)
                previous = result.plan
                endpoint = result.plan
            if endpoint is None:
                raise RuntimeError("empty WP4 path")
            endpoint_score, endpoint_ok = fixed_row_distance(base.plan, endpoint, baseline_mass, float(config["wp4"]["path_estimable_min"]))
            endpoint_scores[condition_index] = endpoint_score
            length = np.nansum(np.stack(segment_distances), axis=0)
            eta = np.full(n_source, np.nan)
            path_ok = endpoint_ok & np.isfinite(length) & (length > float(config["wp4"]["path_estimable_min"]))
            eta[path_ok] = endpoint_score[path_ok]/length[path_ok]
            early = speed_cube[condition_index, segment_starts <= float(config["wp4"]["early_segment_start_max"])]
            late = speed_cube[condition_index, segment_starts >= float(config["wp4"]["late_segment_start_min"])]
            early_mean = np.nanmean(early, axis=0)
            late_mean = np.nanmean(late, axis=0)
            kappa = np.full(n_source, np.nan)
            kappa_ok = np.isfinite(early_mean) & np.isfinite(late_mean) & (early_mean > float(config["wp4"]["path_estimable_min"]))
            kappa[kappa_ok] = late_mean[kappa_ok]/early_mean[kappa_ok]
            eta_rows[condition_index] = eta
            kappa_rows[condition_index] = kappa
            length_rows[condition_index] = length
            h001_index = int(np.flatnonzero(np.isclose(h_grid, 0.01))[0])
            local_to_endpoint = scalar_fidelity(endpoint_score, local_reference[condition_index], 0.10)
            observed_to_endpoint = scalar_fidelity(endpoint_score, wp1_scores[condition_index, h001_index], 0.10)
            wp4_rows.append({
                **task,
                "arm": arm,
                "intervention": intervention,
                **{f"local_to_endpoint_{key}": value for key, value in local_to_endpoint.items()},
                **{f"h001_to_endpoint_{key}": value for key, value in observed_to_endpoint.items()},
                **{f"path_eta_{key}": value for key, value in finite_summary(eta).items()},
                **{f"path_kappa_{key}": value for key, value in finite_summary(kappa).items()},
                **{f"path_length_{key}": value for key, value in finite_summary(length).items()},
            })
            if task["method"] == "uot":
                threshold = float(config["wp6"]["row_mass_estimable_min"])
                h_large, h_small = map(float, config["wp6"]["local_steps"])
                local_plans = []
                observed_plan = None
                for h in (h_large, h_small, float(config["wp6"]["observed_step"])):
                    result = solve_p1(mixed_cost(expression, spatial, arm_weights(arm, intervention, h)), a, b, parameters)
                    solver_rows.append({**task, "stage": "uot_local_decomposition", "arm": arm, "intervention": intervention, "t": h, "converged": result.converged, "iterations": result.iterations, "seconds": result.seconds})
                    if not result.converged:
                        raise RuntimeError(f"WP6 UOT solve failed for {arm}/{intervention}/h={h}")
                    if h in (h_large, h_small):
                        local_plans.append((result.plan-base.plan)/h)
                    else:
                        observed_plan = result.plan
                if observed_plan is None or len(local_plans) != 2:
                    raise RuntimeError("WP6 local states missing")
                derivative = 2.0*local_plans[1]-local_plans[0]
                q0, m0, ok0 = conditional(base.plan, threshold)
                q1, m1, ok1 = conditional(endpoint, threshold)
                qh, mh, okh = conditional(observed_plan, threshold)
                mdot = derivative.sum(axis=1)
                qdot = np.full_like(derivative, np.nan)
                local_ok = ok0
                qdot[local_ok] = (derivative[local_ok]*m0[local_ok,None]-base.plan[local_ok]*mdot[local_ok,None])/(m0[local_ok,None]**2)
                loc_mass = np.full(n_source, np.nan)
                loc_shape = np.full(n_source, np.nan)
                loc_mass[local_ok] = np.abs(mdot[local_ok])/(2.0*m0[local_ok])
                loc_shape[local_ok] = 0.5*np.abs(qdot[local_ok]).sum(axis=1)
                h_obs = float(config["wp6"]["observed_step"])
                obs_ok = ok0 & okh
                obs_mass = np.full(n_source, np.nan)
                obs_shape = np.full(n_source, np.nan)
                obs_combined = np.full(n_source, np.nan)
                obs_mass[obs_ok] = np.abs(mh[obs_ok]-m0[obs_ok])/(h_obs*(mh[obs_ok]+m0[obs_ok]))
                obs_shape[obs_ok] = 0.5*np.abs(qh[obs_ok]-q0[obs_ok]).sum(axis=1)/h_obs
                obs_combined[obs_ok] = np.abs(observed_plan[obs_ok]-base.plan[obs_ok]).sum(axis=1)/(h_obs*(mh[obs_ok]+m0[obs_ok]))
                end_ok = ok0 & ok1
                end_mass = np.full(n_source, np.nan)
                end_shape = np.full(n_source, np.nan)
                end_combined = np.full(n_source, np.nan)
                end_mass[end_ok] = np.abs(m1[end_ok]-m0[end_ok])/(m1[end_ok]+m0[end_ok])
                end_shape[end_ok] = 0.5*np.abs(q1[end_ok]-q0[end_ok]).sum(axis=1)
                end_combined[end_ok] = np.abs(endpoint[end_ok]-base.plan[end_ok]).sum(axis=1)/(m1[end_ok]+m0[end_ok])
                mass_transfer = scalar_fidelity(end_mass, loc_mass, 0.10)
                shape_transfer = scalar_fidelity(end_shape, loc_shape, 0.10)
                wp6_rows.append({
                    **task,
                    "arm": arm,
                    "intervention": intervention,
                    **{f"mass_local_to_endpoint_{key}": value for key, value in mass_transfer.items()},
                    **{f"shape_local_to_endpoint_{key}": value for key, value in shape_transfer.items()},
                    **{f"local_mass_{key}": value for key, value in finite_summary(loc_mass).items()},
                    **{f"local_shape_{key}": value for key, value in finite_summary(loc_shape).items()},
                    **{f"endpoint_mass_{key}": value for key, value in finite_summary(end_mass).items()},
                    **{f"endpoint_shape_{key}": value for key, value in finite_summary(end_shape).items()},
                })
                for name, value in {
                    "local_mass": loc_mass,
                    "local_shape": loc_shape,
                    "observed_mass_h001": obs_mass,
                    "observed_shape_h001": obs_shape,
                    "observed_combined_h001": obs_combined,
                    "endpoint_mass": end_mass,
                    "endpoint_shape": end_shape,
                    "endpoint_combined": end_combined,
                }.items():
                    uot_arrays[f"{expected}__{name}"] = value
        temporary = array_path.with_suffix(".npz.tmp")
        temporary.parent.mkdir(parents=True, exist_ok=True)
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, condition_names=condition_names, endpoint_scores=endpoint_scores, path_eta=eta_rows, path_kappa=kappa_rows, path_length=length_rows, path_speed=speed_cube, **uot_arrays)
        temporary.replace(array_path)
        payload = {"status": "COMPLETED", "config_sha256": config_hash, "task_id": identifier, "seconds": time.perf_counter()-started, "wp4": wp4_rows, "wp6": wp6_rows, "solver": solver_rows, "failures": []}
    except Exception as exc:
        payload = {"status": "FAILED", "config_sha256": config_hash, "task_id": identifier, "seconds": time.perf_counter()-started, "wp4": [], "wp6": [], "solver": [], "failures": [{**task, "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc()}]}
    dump_json(checkpoint, payload)
    return payload


def aggregate_levels(direction: pd.DataFrame, metrics: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "biological_pair_id" not in direction.columns:
        direction = direction.assign(biological_pair_id=direction["pair_id"])
    pair_keys = ["dataset", "pair_type", "biological_pair_id", "independent_unit_id", "cohort_role", "method", "epsilon", "tau", "arm", "intervention"]
    pair = direction.groupby(pair_keys, dropna=False)[metrics].mean().reset_index()
    unit_keys = ["independent_unit_id", "cohort_role", "method", "epsilon", "tau", "arm", "intervention"]
    unit = pair.groupby(unit_keys, dropna=False).agg(**{metric: (metric, "median") for metric in metrics}, biological_pairs=("biological_pair_id", "nunique")).reset_index()
    return pair, unit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not args.run:
        raise SystemExit("use --run after WP3 completes Gate A")
    config, parent, config_hash, wp1_hash = load_configuration()
    output = output_paths(config)
    for key in ("checkpoints", "arrays", "logs", "wp4_results", "wp6_results"):
        output[key].mkdir(parents=True, exist_ok=True)
    tasks = registry(config, parent)
    pd.DataFrame(tasks).to_csv(output["analysis"] / "wp4_wp6" / "TASK_REGISTRY.tsv", sep="\t", index=False)
    payloads = []
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=int(config["execution"]["workers"])) as executor:
        futures = {executor.submit(compute, task, config, config_hash, wp1_hash): task for task in tasks}
        for index, future in enumerate(as_completed(futures), 1):
            payloads.append(future.result())
            progress = {"completed": index, "planned": len(tasks), "successful": sum(p["status"] == "COMPLETED" for p in payloads), "failed": sum(p["status"] == "FAILED" for p in payloads), "wall_seconds": time.perf_counter()-started}
            dump_json(output["logs"] / "wp4_wp6_progress.json", progress)
            print(json.dumps(progress), flush=True)
    completed = [p for p in payloads if p["status"] == "COMPLETED"]
    wp4 = pd.DataFrame([row for p in completed for row in p["wp4"]])
    wp6 = pd.DataFrame([row for p in completed for row in p["wp6"]])
    solver = pd.DataFrame([row for p in completed for row in p["solver"]])
    failures = pd.DataFrame([row for p in payloads for row in p["failures"]])
    wp4_metrics = [column for column in wp4.columns if column.startswith(("local_to_endpoint_", "h001_to_endpoint_", "path_eta_", "path_kappa_", "path_length_"))]
    wp4_pair, wp4_unit = aggregate_levels(wp4, wp4_metrics)
    wp4.to_csv(output["wp4_results"] / "wp4_endpoint_path_direction.tsv", sep="\t", index=False)
    wp4_pair.to_csv(output["wp4_results"] / "wp4_endpoint_transportability_pair.tsv", sep="\t", index=False)
    wp4_unit.to_csv(output["wp4_results"] / "wp4_path_geometry_unit.tsv", sep="\t", index=False)
    solver.to_csv(output["wp4_results"] / "wp4_wp6_solver_diagnostics.tsv", sep="\t", index=False)
    failures.to_csv(output["wp4_results"] / "wp4_wp6_failures.tsv", sep="\t", index=False)
    if len(wp6):
        wp6_metrics = [column for column in wp6.columns if column.startswith(("mass_", "shape_", "local_mass_", "local_shape_", "endpoint_mass_", "endpoint_shape_"))]
        wp6_pair, wp6_unit = aggregate_levels(wp6, wp6_metrics)
        wp6.to_csv(output["wp6_results"] / "wp6_uot_mass_shape_direction.tsv", sep="\t", index=False)
        wp6_pair.to_csv(output["wp6_results"] / "wp6_uot_mass_shape_pair.tsv", sep="\t", index=False)
        wp6_unit.to_csv(output["wp6_results"] / "wp6_uot_mass_shape_unit.tsv", sep="\t", index=False)
    gate = {"tasks": len(payloads), "completed": len(completed), "failed": len(payloads)-len(completed), "wp4_conditions": len(wp4), "wp6_conditions": len(wp6), "solver_calls": len(solver), "solver_nonconvergence": int((~solver.converged.astype(bool)).sum()) if len(solver) else 0, "gate_b_computational_pass": bool(not len(failures) and len(wp4) == len(tasks)*len(CONDITIONS) and len(wp6) == sum(task["method"] == "uot" for task in tasks)*len(CONDITIONS))}
    dump_json(output["wp4_results"] / "WP4_WP6_GATE_B.json", gate)
    print(json.dumps(gate, indent=2), flush=True)
    return 0 if gate["gate_b_computational_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
