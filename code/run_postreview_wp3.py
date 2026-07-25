"""Run WP3 local-fidelity aggregation and local-neighborhood prediction."""

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
from validot.postreview import CONDITIONS, finite_summary, row_response_rate
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
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    parent = json.loads(PARENT_CONFIG_PATH.read_text(encoding="utf-8"))
    return config, parent, sha256(CONFIG_PATH), sha256(WP1_CONFIG_PATH)


def paths(config: dict[str, Any]) -> dict[str, Path]:
    analysis = ROOT / config["output_root"]
    return {
        "analysis": analysis,
        "wp1_checkpoints": analysis / "wp1" / "checkpoints",
        "wp1_arrays": analysis / "wp1" / "arrays",
        "checkpoints": analysis / "wp3" / "checkpoints",
        "arrays": analysis / "wp3" / "arrays",
        "logs": analysis / "logs",
        "results": ROOT / config["results_root"] / "wp3",
    }


def registry(config: dict[str, Any], parent: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    for pair in parent["pairs"]:
        for direction in parent["directions"]:
            pair_file_id = pair["pair_id"] + ("__reverse" if direction == "reverse" else "")
            for method, parameters in config["methods"].items():
                result.append({**pair, "direction": direction, "pair_file_id": pair_file_id, "method": method, "epsilon": parameters["epsilon"], "tau": parameters["tau"]})
    return result


def task_id(task: dict[str, Any]) -> str:
    return f"{task['pair_file_id']}__{task['method']}"


def compute_neighborhood(task: dict[str, Any], config: dict[str, Any], config_hash: str, wp1_hash: str) -> dict[str, Any]:
    output = paths(config)
    identifier = task_id(task)
    checkpoint = output["checkpoints"] / f"{identifier}.json"
    array_output = output["arrays"] / f"{identifier}.npz"
    if checkpoint.is_file() and array_output.is_file():
        prior = json.loads(checkpoint.read_text(encoding="utf-8"))
        if prior.get("status") == "COMPLETED" and prior.get("config_sha256") == config_hash:
            return prior
    started = time.perf_counter()
    try:
        wp1_checkpoint = output["wp1_checkpoints"] / f"{identifier}.json"
        wp1_array = output["wp1_arrays"] / f"{identifier}.npz"
        if not wp1_checkpoint.is_file() or not wp1_array.is_file():
            raise FileNotFoundError(f"WP1 artifact missing for {identifier}")
        wp1_payload = json.loads(wp1_checkpoint.read_text(encoding="utf-8"))
        if wp1_payload.get("status") != "COMPLETED" or wp1_payload.get("config_sha256") != wp1_hash:
            raise RuntimeError(f"WP1 checkpoint is not valid for {identifier}")
        with np.load(wp1_array, allow_pickle=False) as stored:
            h_grid = stored["h_grid"].copy()
            condition_names = stored["condition_names"].astype(str)
            scores = stored["scores"].copy()
            baseline_row_mass_stored = stored["baseline_row_mass"].copy()
        pair, _ = load_pair(Path(config["processed_pair_root"]) / f"{task['pair_file_id']}.npz")
        components = cost_components(pair.source_x, pair.target_x, pair.source_xy, pair.target_xy)
        expression, spatial = components["expression"], components["spatial_cross"]
        n_source, n_target = len(pair.source_x), len(pair.target_x)
        a = np.full(n_source, 1.0 / n_source)
        b = np.full(n_target, 1.0 / n_target)
        parameters = P1Parameters(task["method"], float(task["epsilon"]), task.get("tau"), int(config["solver"]["max_iter"]), float(config["solver"]["tolerance"]))
        base = solve_p1(mixed_cost(expression, spatial, (0.5, 0.5)), a, b, parameters)
        if not base.converged:
            raise RuntimeError("WP3 baseline did not converge")
        baseline_mass = base.plan.sum(axis=1)
        if not np.allclose(baseline_mass, baseline_row_mass_stored, atol=1e-10, rtol=1e-8):
            raise RuntimeError("WP3 recomputed baseline mass differs from WP1")
        t_values = np.asarray(config["wp3"]["local_neighborhood_t"], dtype=float)
        row_errors = np.full((len(CONDITIONS), n_source), np.nan)
        summaries = []
        solver_rows = [{**task, "stage": "base", "arm": "BASE", "intervention": "NONE", "t": 0.0, "converged": base.converged, "iterations": base.iterations, "seconds": base.seconds}]
        for condition_index, (arm, intervention) in enumerate(CONDITIONS):
            expected_name = f"{arm}__{intervention}"
            if condition_names[condition_index] != expected_name:
                raise RuntimeError("WP1 condition order mismatch")
            actual = []
            for t in t_values:
                match = np.flatnonzero(np.isclose(h_grid, t))
                if len(match):
                    actual.append(float(t) * scores[condition_index, int(match[0])])
                else:
                    weights = arm_weights(arm, intervention, float(t))
                    result = solve_p1(mixed_cost(expression, spatial, weights), a, b, parameters)
                    solver_rows.append({**task, "stage": "local_neighborhood", "arm": arm, "intervention": intervention, "t": float(t), "converged": result.converged, "iterations": result.iterations, "seconds": result.seconds})
                    if not result.converged:
                        raise RuntimeError(f"WP3 solve failed for {arm}/{intervention}/t={t}")
                    rate, _ = row_response_rate((result.plan-base.plan)/float(t), baseline_mass, 1e-12)
                    actual.append(float(t) * rate)
            actual_matrix = np.stack(actual, axis=0)
            h001_index = int(np.flatnonzero(np.isclose(h_grid, 0.01))[0])
            prediction = t_values[:, None] * scores[condition_index, h001_index][None, :]
            numerator = np.mean(np.abs(actual_matrix-prediction), axis=0)
            denominator = np.mean(np.abs(actual_matrix), axis=0)
            estimable = np.isfinite(denominator) & (denominator > 1e-12)
            error = np.full(n_source, np.nan)
            error[estimable] = numerator[estimable]/denominator[estimable]
            row_errors[condition_index] = error
            summaries.append({**task, "arm": arm, "intervention": intervention, **{f"neighborhood_error_{key}": value for key, value in finite_summary(error).items()}})
        temporary = array_output.with_suffix(".npz.tmp")
        temporary.parent.mkdir(parents=True, exist_ok=True)
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, condition_names=condition_names, row_neighborhood_error=row_errors, t_values=t_values)
        temporary.replace(array_output)
        payload = {"status": "COMPLETED", "config_sha256": config_hash, "task_id": identifier, "seconds": time.perf_counter()-started, "summary": summaries, "solver": solver_rows, "failures": []}
    except Exception as exc:
        payload = {"status": "FAILED", "config_sha256": config_hash, "task_id": identifier, "seconds": time.perf_counter()-started, "summary": [], "solver": [], "failures": [{**task, "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc()}]}
    dump_json(checkpoint, payload)
    return payload


def aggregate(config: dict[str, Any], tasks: list[dict[str, Any]], payloads: list[dict[str, Any]], wp1_hash: str) -> dict[str, Any]:
    output = paths(config)
    completed = [p for p in payloads if p["status"] == "COMPLETED"]
    neighborhood = pd.DataFrame([row for p in completed for row in p["summary"]])
    solver = pd.DataFrame([row for p in completed for row in p["solver"]])
    failures = pd.DataFrame([row for p in payloads for row in p["failures"]])
    wp1_rows = []
    for task in tasks:
        path = output["wp1_checkpoints"] / f"{task_id(task)}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "COMPLETED" or payload.get("config_sha256") != wp1_hash:
            raise RuntimeError(f"invalid WP1 artifact during WP3 aggregation: {task_id(task)}")
        wp1_rows.extend(payload["summary"])
    local = pd.DataFrame(wp1_rows)
    # The frozen pair registry uses ``pair_id`` for the biological pair and
    # ``pair_file_id`` for its direction-specific file.  Earlier aggregation
    # expected a redundant ``biological_pair_id`` field that is not present in
    # either checkpoint schema.  Reconstruct that alias deterministically;
    # this changes no computed response or outcome value.
    for frame in (local, neighborhood):
        if "biological_pair_id" not in frame.columns:
            if "pair_id" not in frame.columns:
                raise RuntimeError("cannot reconstruct biological_pair_id: pair_id missing")
            frame["biological_pair_id"] = frame["pair_id"].astype(str)
    keys = ["dataset", "pair_type", "biological_pair_id", "independent_unit_id", "cohort_role", "direction", "pair_file_id", "method", "epsilon", "tau", "arm", "intervention"]
    direction = local.merge(neighborhood[keys + ["neighborhood_error_median", "neighborhood_error_q90"]], on=keys, validate="one_to_one")
    metrics = ["h001_spearman", "h001_top_overlap", "h001_rmae", "vector_relative_l1_median", "direction_cosine_median", "neighborhood_error_median", "neighborhood_error_q90"]
    pair_keys = [key for key in keys if key not in {"direction", "pair_file_id"}]
    pair = direction.groupby(pair_keys, dropna=False)[metrics].mean().reset_index()
    unit_keys = ["independent_unit_id", "cohort_role", "method", "epsilon", "tau", "arm", "intervention"]
    unit = pair.groupby(unit_keys, dropna=False).agg(**{metric: (metric, "median") for metric in metrics}, biological_pairs=("biological_pair_id", "nunique")).reset_index()
    thresholds = config["wp3"]
    unit["gate_pass"] = (
        (unit.h001_spearman >= float(thresholds["spearman_min"]))
        & (unit.h001_top_overlap >= float(thresholds["top_decile_overlap_min"]))
        & (unit.h001_rmae <= float(thresholds["rmae_max"]))
        & (unit.vector_relative_l1_median <= float(thresholds["vector_relative_l1_median_max"]))
        & (unit.direction_cosine_median >= float(thresholds["direction_cosine_median_min"]))
        & (unit.neighborhood_error_median <= float(thresholds["neighborhood_error_median_max"]))
    )
    family_keys = ["method", "epsilon", "tau", "arm", "intervention"]
    family = unit.groupby(family_keys, dropna=False).agg(**{metric: (metric, "median") for metric in metrics}, independent_units=("independent_unit_id", "nunique"), unit_pass_fraction=("gate_pass", "mean")).reset_index()
    family["family_gate_pass"] = (
        (family.h001_spearman >= float(thresholds["spearman_min"]))
        & (family.h001_top_overlap >= float(thresholds["top_decile_overlap_min"]))
        & (family.h001_rmae <= float(thresholds["rmae_max"]))
        & (family.vector_relative_l1_median <= float(thresholds["vector_relative_l1_median_max"]))
        & (family.direction_cosine_median >= float(thresholds["direction_cosine_median_min"]))
        & (family.neighborhood_error_median <= float(thresholds["neighborhood_error_median_max"]))
    )
    output["results"].mkdir(parents=True, exist_ok=True)
    direction.to_csv(output["results"] / "wp3_local_fidelity_direction.tsv", sep="\t", index=False)
    pair.to_csv(output["results"] / "wp3_local_fidelity_pair.tsv", sep="\t", index=False)
    unit.to_csv(output["results"] / "wp3_local_fidelity_unit.tsv", sep="\t", index=False)
    family.to_csv(output["results"] / "wp3_local_fidelity_family.tsv", sep="\t", index=False)
    solver.to_csv(output["results"] / "wp3_solver_diagnostics.tsv", sep="\t", index=False)
    failures.to_csv(output["results"] / "wp3_failures.tsv", sep="\t", index=False)
    gate = {"tasks": len(payloads), "completed": len(completed), "failed": len(payloads)-len(completed), "independent_units": int(unit.independent_unit_id.nunique()) if len(unit) else 0, "family_conditions": len(family), "family_passes": int(family.family_gate_pass.sum()) if len(family) else 0, "gate_a_pass": bool(not len(failures) and len(family) and family.family_gate_pass.all())}
    dump_json(output["results"] / "WP3_GATE_A_LOCAL_FIDELITY.json", gate)
    return gate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not args.run:
        raise SystemExit("use --run after full WP1 Gate A passes")
    config, parent, config_hash, wp1_hash = load_configuration()
    output = paths(config)
    for key in ("checkpoints", "arrays", "logs", "results"):
        output[key].mkdir(parents=True, exist_ok=True)
    tasks = registry(config, parent)
    pd.DataFrame(tasks).to_csv(output["analysis"] / "wp3" / "TASK_REGISTRY.tsv", sep="\t", index=False)
    payloads = []
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=int(config["execution"]["workers"])) as executor:
        futures = {executor.submit(compute_neighborhood, task, config, config_hash, wp1_hash): task for task in tasks}
        for index, future in enumerate(as_completed(futures), 1):
            payloads.append(future.result())
            progress = {"completed": index, "planned": len(tasks), "successful": sum(p["status"] == "COMPLETED" for p in payloads), "failed": sum(p["status"] == "FAILED" for p in payloads), "wall_seconds": time.perf_counter()-started}
            dump_json(output["logs"] / "wp3_progress.json", progress)
            print(json.dumps(progress), flush=True)
    gate = aggregate(config, tasks, payloads, wp1_hash)
    print(json.dumps(gate, indent=2), flush=True)
    return 0 if gate["gate_a_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
