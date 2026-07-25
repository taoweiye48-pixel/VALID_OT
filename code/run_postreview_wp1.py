"""Run VALID-OT post-review WP1 without touching frozen P0/P1 outputs."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import platform
import sys
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
import psutil

from validot.io import load_pair
from validot.p1 import P1Parameters, arm_weights, mixed_cost, solve_p1
from validot.postreview import (
    CONDITIONS,
    cost_direction,
    finite_summary,
    row_direction_cosine,
    row_relative_l1,
    row_response_rate,
    row_softmax_plan_derivative,
    scalar_fidelity,
)
from validot.solvers import cost_components


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "postreview_wp1_wp10_v1.json"
PARENT_CONFIG_PATH = ROOT / "configs" / "p1_scale_regularization_v2.yaml"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_configs() -> tuple[dict[str, Any], dict[str, Any], str]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    parent = json.loads(PARENT_CONFIG_PATH.read_text(encoding="utf-8"))
    plan = Path(config["analysis_plan"]["path"])
    if sha256(plan).lower() != str(config["analysis_plan"]["sha256"]).lower():
        raise RuntimeError("analysis-plan hash mismatch; refusing to run")
    return config, parent, sha256(CONFIG_PATH)


def output_paths(config: dict[str, Any]) -> dict[str, Path]:
    analysis = ROOT / config["output_root"]
    results = ROOT / config["results_root"]
    return {
        "analysis": analysis,
        "results": results,
        "checkpoints": analysis / "wp1" / "checkpoints",
        "arrays": analysis / "wp1" / "arrays",
        "rows": analysis / "wp1" / "rows",
        "logs": analysis / "logs",
    }


def prepare(config: dict[str, Any], parent: dict[str, Any], config_hash: str) -> None:
    paths = output_paths(config)
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    snapshot = paths["analysis"] / "config_snapshot.json"
    if snapshot.exists() and sha256(snapshot) != sha256(CONFIG_PATH):
        raise RuntimeError("existing config snapshot differs from frozen execution config")
    if not snapshot.exists():
        snapshot.write_bytes(CONFIG_PATH.read_bytes())
    pair_root = Path(config["processed_pair_root"])
    pair_rows = []
    for pair_spec in parent["pairs"]:
        for direction in parent["directions"]:
            pair_file_id = pair_spec["pair_id"] + ("__reverse" if direction == "reverse" else "")
            path = pair_root / f"{pair_file_id}.npz"
            if not path.is_file():
                raise FileNotFoundError(path)
            pair_rows.append(
                {
                    **pair_spec,
                    "direction": direction,
                    "pair_file_id": pair_file_id,
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    pd.DataFrame(pair_rows).to_csv(paths["analysis"] / "PAIR_MANIFEST.tsv", sep="\t", index=False)
    environment = {
        "analysis_version": config["analysis_version"],
        "config_sha256": config_hash,
        "python": sys.version,
        "platform": platform.platform(),
        "logical_cpu_count": psutil.cpu_count(logical=True),
        "physical_cpu_count": psutil.cpu_count(logical=False),
        "total_ram_gb": psutil.virtual_memory().total / 2**30,
        "packages": {},
    }
    for name in ("numpy", "pandas", "scipy", "sklearn", "statsmodels", "ot", "psutil"):
        module = __import__(name)
        environment["packages"][name] = getattr(module, "__version__", "unknown")
    json_dump(paths["analysis"] / "ENVIRONMENT.json", environment)


def task_registry(config: dict[str, Any], parent: dict[str, Any], pilot: bool) -> list[dict[str, Any]]:
    pilot_ids = set(config["wp1"]["pilot_pair_file_ids"])
    tasks: list[dict[str, Any]] = []
    for pair_spec in parent["pairs"]:
        for direction in parent["directions"]:
            pair_file_id = pair_spec["pair_id"] + ("__reverse" if direction == "reverse" else "")
            if pilot and pair_file_id not in pilot_ids:
                continue
            for method, parameters in config["methods"].items():
                tasks.append(
                    {
                        **pair_spec,
                        "direction": direction,
                        "pair_file_id": pair_file_id,
                        "method": method,
                        "epsilon": float(parameters["epsilon"]),
                        "tau": parameters["tau"],
                    }
                )
    return tasks


def task_id(task: dict[str, Any]) -> str:
    return f"{task['pair_file_id']}__{task['method']}"


def solver_metadata(result: Any) -> dict[str, Any]:
    diagnostics = result.diagnostics or {}
    return {
        "converged": bool(result.converged),
        "iterations": int(result.iterations),
        "seconds": float(result.seconds),
        "last_log_scaling_error": float(diagnostics.get("last_log_scaling_error", np.nan)),
        "row_mass_l1": float(diagnostics.get("row_mass_l1", np.nan)),
        "col_mass_l1": float(diagnostics.get("col_mass_l1", np.nan)),
        "transported_mass": float(result.plan.sum()),
    }


def write_rows(path: Path, rows: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8", newline="") as handle:
        rows.to_csv(handle, sep="\t", index=False)
    temporary.replace(path)


def compute_task(task: dict[str, Any], config: dict[str, Any], config_hash: str) -> dict[str, Any]:
    paths = output_paths(config)
    identifier = task_id(task)
    checkpoint = paths["checkpoints"] / f"{identifier}.json"
    array_path = paths["arrays"] / f"{identifier}.npz"
    row_path = paths["rows"] / f"{identifier}.tsv.gz"
    if checkpoint.is_file() and array_path.is_file() and row_path.is_file():
        prior = json.loads(checkpoint.read_text(encoding="utf-8"))
        if prior.get("status") == "COMPLETED" and prior.get("config_sha256") == config_hash:
            return prior
    started = time.perf_counter()
    try:
        pair_path = Path(config["processed_pair_root"]) / f"{task['pair_file_id']}.npz"
        pair, extras = load_pair(pair_path)
        components = cost_components(pair.source_x, pair.target_x, pair.source_xy, pair.target_xy)
        expression = components["expression"]
        spatial = components["spatial_cross"]
        del components
        n_source = len(pair.source_x)
        n_target = len(pair.target_x)
        a = np.full(n_source, 1.0 / n_source, dtype=np.float64)
        b = np.full(n_target, 1.0 / n_target, dtype=np.float64)
        parameters = P1Parameters(
            task["method"],
            task["epsilon"],
            task.get("tau"),
            int(config["solver"]["max_iter"]),
            float(config["solver"]["tolerance"]),
        )
        base_cost = mixed_cost(expression, spatial, (0.5, 0.5))
        base = solve_p1(base_cost, a, b, parameters)
        if not base.converged:
            raise RuntimeError(f"baseline solver did not converge for {identifier}")
        baseline_row_mass = base.plan.sum(axis=1)
        source_ids = extras.get("source_ids", np.arange(n_source).astype(str)).astype(str)
        h_grid = np.asarray(config["wp1"]["step_grid"], dtype=np.float64)
        if not np.all(np.diff(h_grid) < 0):
            raise ValueError("WP1 step grid must be strictly descending")
        condition_names: list[str] = []
        score_cube = np.full((len(CONDITIONS), len(h_grid), n_source), np.nan, dtype=np.float64)
        reference_scores = np.full((len(CONDITIONS), n_source), np.nan, dtype=np.float64)
        reference_estimable = np.zeros((len(CONDITIONS), n_source), dtype=bool)
        row_frames: list[pd.DataFrame] = []
        summaries: list[dict[str, Any]] = []
        convergence: list[dict[str, Any]] = []
        solver_rows: list[dict[str, Any]] = [
            {**task, "stage": "base", "arm": "BASE", "intervention": "NONE", "h": 0.0, **solver_metadata(base)}
        ]
        for condition_index, (arm, intervention) in enumerate(CONDITIONS):
            condition = f"{arm}__{intervention}"
            condition_names.append(condition)
            previous_v: np.ndarray | None = None
            previous_h: float | None = None
            v_h001: np.ndarray | None = None
            v_penultimate: np.ndarray | None = None
            v_smallest: np.ndarray | None = None
            row_records: list[pd.DataFrame] = []
            for h_index, h in enumerate(h_grid):
                weights = arm_weights(arm, intervention, float(h))
                cost = mixed_cost(expression, spatial, weights)
                result = solve_p1(cost, a, b, parameters)
                solver_rows.append(
                    {**task, "stage": "finite_step", "arm": arm, "intervention": intervention, "h": float(h), **solver_metadata(result)}
                )
                if not result.converged:
                    raise RuntimeError(f"solver did not converge for {identifier}/{condition}/h={h}")
                v = (result.plan - base.plan) / float(h)
                score, estimable_mass = row_response_rate(
                    v, baseline_row_mass, float(config["wp1"]["row_mass_min"])
                )
                score_cube[condition_index, h_index] = score
                adjacent_error = np.full(n_source, np.nan, dtype=np.float64)
                adjacent_estimable = np.zeros(n_source, dtype=bool)
                if previous_v is not None and previous_h is not None:
                    adjacent_error, adjacent_estimable = row_relative_l1(
                        previous_v,
                        v,
                        float(config["wp1"]["row_derivative_l1_min"]),
                    )
                    stats = finite_summary(adjacent_error)
                    convergence.append(
                        {
                            **task,
                            "arm": arm,
                            "intervention": intervention,
                            "h_large": float(previous_h),
                            "h_small": float(h),
                            **{f"relative_l1_{key}": value for key, value in stats.items()},
                        }
                    )
                row_records.append(
                    pd.DataFrame(
                        {
                            "pair_file_id": task["pair_file_id"],
                            "independent_unit_id": task["independent_unit_id"],
                            "cohort_role": task["cohort_role"],
                            "direction": task["direction"],
                            "method": task["method"],
                            "arm": arm,
                            "intervention": intervention,
                            "h": float(h),
                            "row_index": np.arange(n_source),
                            "source_id": source_ids,
                            "score": score,
                            "mass_estimable": estimable_mass,
                            "adjacent_relative_l1": adjacent_error,
                            "adjacent_estimable": adjacent_estimable,
                        }
                    )
                )
                if np.isclose(h, 0.01):
                    v_h001 = v.copy()
                if h_index == len(h_grid) - 2:
                    v_penultimate = v.copy()
                if h_index == len(h_grid) - 1:
                    v_smallest = v.copy()
                previous_v = v
                previous_h = float(h)
            if v_penultimate is None or v_smallest is None or v_h001 is None:
                raise RuntimeError("required WP1 derivative states were not retained")
            v_reference = 2.0 * v_smallest - v_penultimate
            reference_score, reference_ok = row_response_rate(
                v_reference, baseline_row_mass, float(config["wp1"]["row_mass_min"])
            )
            reference_scores[condition_index] = reference_score
            reference_estimable[condition_index] = reference_ok
            vector_error, vector_ok = row_relative_l1(
                v_h001, v_reference, float(config["wp1"]["row_derivative_l1_min"])
            )
            direction_cosine, direction_ok = row_direction_cosine(
                v_h001, v_reference, float(config["wp1"]["row_derivative_l1_min"])
            )
            fidelity = scalar_fidelity(
                reference_score,
                score_cube[condition_index, int(np.where(np.isclose(h_grid, 0.01))[0][0])],
                float(config["wp1"]["top_fraction"]),
            )
            summary = {
                **task,
                "arm": arm,
                "intervention": intervention,
                "reference_type": "richardson_vector",
                **{f"h001_{key}": value for key, value in fidelity.items()},
                **{f"vector_relative_l1_{key}": value for key, value in finite_summary(vector_error).items()},
                **{f"direction_cosine_{key}": value for key, value in finite_summary(direction_cosine).items()},
                "vector_rows_estimable": int(np.sum(vector_ok)),
                "direction_rows_estimable": int(np.sum(direction_ok)),
                "reference_rows_estimable": int(np.sum(reference_ok)),
            }
            if task["method"] == "row_softmax":
                analytic = row_softmax_plan_derivative(
                    base.plan,
                    a,
                    cost_direction(expression, spatial, arm, intervention),
                    task["epsilon"],
                )
                analytic_error, _ = row_relative_l1(
                    v_reference,
                    analytic,
                    float(config["wp1"]["row_derivative_l1_min"]),
                )
                analytic_score, _ = row_response_rate(
                    analytic, baseline_row_mass, float(config["wp1"]["row_mass_min"])
                )
                analytic_fidelity = scalar_fidelity(
                    analytic_score,
                    reference_score,
                    float(config["wp1"]["top_fraction"]),
                )
                summary.update(
                    {f"analytic_relative_l1_{key}": value for key, value in finite_summary(analytic_error).items()}
                )
                summary.update(
                    {f"analytic_scalar_{key}": value for key, value in analytic_fidelity.items()}
                )
            summaries.append(summary)
            row_frames.extend(row_records)
        arrays_temporary = array_path.with_suffix(".npz.tmp")
        with arrays_temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                h_grid=h_grid,
                condition_names=np.asarray(condition_names),
                source_ids=source_ids,
                scores=score_cube,
                reference_scores=reference_scores,
                reference_estimable=reference_estimable,
                baseline_row_mass=baseline_row_mass,
            )
        arrays_temporary.replace(array_path)
        write_rows(row_path, pd.concat(row_frames, ignore_index=True))
        payload = {
            "status": "COMPLETED",
            "config_sha256": config_hash,
            "task_id": identifier,
            "seconds": time.perf_counter() - started,
            "n_source": n_source,
            "n_target": n_target,
            "summary": summaries,
            "convergence": convergence,
            "solver": solver_rows,
            "array_path": str(array_path.relative_to(ROOT)),
            "row_path": str(row_path.relative_to(ROOT)),
            "failures": [],
        }
    except Exception as exc:
        payload = {
            "status": "FAILED",
            "config_sha256": config_hash,
            "task_id": identifier,
            "seconds": time.perf_counter() - started,
            "summary": [],
            "convergence": [],
            "solver": [],
            "failures": [
                {
                    **task,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            ],
        }
    json_dump(checkpoint, payload)
    return payload


def aggregate(config: dict[str, Any], payloads: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    paths = output_paths(config)
    result_dir = paths["results"] / "wp1"
    result_dir.mkdir(parents=True, exist_ok=True)
    completed = [payload for payload in payloads if payload.get("status") == "COMPLETED"]
    failures = [row for payload in payloads for row in payload.get("failures", [])]
    summary = pd.DataFrame([row for payload in completed for row in payload["summary"]])
    convergence = pd.DataFrame([row for payload in completed for row in payload["convergence"]])
    solver = pd.DataFrame([row for payload in completed for row in payload["solver"]])
    summary.to_csv(result_dir / f"wp1_{mode}_local_reference_summary.tsv", sep="\t", index=False)
    convergence.to_csv(result_dir / f"wp1_{mode}_step_convergence_unit.tsv", sep="\t", index=False)
    solver.to_csv(result_dir / f"wp1_{mode}_solver_diagnostics.tsv", sep="\t", index=False)
    pd.DataFrame(failures).to_csv(result_dir / f"wp1_{mode}_failures.tsv", sep="\t", index=False)
    gates: dict[str, Any] = {
        "mode": mode,
        "tasks_total": len(payloads),
        "tasks_completed": len(completed),
        "tasks_failed": len(payloads) - len(completed),
        "solver_calls": int(len(solver)),
        "solver_failures_or_nonconvergence": int((~solver["converged"].astype(bool)).sum()) if len(solver) else 0,
        "row_softmax_analytic_gate": None,
        "smallest_step_adjacent_gate": None,
        "gate_a_numerical_pass": False,
    }
    if len(summary):
        analytic = summary.loc[summary.method == "row_softmax"]
        analytic_pass = bool(
            len(analytic)
            and (analytic["analytic_relative_l1_median"] <= float(config["wp1"]["analytic_row_softmax_median_relative_l1_max"])).all()
            and (analytic["analytic_relative_l1_q90"] <= float(config["wp1"]["analytic_row_softmax_q90_relative_l1_max"])).all()
        )
        gates["row_softmax_analytic_gate"] = {
            "conditions": int(len(analytic)),
            "pass": analytic_pass,
            "maximum_median_relative_l1": float(analytic["analytic_relative_l1_median"].max()),
            "maximum_q90_relative_l1": float(analytic["analytic_relative_l1_q90"].max()),
        }
    else:
        analytic_pass = False
    if len(convergence):
        smallest = float(np.min(config["wp1"]["step_grid"]))
        adjacent = convergence.loc[np.isclose(convergence.h_small, smallest)]
        adjacent_pass = bool(
            len(adjacent)
            and (adjacent["relative_l1_median"] <= float(config["wp1"]["smallest_step_adjacent_median_relative_l1_max"])).all()
            and (adjacent["relative_l1_q90"] <= float(config["wp1"]["smallest_step_adjacent_q90_relative_l1_max"])).all()
        )
        gates["smallest_step_adjacent_gate"] = {
            "conditions": int(len(adjacent)),
            "pass": adjacent_pass,
            "maximum_median_relative_l1": float(adjacent["relative_l1_median"].max()),
            "maximum_q90_relative_l1": float(adjacent["relative_l1_q90"].max()),
        }
    else:
        adjacent_pass = False
    gates["gate_a_numerical_pass"] = bool(
        not failures
        and gates["solver_failures_or_nonconvergence"] == 0
        and analytic_pass
        and adjacent_pass
    )
    json_dump(result_dir / f"WP1_{mode.upper()}_GATE.json", gates)
    return gates


def collect_existing(config: dict[str, Any], tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    paths = output_paths(config)
    payloads = []
    for task in tasks:
        checkpoint = paths["checkpoints"] / f"{task_id(task)}.json"
        if checkpoint.is_file():
            payloads.append(json.loads(checkpoint.read_text(encoding="utf-8")))
    return payloads


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("pilot", "full", "status"), required=True)
    args = parser.parse_args()
    config, parent, config_hash = load_configs()
    prepare(config, parent, config_hash)
    pilot = args.mode == "pilot"
    tasks = task_registry(config, parent, pilot=pilot)
    if args.mode == "status":
        tasks = task_registry(config, parent, pilot=False)
        payloads = collect_existing(config, tasks)
        print(json.dumps({"planned": len(tasks), "checkpoints": len(payloads), "completed": sum(p.get("status") == "COMPLETED" for p in payloads), "failed": sum(p.get("status") == "FAILED" for p in payloads)}, indent=2))
        return 0
    paths = output_paths(config)
    pd.DataFrame(tasks).to_csv(paths["analysis"] / "wp1" / f"TASK_REGISTRY_{args.mode}.tsv", sep="\t", index=False)
    workers = int(config["execution"]["workers"])
    payloads: list[dict[str, Any]] = []
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(compute_task, task, config, config_hash): task for task in tasks}
        for index, future in enumerate(as_completed(futures), 1):
            payload = future.result()
            payloads.append(payload)
            progress = {
                "mode": args.mode,
                "completed_futures": index,
                "planned_tasks": len(tasks),
                "successful_tasks": sum(item.get("status") == "COMPLETED" for item in payloads),
                "failed_tasks": sum(item.get("status") == "FAILED" for item in payloads),
                "wall_seconds": time.perf_counter() - started,
                "last_task": payload.get("task_id"),
            }
            json_dump(paths["logs"] / f"wp1_{args.mode}_progress.json", progress)
            print(json.dumps(progress), flush=True)
    gates = aggregate(config, payloads, args.mode)
    print(json.dumps(gates, indent=2), flush=True)
    return 0 if gates["gate_a_numerical_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
