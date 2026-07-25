"""Run WP2 matrix-free balanced-OT implicit differentiation checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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

from validot.io import load_pair
from validot.p1 import P1Parameters, arm_weights, mixed_cost, solve_p1
from validot.postreview import (
    CONDITIONS,
    balanced_plan_implicit_derivative,
    cost_direction,
    finite_summary,
    row_direction_cosine,
    row_relative_l1,
    row_response_rate,
    scalar_fidelity,
)
from validot.solvers import cost_components


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "postreview_wp2_wp10_v1.json"
PARENT_CONFIG_PATH = ROOT / "configs" / "p1_scale_regularization_v2.yaml"


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


def load_configuration() -> tuple[dict[str, Any], dict[str, Any], str]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    parent = json.loads(PARENT_CONFIG_PATH.read_text(encoding="utf-8"))
    return config, parent, sha256(CONFIG_PATH)


def paths(config: dict[str, Any]) -> dict[str, Path]:
    analysis = ROOT / config["output_root"]
    return {
        "analysis": analysis,
        "checkpoints": analysis / "wp2" / "checkpoints",
        "logs": analysis / "logs",
        "results": ROOT / config["results_root"] / "wp2",
    }


def task_id(task: dict[str, Any]) -> str:
    return f"{task['pair_file_id']}__balanced_ot"


def tasks(config: dict[str, Any], parent: dict[str, Any]) -> list[dict[str, Any]]:
    selected = set(config["wp2"]["balanced_implicit_unit_ids"])
    result = []
    for pair in parent["pairs"]:
        if pair["independent_unit_id"] not in selected:
            continue
        for direction in parent["directions"]:
            pair_file_id = pair["pair_id"] + ("__reverse" if direction == "reverse" else "")
            result.append({**pair, "direction": direction, "pair_file_id": pair_file_id, "method": "balanced_ot"})
    observed = {row["independent_unit_id"] for row in result}
    if observed != selected:
        raise RuntimeError(f"WP2 selected-unit mismatch: missing={sorted(selected-observed)}")
    return result


def compute(task: dict[str, Any], config: dict[str, Any], config_hash: str) -> dict[str, Any]:
    output = paths(config)
    identifier = task_id(task)
    checkpoint = output["checkpoints"] / f"{identifier}.json"
    if checkpoint.is_file():
        prior = json.loads(checkpoint.read_text(encoding="utf-8"))
        if prior.get("status") == "COMPLETED" and prior.get("config_sha256") == config_hash:
            return prior
    started = time.perf_counter()
    try:
        pair, _ = load_pair(Path(config["processed_pair_root"]) / f"{task['pair_file_id']}.npz")
        components = cost_components(pair.source_x, pair.target_x, pair.source_xy, pair.target_xy)
        expression = components["expression"]
        spatial = components["spatial_cross"]
        n_source, n_target = len(pair.source_x), len(pair.target_x)
        a = np.full(n_source, 1.0 / n_source)
        b = np.full(n_target, 1.0 / n_target)
        method = config["methods"]["balanced_ot"]
        parameters = P1Parameters("balanced_ot", method["epsilon"], None, int(config["solver"]["max_iter"]), float(config["solver"]["tolerance"]))
        base_cost = mixed_cost(expression, spatial, (0.5, 0.5))
        base = solve_p1(base_cost, a, b, parameters)
        if not base.converged:
            raise RuntimeError("balanced baseline did not converge")
        mass = base.plan.sum(axis=1)
        rows = []
        solve_rows = [{**task, "stage": "base", "arm": "BASE", "intervention": "NONE", "h": 0.0, "converged": base.converged, "iterations": base.iterations, "seconds": base.seconds}]
        h_large, h_small = map(float, config["wp2"]["richardson_steps"])
        for arm, intervention in CONDITIONS:
            finite = []
            for h in (h_large, h_small):
                weights = arm_weights(arm, intervention, h)
                result = solve_p1(mixed_cost(expression, spatial, weights), a, b, parameters)
                solve_rows.append({**task, "stage": "finite_step", "arm": arm, "intervention": intervention, "h": h, "converged": result.converged, "iterations": result.iterations, "seconds": result.seconds})
                if not result.converged:
                    raise RuntimeError(f"finite solve failed for {arm}/{intervention}/h={h}")
                finite.append((result.plan - base.plan) / h)
            richardson = 2.0 * finite[1] - finite[0]
            implicit, implicit_diag = balanced_plan_implicit_derivative(
                base.plan,
                cost_direction(expression, spatial, arm, intervention),
                method["epsilon"],
                float(config["wp2"]["linear_solver_relative_tolerance"]),
                int(config["wp2"]["linear_solver_max_iterations"]),
            )
            relative, relative_ok = row_relative_l1(richardson, implicit, 1e-12)
            cosine, cosine_ok = row_direction_cosine(richardson, implicit, 1e-12)
            rich_score, _ = row_response_rate(richardson, mass, 1e-12)
            implicit_score, _ = row_response_rate(implicit, mass, 1e-12)
            fidelity = scalar_fidelity(implicit_score, rich_score, 0.10)
            denominator = float(np.abs(implicit).sum())
            rows.append({
                **task,
                "arm": arm,
                "intervention": intervention,
                "global_plan_relative_l1": float(np.abs(richardson-implicit).sum()/max(denominator,1e-300)),
                **{f"row_relative_l1_{key}": value for key, value in finite_summary(relative).items()},
                **{f"row_direction_cosine_{key}": value for key, value in finite_summary(cosine).items()},
                "row_relative_estimable": int(np.sum(relative_ok)),
                "row_direction_estimable": int(np.sum(cosine_ok)),
                **{f"scalar_{key}": value for key, value in fidelity.items()},
                **implicit_diag,
            })
        payload = {"status": "COMPLETED", "config_sha256": config_hash, "task_id": identifier, "seconds": time.perf_counter()-started, "rows": rows, "solver": solve_rows, "failures": []}
    except Exception as exc:
        payload = {"status": "FAILED", "config_sha256": config_hash, "task_id": identifier, "seconds": time.perf_counter()-started, "rows": [], "solver": [], "failures": [{**task, "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc()}]}
    dump_json(checkpoint, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not args.run:
        raise SystemExit("use --run after WP1 Gate A passes")
    config, parent, config_hash = load_configuration()
    output = paths(config)
    for key in ("checkpoints", "logs", "results"):
        output[key].mkdir(parents=True, exist_ok=True)
    registry = tasks(config, parent)
    pd.DataFrame(registry).to_csv(output["analysis"] / "wp2" / "TASK_REGISTRY.tsv", sep="\t", index=False)
    payloads = []
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=int(config["execution"]["workers"])) as executor:
        future_map = {executor.submit(compute, task, config, config_hash): task for task in registry}
        for index, future in enumerate(as_completed(future_map), 1):
            payloads.append(future.result())
            progress = {"completed": index, "planned": len(registry), "successful": sum(p["status"] == "COMPLETED" for p in payloads), "failed": sum(p["status"] == "FAILED" for p in payloads), "wall_seconds": time.perf_counter()-started}
            dump_json(output["logs"] / "wp2_progress.json", progress)
            print(json.dumps(progress), flush=True)
    completed = [p for p in payloads if p["status"] == "COMPLETED"]
    rows = pd.DataFrame([row for p in completed for row in p["rows"]])
    solver = pd.DataFrame([row for p in completed for row in p["solver"]])
    failures = pd.DataFrame([row for p in payloads for row in p["failures"]])
    rows.to_csv(output["results"] / "wp2_derivative_cross_validation.tsv", sep="\t", index=False)
    solver.to_csv(output["results"] / "wp2_solver_diagnostics.tsv", sep="\t", index=False)
    failures.to_csv(output["results"] / "wp2_failures.tsv", sep="\t", index=False)
    gate = {
        "tasks": len(payloads),
        "completed": len(completed),
        "failed": len(payloads)-len(completed),
        "conditions": len(rows),
        "selected_independent_units": int(rows.independent_unit_id.nunique()) if len(rows) else 0,
        "all_linear_solves_converged": bool(len(rows) and rows.converged.astype(bool).all()),
        "global_plan_relative_l1_pass": bool(len(rows) and (rows.global_plan_relative_l1 <= float(config["wp2"]["global_plan_relative_l1_max"])).all()),
        "row_relative_l1_pass": bool(len(rows) and (rows.row_relative_l1_median <= float(config["wp2"]["row_median_relative_l1_max"])).all() and (rows.row_relative_l1_q90 <= float(config["wp2"]["row_q90_relative_l1_max"])).all()),
        "row_direction_cosine_pass": bool(len(rows) and (rows.row_direction_cosine_median >= float(config["wp2"]["row_median_direction_cosine_min"])).all()),
    }
    gate["wp2_pass"] = bool(not len(failures) and all(value for key, value in gate.items() if key.endswith("_pass") or key == "all_linear_solves_converged"))
    dump_json(output["results"] / "WP2_GATE.json", gate)
    print(json.dumps(gate, indent=2), flush=True)
    return 0 if gate["wp2_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
