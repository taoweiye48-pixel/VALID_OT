"""WP8 held-out gene split robustness on the frozen 600-gene universe."""

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
from validot.metrics import normalized_excess_aurc
from validot.p1 import P1Parameters, arm_weights, mixed_cost, solve_p1
from validot.postreview import row_response_rate, scalar_fidelity
from validot.solvers import cost_components


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "postreview_wp2_wp10_v1.json"
WP1_CONFIG_PATH = ROOT / "configs" / "postreview_wp1_wp10_v1.json"
PARENT_CONFIG_PATH = ROOT / "configs" / "p1_scale_regularization_v2.yaml"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_names(names: np.ndarray) -> str:
    return hashlib.sha256("\n".join(map(str, names)).encode("utf-8")).hexdigest()


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def output_paths(config: dict[str, Any]) -> dict[str, Path]:
    analysis = ROOT / config["output_root"]
    return {
        "wp1_checkpoints": analysis / "wp1" / "checkpoints",
        "wp1_arrays": analysis / "wp1" / "arrays",
        "wp46_checkpoints": analysis / "wp4_wp6" / "checkpoints",
        "wp46_arrays": analysis / "wp4_wp6" / "arrays",
        "checkpoints": analysis / "wp8" / "checkpoints",
        "arrays": analysis / "wp8" / "arrays",
        "logs": analysis / "logs",
        "results": ROOT / config["results_root"] / "wp8",
    }


def row_normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def eligible(pair: Any, extras: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source = np.column_stack([pair.source_x, extras["source_heldout"]])
    target = np.column_stack([pair.target_x, extras["target_heldout"]])
    genes = np.concatenate([extras["cost_genes"].astype(str), extras["heldout_genes"].astype(str)])
    if source.shape[1] != 600 or target.shape[1] != 600 or len(genes) != 600:
        raise RuntimeError("WP8 requires the frozen 600-gene eligible universe")
    if len(np.unique(genes)) != len(genes):
        raise RuntimeError("eligible gene names are not unique")
    order = np.argsort(genes, kind="mergesort")
    return source[:, order], target[:, order], genes[order]


def panel(pair: Any, extras: dict[str, np.ndarray], split: str, config: dict[str, Any]) -> dict[str, Any]:
    source, target, genes = eligible(pair, extras)
    n_cost = int(config["wp8"]["cost_features"])
    if split == "historical":
        historical_cost = set(extras["cost_genes"].astype(str))
        cost_index = np.asarray([index for index, gene in enumerate(genes) if gene in historical_cost], dtype=int)
        heldout_index = np.asarray([index for index, gene in enumerate(genes) if gene not in historical_cost], dtype=int)
    elif split == "source_only":
        variance = np.var(source, axis=0)
        order = np.lexsort((genes, -variance))
        cost_index, heldout_index = order[:n_cost], order[n_cost:]
    elif split.startswith("random_"):
        seed = int(split.rsplit("_", 1)[1])
        order = np.random.default_rng(seed).permutation(len(genes))
        cost_index, heldout_index = order[:n_cost], order[n_cost:]
    else:
        raise ValueError(split)
    if len(cost_index) != n_cost or len(heldout_index) != int(config["wp8"]["heldout_features"]):
        raise RuntimeError("invalid WP8 panel sizes")
    return {
        "source_cost": row_normalize(source[:, cost_index]),
        "target_cost": row_normalize(target[:, cost_index]),
        "source_heldout": row_normalize(source[:, heldout_index]),
        "target_heldout": row_normalize(target[:, heldout_index]),
        "cost_genes": genes[cost_index],
        "heldout_genes": genes[heldout_index],
        "cost_hash": hash_names(genes[cost_index]),
        "heldout_hash": hash_names(genes[heldout_index]),
        "historical_cost_overlap": float(np.mean(np.isin(genes[cost_index], extras["cost_genes"].astype(str)))),
    }


def registry(config: dict[str, Any], parent: dict[str, Any]) -> list[dict[str, Any]]:
    tasks = []
    for pair in parent["pairs"]:
        for direction in parent["directions"]:
            pair_file_id = pair["pair_id"] + ("__reverse" if direction == "reverse" else "")
            for method in config["wp8"]["method_scope"]:
                values = config["methods"][method]
                for split in config["wp8"]["primary_splits"]:
                    tasks.append({**pair, "direction": direction, "pair_file_id": pair_file_id, "method": method, "epsilon": values["epsilon"], "tau": values["tau"], "split": split})
    return tasks


def dependency(task: dict[str, Any], output: dict[str, Path], config_hash: str, wp1_hash: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    identifier = f"{task['pair_file_id']}__{task['method']}"
    wp1_status = json.loads((output["wp1_checkpoints"] / f"{identifier}.json").read_text(encoding="utf-8"))
    wp46_status = json.loads((output["wp46_checkpoints"] / f"{identifier}.json").read_text(encoding="utf-8"))
    if wp1_status.get("status") != "COMPLETED" or wp1_status.get("config_sha256") != wp1_hash:
        raise RuntimeError("invalid WP1 dependency")
    if wp46_status.get("status") != "COMPLETED" or wp46_status.get("config_sha256") != config_hash:
        raise RuntimeError("invalid WP4 dependency")
    with np.load(output["wp1_arrays"] / f"{identifier}.npz", allow_pickle=False) as wp1, np.load(output["wp46_arrays"] / f"{identifier}.npz", allow_pickle=False) as wp46:
        names = wp1["condition_names"].astype(str)
        condition = int(np.flatnonzero(names == "R__I_EXPR")[0])
        h_index = int(np.flatnonzero(np.isclose(wp1["h_grid"], 0.01))[0])
        return wp1["scores"][condition, h_index].copy(), wp1["reference_scores"][condition].copy(), wp46["endpoint_scores"][condition].copy()


def heldout_loss(plan: np.ndarray, source: np.ndarray, target: np.ndarray) -> np.ndarray:
    q = plan / np.maximum(plan.sum(axis=1, keepdims=True), 1e-300)
    transported = q @ target
    transported = row_normalize(transported)
    return 1.0 - np.sum(row_normalize(source) * transported, axis=1)


def compute(task: dict[str, Any], config: dict[str, Any], config_hash: str, wp1_hash: str) -> dict[str, Any]:
    output = output_paths(config)
    identifier = f"{task['pair_file_id']}__{task['method']}__{task['split']}"
    checkpoint = output["checkpoints"] / f"{identifier}.json"
    array_path = output["arrays"] / f"{identifier}.npz"
    if checkpoint.is_file() and array_path.is_file():
        prior = json.loads(checkpoint.read_text(encoding="utf-8"))
        if prior.get("status") == "COMPLETED" and prior.get("config_sha256") == config_hash:
            return prior
    started = time.perf_counter()
    try:
        pair, extras = load_pair(Path(config["processed_pair_root"]) / f"{task['pair_file_id']}.npz")
        selected = panel(pair, extras, task["split"], config)
        components = cost_components(selected["source_cost"], selected["target_cost"], pair.source_xy, pair.target_xy)
        expression, spatial = components["expression"], components["spatial_cross"]
        a = np.full(len(pair.source_x), 1.0/len(pair.source_x))
        b = np.full(len(pair.target_x), 1.0/len(pair.target_x))
        param = P1Parameters(task["method"], float(task["epsilon"]), task.get("tau"), int(config["solver"]["max_iter"]), float(config["solver"]["tolerance"]))
        base = solve_p1(mixed_cost(expression, spatial, (0.5, 0.5)), a, b, param)
        solver = [{"stage": "base", "converged": base.converged, "iterations": base.iterations, "seconds": base.seconds}]
        if not base.converged:
            raise RuntimeError("WP8 baseline failed")
        mass = base.plan.sum(axis=1)
        if task["split"] == "historical":
            finite, local, endpoint = dependency(task, output, config_hash, wp1_hash)
        else:
            derivatives = []
            observed = None
            endpoint_plan = None
            h_large, h_small = map(float, config["wp8"]["local_steps"])
            observed_h = float(config["wp8"]["observed_step"])
            for t in (h_large, h_small, observed_h, 1.0):
                result = solve_p1(mixed_cost(expression, spatial, arm_weights("R", "I_EXPR", t)), a, b, param)
                solver.append({"stage": "split_response", "t": t, "converged": result.converged, "iterations": result.iterations, "seconds": result.seconds})
                if not result.converged:
                    raise RuntimeError(f"WP8 solve failed t={t}")
                if t in (h_large, h_small):
                    derivatives.append((result.plan-base.plan)/t)
                elif t == observed_h:
                    observed = result.plan
                else:
                    endpoint_plan = result.plan
            derivative = 2.0*derivatives[1]-derivatives[0]
            local, _ = row_response_rate(derivative, mass, 1e-12)
            finite, _ = row_response_rate((observed-base.plan)/observed_h, mass, 1e-12)
            endpoint, _ = row_response_rate(endpoint_plan-base.plan, mass, 1e-12)
        loss = heldout_loss(base.plan, selected["source_heldout"], selected["target_heldout"])
        local_fidelity = scalar_fidelity(local, finite, 0.10)
        endpoint_transfer = scalar_fidelity(endpoint, local, 0.10)
        utilities = {name: normalized_excess_aurc(loss, score) for name, score in (("finite_response_h001", finite), ("local_reference", local), ("endpoint_response", endpoint))}
        rows = []
        for score_name, values in utilities.items():
            rows.append({
                **task,
                "cost_gene_hash": selected["cost_hash"],
                "heldout_gene_hash": selected["heldout_hash"],
                "historical_cost_overlap": selected["historical_cost_overlap"],
                "score": score_name,
                **{f"local_fidelity_{key}": value for key, value in local_fidelity.items()},
                **{f"endpoint_transfer_{key}": value for key, value in endpoint_transfer.items()},
                **{f"heldout_{key}": value for key, value in values.items()},
            })
        array_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = array_path.with_suffix(".npz.tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, finite_response_h001=finite, local_reference=local, endpoint_response=endpoint, heldout_expression_loss=loss, cost_genes=selected["cost_genes"], heldout_genes=selected["heldout_genes"])
        temporary.replace(array_path)
        payload = {"status": "COMPLETED", "config_sha256": config_hash, "task_id": identifier, "seconds": time.perf_counter()-started, "rows": rows, "solver": [{**task, **value} for value in solver], "failures": []}
    except Exception as exc:
        payload = {"status": "FAILED", "config_sha256": config_hash, "task_id": identifier, "seconds": time.perf_counter()-started, "rows": [], "solver": [], "failures": [{**task, "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc()}]}
    dump_json(checkpoint, payload)
    return payload


def panel_registry(tasks: list[dict[str, Any]], config: dict[str, Any]) -> pd.DataFrame:
    rows = []
    unique = {(task["pair_file_id"], task["split"]): task for task in tasks}
    for (pair_file_id, split), task in unique.items():
        pair, extras = load_pair(Path(config["processed_pair_root"]) / f"{pair_file_id}.npz")
        selected = panel(pair, extras, split, config)
        for role, genes in (("cost", selected["cost_genes"]), ("heldout", selected["heldout_genes"])):
            for order, gene in enumerate(genes):
                rows.append({"pair_file_id": pair_file_id, "direction": task["direction"], "split": split, "role": role, "order": order, "gene": str(gene), "panel_hash": selected[f"{role}_hash"]})
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not args.run:
        raise SystemExit("use --run after WP4/WP6")
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    parent = json.loads(PARENT_CONFIG_PATH.read_text(encoding="utf-8"))
    config_hash, wp1_hash = sha256(CONFIG_PATH), sha256(WP1_CONFIG_PATH)
    output = output_paths(config)
    for key in ("checkpoints", "arrays", "logs", "results"):
        output[key].mkdir(parents=True, exist_ok=True)
    tasks = registry(config, parent)
    pd.DataFrame(tasks).to_csv(output["results"] / "wp8_task_registry.tsv", sep="\t", index=False)
    panel_registry(tasks, config).to_csv(output["results"] / "wp8_gene_panels.tsv.gz", sep="\t", index=False, compression="gzip")
    payloads = []
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=int(config["execution"]["workers"])) as executor:
        futures = {executor.submit(compute, task, config, config_hash, wp1_hash): task for task in tasks}
        for index, future in enumerate(as_completed(futures), 1):
            payloads.append(future.result())
            progress = {"completed": index, "planned": len(tasks), "successful": sum(value["status"] == "COMPLETED" for value in payloads), "failed": sum(value["status"] == "FAILED" for value in payloads), "wall_seconds": time.perf_counter()-started}
            dump_json(output["logs"] / "wp8_progress.json", progress)
            print(json.dumps(progress), flush=True)
    completed = [value for value in payloads if value["status"] == "COMPLETED"]
    rows = pd.DataFrame([row for value in completed for row in value["rows"]])
    solver = pd.DataFrame([row for value in completed for row in value["solver"]])
    failures = pd.DataFrame([row for value in payloads for row in value["failures"]])
    rows.to_csv(output["results"] / "wp8_heldout_split_robustness_direction.tsv", sep="\t", index=False)
    solver.to_csv(output["results"] / "wp8_solver_diagnostics.tsv", sep="\t", index=False)
    failures.to_csv(output["results"] / "wp8_failures.tsv", sep="\t", index=False)
    metrics = [
        column
        for column in rows.columns
        if column.startswith(("local_fidelity_", "endpoint_transfer_", "heldout_"))
        and pd.api.types.is_numeric_dtype(rows[column])
    ]
    if "biological_pair_id" not in rows.columns:
        rows = rows.assign(biological_pair_id=rows["pair_id"])
    pair_keys = ["dataset", "pair_type", "biological_pair_id", "independent_unit_id", "cohort_role", "method", "epsilon", "tau", "split", "score"]
    pair = rows.groupby(pair_keys, dropna=False)[metrics].mean().reset_index()
    unit_keys = ["independent_unit_id", "cohort_role", "method", "epsilon", "tau", "split", "score"]
    unit = pair.groupby(unit_keys, dropna=False)[metrics].median().reset_index()
    pair.to_csv(output["results"] / "wp8_heldout_split_robustness_pair.tsv", sep="\t", index=False)
    unit.to_csv(output["results"] / "wp8_heldout_split_robustness_unit.tsv", sep="\t", index=False)
    gate = {"planned_tasks": len(tasks), "completed_tasks": len(completed), "failed_tasks": len(tasks)-len(completed), "gene_splits": len(config["wp8"]["primary_splits"]), "random_repeats": len(config["wp8"]["random_seeds"]), "solver_calls": len(solver), "solver_nonconvergence": int((~solver.converged.astype(bool)).sum()) if len(solver) else 0, "computational_pass": bool(len(completed) == len(tasks) and not len(failures))}
    dump_json(output["results"] / "WP8_GATE_C_GENE_SPLIT.json", gate)
    print(json.dumps(gate, indent=2), flush=True)
    return 0 if gate["computational_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
