"""Run WP5, WP9 and the primary (stored-severity) WP10 analyses.

The runner is deliberately downstream of WP1 and WP4/WP6.  It reuses their
frozen row responses and solves only the baseline plans needed for static
scores and controlled HER2ST correspondence truth.  Manual-layer donors are
computed in a separate branch and are never pooled with the 21 main units.
"""

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
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

from validot.evaluation import boundary_proximity
from validot.io import load_pair
from validot.metrics import fixed_budget_retained_loss, normalized_excess_aurc
from validot.p1 import P1Parameters, arm_weights, mixed_cost, solve_p1
from validot.postreview import CONDITIONS, row_response_rate, scalar_fidelity
from validot.solvers import cost_components


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "postreview_wp2_wp10_v1.json"
WP1_CONFIG_PATH = ROOT / "configs" / "postreview_wp1_wp10_v1.json"
PARENT_CONFIG_PATH = ROOT / "configs" / "p1_scale_regularization_v2.yaml"
MANUAL_ROOT = Path(
    os.environ.get(
        "VALIDOT_MANUAL_LAYER_PAIRS_ROOT",
        str(ROOT / "data" / "processed" / "p1_v2_manual_layer_pairs"),
    )
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def output_paths(config: dict[str, Any]) -> dict[str, Path]:
    analysis = ROOT / config["output_root"]
    results = ROOT / config["results_root"]
    return {
        "analysis": analysis,
        "wp1_checkpoints": analysis / "wp1" / "checkpoints",
        "wp1_arrays": analysis / "wp1" / "arrays",
        "wp46_checkpoints": analysis / "wp4_wp6" / "checkpoints",
        "wp46_arrays": analysis / "wp4_wp6" / "arrays",
        "checkpoints": analysis / "wp5_wp9_wp10" / "checkpoints",
        "arrays": analysis / "wp5_wp9_wp10" / "arrays",
        "logs": analysis / "logs",
        "wp5": results / "wp5",
        "wp9": results / "wp9",
        "wp10": results / "wp10",
    }


def parameters(method: str, config: dict[str, Any]) -> P1Parameters:
    values = config["methods"][method]
    return P1Parameters(
        method,
        float(values["epsilon"]),
        values.get("tau"),
        int(config["solver"]["max_iter"]),
        float(config["solver"]["tolerance"]),
    )


def conditional(plan: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mass = np.asarray(plan, dtype=float).sum(axis=1)
    q = plan / np.maximum(mass[:, None], 1e-300)
    return q, mass


def heldout_loss(q: np.ndarray, extras: dict[str, np.ndarray]) -> np.ndarray:
    transported = q @ extras["target_heldout"]
    transported /= np.maximum(np.linalg.norm(transported, axis=1, keepdims=True), 1e-12)
    source = np.asarray(extras["source_heldout"], dtype=float)
    source /= np.maximum(np.linalg.norm(source, axis=1, keepdims=True), 1e-12)
    return 1.0 - np.sum(source * transported, axis=1)


def local_scores(
    pair: Any,
    plan: np.ndarray,
    total_cost: np.ndarray,
    uniform_source_mass: np.ndarray,
) -> dict[str, np.ndarray]:
    q, mass = conditional(plan)
    rows = np.arange(len(q))
    if q.shape[1] > 1:
        top_two = np.partition(q, -2, axis=1)[:, -2:]
        best = np.max(top_two, axis=1)
        second = np.min(top_two, axis=1)
    else:
        best = q[:, 0]
        second = q[:, 0]
    predicted = np.argmax(q, axis=1)
    entropy = -np.sum(q * np.log(np.maximum(q, 1e-300)), axis=1) / max(np.log(q.shape[1]), 1.0)
    barycentric = q @ pair.target_xy
    tissue_scale = max(float(np.linalg.norm(np.ptp(pair.target_xy, axis=0))), 1e-12)
    return {
        "conditional_entropy": entropy,
        "low_max_probability": 1.0 - best,
        "probability_margin_risk": 1.0 - (best - second),
        "assigned_raw_cost": total_cost[rows, predicted],
        "barycentric_displacement": np.linalg.norm(barycentric - pair.source_xy, axis=1) / tissue_scale,
        "transported_mass_deficit": 1.0 - mass / np.maximum(uniform_source_mass, 1e-300),
        "source_boundary_proximity": boundary_proximity(pair.source_xy),
    }


def utility(
    loss: np.ndarray,
    risk: np.ndarray,
    *,
    eligible: np.ndarray | None = None,
    source_index: np.ndarray | None = None,
) -> dict[str, float]:
    keep = np.isfinite(loss) & np.isfinite(risk)
    if eligible is not None:
        eligible = np.asarray(eligible, dtype=bool)
        if eligible.shape != keep.shape:
            raise ValueError("eligible must align with loss and risk")
        keep &= eligible
    if int(keep.sum()) < 3 or np.unique(loss[keep]).size < 2:
        return {
            "n_estimable": int(keep.sum()),
            "spearman": float("nan"),
            "aurc": float("nan"),
            "oracle_aurc": float("nan"),
            "random_aurc": float("nan"),
            "normalized_excess_aurc": float("nan"),
            "retained_loss_at_80pct_coverage": float("nan"),
            "retained_loss_at_90pct_coverage": float("nan"),
        }
    values = normalized_excess_aurc(loss[keep], risk[keep])
    if source_index is not None:
        source_index = np.asarray(source_index)
        if source_index.shape != keep.shape:
            raise ValueError("source_index must align with loss and risk")
        coverages, retained = fixed_budget_retained_loss(
            loss[keep], risk[keep], source_index[keep], np.asarray([0.8, 0.9])
        )
        for coverage, value in zip(coverages, retained):
            values[f"retained_loss_at_{int(round(coverage * 100))}pct_coverage"] = float(value)
    rho = spearmanr(loss[keep], risk[keep]).statistic
    return {"n_estimable": int(keep.sum()), "spearman": float(rho), **values}


def binary_utility(
    label: np.ndarray,
    risk: np.ndarray,
    *,
    eligible: np.ndarray | None = None,
    source_index: np.ndarray | None = None,
) -> dict[str, float]:
    keep = np.isfinite(label) & np.isfinite(risk)
    if eligible is not None:
        eligible = np.asarray(eligible, dtype=bool)
        if eligible.shape != keep.shape:
            raise ValueError("eligible must align with label and risk")
        keep &= eligible
    y = label[keep].astype(int)
    score = risk[keep]
    selected_index = np.asarray(source_index)[keep] if source_index is not None else None
    base = utility(y.astype(float), score, source_index=selected_index)
    if len(y) < 3 or np.unique(y).size < 2:
        return {
            **base,
            "auroc": float("nan"),
            "auprc": float("nan"),
            "prevalence": float(np.mean(y)) if len(y) else float("nan"),
            "precision_at_prevalence": float("nan"),
            "recall_at_prevalence": float("nan"),
        }
    prevalence = float(np.mean(y))
    selected_n = max(1, int(np.ceil(prevalence * len(y))))
    order = np.argsort(-score, kind="mergesort")[:selected_n]
    true_positive = int(y[order].sum())
    return {
        **base,
        "auroc": float(roc_auc_score(y, score)),
        "auprc": float(average_precision_score(y, score)),
        "prevalence": prevalence,
        "precision_at_prevalence": true_positive / selected_n,
        "recall_at_prevalence": true_positive / max(int(y.sum()), 1),
    }


def metadata(task: dict[str, Any], pair: Any) -> dict[str, Any]:
    return {
        "dataset": task.get("dataset", pair.dataset),
        "pair_type": task.get("pair_type", pair.metadata.get("pair_type", "")),
        "biological_pair_id": task.get("biological_pair_id", pair.metadata.get("biological_pair_id", pair.pair_id)),
        "independent_unit_id": task.get("independent_unit_id", pair.metadata.get("independent_unit_id", "")),
        "cohort_role": task.get("cohort_role", pair.metadata.get("cohort_role", "")),
        "pair_id": task["pair_file_id"],
        "direction": task.get("direction", pair.metadata.get("direction", "forward")),
        "method": task["method"],
        "epsilon": task["epsilon"],
        "tau": task.get("tau"),
    }


def main_registry(config: dict[str, Any], parent: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    for pair in parent["pairs"]:
        for direction in parent["directions"]:
            pair_file_id = pair["pair_id"] + ("__reverse" if direction == "reverse" else "")
            for method, values in config["methods"].items():
                result.append({
                    **pair,
                    "branch": "main21",
                    "direction": direction,
                    "pair_file_id": pair_file_id,
                    "pair_path": str(Path(config["processed_pair_root"]) / f"{pair_file_id}.npz"),
                    "method": method,
                    "epsilon": values["epsilon"],
                    "tau": values["tau"],
                })
    return result


def manual_registry(config: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    for path in sorted(MANUAL_ROOT.glob("*.npz")):
        pair, _ = load_pair(path)
        direction = str(pair.metadata.get("direction", "reverse" if "__reverse" in path.stem else "forward"))
        for method, values in config["methods"].items():
            result.append({
                "branch": "manual3",
                "dataset": pair.dataset,
                "pair_type": pair.metadata.get("pair_type", "manual_layer_adjacent"),
                "biological_pair_id": pair.metadata.get("biological_pair_id", pair.pair_id.replace("__reverse", "")),
                "independent_unit_id": pair.metadata.get("independent_unit_id", ""),
                "cohort_role": "manual_layer_truth",
                "direction": direction,
                "pair_file_id": path.stem,
                "pair_path": str(path),
                "method": method,
                "epsilon": values["epsilon"],
                "tau": values["tau"],
            })
    return result


def dependency_scores(task: dict[str, Any], output: dict[str, Path], config_hash: str, wp1_hash: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    identifier = f"{task['pair_file_id']}__{task['method']}"
    wp1_checkpoint = json.loads((output["wp1_checkpoints"] / f"{identifier}.json").read_text(encoding="utf-8"))
    wp46_checkpoint = json.loads((output["wp46_checkpoints"] / f"{identifier}.json").read_text(encoding="utf-8"))
    if wp1_checkpoint.get("status") != "COMPLETED" or wp1_checkpoint.get("config_sha256") != wp1_hash:
        raise RuntimeError("invalid WP1 dependency")
    if wp46_checkpoint.get("status") != "COMPLETED" or wp46_checkpoint.get("config_sha256") != config_hash:
        raise RuntimeError("invalid WP4/WP6 dependency")
    with np.load(output["wp1_arrays"] / f"{identifier}.npz", allow_pickle=False) as stored:
        names = stored["condition_names"].astype(str)
        condition = int(np.flatnonzero(names == "R__I_EXPR")[0])
        h_grid = stored["h_grid"]
        h_index = int(np.flatnonzero(np.isclose(h_grid, 0.01))[0])
        finite = stored["scores"][condition, h_index].copy()
        local = stored["reference_scores"][condition].copy()
    with np.load(output["wp46_arrays"] / f"{identifier}.npz", allow_pickle=False) as stored:
        endpoint = stored["endpoint_scores"][condition].copy()
        uot = {key: stored[key].copy() for key in stored.files if key.startswith("R__I_EXPR__")}
    return finite, local, endpoint, uot


def manual_responses(pair: Any, task: dict[str, Any], config: dict[str, Any], expression: np.ndarray, spatial: np.ndarray, a: np.ndarray, b: np.ndarray) -> tuple[Any, np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    param = parameters(task["method"], config)
    base = solve_p1(mixed_cost(expression, spatial, (0.5, 0.5)), a, b, param)
    if not base.converged:
        raise RuntimeError("manual baseline failed")
    mass = base.plan.sum(axis=1)
    derivatives = []
    solver = [{"stage": "base", "converged": base.converged, "iterations": base.iterations, "seconds": base.seconds}]
    observed = None
    endpoint_plan = None
    for t in (0.00125, 0.000625, 0.01, 1.0):
        result = solve_p1(mixed_cost(expression, spatial, arm_weights("R", "I_EXPR", t)), a, b, param)
        solver.append({"stage": "manual_response", "t": t, "converged": result.converged, "iterations": result.iterations, "seconds": result.seconds})
        if not result.converged:
            raise RuntimeError(f"manual response failed t={t}")
        if t in (0.00125, 0.000625):
            derivatives.append((result.plan - base.plan) / t)
        elif t == 0.01:
            observed = result.plan
        else:
            endpoint_plan = result.plan
    derivative = 2.0 * derivatives[1] - derivatives[0]
    local, _ = row_response_rate(derivative, mass, 1e-12)
    finite, _ = row_response_rate((observed - base.plan) / 0.01, mass, 1e-12)
    endpoint, _ = row_response_rate(endpoint_plan - base.plan, mass, 1e-12)
    return base, finite, local, endpoint, solver


def compute(task: dict[str, Any], config: dict[str, Any], config_hash: str, wp1_hash: str) -> dict[str, Any]:
    output = output_paths(config)
    identifier = f"{task['branch']}__{task['pair_file_id']}__{task['method']}"
    checkpoint = output["checkpoints"] / f"{identifier}.json"
    array_path = output["arrays"] / f"{identifier}.npz"
    if checkpoint.is_file() and array_path.is_file():
        prior = json.loads(checkpoint.read_text(encoding="utf-8"))
        if prior.get("status") == "COMPLETED" and prior.get("config_sha256") == config_hash:
            return prior
    started = time.perf_counter()
    try:
        pair, extras = load_pair(Path(task["pair_path"]))
        components = cost_components(pair.source_x, pair.target_x, pair.source_xy, pair.target_xy)
        expression, spatial = components["expression"], components["spatial_cross"]
        total_cost = mixed_cost(expression, spatial, (0.5, 0.5))
        a = np.full(len(pair.source_x), 1.0 / len(pair.source_x))
        b = np.full(len(pair.target_x), 1.0 / len(pair.target_x))
        solver_rows: list[dict[str, Any]] = []
        uot_arrays: dict[str, np.ndarray] = {}
        if task["branch"] == "main21":
            finite, local, endpoint, uot_arrays = dependency_scores(task, output, config_hash, wp1_hash)
            base = solve_p1(total_cost, a, b, parameters(task["method"], config))
            solver_rows.append({"stage": "baseline", "converged": base.converged, "iterations": base.iterations, "seconds": base.seconds})
            if not base.converged:
                raise RuntimeError("main baseline failed")
        else:
            base, finite, local, endpoint, solver_rows = manual_responses(pair, task, config, expression, spatial, a, b)
        q, mass = conditional(base.plan)
        scores = {
            "finite_response_h001": finite,
            "local_reference": local,
            "endpoint_response": endpoint,
            **local_scores(pair, base.plan, total_cost, a),
        }
        witnesses = {"heldout_expression_loss": heldout_loss(q, extras)}
        if task["branch"] == "manual3":
            predicted = np.argmax(q, axis=1)
            witnesses["manual_layer_mismatch"] = (pair.source_labels.astype(str) != pair.target_labels.astype(str)[predicted]).astype(float)
        meta = metadata(task, pair)
        wp5_rows = []
        for witness_name, loss in witnesses.items():
            for score_name in ("finite_response_h001", "local_reference", "endpoint_response", "source_boundary_proximity"):
                wp5_rows.append({**meta, "branch": task["branch"], "witness": witness_name, "score": score_name, **utility(loss, scores[score_name])})
        wp9_internal = []
        for score_name, score in scores.items():
            for reference_name, reference in (("local_reference", local), ("endpoint_response", endpoint)):
                values = scalar_fidelity(reference, score, 0.10)
                wp9_internal.append({
                    **meta,
                    "branch": task["branch"],
                    "score": score_name,
                    "reference": reference_name,
                    "amplitude_comparable": score_name in {"finite_response_h001", "local_reference", "endpoint_response"},
                    **values,
                })
        wp9_external = []
        for witness_name, loss in witnesses.items():
            for score_name, score in scores.items():
                wp9_external.append({**meta, "branch": task["branch"], "witness": witness_name, "score": score_name, **utility(loss, score)})
        wp10_corr: list[dict[str, Any]] = []
        wp10_missing: list[dict[str, Any]] = []
        truth_arrays: dict[str, np.ndarray] = {}
        if task["branch"] == "main21" and str(meta["dataset"]).startswith("HER2ST"):
            truth = np.asarray(pair.truth_target, dtype=int)
            valid = (truth >= 0) & (truth < q.shape[1])
            ranks = np.full(len(q), np.nan)
            true_probability = np.full(len(q), np.nan)
            top_hits = {int(k): np.full(len(q), np.nan) for k in config["wp10"]["top_k"]}
            bary_error = np.full(len(q), np.nan)
            if np.any(valid):
                row_indices = np.flatnonzero(valid)
                true_probability[valid] = q[row_indices, truth[valid]]
                order = np.argsort(-q[valid], axis=1, kind="mergesort")
                ranks[valid] = np.argmax(order == truth[valid][:, None], axis=1) + 1
                for k, values in top_hits.items():
                    values[valid] = (ranks[valid] <= k).astype(float)
                tissue_scale = max(float(np.linalg.norm(np.ptp(pair.target_xy, axis=0))), 1e-12)
                bary = q @ pair.target_xy
                bary_error[valid] = np.linalg.norm(bary[valid] - pair.target_xy[truth[valid]], axis=1) / tissue_scale
            wp10_corr.append({
                **meta,
                "n_truth": int(valid.sum()),
                "top1": float(np.nanmean(top_hits[1])) if 1 in top_hits else float("nan"),
                "top5": float(np.nanmean(top_hits[5])) if 5 in top_hits else float("nan"),
                "top10": float(np.nanmean(top_hits[10])) if 10 in top_hits else float("nan"),
                "median_true_probability": float(np.nanmedian(true_probability)),
                "median_reciprocal_rank": float(np.nanmedian(1.0 / ranks)),
                "median_normalized_barycentric_error": float(np.nanmedian(bary_error)),
            })
            truth_arrays = {"truth_rank": ranks, "truth_probability": true_probability, "truth_barycentric_error": bary_error, **{f"truth_top{k}": values for k, values in top_hits.items()}}
            missing = np.asarray(pair.truth_missing, dtype=bool).astype(int)
            crop_scores = dict(scores)
            for key, value in uot_arrays.items():
                clean = key.replace("R__I_EXPR__", "")
                crop_scores[f"uot_{clean}"] = value
            if np.unique(missing).size == 2:
                for score_name, score in crop_scores.items():
                    wp10_missing.append({**meta, "score": score_name, **binary_utility(missing, score)})
            top1_error = np.full(len(q), np.nan)
            top1_error[valid] = 1.0 - top_hits[1][valid]
            fixed_budget_eligible = valid & (mass > 1e-12)
            frozen_source_index = np.arange(len(q), dtype=int)
            for score_name, score in scores.items():
                if np.unique(top1_error[np.isfinite(top1_error)]).size == 2:
                    wp10_corr.append({
                        **meta,
                        "metric": "top1_error_detection",
                        "score": score_name,
                        **binary_utility(
                            top1_error,
                            score,
                            eligible=fixed_budget_eligible,
                            source_index=frozen_source_index,
                        ),
                    })
                wp10_corr.append({**meta, "metric": "barycentric_error_ranking", "score": score_name, **utility(bary_error, score)})
        array_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = array_path.with_suffix(".npz.tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **scores, **witnesses, **truth_arrays)
        temporary.replace(array_path)
        payload = {
            "status": "COMPLETED",
            "config_sha256": config_hash,
            "task_id": identifier,
            "seconds": time.perf_counter() - started,
            "wp5": wp5_rows,
            "wp9_internal": wp9_internal,
            "wp9_external": wp9_external,
            "wp10_correspondence": wp10_corr,
            "wp10_missingness": wp10_missing,
            "solver": [{**meta, **row} for row in solver_rows],
            "failures": [],
        }
    except Exception as exc:
        payload = {
            "status": "FAILED",
            "config_sha256": config_hash,
            "task_id": identifier,
            "seconds": time.perf_counter() - started,
            "wp5": [],
            "wp9_internal": [],
            "wp9_external": [],
            "wp10_correspondence": [],
            "wp10_missingness": [],
            "solver": [],
            "failures": [{**task, "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc()}],
        }
    dump_json(checkpoint, payload)
    return payload


def aggregate_direction(frame: pd.DataFrame, metric_columns: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    dimensions = [column for column in frame.columns if column not in metric_columns and column not in {"pair_id", "direction"}]
    pair_dimensions = [column for column in dimensions if column not in {"independent_unit_id", "cohort_role"}] + ["independent_unit_id", "cohort_role"]
    pair = frame.groupby(pair_dimensions, dropna=False)[metric_columns].mean().reset_index()
    unit_dimensions = [column for column in dimensions if column not in {"biological_pair_id", "dataset", "pair_type"}]
    unit = pair.groupby(unit_dimensions, dropna=False)[metric_columns].median().reset_index()
    return pair, unit


def write_frame(rows: list[dict[str, Any]], path: Path) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep="\t", index=False)
    return frame


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not args.run:
        raise SystemExit("use --run after WP4/WP6 completion")
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    parent = json.loads(PARENT_CONFIG_PATH.read_text(encoding="utf-8"))
    config_hash, wp1_hash = sha256(CONFIG_PATH), sha256(WP1_CONFIG_PATH)
    output = output_paths(config)
    for key in ("checkpoints", "arrays", "logs", "wp5", "wp9", "wp10"):
        output[key].mkdir(parents=True, exist_ok=True)
    tasks = main_registry(config, parent) + manual_registry(config)
    pd.DataFrame(tasks).to_csv(output["analysis"] / "wp5_wp9_wp10" / "TASK_REGISTRY.tsv", sep="\t", index=False)
    payloads = []
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=int(config["execution"]["workers"])) as executor:
        futures = {executor.submit(compute, task, config, config_hash, wp1_hash): task for task in tasks}
        for index, future in enumerate(as_completed(futures), 1):
            payloads.append(future.result())
            progress = {
                "completed": index,
                "planned": len(tasks),
                "successful": sum(value["status"] == "COMPLETED" for value in payloads),
                "failed": sum(value["status"] == "FAILED" for value in payloads),
                "wall_seconds": time.perf_counter() - started,
            }
            dump_json(output["logs"] / "wp5_wp9_wp10_progress.json", progress)
            print(json.dumps(progress), flush=True)
    completed = [value for value in payloads if value["status"] == "COMPLETED"]
    failures = [row for value in payloads for row in value["failures"]]
    wp5 = write_frame([row for value in completed for row in value["wp5"]], output["wp5"] / "wp5_same_score_external_utility_direction.tsv")
    wp9_internal = write_frame([row for value in completed for row in value["wp9_internal"]], output["wp9"] / "wp9_real_score_internal_audit_direction.tsv")
    wp9_external = write_frame([row for value in completed for row in value["wp9_external"]], output["wp9"] / "wp9_real_score_external_audit_direction.tsv")
    write_frame([row for value in completed for row in value["wp10_correspondence"]], output["wp10"] / "wp10_her2st_correspondence_truth_direction.tsv")
    write_frame([row for value in completed for row in value["wp10_missingness"]], output["wp10"] / "wp10_crop_missingness_utility_direction.tsv")
    write_frame([row for value in completed for row in value["solver"]], output["wp5"] / "wp5_wp9_wp10_solver_diagnostics.tsv")
    write_frame(failures, output["wp5"] / "wp5_wp9_wp10_failures.tsv")
    registry = pd.DataFrame([
        {"score": name, "risk_direction": "larger_is_higher_risk", "primary": name in config["wp9"]["core_scores"]}
        for name in sorted(set(wp9_internal.score))
    ])
    registry.to_csv(output["wp9"] / "wp9_score_registry.tsv", sep="\t", index=False)
    for label, frame, result_path in (
        ("wp5", wp5, output["wp5"] / "wp5_same_score_external_utility_unit.tsv"),
        ("wp9_internal", wp9_internal, output["wp9"] / "wp9_real_score_internal_audit_unit.tsv"),
        ("wp9_external", wp9_external, output["wp9"] / "wp9_real_score_external_audit_unit.tsv"),
    ):
        if len(frame):
            metrics = [column for column in frame.columns if column in {"n_estimable", "spearman", "aurc", "oracle_aurc", "random_aurc", "normalized_excess_aurc", "retained_loss_at_80pct_coverage", "retained_loss_at_90pct_coverage", "top_overlap", "raw_mae", "rmae", "intercept", "slope", "r2"}]
            _, unit = aggregate_direction(frame, metrics)
            unit.to_csv(result_path, sep="\t", index=False)
    gate = {
        "planned_tasks": len(tasks),
        "completed_tasks": len(completed),
        "failed_tasks": len(payloads) - len(completed),
        "main_tasks": sum(task["branch"] == "main21" for task in tasks),
        "manual_layer_tasks": sum(task["branch"] == "manual3" for task in tasks),
        "wp5_rows": len(wp5),
        "wp9_internal_rows": len(wp9_internal),
        "wp9_external_rows": len(wp9_external),
        "computational_pass": bool(len(completed) == len(tasks) and not failures),
    }
    dump_json(output["wp10"] / "WP5_WP9_GATE_D_AND_WP10_GATE_E_PRIMARY.json", gate)
    print(json.dumps(gate, indent=2), flush=True)
    return 0 if gate["computational_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
