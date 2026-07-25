"""Run the frozen sparse full-cohort WP11 response-surface analysis."""

from __future__ import annotations

import argparse
import copy
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

from validot.io import load_pair
from validot.metrics import normalized_excess_aurc
from validot.p1 import P1Parameters, mixed_cost, solve_p1
from validot.postreview import finite_summary, scalar_fidelity
from validot.solvers import cost_components
from validot.wp11 import (
    build_condition_registries,
    factor_coordinates,
    factorial_effect,
    interpolate_from_baseline,
    regularization,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "postreview_wp11_v1.json"


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


def load_configuration() -> tuple[dict[str, Any], dict[str, Any], str]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    parent_path = ROOT / config["parent_config"]
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    return config, parent, sha256(CONFIG_PATH)


def validate_dependencies(config: dict[str, Any]) -> None:
    status = json.loads((ROOT / config["wp2_wp10_status"]).read_text(encoding="utf-8"))
    if status.get("state") != "COMPLETE" or not status.get("stages", {}).get("PIPELINE", {}).get("passed"):
        raise RuntimeError("WP1-WP10 pipeline is not complete")


def output_paths(config: dict[str, Any]) -> dict[str, Path]:
    analysis = ROOT / config["output_root"]
    results = ROOT / config["results_root"] / "wp11"
    return {
        "analysis": analysis,
        "checkpoints": analysis / "checkpoints",
        "arrays": analysis / "arrays",
        "logs": analysis / "logs",
        "results": results,
        "status": analysis / "WP11_STATUS.json",
    }


def task_registry(config: dict[str, Any], parent: dict[str, Any]) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for pair in parent["pairs"]:
        for direction in parent["directions"]:
            pair_file_id = pair["pair_id"] + ("__reverse" if direction == "reverse" else "")
            for method, values in config["methods"].items():
                tasks.append(
                    {
                        **pair,
                        "direction": direction,
                        "pair_file_id": pair_file_id,
                        "method": method,
                        "epsilon0": float(values["epsilon"]),
                        "tau0": values.get("tau"),
                    }
                )
    return tasks


def task_id(task: dict[str, Any]) -> str:
    return f"{task['pair_file_id']}__{task['method']}"


def parameters(
    task: dict[str, Any],
    config: dict[str, Any],
    u: float,
    regime: str,
) -> P1Parameters:
    epsilon, tau = regularization(
        task["method"], task["epsilon0"], task.get("tau0"), u, regime
    )
    return P1Parameters(
        task["method"],
        epsilon,
        tau,
        int(config["solver"]["max_iter"]),
        float(config["solver"]["tolerance"]),
    )


def fixed_row_distance(
    baseline: np.ndarray,
    comparison: np.ndarray,
    baseline_mass: np.ndarray,
    threshold: float,
    divisor: float = 1.0,
) -> np.ndarray:
    estimable = np.isfinite(baseline_mass) & (baseline_mass > float(threshold))
    score = np.full(len(baseline_mass), np.nan, dtype=float)
    score[estimable] = (
        np.abs(comparison[estimable] - baseline[estimable]).sum(axis=1)
        / (2.0 * baseline_mass[estimable] * float(divisor))
    )
    return score


def conditional(plan: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mass = np.asarray(plan, dtype=float).sum(axis=1)
    estimable = np.isfinite(mass) & (mass > float(threshold))
    q = np.full_like(plan, np.nan, dtype=float)
    q[estimable] = plan[estimable] / mass[estimable, None]
    return q, mass, estimable


def heldout_loss(q: np.ndarray, extras: dict[str, np.ndarray]) -> np.ndarray | None:
    if "source_heldout" not in extras or "target_heldout" not in extras:
        return None
    source = np.asarray(extras["source_heldout"], dtype=float)
    target = np.asarray(extras["target_heldout"], dtype=float)
    transported = q @ target
    transported /= np.maximum(np.linalg.norm(transported, axis=1, keepdims=True), 1e-12)
    source = source / np.maximum(np.linalg.norm(source, axis=1, keepdims=True), 1e-12)
    return 1.0 - np.sum(source * transported, axis=1)


def controlled_truth_loss(pair: Any, q: np.ndarray) -> np.ndarray | None:
    truth = np.asarray(pair.truth_target, dtype=int)
    valid = (~np.asarray(pair.truth_missing, dtype=bool)) & (truth >= 0) & (truth < len(pair.target_xy))
    if not np.any(valid):
        return None
    loss = np.full(len(truth), np.nan, dtype=float)
    barycenter = q @ np.asarray(pair.target_xy, dtype=float)
    scale = max(float(np.linalg.norm(np.ptp(pair.target_xy, axis=0))), 1e-12)
    loss[valid] = np.linalg.norm(barycenter[valid] - pair.target_xy[truth[valid]], axis=1) / scale
    return loss


def utility(loss: np.ndarray | None, risk: np.ndarray) -> dict[str, float]:
    empty = {
        "n_estimable": 0,
        "spearman": float("nan"),
        "aurc": float("nan"),
        "oracle_aurc": float("nan"),
        "random_aurc": float("nan"),
        "normalized_excess_aurc": float("nan"),
        "retained_loss_at_80pct_coverage": float("nan"),
        "retained_loss_at_90pct_coverage": float("nan"),
    }
    if loss is None:
        return empty
    keep = np.isfinite(loss) & np.isfinite(risk)
    if int(keep.sum()) < 3 or np.unique(loss[keep]).size < 2:
        return {**empty, "n_estimable": int(keep.sum())}
    values = normalized_excess_aurc(loss[keep], risk[keep])
    rho = spearmanr(loss[keep], risk[keep]).statistic
    return {"n_estimable": int(keep.sum()), "spearman": float(rho), **values}


def binary_utility(labels: np.ndarray, risk: np.ndarray) -> dict[str, float]:
    keep = np.isfinite(risk)
    y = np.asarray(labels, dtype=bool)[keep].astype(int)
    score = np.asarray(risk, dtype=float)[keep]
    if len(y) < 3 or np.unique(y).size < 2:
        return {"n_estimable": int(len(y)), "auroc": float("nan"), "auprc": float("nan")}
    return {
        "n_estimable": int(len(y)),
        "auroc": float(roc_auc_score(y, score)),
        "auprc": float(average_precision_score(y, score)),
    }


def uot_components(
    baseline: np.ndarray,
    comparison: np.ndarray,
    threshold: float,
    divisor: float,
) -> dict[str, np.ndarray]:
    q0, m0, ok0 = conditional(baseline, threshold)
    q1, m1, ok1 = conditional(comparison, threshold)
    ok = ok0 & ok1
    mass = np.full(len(m0), np.nan)
    shape = np.full(len(m0), np.nan)
    combined = np.full(len(m0), np.nan)
    denominator = np.maximum(m1[ok] + m0[ok], 1e-300) * float(divisor)
    mass[ok] = np.abs(m1[ok] - m0[ok]) / denominator
    shape[ok] = 0.5 * np.abs(q1[ok] - q0[ok]).sum(axis=1) / float(divisor)
    combined[ok] = np.abs(comparison[ok] - baseline[ok]).sum(axis=1) / denominator
    return {"mass": mass, "shape": shape, "combined": combined}


def prefixed_summary(prefix: str, values: np.ndarray) -> dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in finite_summary(values).items()}


def compute_task(
    task: dict[str, Any],
    config: dict[str, Any],
    config_hash: str,
) -> dict[str, Any]:
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
        pair, extras = load_pair(Path(config["processed_pair_root"]) / f"{task['pair_file_id']}.npz")
        components = cost_components(pair.source_x, pair.target_x, pair.source_xy, pair.target_xy)
        expression = components["expression"]
        spatial = components["spatial_cross"]
        a = np.full(len(pair.source_x), 1.0 / len(pair.source_x))
        b = np.full(len(pair.target_x), 1.0 / len(pair.target_x))
        threshold = float(config["wp11"]["row_mass_estimable_min"])
        h = float(config["wp11"]["finite_step"])
        logical, physical = build_condition_registries(config["wp11"])
        base_parameters = parameters(task, config, 1.0, "fixed")
        base = solve_p1(mixed_cost(expression, spatial, (0.5, 0.5)), a, b, base_parameters)
        if not base.converged:
            raise RuntimeError("WP11 baseline did not converge")
        q0, baseline_mass, _ = conditional(base.plan, threshold)
        heldout = heldout_loss(q0, extras)
        truth_loss = controlled_truth_loss(pair, q0)
        crop_truth = np.asarray(pair.truth_missing, dtype=bool)
        solver_rows = [
            {
                **task,
                "stage": "baseline",
                "physical_id": "baseline",
                "alpha": 0.5,
                "beta": 0.5,
                "u": 1.0,
                "v": 0.5,
                "regularization_regime": "fixed",
                "epsilon": base_parameters.epsilon,
                "tau": base_parameters.tau,
                "converged": base.converged,
                "iterations": base.iterations,
                "seconds": base.seconds,
            }
        ]
        physical_results: dict[str, dict[str, Any]] = {}
        endpoint_arrays: list[np.ndarray] = []
        h_arrays: list[np.ndarray] = []
        uot_endpoint_mass: list[np.ndarray] = []
        uot_endpoint_shape: list[np.ndarray] = []
        uot_endpoint_combined: list[np.ndarray] = []
        uot_h_mass: list[np.ndarray] = []
        uot_h_shape: list[np.ndarray] = []
        uot_h_combined: list[np.ndarray] = []
        physical_ids: list[str] = []
        for condition in physical:
            alpha = float(condition["alpha"])
            beta = float(condition["beta"])
            u = float(condition["u"])
            regime = str(condition["regularization_regime"])
            is_baseline = bool(np.isclose(alpha, 0.5) and np.isclose(beta, 0.5))
            endpoint_parameters = parameters(task, config, u, regime)
            if is_baseline:
                endpoint = base
                observed = base
                alpha_h, beta_h, u_h = 0.5, 0.5, 1.0
                h_parameters = base_parameters
            else:
                endpoint = solve_p1(
                    mixed_cost(expression, spatial, (alpha, beta)),
                    a,
                    b,
                    endpoint_parameters,
                )
                if not endpoint.converged:
                    raise RuntimeError(f"WP11 endpoint did not converge: {condition['physical_id']}")
                alpha_h, beta_h, u_h = interpolate_from_baseline(alpha, beta, h)
                h_parameters = parameters(task, config, u_h, regime)
                observed = solve_p1(
                    mixed_cost(expression, spatial, (alpha_h, beta_h)),
                    a,
                    b,
                    h_parameters,
                )
                if not observed.converged:
                    raise RuntimeError(f"WP11 h=0.01 state did not converge: {condition['physical_id']}")
                for stage, result, pars, ca, cb, cu in (
                    ("endpoint", endpoint, endpoint_parameters, alpha, beta, u),
                    ("finite_h001", observed, h_parameters, alpha_h, beta_h, u_h),
                ):
                    solver_rows.append(
                        {
                            **task,
                            "stage": stage,
                            "physical_id": condition["physical_id"],
                            "alpha": ca,
                            "beta": cb,
                            "u": cu,
                            "v": cb / max(ca + cb, 1e-300),
                            "regularization_regime": regime,
                            "epsilon": pars.epsilon,
                            "tau": pars.tau,
                            "converged": result.converged,
                            "iterations": result.iterations,
                            "seconds": result.seconds,
                        }
                    )
            endpoint_score = fixed_row_distance(base.plan, endpoint.plan, baseline_mass, threshold)
            h_score = fixed_row_distance(base.plan, observed.plan, baseline_mass, threshold, h)
            fidelity = scalar_fidelity(endpoint_score, h_score, float(config["wp11"]["top_fraction"]))
            row = {
                "physical_id": condition["physical_id"],
                "endpoint_epsilon": endpoint_parameters.epsilon,
                "endpoint_tau": endpoint_parameters.tau,
                "h001_alpha": alpha_h,
                "h001_beta": beta_h,
                "h001_u": u_h,
                "h001_epsilon": h_parameters.epsilon,
                "h001_tau": h_parameters.tau,
                **prefixed_summary("endpoint_response", endpoint_score),
                **prefixed_summary("h001_response", h_score),
                **{f"h001_to_endpoint_{key}": value for key, value in fidelity.items()},
                **{f"heldout_endpoint_{key}": value for key, value in utility(heldout, endpoint_score).items()},
                **{f"heldout_h001_{key}": value for key, value in utility(heldout, h_score).items()},
                **{f"truth_endpoint_{key}": value for key, value in utility(truth_loss, endpoint_score).items()},
                **{f"truth_h001_{key}": value for key, value in utility(truth_loss, h_score).items()},
                **{f"crop_endpoint_{key}": value for key, value in binary_utility(crop_truth, endpoint_score).items()},
                **{f"crop_h001_{key}": value for key, value in binary_utility(crop_truth, h_score).items()},
            }
            if task["method"] == "uot":
                endpoint_components = uot_components(base.plan, endpoint.plan, threshold, 1.0)
                h_components = uot_components(base.plan, observed.plan, threshold, h)
                for name in ("mass", "shape", "combined"):
                    row.update(prefixed_summary(f"uot_endpoint_{name}", endpoint_components[name]))
                    row.update(prefixed_summary(f"uot_h001_{name}", h_components[name]))
                uot_endpoint_mass.append(endpoint_components["mass"])
                uot_endpoint_shape.append(endpoint_components["shape"])
                uot_endpoint_combined.append(endpoint_components["combined"])
                uot_h_mass.append(h_components["mass"])
                uot_h_shape.append(h_components["shape"])
                uot_h_combined.append(h_components["combined"])
            physical_results[condition["physical_id"]] = row
            physical_ids.append(condition["physical_id"])
            endpoint_arrays.append(endpoint_score)
            h_arrays.append(h_score)
        rows = [{**task, **condition, **physical_results[condition["physical_id"]]} for condition in logical]
        array_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = array_path.with_suffix(".npz.tmp")
        arrays: dict[str, Any] = {
            "physical_ids": np.asarray(physical_ids),
            "endpoint_response": np.stack(endpoint_arrays),
            "h001_response": np.stack(h_arrays),
            "baseline_heldout_loss": heldout if heldout is not None else np.full(len(a), np.nan),
            "controlled_truth_loss": truth_loss if truth_loss is not None else np.full(len(a), np.nan),
            "crop_truth": crop_truth,
        }
        if task["method"] == "uot":
            arrays.update(
                {
                    "uot_endpoint_mass": np.stack(uot_endpoint_mass),
                    "uot_endpoint_shape": np.stack(uot_endpoint_shape),
                    "uot_endpoint_combined": np.stack(uot_endpoint_combined),
                    "uot_h001_mass": np.stack(uot_h_mass),
                    "uot_h001_shape": np.stack(uot_h_shape),
                    "uot_h001_combined": np.stack(uot_h_combined),
                }
            )
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        temporary.replace(array_path)
        payload = {
            "status": "COMPLETED",
            "config_sha256": config_hash,
            "task_id": identifier,
            "seconds": time.perf_counter() - started,
            "rows": rows,
            "solver": solver_rows,
            "failures": [],
        }
    except Exception as exc:
        payload = {
            "status": "FAILED",
            "config_sha256": config_hash,
            "task_id": identifier,
            "seconds": time.perf_counter() - started,
            "rows": [],
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
    dump_json(checkpoint, payload)
    return payload


SURFACE_METRIC_PREFIXES = (
    "endpoint_response_",
    "h001_response_",
    "h001_to_endpoint_",
    "heldout_endpoint_",
    "heldout_h001_",
    "truth_endpoint_",
    "truth_h001_",
    "crop_endpoint_",
    "crop_h001_",
    "uot_endpoint_",
    "uot_h001_",
)

CONTRAST_METRICS = (
    "endpoint_response_median",
    "endpoint_response_q90",
    "endpoint_response_mean",
    "h001_response_median",
    "h001_response_q90",
    "h001_response_mean",
    "uot_endpoint_mass_median",
    "uot_endpoint_shape_median",
    "uot_endpoint_combined_median",
    "uot_h001_mass_median",
    "uot_h001_shape_median",
    "uot_h001_combined_median",
)


def coordinate_row(frame: pd.DataFrame, alpha: float, beta: float) -> pd.Series:
    selected = frame[np.isclose(frame["alpha"], alpha) & np.isclose(frame["beta"], beta)]
    if len(selected) != 1:
        raise RuntimeError(f"expected one logical row at alpha={alpha}, beta={beta}; found {len(selected)}")
    return selected.iloc[0]


def factorial_contrasts(rows: pd.DataFrame) -> pd.DataFrame:
    output: list[dict[str, Any]] = []
    group_keys = [
        "dataset", "pair_id", "pair_type", "independent_unit_id", "cohort_role",
        "direction", "pair_file_id", "method", "epsilon0", "tau0",
    ]
    for keys, group in rows.groupby(group_keys, dropna=False):
        metadata = dict(zip(group_keys, keys if isinstance(keys, tuple) else (keys,)))
        for regime in ("fixed", "coregularized"):
            regime_frame = group[group["regularization_regime"] == regime]
            for channel in ("expression", "spatial"):
                coords = factor_coordinates(channel)
                selected = {name: coordinate_row(regime_frame, *coordinate) for name, coordinate in coords.items()}
                for metric in CONTRAST_METRICS:
                    if metric not in rows.columns or any(not np.isfinite(float(row.get(metric, np.nan))) for row in selected.values()):
                        continue
                    effects = factorial_effect({name: float(row[metric]) for name, row in selected.items()})
                    output.append({**metadata, "channel": channel, "regularization_regime": regime, "metric": metric, **effects})
    return pd.DataFrame(output)


def scale_regularization_contrasts(rows: pd.DataFrame) -> pd.DataFrame:
    output: list[dict[str, Any]] = []
    group_keys = [
        "dataset", "pair_id", "pair_type", "independent_unit_id", "cohort_role",
        "direction", "pair_file_id", "method", "epsilon0", "tau0",
    ]
    for keys, group in rows.groupby(group_keys, dropna=False):
        metadata = dict(zip(group_keys, keys if isinstance(keys, tuple) else (keys,)))
        grid = group[group["condition_family"] == "grid"]
        for regime in ("fixed", "coregularized"):
            regime_grid = grid[grid["regularization_regime"] == regime]
            for v in sorted(regime_grid["v"].unique()):
                at_v = regime_grid[np.isclose(regime_grid["v"], v)]
                reference = at_v[np.isclose(at_v["u"], 1.0)].iloc[0]
                for u in (0.5, 0.75):
                    comparison = at_v[np.isclose(at_v["u"], u)].iloc[0]
                    for metric in CONTRAST_METRICS:
                        if metric in rows.columns and np.isfinite(float(reference.get(metric, np.nan))) and np.isfinite(float(comparison.get(metric, np.nan))):
                            output.append({**metadata, "contrast_type": "fixed_v_scale", "regularization_regime": regime, "u": u, "v": float(v), "alpha": float(comparison.alpha), "beta": float(comparison.beta), "metric": metric, "reference_value": float(reference[metric]), "comparison_value": float(comparison[metric]), "contrast": float(comparison[metric]-reference[metric])})
        coordinate_keys = ["condition_family", "alpha", "beta", "u", "v"]
        for coordinates, at_coordinate in group.groupby(coordinate_keys, dropna=False):
            if set(at_coordinate["regularization_regime"]) != {"fixed", "coregularized"}:
                continue
            fixed = at_coordinate[at_coordinate["regularization_regime"] == "fixed"].iloc[0]
            coreg = at_coordinate[at_coordinate["regularization_regime"] == "coregularized"].iloc[0]
            for metric in CONTRAST_METRICS:
                if metric in rows.columns and np.isfinite(float(fixed.get(metric, np.nan))) and np.isfinite(float(coreg.get(metric, np.nan))):
                    output.append({**metadata, "contrast_type": "coregularized_minus_fixed", "regularization_regime": "coregularized-fixed", "u": float(coreg.u), "v": float(coreg.v), "alpha": float(coreg.alpha), "beta": float(coreg.beta), "metric": metric, "reference_value": float(fixed[metric]), "comparison_value": float(coreg[metric]), "contrast": float(coreg[metric]-fixed[metric])})
    return pd.DataFrame(output)


def hierarchical_surface(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = rows.copy()
    rows["biological_pair_id"] = rows["pair_id"]
    metrics = [column for column in rows.columns if column.startswith(SURFACE_METRIC_PREFIXES) and pd.api.types.is_numeric_dtype(rows[column])]
    condition_keys = ["condition_id", "physical_id", "condition_family", "u", "v", "alpha", "beta", "regularization_regime", "shared_at_u1"]
    pair_keys = ["dataset", "pair_type", "biological_pair_id", "independent_unit_id", "cohort_role", "method", "epsilon0", "tau0", *condition_keys]
    pair = rows.groupby(pair_keys, dropna=False)[metrics].mean().reset_index()
    unit_keys = ["independent_unit_id", "cohort_role", "method", "epsilon0", "tau0", *condition_keys]
    unit = pair.groupby(unit_keys, dropna=False)[metrics].median().reset_index()
    return pair, unit


def hierarchical_long(rows: pd.DataFrame, condition_keys: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = rows.copy()
    rows["biological_pair_id"] = rows["pair_id"]
    value_columns = [column for column in ("baseline", "delta_removal", "delta_compensation", "delta_joint", "delta_interaction", "reference_value", "comparison_value", "contrast") if column in rows.columns]
    pair_keys = ["dataset", "pair_type", "biological_pair_id", "independent_unit_id", "cohort_role", "method", "epsilon0", "tau0", *condition_keys]
    pair = rows.groupby(pair_keys, dropna=False)[value_columns].mean().reset_index()
    unit_keys = ["independent_unit_id", "cohort_role", "method", "epsilon0", "tau0", *condition_keys]
    unit = pair.groupby(unit_keys, dropna=False)[value_columns].median().reset_index()
    return pair, unit


def run(config: dict[str, Any], parent: dict[str, Any], config_hash: str, smoke: bool) -> int:
    validate_dependencies(config)
    output = output_paths(config)
    for key in ("analysis", "checkpoints", "arrays", "logs", "results"):
        output[key].mkdir(parents=True, exist_ok=True)
    logical, physical = build_condition_registries(config["wp11"])
    if len(logical) != int(config["wp11"]["expected_logical_conditions_per_task"]):
        raise RuntimeError("logical WP11 condition count mismatch")
    if len(physical) != int(config["wp11"]["expected_physical_conditions_per_task"]):
        raise RuntimeError("physical WP11 condition count mismatch")
    pd.DataFrame(logical).to_csv(output["results"] / "wp11_condition_registry.tsv", sep="\t", index=False)
    tasks = task_registry(config, parent)
    if smoke:
        tasks = tasks[:3]
    pd.DataFrame(tasks).to_csv(output["analysis"] / "TASK_REGISTRY.tsv", sep="\t", index=False)
    dump_json(output["status"], {"state": "RUNNING", "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "smoke": smoke, "tasks": len(tasks), "config_sha256": config_hash})
    payloads: list[dict[str, Any]] = []
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=int(config["execution"]["workers"])) as executor:
        futures = {executor.submit(compute_task, task, config, config_hash): task for task in tasks}
        for index, future in enumerate(as_completed(futures), 1):
            payloads.append(future.result())
            progress = {"completed": index, "planned": len(tasks), "successful": sum(value["status"] == "COMPLETED" for value in payloads), "failed": sum(value["status"] == "FAILED" for value in payloads), "wall_seconds": time.perf_counter()-started}
            dump_json(output["logs"] / "wp11_progress.json", progress)
            print(json.dumps(progress), flush=True)
    completed = [value for value in payloads if value["status"] == "COMPLETED"]
    rows = pd.DataFrame([row for value in completed for row in value["rows"]])
    solver = pd.DataFrame([row for value in completed for row in value["solver"]])
    failures = pd.DataFrame([row for value in payloads for row in value["failures"]])
    rows.to_csv(output["results"] / "wp11_alpha_beta_surface_direction.tsv", sep="\t", index=False)
    solver.to_csv(output["results"] / "wp11_solver_diagnostics.tsv", sep="\t", index=False)
    failures.to_csv(output["results"] / "wp11_failures.tsv", sep="\t", index=False)
    factorial = factorial_contrasts(rows) if len(rows) else pd.DataFrame()
    scale_regularization = scale_regularization_contrasts(rows) if len(rows) else pd.DataFrame()
    factorial.to_csv(output["results"] / "wp11_factorial_contrasts_direction.tsv", sep="\t", index=False)
    scale_regularization.to_csv(output["results"] / "wp11_scale_regularization_contrasts_direction.tsv", sep="\t", index=False)
    if len(rows):
        surface_pair, surface_unit = hierarchical_surface(rows)
        factorial_pair, factorial_unit = hierarchical_long(factorial, ["channel", "regularization_regime", "metric"])
        scale_pair, scale_unit = hierarchical_long(scale_regularization, ["contrast_type", "regularization_regime", "u", "v", "alpha", "beta", "metric"])
        surface_pair.to_csv(output["results"] / "wp11_alpha_beta_surface_pair.tsv", sep="\t", index=False)
        surface_unit.to_csv(output["results"] / "wp11_alpha_beta_surface_unit.tsv", sep="\t", index=False)
        factorial_pair.to_csv(output["results"] / "wp11_factorial_contrasts_pair.tsv", sep="\t", index=False)
        factorial_unit.to_csv(output["results"] / "wp11_factorial_contrasts_unit.tsv", sep="\t", index=False)
        scale_pair.to_csv(output["results"] / "wp11_scale_regularization_contrasts_pair.tsv", sep="\t", index=False)
        scale_unit.to_csv(output["results"] / "wp11_scale_regularization_contrasts_unit.tsv", sep="\t", index=False)
    expected_rows = len(tasks) * len(logical)
    expected_solver_calls = len(tasks) * (1 + 2 * (len(physical) - 1))
    baseline_max = float(rows.loc[np.isclose(rows["alpha"], 0.5) & np.isclose(rows["beta"], 0.5), "endpoint_response_maximum"].max()) if len(rows) else float("nan")
    gate = {
        "smoke": smoke,
        "planned_tasks": len(tasks),
        "completed_tasks": len(completed),
        "failed_tasks": len(tasks)-len(completed),
        "logical_conditions_per_task": len(logical),
        "physical_conditions_per_task": len(physical),
        "surface_rows": len(rows),
        "expected_surface_rows": expected_rows,
        "solver_calls": len(solver),
        "expected_solver_calls": expected_solver_calls,
        "solver_nonconvergence": int((~solver["converged"].astype(bool)).sum()) if len(solver) else 0,
        "baseline_endpoint_response_maximum": baseline_max,
        "computational_pass": bool(len(completed) == len(tasks) and not len(failures) and len(rows) == expected_rows and len(solver) == expected_solver_calls and baseline_max <= 1e-12),
    }
    dump_json(output["results"] / ("WP11_SMOKE_GATE.json" if smoke else "WP11_GATE.json"), gate)
    dump_json(output["status"], {"state": "COMPLETE" if gate["computational_pass"] else "STOPPED_GATE_FAILURE", "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "smoke": smoke, "gate": gate, "config_sha256": config_hash})
    print(json.dumps(gate, indent=2), flush=True)
    return 0 if gate["computational_pass"] else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config, parent, config_hash = load_configuration()
    if args.smoke:
        config = copy.deepcopy(config)
        config["output_root"] = "analysis/postreview_wp11_smoke"
        config["results_root"] = "results/postreview_wp11_smoke"
    return run(config, parent, config_hash, bool(args.smoke))


if __name__ == "__main__":
    raise SystemExit(main())
