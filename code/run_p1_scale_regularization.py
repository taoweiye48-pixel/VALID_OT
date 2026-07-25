from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
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

from validot.controls import adjusted_association, permutation_controls, spatial_blocks
from validot.evaluation import boundary_proximity, local_sparsity
from validot.io import load_pair
from validot.metrics import conditional_plan, exact_row_response, normalized_excess_aurc
from validot.p1 import (
    P1Parameters,
    arm_weights,
    exact_and_fd_response,
    mixed_cost,
    plan_difference,
    positive_median,
    primal_objective,
    response_metrics,
    solve_p1,
    solver_diagnostics,
)
from validot.solvers import cost_components


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "p1_scale_regularization_v1.yaml"
ANALYSIS = ROOT / "analysis" / "p1_scale_regularization"
RESULTS = ROOT / "results" / "p1_scale_regularization"
BUILD = ROOT / "build" / "p1_scale_regularization"
CHECKPOINTS = BUILD / "checkpoints"
LOGS = ANALYSIS / "logs"
EXPECTED_CONFIG_HASH = "dd72774f6a525ec380d8bf32a5fc3c6e5edbde4687a16f569abccb892687e3b1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True) + "\n", encoding="utf-8")


def load_config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def verify_p0_manifest() -> dict[str, Any]:
    manifest = ROOT / "analysis_freeze" / "p0-pre-reanalysis" / "manifest.sha256"
    missing, mismatched = [], []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split("  ", 1)
        path = ROOT / Path(relative)
        if not path.is_file():
            missing.append(relative)
        elif sha256(path) != expected:
            mismatched.append(relative)
    result = {
        "manifest": str(manifest.relative_to(ROOT)),
        "manifest_sha256": sha256(manifest),
        "entries": sum(1 for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()),
        "missing": missing,
        "mismatched": mismatched,
        "pass": not missing and not mismatched,
    }
    return result


def protected_paths(config: dict[str, Any]) -> list[Path]:
    paths: list[Path] = []
    frozen = ROOT / "analysis_freeze" / "p0-pre-reanalysis" / "frozen_artifacts.txt"
    for line in frozen.read_text(encoding="utf-8").splitlines():
        if re.match(r"^[0-9a-f]{64}\t", line):
            paths.append(ROOT / line.split("\t", 1)[1])
    source = Path(config["source_installation"])
    paths.extend(
        [
            source / "15_v1_3_correction" / "00_v1_2_snapshot" / "frozen_config_v1.2.json",
            source / "12_E8_statistics" / "registered_real_external_gate.tsv",
            source / "00_protocol" / "frozen_config.json",
            ROOT / "analysis_freeze" / "p0-pre-reanalysis" / "manifest.sha256",
        ]
    )
    return list(dict.fromkeys(path.resolve() for path in paths))


def protection_snapshot(config: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"path": str(path), "exists": path.is_file(), "sha256": sha256(path) if path.is_file() else None}
        for path in protected_paths(config)
    ]


def prepare_freeze() -> int:
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    BUILD.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    actual = sha256(CONFIG_PATH)
    if actual != EXPECTED_CONFIG_HASH:
        raise RuntimeError(f"P1 v1 config hash changed: {actual} != {EXPECTED_CONFIG_HASH}")
    p0 = verify_p0_manifest()
    json_dump(ANALYSIS / "logs" / "p0_manifest_verification_pre_run.json", p0)
    if not p0["pass"]:
        raise RuntimeError("P0 manifest verification failed; P1 is stopped")
    shutil.copyfile(CONFIG_PATH, ANALYSIS / "config_snapshot.yaml")
    (ANALYSIS / "config_snapshot.sha256").write_text(actual + "  config_snapshot.yaml\n", encoding="utf-8")
    code_paths = [
        ROOT / "code" / "run_p1_scale_regularization.py",
        ROOT / "code" / "validot" / "p1.py",
        ROOT / "code" / "validot" / "solvers.py",
        ROOT / "code" / "validot" / "metrics.py",
        ROOT / "code" / "validot" / "controls.py",
        CONFIG_PATH,
    ]
    code_paths.extend(sorted((ROOT / "code" / "tests").glob("test_p1_*.py")))
    code_paths.extend(sorted((ROOT / "code" / "tests").glob("test_cost_epsilon*.py")))
    code_paths.extend(sorted((ROOT / "code" / "tests").glob("test_row_softmax_temperature_scaling.py")))
    code_paths.extend(sorted((ROOT / "code" / "tests").glob("test_*path_endpoints.py")))
    code_paths = list(dict.fromkeys(code_paths))
    (ANALYSIS / "code_manifest.sha256").write_text(
        "".join(f"{sha256(path)}  {path.relative_to(ROOT).as_posix()}\n" for path in code_paths),
        encoding="utf-8",
    )
    json_dump(ANALYSIS / "logs" / "protected_hashes_pre_run.json", protection_snapshot(load_config()))
    return 0


def parameter_specs(config: dict[str, Any]) -> list[dict[str, Any]]:
    specs = []
    for method in config["methods"]:
        for index, raw in enumerate(config["parameter_grid"][method]):
            epsilon = float(raw.get("epsilon", raw.get("temperature")))
            specs.append(
                {
                    "method": method,
                    "epsilon": epsilon,
                    "tau": float(raw["tau"]) if "tau" in raw else None,
                    "grid_role": raw.get("grid_role", "epsilon_scan" if epsilon != 0.25 else "baseline"),
                    "parameter_index": index,
                }
            )
    return specs


def task_id(pair_file_id: str, spec: dict[str, Any]) -> str:
    tau = "none" if spec["tau"] is None else str(spec["tau"]).replace(".", "p")
    eps = str(spec["epsilon"]).replace(".", "p")
    return f"{pair_file_id}__{spec['method']}__e{eps}__t{tau}"


def condition_registry(config: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for pair in config["pairs"]:
        for direction in config["directions"]:
            pair_file_id = pair["pair_id"] + ("__reverse" if direction == "reverse" else "")
            for spec in parameter_specs(config):
                for arm in ("R", "N"):
                    rows.append(
                        {
                            **pair,
                            "direction": direction,
                            "pair_file_id": pair_file_id,
                            "task_id": task_id(pair_file_id, spec),
                            **spec,
                            "arm": arm,
                            "claim_status": config["claim_status"],
                            "planned_status": "PLANNED",
                        }
                    )
                if spec["grid_role"] == "baseline":
                    rows.append(
                        {
                            **pair,
                            "direction": direction,
                            "pair_file_id": pair_file_id,
                            "task_id": task_id(pair_file_id, spec),
                            **spec,
                            "arm": "S",
                            "claim_status": config["claim_status"],
                            "planned_status": "PLANNED_EQUIVALENCE_ONLY",
                        }
                    )
    return pd.DataFrame(rows)


def stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("::".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little")


def metadata_from_pair(pair: Any, independent_unit_id: str) -> dict[str, Any]:
    return {
        "dataset": pair.dataset,
        "pair_type": pair.metadata.get("pair_type", ""),
        "biological_pair_id": pair.metadata.get("biological_pair_id", pair.pair_id.replace("__reverse", "")),
        "direction": pair.metadata.get("direction", "forward"),
        "pair_id": pair.pair_id,
        "independent_unit_id": independent_unit_id,
    }


def witness_data(pair: Any, extras: dict[str, np.ndarray], base_plan: np.ndarray) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, np.ndarray], dict[str, float]]:
    conditional, _ = conditional_plan(base_plan)
    predicted = np.argmax(conditional, axis=1)
    source_labels = pair.source_labels.astype(str)
    target_labels = pair.target_labels.astype(str)
    label_error = (source_labels != target_labels[predicted]).astype(float)
    shared_labels = np.intersect1d(np.unique(source_labels), np.unique(target_labels))
    shared = np.isin(source_labels, shared_labels)
    source_only = ~shared
    transported = conditional @ extras["target_heldout"]
    transported /= np.maximum(np.linalg.norm(transported, axis=1, keepdims=True), 1e-12)
    source = extras["source_heldout"]
    source = source / np.maximum(np.linalg.norm(source, axis=1, keepdims=True), 1e-12)
    heldout = 1.0 - np.sum(source * transported, axis=1)
    witnesses = {
        "heldout_loss": (heldout, np.ones(len(heldout), dtype=bool)),
        "label_error_shared_closed_set": (label_error, shared),
        "source_only_open_set": (source_only.astype(float), np.ones(len(source_only), dtype=bool)),
    }
    qc = {
        "source_boundary_proximity": boundary_proximity(pair.source_xy),
        "source_sparsity": local_sparsity(pair.source_xy),
        "matched_target_sparsity": local_sparsity(pair.target_xy)[predicted],
    }
    coverage = {
        "shared_label_coverage": float(np.mean(shared)),
        "source_only_fraction": float(np.mean(source_only)),
        "shared_label_count": int(len(shared_labels)),
    }
    return witnesses, qc, coverage


def external_records(
    config: dict[str, Any],
    metadata: dict[str, Any],
    spec: dict[str, Any],
    arm: str,
    pair: Any,
    extras: dict[str, np.ndarray],
    base_plan: np.ndarray,
    exact: dict[str, np.ndarray],
    matched_fd: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    witnesses, qc, coverage = witness_data(pair, extras, base_plan)
    scores = {
        "exact_I_EXPR": exact["I_EXPR"],
        "exact_I_SPATIAL": exact["I_SPATIAL"],
        "exact_combined": np.maximum(exact["I_EXPR"], exact["I_SPATIAL"]),
        "matched_fd_I_EXPR": matched_fd["I_EXPR"],
        "matched_fd_I_SPATIAL": matched_fd["I_SPATIAL"],
        "matched_fd_combined": np.maximum(matched_fd["I_EXPR"], matched_fd["I_SPATIAL"]),
        "source_boundary_proximity": qc["source_boundary_proximity"],
    }
    rows: list[dict[str, Any]] = []
    for witness, (loss, mask) in witnesses.items():
        witness_rows = []
        for score_name, score in scores.items():
            valid = mask & np.isfinite(loss) & np.isfinite(score)
            base = {
                **metadata,
                **spec,
                "arm": arm,
                "witness": witness,
                "score": score_name,
                "n": int(valid.sum()),
                "witness_coverage": float(np.mean(valid)),
                **coverage,
                "claim_status": config["claim_status"],
            }
            if valid.sum() < 2 or np.unique(loss[valid]).size < 2:
                utility = {key: float("nan") for key in ["aurc", "oracle_aurc", "random_aurc", "normalized_excess_aurc", "retained_loss_at_80pct_coverage", "retained_loss_at_90pct_coverage"]}
                control = {"control_status": "INSUFFICIENT_VARIATION", "adjusted_status": "INSUFFICIENT_VARIATION"}
            else:
                utility = normalized_excess_aurc(loss[valid], score[valid])
                control = permutation_controls(
                    loss[valid], score[valid], pair.source_xy[valid],
                    repeats=int(config["controls"]["permutation_repeats"]),
                    seed=stable_seed(config["analysis_version"], metadata["pair_id"], spec["method"], spec["epsilon"], spec.get("tau"), arm, witness, score_name),
                )
                confounds = {
                    "boundary": qc["source_boundary_proximity"][valid],
                    "source_sparsity": qc["source_sparsity"][valid],
                    "matched_target_sparsity": qc["matched_target_sparsity"][valid],
                    "log_library_size": np.log1p(extras["source_library_size"][valid]),
                    "log_region_size": np.log1p(extras["source_region_size"][valid]),
                }
                if score_name == "source_boundary_proximity":
                    confounds.pop("boundary")
                control.update(adjusted_association(loss[valid], score[valid], confounds, cluster_groups=spatial_blocks(pair.source_xy[valid])))
            row = {**base, **utility, **control}
            row["relative_improvement_over_random"] = 1.0 - row["normalized_excess_aurc"] if np.isfinite(row["normalized_excess_aurc"]) else float("nan")
            row["negative_control_pass"] = bool(
                row.get("control_status") == "COMPLETED"
                and row.get("primary_normalized_excess_aurc", np.inf) < row.get("label_shuffle_median", -np.inf)
                and row.get("primary_normalized_excess_aurc", np.inf) < row.get("within_block_permutation_median", -np.inf)
                and row.get("label_shuffle_p_lower", 1.0) <= 0.05
                and row.get("within_block_p_lower", 1.0) <= 0.05
            )
            row["leakage_control_pass"] = bool(abs(row.get("leakage_positive_control_normalized_excess_aurc", np.inf)) <= 1e-8)
            row["confound_direction_pass"] = bool(row.get("risk_positive_after_adjustment", False))
            row["confound_significance_pass"] = bool(row.get("risk_rank_pvalue", 1.0) <= 0.05)
            witness_rows.append(row)
        fixed = next((row["normalized_excess_aurc"] for row in witness_rows if row["score"] == "source_boundary_proximity"), float("nan"))
        for row in witness_rows:
            row["fixed_qc_nex_aurc"] = fixed
            row["fixed_qc_gain"] = fixed - row["normalized_excess_aurc"] if np.isfinite(fixed) and np.isfinite(row["normalized_excess_aurc"]) else float("nan")
            row["relative_fixed_qc_gain"] = row["fixed_qc_gain"] / max(abs(fixed), 1e-12) if np.isfinite(row["fixed_qc_gain"]) else float("nan")
        rows.extend(witness_rows)
    return rows


def rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / (1024.0 * 1024.0)


def solve_record(
    result: Any,
    cost: np.ndarray,
    expression: np.ndarray,
    spatial: np.ndarray,
    metadata: dict[str, Any],
    spec: dict[str, Any],
    arm: str,
    stage: str,
    intervention: str,
    weights: tuple[float, float],
) -> dict[str, Any]:
    return {
        **metadata,
        **spec,
        "arm": arm,
        "stage": stage,
        "intervention": intervention,
        "expression_weight": weights[0],
        "spatial_weight": weights[1],
        "ce_positive_median": positive_median(expression),
        "cs_positive_median": positive_median(spatial),
        **solver_diagnostics(result, cost, spec["epsilon"], rss_mb()),
    }


def compute_task(task: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    task_name = task_id(task["pair_file_id"], task)
    checkpoint = CHECKPOINTS / f"{task_name}.json"
    if checkpoint.is_file():
        payload = json.loads(checkpoint.read_text(encoding="utf-8"))
        if payload.get("config_hash") == EXPECTED_CONFIG_HASH and payload.get("status") == "COMPLETED":
            return payload
    try:
        if sha256(CONFIG_PATH) != EXPECTED_CONFIG_HASH:
            raise RuntimeError("frozen P1 configuration changed during execution")
        pair_root = Path(config["processed_pair_root"])
        pair, extras = load_pair(pair_root / f"{task['pair_file_id']}.npz")
        metadata = metadata_from_pair(pair, task["independent_unit_id"])
        components = cost_components(pair.source_x, pair.target_x, pair.source_xy, pair.target_xy)
        expression, spatial = components["expression"], components["spatial_cross"]
        a = np.full(len(pair.source_x), 1.0 / len(pair.source_x))
        b = np.full(len(pair.target_x), 1.0 / len(pair.target_x))
        params = P1Parameters(task["method"], task["epsilon"], task.get("tau"), config["solver"]["max_iter"], config["solver"]["tolerance"])
        base_cost = mixed_cost(expression, spatial, (0.5, 0.5))
        base = solve_p1(base_cost, a, b, params)
        diagnostics = [solve_record(base, base_cost, expression, spatial, metadata, task, "shared", "base", "NONE", (0.5, 0.5))]
        plans: dict[str, dict[str, np.ndarray]] = {}
        responses: dict[str, dict[str, dict[str, np.ndarray]]] = {}
        internal: list[dict[str, Any]] = []
        h = float(config["paths"]["h"])
        gate = config["fidelity_gate"]
        for arm in ("R", "N"):
            plans[arm], responses[arm] = {}, {}
            for intervention in ("I_EXPR", "I_SPATIAL"):
                endpoint_weights = arm_weights(arm, intervention, 1.0)
                fd_weights = arm_weights(arm, intervention, h)
                endpoint_cost = mixed_cost(expression, spatial, endpoint_weights)
                fd_cost = mixed_cost(expression, spatial, fd_weights)
                endpoint = solve_p1(endpoint_cost, a, b, params)
                fd = solve_p1(fd_cost, a, b, params)
                exact_response, fd_response = exact_and_fd_response(base.plan, endpoint.plan, fd.plan, h)
                plans[arm][intervention] = endpoint.plan
                responses[arm][intervention] = {"exact": exact_response, "matched": fd_response}
                diagnostics.extend([
                    solve_record(endpoint, endpoint_cost, expression, spatial, metadata, task, arm, "full_endpoint", intervention, endpoint_weights),
                    solve_record(fd, fd_cost, expression, spatial, metadata, task, arm, "matched_fd", intervention, fd_weights),
                ])
        for arm in ("R", "N"):
            for intervention in ("I_EXPR", "I_SPATIAL"):
                reference = responses[arm][intervention]["exact"]
                score_map = {
                    "matched_finite_difference": responses[arm][intervention]["matched"],
                    "registered_finite_difference_fixed_score": responses["R"][intervention]["matched"],
                }
                for score_name, score in score_map.items():
                    internal.append({**metadata, **task, "arm": arm, "intervention": intervention, "score": score_name, **response_metrics(reference, score, gate), "claim_status": config["claim_status"]})
        external = []
        for arm in ("R", "N"):
            external.extend(external_records(config, metadata, task, arm, pair, extras, base.plan, {key: responses[arm][key]["exact"] for key in responses[arm]}, {key: responses[arm][key]["matched"] for key in responses[arm]}))
        invariance: list[dict[str, Any]] = []
        reproduction: list[dict[str, Any]] = []
        if task["grid_role"] == "baseline":
            if task["method"] == "uot":
                s_params = P1Parameters("uot", 0.125, 1.0, config["solver"]["max_iter"], config["solver"]["tolerance"])
            else:
                s_params = P1Parameters(task["method"], 0.125, None, config["solver"]["max_iter"], config["solver"]["tolerance"])
            tol = max(1e-7, 10.0 * float(config["solver"]["tolerance"]))
            for intervention in ("I_EXPR", "I_SPATIAL"):
                s_weights = arm_weights("R", intervention, 1.0)
                s_cost = mixed_cost(expression, spatial, s_weights)
                s_result = solve_p1(s_cost, a, b, s_params)
                s_spec = {**task, "epsilon": 0.125, "tau": 1.0 if task["method"] == "uot" else None, "grid_role": "scale_equivalence"}
                diagnostics.append(solve_record(s_result, s_cost, expression, spatial, metadata, s_spec, "S", "full_endpoint", intervention, s_weights))
                diff = plan_difference(plans["N"][intervention], s_result.plan, tol)
                n_response = responses["N"][intervention]["exact"]
                s_response = exact_row_response(base.plan, s_result.plan)
                response_eval = response_metrics(n_response, s_response, {"spearman_min": -1.0, "top_decile_overlap_min": 0.0, "nmae_max": float("inf")})
                n_internal = next(row for row in internal if row["arm"] == "N" and row["intervention"] == intervention and row["score"] == "matched_finite_difference")
                s_gate = response_metrics(s_response, responses["N"][intervention]["matched"], gate)
                objective_n = primal_objective(task["method"], plans["N"][intervention], mixed_cost(expression, spatial, arm_weights("N", intervention, 1.0)), a, b, task["epsilon"], task.get("tau")) if task["method"] != "row_softmax" else float("nan")
                objective_s = primal_objective(task["method"], s_result.plan, s_cost, a, b, s_params.epsilon, s_params.tau) if task["method"] != "row_softmax" else float("nan")
                invariance.append({**metadata, **task, "intervention": intervention, **diff, "row_response_spearman": response_eval["spearman"], "row_response_nmae": response_eval["nmae"], "transported_mass_difference": float(abs(plans["N"][intervention].sum() - s_result.plan.sum())), "objective_n": objective_n, "objective_s": objective_s, "objective_ratio_s_over_n": objective_s / objective_n if np.isfinite(objective_n) and abs(objective_n) > 1e-300 else float("nan"), "gate_n": bool(n_internal["gate_pass"]), "gate_s_reference": bool(s_gate["gate_pass"]), "gate_conclusion_consistent": bool(n_internal["gate_pass"] == s_gate["gate_pass"]), "claim_status": config["claim_status"]})
            frozen_path = Path(config["source_installation"]) / "10_E6_real_external" / task["pair_file_id"] / task["method"] / "row_responses.npz"
            if frozen_path.is_file():
                with np.load(frozen_path, allow_pickle=False) as frozen:
                    for intervention in ("I_EXPR", "I_SPATIAL"):
                        reproduction.append({**metadata, **task, "intervention": intervention, "exact_max_abs_difference": float(np.max(np.abs(responses["R"][intervention]["exact"] - frozen[f"exact_{intervention}"]))), "fd_max_abs_difference": float(np.max(np.abs(responses["R"][intervention]["matched"] - frozen[f"endpoint_{intervention}"]))), "frozen_path_exists": True})
            else:
                reproduction.append({**metadata, **task, "intervention": "ALL", "exact_max_abs_difference": float("nan"), "fd_max_abs_difference": float("nan"), "frozen_path_exists": False})
        payload = {"status": "COMPLETED", "config_hash": EXPECTED_CONFIG_HASH, "task_id": task_name, "seconds": time.perf_counter() - started, "internal": internal, "external": external, "diagnostics": diagnostics, "invariance": invariance, "reproduction": reproduction, "failures": []}
    except Exception as exc:
        payload = {"status": "FAILED", "config_hash": EXPECTED_CONFIG_HASH, "task_id": task_name, "seconds": time.perf_counter() - started, "internal": [], "external": [], "diagnostics": [], "invariance": [], "reproduction": [], "failures": [{**task, "task_id": task_name, "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc(), "claim_status": config["claim_status"]}]}
    json_dump(checkpoint, payload)
    return payload


def aggregate_internal(direction: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["dataset", "pair_type", "biological_pair_id", "independent_unit_id", "method", "epsilon", "tau", "grid_role", "arm", "intervention", "score", "claim_status"]
    numeric = ["spearman", "top_decile_overlap", "raw_mae", "reference_mad", "nmae", "reference_response_median", "reference_response_iqr"]
    pair = direction.groupby(keys, dropna=False)[numeric].mean().reset_index()
    pair["directions"] = direction.groupby(keys, dropna=False).direction.nunique().to_numpy()
    add_gate(pair, config)
    unit_keys = [key for key in keys if key not in {"dataset", "pair_type", "biological_pair_id"}]
    aggregations = {column: "median" for column in numeric}
    unit = pair.groupby(unit_keys, dropna=False).agg(**{column: (column, function) for column, function in aggregations.items()}, biological_pairs=("biological_pair_id", "nunique"), technology=("dataset", "first")).reset_index()
    add_gate(unit, config)
    return pair, unit


def add_gate(table: pd.DataFrame, config: dict[str, Any]) -> None:
    gate = config["fidelity_gate"]
    table["estimable"] = np.isfinite(table["nmae"])
    table["spearman_gate_pass"] = table["spearman"] >= gate["spearman_min"]
    table["overlap_gate_pass"] = table["top_decile_overlap"] >= gate["top_decile_overlap_min"]
    table["nmae_gate_pass"] = table["estimable"] & (table["nmae"] <= gate["nmae_max"])
    table["gate_pass"] = table[["spearman_gate_pass", "overlap_gate_pass", "nmae_gate_pass"]].all(axis=1)


def aggregate_external(direction: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["dataset", "pair_type", "biological_pair_id", "independent_unit_id", "method", "epsilon", "tau", "grid_role", "arm", "witness", "score", "claim_status"]
    numeric = ["normalized_excess_aurc", "relative_improvement_over_random", "fixed_qc_nex_aurc", "fixed_qc_gain", "relative_fixed_qc_gain", "negative_control_pass", "leakage_control_pass", "confound_direction_pass", "confound_significance_pass"]
    pair = direction.groupby(keys, dropna=False)[numeric].mean().reset_index()
    pair["directions"] = direction.groupby(keys, dropna=False).direction.nunique().to_numpy()
    unit_keys = [key for key in keys if key not in {"dataset", "pair_type", "biological_pair_id"}]
    unit = pair.groupby(unit_keys, dropna=False).agg(**{column: (column, "mean") for column in numeric}, biological_pairs=("biological_pair_id", "nunique"), technology=("dataset", "first")).reset_index()
    return pair, unit


def leave_one_unit_low(values: pd.Series, units: pd.Series) -> float:
    estimates = []
    for omitted in pd.unique(units):
        retained = values[units != omitted].dropna()
        if len(retained):
            estimates.append(float(retained.median()))
    return float(min(estimates)) if estimates else float("nan")


def finite_mean(values: pd.Series) -> float:
    """Mean of available numeric values without all-missing runtime warnings."""
    available = pd.to_numeric(values, errors="coerce").dropna()
    return float(available.mean()) if len(available) else float("nan")


def finite_median(values: pd.Series) -> float:
    """Median of available numeric values without all-missing warnings."""
    available = pd.to_numeric(values, errors="coerce").dropna()
    return float(available.median()) if len(available) else float("nan")


def gate_summary(internal_unit: pd.DataFrame, external_unit: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    rows = []
    ikeys = ["method", "epsilon", "tau", "grid_role", "arm", "intervention", "score", "claim_status"]
    for keys, group in internal_unit.groupby(ikeys, dropna=False):
        record = dict(zip(ikeys, keys))
        record.update(gate_family="internal_fidelity", independent_units=group.independent_unit_id.nunique(), median_spearman=float(group.spearman.median()), median_top_decile_overlap=float(group.top_decile_overlap.median()), median_nmae=float(group.nmae.median()), unit_gate_pass_fraction=float(group.gate_pass.mean()))
        # add_gate() operates on canonical metric names.  The family-level
        # record intentionally uses median_* names for reporting, so construct
        # an explicit canonical view instead of passing the reporting record.
        temp = pd.DataFrame([{
            "spearman": record["median_spearman"],
            "top_decile_overlap": record["median_top_decile_overlap"],
            "nmae": record["median_nmae"],
        }])
        add_gate(temp, config)
        record.update({name: temp.iloc[0][name] for name in ["spearman_gate_pass", "overlap_gate_pass", "nmae_gate_pass", "gate_pass"]})
        rows.append(record)
    ekeys = ["method", "epsilon", "tau", "grid_role", "arm", "witness", "score", "claim_status"]
    threshold = config["external_gate"]
    for keys, group in external_unit.groupby(ekeys, dropna=False):
        record = dict(zip(ekeys, keys))
        technology_direction = group.groupby("technology").fixed_qc_gain.apply(finite_median)
        loo = leave_one_unit_low(group.fixed_qc_gain, group.independent_unit_id)
        median_relative_gain = finite_median(group.relative_fixed_qc_gain)
        base_pass = bool(median_relative_gain >= threshold["median_relative_aurc_improvement_min"] and np.sum(technology_direction > 0) >= threshold["minimum_positive_technology_count"] and np.mean(group.fixed_qc_gain > 0) >= threshold["minimum_positive_unit_fraction"] and loo >= 0)
        negative_control_fraction = finite_mean(group.negative_control_pass)
        leakage_control_fraction = finite_mean(group.leakage_control_pass)
        confound_direction_fraction = finite_mean(group.confound_direction_pass)
        controls_pass = bool(negative_control_fraction >= 2/3 and leakage_control_fraction == 1 and confound_direction_fraction >= 2/3)
        record.update(gate_family="external_utility", independent_units=group.independent_unit_id.nunique(), median_nex_aurc=finite_median(group.normalized_excess_aurc), median_relative_fixed_qc_gain=median_relative_gain, positive_unit_fraction=float(np.mean(group.fixed_qc_gain > 0)), positive_technology_count=int(np.sum(technology_direction > 0)), leave_one_unit_out_fixed_qc_low=loo, negative_control_pass_fraction=negative_control_fraction, leakage_control_pass_fraction=leakage_control_fraction, confound_direction_pass_fraction=confound_direction_fraction, sensitivity_external_gate=base_pass, sensitivity_external_gate_with_controls=bool(base_pass and controls_pass), registered_gate_status="immutable; see registered v1.2 frozen sheet", confirmatory=False)
        rows.append(record)
    return pd.DataFrame(rows)


def write_result_manifest() -> None:
    paths = sorted(path for path in RESULTS.rglob("*") if path.is_file() and path.name != "p1_result_manifest.sha256")
    (RESULTS / "p1_result_manifest.sha256").write_text("".join(f"{sha256(path)}  {path.relative_to(RESULTS).as_posix()}\n" for path in paths), encoding="utf-8")


def finalize(config: dict[str, Any], payloads: list[dict[str, Any]], registry: pd.DataFrame) -> dict[str, Any]:
    internal = pd.DataFrame([row for payload in payloads for row in payload["internal"]])
    external = pd.DataFrame([row for payload in payloads for row in payload["external"]])
    diagnostics = pd.DataFrame([row for payload in payloads for row in payload["diagnostics"]])
    invariance = pd.DataFrame([row for payload in payloads for row in payload["invariance"]])
    reproduction = pd.DataFrame([row for payload in payloads for row in payload["reproduction"]])
    failures = pd.DataFrame([row for payload in payloads for row in payload["failures"]])
    internal_pair, internal_unit = aggregate_internal(internal, config)
    external_pair, external_unit = aggregate_external(external)
    gates = gate_summary(internal_unit, external_unit, config)
    registry.to_csv(RESULTS / "p1_condition_registry.csv", index=False)
    internal.to_csv(RESULTS / "p1_internal_direction_level.csv", index=False)
    internal_pair.to_csv(RESULTS / "p1_internal_pair_level.csv", index=False)
    internal_unit.to_csv(RESULTS / "p1_internal_unit_level.csv", index=False)
    external.to_csv(RESULTS / "p1_external_direction_level.csv", index=False)
    external_pair.to_csv(RESULTS / "p1_external_pair_level.csv", index=False)
    external_unit.to_csv(RESULTS / "p1_external_unit_level.csv", index=False)
    gates.to_csv(RESULTS / "p1_gate_summary.csv", index=False)
    invariance.to_csv(RESULTS / "p1_scale_invariance.csv", index=False)
    diagnostics.to_csv(RESULTS / "p1_cost_scale_diagnostics.csv", index=False)
    runtime_columns = [column for column in diagnostics.columns if column in {"dataset", "pair_type", "biological_pair_id", "independent_unit_id", "direction", "pair_id", "method", "epsilon", "tau", "grid_role", "arm", "stage", "intervention", "solver_iterations", "converged", "runtime_seconds", "peak_memory_mb", "memory_measurement", "solver_last_error", "row_mass_l1", "col_mass_l1"}]
    diagnostics[runtime_columns].to_csv(RESULTS / "p1_solver_runtime.csv", index=False)
    failures.to_csv(RESULTS / "p1_failures.csv", index=False)
    reproduction.to_csv(RESULTS / "p1_registered_reproduction.csv", index=False)
    if len(reproduction):
        reproduction_values = pd.to_numeric(
            reproduction[["exact_max_abs_difference", "fd_max_abs_difference"]].stack(),
            errors="coerce",
        ).dropna()
        reproduction_max = float(reproduction_values.max()) if len(reproduction_values) else None
    else:
        reproduction_max = None
    summary = {
        "analysis_version": config["analysis_version"],
        "claim_status": config["claim_status"],
        "tasks_planned": int(len(payloads)),
        "tasks_successful": int(sum(payload["status"] == "COMPLETED" for payload in payloads)),
        "tasks_failed": int(sum(payload["status"] != "COMPLETED" for payload in payloads)),
        "solver_calls": int(len(diagnostics)),
        "internal_direction_rows": int(len(internal)),
        "external_direction_rows": int(len(external)),
        "independent_units": int(internal_unit.independent_unit_id.nunique()),
        # JSON null is used when an extension has no frozen Arm-R reproduction
        # rows.  Emitting the non-standard JSON token NaN breaks strict readers.
        "arm_r_reproduction_max_abs_difference": reproduction_max,
        "scale_equivalence_failures": int((~invariance.equivalence_pass).sum()) if len(invariance) else 0,
        "registered_external_gate_passes": 0,
        "registered_external_gate_note": "immutable registered v1.2 result; P1 cells are non-confirmatory sensitivity analyses",
        "sensitivity_external_gate_with_controls_passes": int(gates.get("sensitivity_external_gate_with_controls", pd.Series(dtype=bool)).eq(True).sum()),
    }
    json_dump(RESULTS / "p1_summary.json", summary)
    write_result_manifest()
    return summary


def verify_protection(config: dict[str, Any]) -> dict[str, Any]:
    pre_path = ANALYSIS / "logs" / "protected_hashes_pre_run.json"
    pre = {row["path"]: row for row in json.loads(pre_path.read_text(encoding="utf-8"))}
    post_rows = protection_snapshot(config)
    changed = []
    for row in post_rows:
        before = pre.get(row["path"])
        if before is None or before["exists"] != row["exists"] or before["sha256"] != row["sha256"]:
            changed.append({"path": row["path"], "before": before, "after": row})
    result = {"pass": not changed, "checked": len(post_rows), "changed": changed}
    json_dump(ANALYSIS / "logs" / "protected_hashes_post_run.json", post_rows)
    json_dump(ANALYSIS / "logs" / "scientific_artifact_protection_check.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-freeze", action="store_true")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None, help="development only; forbidden for final P1")
    args = parser.parse_args()
    if args.prepare_freeze:
        return prepare_freeze()
    config = load_config()
    if sha256(CONFIG_PATH) != EXPECTED_CONFIG_HASH:
        raise RuntimeError("P1 config hash mismatch")
    p0 = verify_p0_manifest()
    if not p0["pass"]:
        raise RuntimeError("P0 manifest failed before solver execution")
    if not (ANALYSIS / "config_snapshot.sha256").is_file():
        raise RuntimeError("P1 freeze snapshot is absent; run --prepare-freeze first")
    registry = condition_registry(config)
    RESULTS.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    registry.to_csv(RESULTS / "p1_condition_registry.csv", index=False)
    pair_map = {pair["pair_id"]: pair for pair in config["pairs"]}
    tasks = []
    for pair in config["pairs"]:
        for direction in config["directions"]:
            pair_file_id = pair["pair_id"] + ("__reverse" if direction == "reverse" else "")
            for spec in parameter_specs(config):
                tasks.append({**spec, "pair_file_id": pair_file_id, "independent_unit_id": pair["independent_unit_id"]})
    if args.limit is not None:
        tasks = tasks[: args.limit]
    payloads = []
    run_log = LOGS / "p1_execution_progress.json"
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        future_map = {pool.submit(compute_task, task, config): task for task in tasks}
        for completed, future in enumerate(as_completed(future_map), 1):
            payload = future.result()
            payloads.append(payload)
            json_dump(run_log, {"completed": completed, "total": len(tasks), "successful": sum(item["status"] == "COMPLETED" for item in payloads), "failed": sum(item["status"] != "COMPLETED" for item in payloads), "last_task": payload["task_id"], "updated_epoch": time.time()})
            print(f"P1 {completed}/{len(tasks)} {payload['status']} {payload['task_id']}", flush=True)
    summary = finalize(config, payloads, registry)
    protection = verify_protection(config)
    if not protection["pass"]:
        raise RuntimeError("Frozen scientific artifacts changed during P1; final interpretation is blocked")
    print(json.dumps({**summary, "scientific_artifact_protection": protection}, ensure_ascii=False, indent=2))
    return 0 if summary["tasks_failed"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
