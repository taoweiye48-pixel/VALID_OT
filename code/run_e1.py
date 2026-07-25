from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from validot.solvers import (
    SolverResult,
    cost_components,
    log_sinkhorn,
    paste2_partial_fgw,
    paste_fgw,
    row_softmax,
)
from validot.utils import read_json, status_payload, write_json


ROOT = Path(__file__).resolve().parents[1]
CONFIG = read_json(ROOT / "00_protocol" / "frozen_config.json")
OUTPUT = ROOT / "05_E1_solver_validation"


def relative_l1(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.abs(left - right).sum() / max(np.abs(left).sum() + np.abs(right).sum(), 1e-300))


def make_toy(n_source: int, n_target: int, seed: int = 20260716):
    rng = np.random.default_rng(seed)
    angles_source = np.linspace(0, 2 * np.pi, n_source, endpoint=False)
    angles_target = np.linspace(0, 2 * np.pi, n_target, endpoint=False) + 0.08
    xy_source = np.column_stack([np.cos(angles_source), 0.7 * np.sin(angles_source)])
    xy_target = np.column_stack([np.cos(angles_target), 0.7 * np.sin(angles_target)])
    xy_target += 0.02 * rng.normal(size=xy_target.shape)
    centers = rng.normal(size=(6, 12))
    labels_source = np.arange(n_source) % len(centers)
    labels_target = np.arange(n_target) % len(centers)
    x_source = centers[labels_source] + 0.08 * rng.normal(size=(n_source, centers.shape[1]))
    x_target = centers[labels_target] + 0.08 * rng.normal(size=(n_target, centers.shape[1]))
    x_source /= np.maximum(np.linalg.norm(x_source, axis=1, keepdims=True), 1e-12)
    x_target /= np.maximum(np.linalg.norm(x_target, axis=1, keepdims=True), 1e-12)
    return x_source, x_target, xy_source, xy_target


def solve(method: str, components: dict[str, np.ndarray], a: np.ndarray, b: np.ndarray) -> SolverResult:
    settings = CONFIG["solver"]
    expression = components["expression"]
    spatial = components["spatial_cross"]
    total = 0.5 * expression + 0.5 * spatial
    if method == "row_softmax":
        return row_softmax(total, a, settings["epsilon"])
    if method == "balanced_ot":
        return log_sinkhorn(
            total, a, b, settings["epsilon"], max_iter=settings["sinkhorn_max_iter"], tol=settings["sinkhorn_tol"]
        )
    if method == "uot":
        return log_sinkhorn(
            total,
            a,
            b,
            settings["epsilon"],
            settings["uot_tau"],
            settings["uot_tau"],
            settings["sinkhorn_max_iter"],
            settings["sinkhorn_tol"],
        )
    if method == "paste_fgw":
        return paste_fgw(
            expression,
            components["source_structure"],
            components["target_structure"],
            a,
            b,
            settings["fgw_alpha"],
            settings["fgw_max_iter"],
        )
    if method == "paste2_partial_fgw":
        return paste2_partial_fgw(
            expression,
            components["source_structure"],
            components["target_structure"],
            a,
            b,
            settings["fgw_alpha"],
            settings["paste2_overlap"],
            settings["fgw_max_iter"],
        )
    raise KeyError(method)


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    methods = CONFIG["methods"]["required"]
    records: list[dict[str, object]] = []
    all_pass = True
    for n_source, n_target in [(64, 64), (64, 71), (128, 128)]:
        x_source, x_target, xy_source, xy_target = make_toy(n_source, n_target)
        components = cost_components(x_source, x_target, xy_source, xy_target)
        a = np.full(n_source, 1.0 / n_source)
        b = np.full(n_target, 1.0 / n_target)
        for method in methods:
            first = solve(method, components, a, b)
            second = solve(method, components, a, b)
            repeat_l1 = relative_l1(first.plan, second.plan)
            finite = bool(np.all(np.isfinite(first.plan)) and np.min(first.plan) >= 0)
            if method == "paste2_partial_fgw":
                expected_mass = float(CONFIG["solver"]["paste2_overlap"])
            elif method in {"row_softmax", "balanced_ot", "paste_fgw"}:
                expected_mass = 1.0
            else:
                expected_mass = None
            mass_error = abs(float(first.plan.sum()) - expected_mass) if expected_mass is not None else 0.0
            if method in {"balanced_ot", "paste_fgw"}:
                row_error = float(np.abs(first.plan.sum(axis=1) - a).sum())
                col_error = float(np.abs(first.plan.sum(axis=0) - b).sum())
            else:
                row_error = float("nan")
                col_error = float("nan")
            passed = finite and first.converged and repeat_l1 <= 1e-8 and mass_error <= 1e-6
            if method in {"balanced_ot", "paste_fgw"}:
                passed = passed and row_error <= 1e-6 and col_error <= 1e-6
            all_pass &= passed
            records.append(
                {
                    "n_source": n_source,
                    "n_target": n_target,
                    "method": method,
                    "converged": first.converged,
                    "iterations": first.iterations,
                    "seconds": first.seconds,
                    "repeat_relative_l1": repeat_l1,
                    "transported_mass": float(first.plan.sum()),
                    "expected_mass": expected_mass,
                    "mass_error": mass_error,
                    "row_marginal_l1": row_error,
                    "col_marginal_l1": col_error,
                    "finite_nonnegative": finite,
                    "passed": passed,
                }
            )
    import pandas as pd

    table = pd.DataFrame.from_records(records)
    table.to_csv(OUTPUT / "solver_validation.tsv", sep="\t", index=False)
    decision = status_payload(
        "E1",
        "COMPLETED_GO" if all_pass else "FAILED_NUMERIC",
        all_required_methods_passed=bool(all_pass),
        failed_records=table.loc[~table.passed].to_dict(orient="records"),
        record_count=len(table),
    )
    write_json(OUTPUT / "E1_DECISION.json", decision)
    print(json.dumps(decision, ensure_ascii=False, indent=2, default=str))
    return 0 if all_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
