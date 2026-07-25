"""Run independent UOT plan and direction-derivative cross-validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from validot.solvers import log_sinkhorn
from validot.uot_independent import (
    comparison_metrics,
    implicit_plan_derivative,
    solve_uot_dual,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "uot_independent_validation_v1.json"


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


def deterministic_indices(
    size: int, keep: int, pair_file_id: str, axis: str
) -> np.ndarray:
    if keep >= size:
        return np.arange(size, dtype=int)
    ranked = sorted(
        range(size),
        key=lambda index: hashlib.sha256(
            f"{pair_file_id}|{axis}|{index}".encode("utf-8")
        ).hexdigest(),
    )
    return np.sort(np.asarray(ranked[:keep], dtype=int))


def scaled_costs(
    source_x: np.ndarray,
    target_x: np.ndarray,
    source_xy: np.ndarray,
    target_xy: np.ndarray,
    source_indices: np.ndarray,
    target_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    expression_full = cdist(source_x, target_x, metric="sqeuclidean")
    spatial_full = cdist(source_xy, target_xy, metric="sqeuclidean")
    expression_positive = expression_full[expression_full > 0]
    spatial_positive = spatial_full[spatial_full > 0]
    expression_scale = (
        float(np.median(expression_positive))
        if expression_positive.size
        else 1.0
    )
    spatial_scale = (
        float(np.median(spatial_positive))
        if spatial_positive.size
        else 1.0
    )
    selected = np.ix_(source_indices, target_indices)
    return (
        expression_full[selected] / max(expression_scale, 1e-12),
        spatial_full[selected] / max(spatial_scale, 1e-12),
        {
            "expression_positive_median": expression_scale,
            "spatial_positive_median": spatial_scale,
        },
    )


def weights_and_direction(
    expression: np.ndarray,
    spatial: np.ndarray,
    arm: str,
    intervention: str,
) -> tuple[np.ndarray, np.ndarray]:
    baseline = 0.5 * expression + 0.5 * spatial
    if arm == "R" and intervention == "I_EXPR":
        direction = -0.5 * expression
    elif arm == "R" and intervention == "I_SPATIAL":
        direction = -0.5 * spatial
    elif arm == "N" and intervention == "I_EXPR":
        direction = -0.5 * expression + 0.5 * spatial
    elif arm == "N" and intervention == "I_SPATIAL":
        direction = 0.5 * expression - 0.5 * spatial
    else:
        raise ValueError(f"unknown condition {arm}/{intervention}")
    return baseline, direction


def run() -> int:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    config_hash = sha256(CONFIG_PATH)
    analysis_root = ROOT / config["output_root"]
    result_root = ROOT / config["results_root"]
    analysis_root.mkdir(parents=True, exist_ok=True)
    result_root.mkdir(parents=True, exist_ok=True)
    thresholds = config["validation_thresholds"]
    all_rows: list[dict[str, Any]] = []
    solver_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    started = time.perf_counter()

    registry = []
    for pair in config["pairs"]:
        for parameters in config["parameter_conditions"]:
            for arm, intervention in config["conditions"]:
                registry.append(
                    {
                        **pair,
                        **parameters,
                        "arm": arm,
                        "intervention": intervention,
                    }
                )
    pd.DataFrame(registry).to_csv(
        analysis_root / "TASK_REGISTRY.tsv", sep="\t", index=False
    )

    for task_index, pair_spec in enumerate(config["pairs"], 1):
        pair_path = (
            Path(config["processed_pair_root"])
            / f"{pair_spec['pair_file_id']}.npz"
        )
        try:
            with np.load(pair_path, allow_pickle=False) as data:
                source_x = data["source_x"].copy()
                target_x = data["target_x"].copy()
                source_xy = data["source_xy"].copy()
                target_xy = data["target_xy"].copy()
            source_indices = deterministic_indices(
                len(source_x),
                int(config["subsample"]["source_spots"]),
                pair_spec["pair_file_id"],
                "source",
            )
            target_indices = deterministic_indices(
                len(target_x),
                int(config["subsample"]["target_spots"]),
                pair_spec["pair_file_id"],
                "target",
            )
            expression, spatial, cost_scales = scaled_costs(
                source_x,
                target_x,
                source_xy,
                target_xy,
                source_indices,
                target_indices,
            )
            a = np.full(len(source_indices), 1.0 / len(source_indices))
            b = np.full(len(target_indices), 1.0 / len(target_indices))

            for parameter in config["parameter_conditions"]:
                epsilon = float(parameter["epsilon"])
                tau = float(parameter["tau"])
                base_cost = 0.5 * expression + 0.5 * spatial
                independent_base = solve_uot_dual(
                    base_cost,
                    a,
                    b,
                    epsilon,
                    tau,
                    gradient_tolerance=float(
                        config["independent_solver"]["gradient_tolerance"]
                    ),
                    max_iterations=int(
                        config["independent_solver"]["max_iterations"]
                    ),
                )
                production_base = log_sinkhorn(
                    base_cost,
                    a,
                    b,
                    epsilon=epsilon,
                    tau_a=tau,
                    tau_b=tau,
                    max_iter=int(
                        config["production_solver"]["max_iterations"]
                    ),
                    tol=float(config["production_solver"]["tolerance"]),
                )
                baseline_relative_l1 = float(
                    np.abs(
                        production_base.plan - independent_base.plan
                    ).sum()
                    / max(np.abs(independent_base.plan).sum(), 1e-300)
                )
                solver_rows.append(
                    {
                        **pair_spec,
                        **parameter,
                        **cost_scales,
                        "source_spots": len(source_indices),
                        "target_spots": len(target_indices),
                        "independent_converged": independent_base.converged,
                        "production_converged": production_base.converged,
                        "baseline_plan_relative_l1": baseline_relative_l1,
                        **{
                            f"independent_{key}": value
                            for key, value in independent_base.diagnostics.items()
                        },
                        "production_iterations": production_base.iterations,
                        "production_last_error": (
                            production_base.diagnostics or {}
                        ).get("last_log_scaling_error", float("nan")),
                        "production_transported_mass": float(
                            production_base.plan.sum()
                        ),
                    }
                )
                if not independent_base.converged:
                    raise RuntimeError(
                        f"independent baseline failed: {pair_spec['pair_file_id']}, "
                        f"{parameter['label']}"
                    )
                if not production_base.converged:
                    raise RuntimeError(
                        f"production baseline failed: {pair_spec['pair_file_id']}, "
                        f"{parameter['label']}"
                    )

                for arm, intervention in config["conditions"]:
                    _, direction = weights_and_direction(
                        expression, spatial, arm, intervention
                    )
                    independent_derivative, derivative_diagnostics = (
                        implicit_plan_derivative(
                            independent_base.plan,
                            direction,
                            epsilon,
                            tau,
                        )
                    )
                    finite_derivatives = []
                    for h in map(float, config["richardson_steps"]):
                        finite = log_sinkhorn(
                            base_cost + h * direction,
                            a,
                            b,
                            epsilon=epsilon,
                            tau_a=tau,
                            tau_b=tau,
                            max_iter=int(
                                config["production_solver"][
                                    "max_iterations"
                                ]
                            ),
                            tol=float(
                                config["production_solver"]["tolerance"]
                            ),
                        )
                        if not finite.converged:
                            raise RuntimeError(
                                f"production finite solve failed: "
                                f"{pair_spec['pair_file_id']}, "
                                f"{parameter['label']}, {arm}/{intervention}, "
                                f"h={h}"
                            )
                        finite_derivatives.append(
                            (finite.plan - production_base.plan) / h
                        )
                    production_richardson = (
                        2.0 * finite_derivatives[1]
                        - finite_derivatives[0]
                    )
                    metrics = comparison_metrics(
                        production_richardson,
                        independent_derivative,
                        production_base.plan,
                        independent_base.plan,
                        top_fraction=0.10,
                        denominator_minimum=float(
                            config["estimable_minimum"]
                        ),
                    )
                    row = {
                        **pair_spec,
                        **parameter,
                        "arm": arm,
                        "intervention": intervention,
                        "source_spots": len(source_indices),
                        "target_spots": len(target_indices),
                        "baseline_plan_relative_l1": baseline_relative_l1,
                        **derivative_diagnostics,
                        **metrics,
                    }
                    row["validation_pass"] = bool(
                        baseline_relative_l1
                        <= float(
                            thresholds[
                                "baseline_plan_relative_l1_max"
                            ]
                        )
                        and metrics["global_derivative_relative_l1"]
                        <= float(
                            thresholds[
                                "global_derivative_relative_l1_max"
                            ]
                        )
                        and metrics["row_relative_l1_median"]
                        <= float(
                            thresholds[
                                "row_relative_l1_median_max"
                            ]
                        )
                        and metrics["row_relative_l1_q90"]
                        <= float(
                            thresholds["row_relative_l1_q90_max"]
                        )
                        and metrics["row_direction_cosine_median"]
                        >= float(
                            thresholds[
                                "row_direction_cosine_median_min"
                            ]
                        )
                        and derivative_diagnostics[
                            "linear_relative_residual_l2"
                        ]
                        <= float(
                            thresholds[
                                "independent_linear_relative_residual_l2_max"
                            ]
                        )
                        and metrics["scalar_spearman"]
                        >= float(thresholds["scalar_spearman_min"])
                        and metrics["scalar_rmae"]
                        <= float(thresholds["scalar_rmae_max"])
                    )
                    all_rows.append(row)
        except Exception as exc:
            failures.append(
                {
                    **pair_spec,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
        progress = {
            "completed_pairs": task_index,
            "planned_pairs": len(config["pairs"]),
            "conditions_completed": len(all_rows),
            "failures": len(failures),
            "wall_seconds": time.perf_counter() - started,
        }
        dump_json(analysis_root / "progress.json", progress)
        print(json.dumps(progress), flush=True)

    rows = pd.DataFrame(all_rows)
    solvers = pd.DataFrame(solver_rows)
    failure_frame = pd.DataFrame(failures)
    rows.to_csv(
        result_root / "uot_independent_derivative_conditions.tsv",
        sep="\t",
        index=False,
    )
    solvers.to_csv(
        result_root / "uot_independent_solver_comparison.tsv",
        sep="\t",
        index=False,
    )
    failure_frame.to_csv(
        result_root / "uot_independent_failures.tsv",
        sep="\t",
        index=False,
    )
    summary = {
        "analysis_version": config["analysis_version"],
        "config_sha256": config_hash,
        "selected_independent_units": int(
            rows["independent_unit_id"].nunique()
        )
        if len(rows)
        else 0,
        "parameter_conditions": int(rows["label"].nunique())
        if len(rows)
        else 0,
        "arm_channel_conditions": len(rows),
        "validation_passes": int(rows["validation_pass"].sum())
        if len(rows)
        else 0,
        "failed_tasks": len(failures),
        "all_conditions_pass": bool(
            len(rows) == len(registry)
            and rows["validation_pass"].astype(bool).all()
            and not failures
        ),
        "baseline_plan_relative_l1_max": float(
            solvers["baseline_plan_relative_l1"].max()
        )
        if len(solvers)
        else float("nan"),
        "global_derivative_relative_l1_max": float(
            rows["global_derivative_relative_l1"].max()
        )
        if len(rows)
        else float("nan"),
        "row_relative_l1_median_max": float(
            rows["row_relative_l1_median"].max()
        )
        if len(rows)
        else float("nan"),
        "row_relative_l1_q90_max": float(
            rows["row_relative_l1_q90"].max()
        )
        if len(rows)
        else float("nan"),
        "row_direction_cosine_median_min": float(
            rows["row_direction_cosine_median"].min()
        )
        if len(rows)
        else float("nan"),
        "scalar_spearman_min": float(rows["scalar_spearman"].min())
        if len(rows)
        else float("nan"),
        "scalar_rmae_max": float(rows["scalar_rmae"].max())
        if len(rows)
        else float("nan"),
        "nonestimable_fraction_max": float(
            rows["scalar_nonestimable_fraction"].max()
        )
        if len(rows)
        else float("nan"),
    }
    dump_json(
        result_root / "UOT_INDEPENDENT_VALIDATION_SUMMARY.json",
        summary,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0 if summary["all_conditions_pass"] else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not args.run:
        raise SystemExit("use --run to execute the frozen validation design")
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
