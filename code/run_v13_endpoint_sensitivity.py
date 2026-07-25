from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from validot.benchmark import _masses, _solve_common
from validot.io import load_pair
from validot.metrics import exact_row_response, fidelity_metrics, mae_diagnostics
from validot.solvers import cost_components
from validot.utils import read_json, status_payload, write_json


ROOT = Path(__file__).resolve().parents[1]
CONFIG = read_json(ROOT / "00_protocol" / "frozen_config.json")
OUTPUT = ROOT / "15_v1_3_correction" / "04_p1_sensitivity" / "endpoint_step"
PAIR_ROOT = ROOT / "03_data_processed" / "external_pairs"
RESPONSE_ROOT = ROOT / "10_E6_real_external"
PAIR_IDS = ["STAR_8M_D1_D2", "ST_E15_5_S1_S2", "ST_DEV_E13_5_E14_5"]
METHODS = (
    CONFIG["v1_3_correction"]["confirmatory_ot_methods"]
    + CONFIG["v1_3_correction"]["non_ot_stress_test"]
)
STEPS = CONFIG["v1_3_correction"]["endpoint_steps"]


def execute(pair_id: str, method: str) -> list[dict[str, object]]:
    pair, _ = load_pair(PAIR_ROOT / f"{pair_id}.npz")
    with np.load(RESPONSE_ROOT / pair_id / method / "row_responses.npz") as old:
        exact = {
            "I_EXPR": old["exact_I_EXPR"].copy(),
            "I_SPATIAL": old["exact_I_SPATIAL"].copy(),
        }
    components = cost_components(pair.source_x, pair.target_x, pair.source_xy, pair.target_xy)
    a, b = _masses(pair)
    base_started = time.perf_counter()
    base = _solve_common(method, components, a, b, CONFIG["solver"], 0.5, 0.5)
    base_wall_seconds = time.perf_counter() - base_started
    rows = []
    for step in STEPS:
        for intervention in ("I_EXPR", "I_SPATIAL"):
            expression_weight = 0.5 * (1.0 - step) if intervention == "I_EXPR" else 0.5
            spatial_weight = 0.5 if intervention == "I_EXPR" else 0.5 * (1.0 - step)
            started = time.perf_counter()
            endpoint = _solve_common(
                method,
                components,
                a,
                b,
                CONFIG["solver"],
                expression_weight,
                spatial_weight,
            )
            wall_seconds = time.perf_counter() - started
            response = exact_row_response(base.plan, endpoint.plan) / step
            rows.append(
                {
                    "dataset": pair.dataset,
                    "pair_id": pair_id,
                    "pair_type": pair.metadata.get("pair_type", ""),
                    "method": method,
                    "intervention": intervention,
                    "endpoint_step": step,
                    "base_wall_seconds": base_wall_seconds,
                    "endpoint_wall_seconds": wall_seconds,
                    "endpoint_solver_seconds": endpoint.seconds,
                    "endpoint_iterations": endpoint.iterations,
                    "endpoint_converged": endpoint.converged,
                    **fidelity_metrics(exact[intervention], response),
                    **mae_diagnostics(exact[intervention], response),
                }
            )
    return rows


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    specs = [(pair_id, method) for pair_id in PAIR_IDS for method in METHODS]
    rows = []
    with ProcessPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(execute, *spec): spec for spec in specs}
        for completed, future in enumerate(as_completed(futures), start=1):
            rows.extend(future.result())
            print(f"endpoint sensitivity {completed}/{len(specs)}", flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(OUTPUT / "endpoint_step_sensitivity.tsv", sep="\t", index=False)
    summary = (
        table.groupby(["method", "intervention", "endpoint_step"], dropna=False)
        .agg(
            pairs=("pair_id", "nunique"),
            median_spearman=("spearman", "median"),
            median_top_decile_precision=("top_decile_precision", "median"),
            median_normalized_mae=("normalized_mae", "median"),
            median_endpoint_seconds=("endpoint_wall_seconds", "median"),
            all_converged=("endpoint_converged", "all"),
        )
        .reset_index()
    )
    summary.to_csv(OUTPUT / "endpoint_step_summary.tsv", sep="\t", index=False)
    decision = status_payload(
        "V1_3_ENDPOINT_SENSITIVITY",
        "COMPLETED" if bool(table.endpoint_converged.all()) else "FAILED_NUMERIC",
        pairs=PAIR_IDS,
        methods=METHODS,
        endpoint_steps=STEPS,
        rows=len(table),
        interpretation="local finite-difference response; requires additional solver evaluations",
    )
    write_json(OUTPUT / "ENDPOINT_SENSITIVITY_DECISION.json", decision)
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    return 0 if decision["status"] == "COMPLETED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
