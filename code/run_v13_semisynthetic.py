from __future__ import annotations

import json
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd

from run_semisynthetic_benchmark import select_bases
from validot.benchmark import run_audit
from validot.evaluation import (
    external_utility_records,
    internal_fidelity_records,
    missing_records,
    risk_scores,
    semisynthetic_losses,
)
from validot.io import load_pair
from validot.semisynthetic import generate_pair
from validot.utils import read_json, status_payload, write_json


ROOT = Path(__file__).resolve().parents[1]
CONFIG = read_json(ROOT / "00_protocol" / "frozen_config.json")
CORRECTION = CONFIG["v1_3_correction"]
PAIR_SELECTION = ROOT / "01_manifest" / "pair_selection.tsv"
PROCESSED = ROOT / "03_data_processed" / "external_pairs"
OUTPUT = ROOT / "15_v1_3_correction" / "01_semisynthetic_rerun"
METHODS = CORRECTION["confirmatory_ot_methods"] + CORRECTION["non_ot_stress_test"]


@lru_cache(maxsize=8)
def cached_base(path_string: str, side: str):
    pair, _ = load_pair(Path(path_string))
    if side == "source":
        return pair.source_x, pair.source_xy, pair.source_labels
    return pair.target_x, pair.target_xy, pair.target_labels


def execute_task(spec: dict[str, object]) -> dict[str, object]:
    task_id = str(spec["task_id"])
    task_dir = OUTPUT / "checkpoints" / task_id
    checkpoint = task_dir / "status.json"
    if checkpoint.exists() and read_json(checkpoint).get("status") == "COMPLETED":
        return read_json(checkpoint)
    task_dir.mkdir(parents=True, exist_ok=True)
    write_json(checkpoint, status_payload("V1_3_SEMISYNTHETIC_TASK", "RUNNING", task_id=task_id))
    started = time.perf_counter()
    try:
        base_x, base_xy, base_labels = cached_base(str(spec["base_path"]), str(spec["base_side"]))
        pair = generate_pair(
            base_x,
            base_xy,
            base_labels,
            str(spec["scenario"]),
            int(spec["seed"]),
            n=CONFIG["preprocessing"]["semisynthetic_main_n"],
            dataset=str(spec["dataset_id"]),
        )
        audit = run_audit(pair, str(spec["method"]), CONFIG["solver"])
        fidelity = pd.DataFrame(internal_fidelity_records(pair, audit))
        losses = semisynthetic_losses(pair, audit)
        utility = pd.DataFrame(external_utility_records(pair, audit, losses))
        missing = pd.DataFrame(missing_records(pair, audit))
        fidelity.to_csv(task_dir / "fidelity.tsv", sep="\t", index=False)
        utility.to_csv(task_dir / "utility.tsv", sep="\t", index=False)
        missing.to_csv(task_dir / "missing.tsv", sep="\t", index=False)
        score_arrays = risk_scores(pair, audit)
        np.savez_compressed(
            task_dir / "row_responses.npz",
            **{f"loss__{name}": value for name, value in losses.items()},
            **{f"risk__{name}": value for name, value in score_arrays.items()},
            exact_I_EXPR=audit.exact_response["I_EXPR"],
            exact_I_SPATIAL=audit.exact_response["I_SPATIAL"],
            finite_difference_I_EXPR_h001=audit.endpoint_response["I_EXPR"],
            finite_difference_I_SPATIAL_h001=audit.endpoint_response["I_SPATIAL"],
            source_xy=pair.source_xy,
        )
        converged = bool(
            audit.base.converged
            and all(result.converged for result in audit.deleted.values())
            and all(result.converged for result in audit.endpoint.values())
        )
        status = status_payload(
            "V1_3_SEMISYNTHETIC_TASK",
            "COMPLETED" if converged else "FAILED_NUMERIC",
            task_id=task_id,
            method=str(spec["method"]),
            seconds=time.perf_counter() - started,
            converged=converged,
            tie_handling="fractional",
        )
    except Exception as exc:
        status = status_payload(
            "V1_3_SEMISYNTHETIC_TASK",
            "FAILED_NUMERIC",
            task_id=task_id,
            method=str(spec["method"]),
            seconds=time.perf_counter() - started,
            error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(),
        )
    write_json(checkpoint, status)
    return status


def _read_nonempty(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        try:
            frame = pd.read_csv(path, sep="\t")
        except pd.errors.EmptyDataError:
            continue
        if not frame.empty:
            frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    selection = pd.read_csv(PAIR_SELECTION, sep="\t")
    bases = select_bases(selection)
    specs = []
    for base_index, row in enumerate(bases.to_dict(orient="records")):
        for scenario in CONFIG["semisynthetic"]["scenarios"]:
            for seed in CONFIG["semisynthetic"]["seeds"]:
                for method in METHODS:
                    dataset_id = f"{row['dataset']}_base{base_index}"
                    specs.append(
                        {
                            "task_id": f"{dataset_id}_{scenario}_seed{seed}__{method}",
                            "base_path": str(PROCESSED / f"{row['pair_id']}.npz"),
                            "base_side": row["base_side"],
                            "dataset_id": dataset_id,
                            "scenario": scenario,
                            "seed": seed,
                            "method": method,
                        }
                    )
    statuses = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(execute_task, spec) for spec in specs]
        for completed, future in enumerate(as_completed(futures), start=1):
            status = future.result()
            statuses.append(status)
            if status["status"] == "FAILED_NUMERIC":
                for pending in futures:
                    pending.cancel()
                print(json.dumps(status, ensure_ascii=False, indent=2), flush=True)
                return 2
            if completed % 25 == 0 or completed == len(futures):
                print(f"v1.3 semisynthetic progress {completed}/{len(futures)}", flush=True)
    checkpoints = OUTPUT / "checkpoints"
    fidelity = _read_nonempty(list(checkpoints.glob("*/fidelity.tsv")))
    utility = _read_nonempty(list(checkpoints.glob("*/utility.tsv")))
    missing = _read_nonempty(list(checkpoints.glob("*/missing.tsv")))
    fidelity.to_csv(OUTPUT / "internal_fidelity_all.tsv", sep="\t", index=False)
    utility.to_csv(OUTPUT / "external_utility_all.tsv", sep="\t", index=False)
    missing.to_csv(OUTPUT / "missing_all.tsv", sep="\t", index=False)
    decision = status_payload(
        "V1_3_SEMISYNTHETIC",
        "COMPLETED",
        bases=len(bases),
        scenarios=len(CONFIG["semisynthetic"]["scenarios"]),
        seeds=len(CONFIG["semisynthetic"]["seeds"]),
        methods=METHODS,
        tasks=len(specs),
        tie_handling="fractional",
        fgw_status="excluded_pending_objective_consistent_rerun",
    )
    write_json(OUTPUT / "V1_3_SEMISYNTHETIC_DECISION.json", decision)
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
