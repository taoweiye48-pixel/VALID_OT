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

from validot.benchmark import run_audit
from validot.evaluation import (
    external_utility_records,
    internal_fidelity_records,
    missing_records,
    semisynthetic_losses,
)
from validot.io import load_pair
from validot.semisynthetic import generate_pair
from validot.utils import read_json, status_payload, write_json


ROOT = Path(__file__).resolve().parents[1]
CONFIG = read_json(ROOT / "00_protocol" / "frozen_config.json")
PAIR_SELECTION = ROOT / "01_manifest" / "pair_selection.tsv"
PROCESSED = ROOT / "03_data_processed" / "external_pairs"
OUTPUT = ROOT / "08_E4_internal_fidelity"
OUTPUT_EXTERNAL = ROOT / "09_E5_semisynthetic_external"


def select_bases(selection: pd.DataFrame) -> pd.DataFrame:
    bases = []
    for dataset, group in selection.groupby("dataset", sort=True):
        candidates = []
        seen = set()
        for row in group.sort_values("pair_id").to_dict(orient="records"):
            for side in ["source", "target"]:
                path = row[f"{side}_path"]
                if path in seen:
                    continue
                seen.add(path)
                candidates.append({**row, "base_side": side, "base_path": path})
        bases.extend(candidates[:2])
    return pd.DataFrame(bases)


@lru_cache(maxsize=8)
def cached_base(path_string: str, side: str):
    pair, _ = load_pair(Path(path_string))
    if side == "source":
        return pair.source_x, pair.source_xy, pair.source_labels
    return pair.target_x, pair.target_xy, pair.target_labels


def execute_task(spec: dict[str, object]) -> dict[str, object]:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    task_id = str(spec["task_id"])
    task_dir = OUTPUT / "checkpoints" / task_id
    checkpoint = task_dir / "status.json"
    if checkpoint.exists() and read_json(checkpoint).get("status") == "COMPLETED":
        return read_json(checkpoint)
    task_dir.mkdir(parents=True, exist_ok=True)
    write_json(checkpoint, status_payload("E4_E5_TASK", "RUNNING", task_id=task_id))
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
        fidelity_task = pd.DataFrame(internal_fidelity_records(pair, audit))
        losses = semisynthetic_losses(pair, audit)
        utility_task = pd.DataFrame(external_utility_records(pair, audit, losses))
        missing_task = pd.DataFrame(missing_records(pair, audit))
        fidelity_task.to_csv(task_dir / "fidelity.tsv", sep="\t", index=False)
        utility_task.to_csv(task_dir / "utility.tsv", sep="\t", index=False)
        missing_task.to_csv(task_dir / "missing.tsv", sep="\t", index=False)
        converged = bool(
            audit.base.converged
            and all(result.converged for result in audit.deleted.values())
            and all(result.converged for result in audit.endpoint.values())
        )
        status = status_payload(
            "E4_E5_TASK",
            "COMPLETED" if converged else "FAILED_NUMERIC",
            task_id=task_id,
            seconds=time.perf_counter() - started,
            converged=converged,
        )
    except Exception as exc:
        status = status_payload(
            "E4_E5_TASK",
            "FAILED_NUMERIC",
            task_id=task_id,
            seconds=time.perf_counter() - started,
            error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(),
        )
    write_json(checkpoint, status)
    return status


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    OUTPUT_EXTERNAL.mkdir(parents=True, exist_ok=True)
    selection = pd.read_csv(PAIR_SELECTION, sep="\t")
    bases = select_bases(selection)
    if bases.groupby("dataset").size().min() < 2:
        raise RuntimeError("fewer than two base slices for a required technology")
    specs = []
    for base_index, row in enumerate(bases.to_dict(orient="records")):
        for scenario in CONFIG["semisynthetic"]["scenarios"]:
            for seed in CONFIG["semisynthetic"]["seeds"]:
                for method in CONFIG["methods"]["required"]:
                    dataset_id = f"{row['dataset']}_base{base_index}"
                    task_id = f"{dataset_id}_{scenario}_seed{seed}__{method}"
                    specs.append(
                        {
                            "task_id": task_id,
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
                print(json.dumps(status, ensure_ascii=False, indent=2))
                return 2
            if completed % 25 == 0 or completed == len(futures):
                print(f"progress {completed}/{len(futures)}", flush=True)
    fidelity_files = list((OUTPUT / "checkpoints").glob("*/fidelity.tsv"))
    utility_files = list((OUTPUT / "checkpoints").glob("*/utility.tsv"))
    missing_files = list((OUTPUT / "checkpoints").glob("*/missing.tsv"))
    fidelity = pd.concat([pd.read_csv(path, sep="\t") for path in fidelity_files], ignore_index=True)
    utility = pd.concat([pd.read_csv(path, sep="\t") for path in utility_files], ignore_index=True)
    missing_frames = []
    for path in missing_files:
        try:
            frame = pd.read_csv(path, sep="\t")
        except pd.errors.EmptyDataError:
            continue
        if not frame.empty:
            missing_frames.append(frame)
    missing = pd.concat(missing_frames, ignore_index=True) if missing_frames else pd.DataFrame()
    fidelity.to_csv(OUTPUT / "internal_fidelity_all.tsv", sep="\t", index=False)
    utility.to_csv(OUTPUT_EXTERNAL / "external_utility_all.tsv", sep="\t", index=False)
    missing.to_csv(OUTPUT_EXTERNAL / "missing_all.tsv", sep="\t", index=False)
    decision = status_payload(
        "E4_E5",
        "COMPLETED",
        bases=len(bases),
        scenarios=len(CONFIG["semisynthetic"]["scenarios"]),
        seeds=len(CONFIG["semisynthetic"]["seeds"]),
        methods=len(CONFIG["methods"]["required"]),
        tasks=len(fidelity_files),
        numerical_failures=0,
    )
    write_json(OUTPUT / "E4_E5_DECISION.json", decision)
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
