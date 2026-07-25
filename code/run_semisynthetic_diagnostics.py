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
from sklearn.metrics import average_precision_score, roc_auc_score

from run_semisynthetic_benchmark import select_bases
from validot.benchmark import run_audit
from validot.evaluation import external_utility_records, missing_records, semisynthetic_losses
from validot.io import load_pair
from validot.metrics import conditional_plan
from validot.semisynthetic import generate_pair
from validot.utils import read_json, status_payload, write_json


ROOT = Path(__file__).resolve().parents[1]
CONFIG = read_json(ROOT / "00_protocol" / "frozen_config.json")
SELECTION = ROOT / "01_manifest" / "pair_selection.tsv"
PROCESSED = ROOT / "03_data_processed" / "external_pairs"
OUTPUT = ROOT / "09_E5_semisynthetic_external" / "diagnostics"


@lru_cache(maxsize=8)
def cached_base(path_string: str, side: str):
    pair, _ = load_pair(Path(path_string))
    if side == "source":
        return pair.source_x, pair.source_xy, pair.source_labels
    return pair.target_x, pair.target_xy, pair.target_labels


def duplicate_metrics(pair, audit) -> dict[str, object]:
    conditional, _ = conditional_plan(audit.base.plan)
    predicted = np.argmax(conditional, axis=1)
    ambiguous = np.asarray([len(targets) == 2 for targets in pair.equivalent_targets], dtype=bool)
    asymmetry = []
    both_top2 = []
    strict_error = []
    for source_index in np.flatnonzero(ambiguous):
        first, second = pair.equivalent_targets[source_index]
        p, q = conditional[source_index, [first, second]]
        asymmetry.append(abs(p - q) / max(p + q, 1e-12))
        top2 = set(np.argpartition(conditional[source_index], -2)[-2:].tolist())
        both_top2.append({first, second}.issubset(top2))
        strict_error.append(predicted[source_index] != pair.truth_target[source_index])
    entropy = audit.proxies["conditional_entropy"]
    margin = audit.proxies["probability_margin"]
    return {
        "ambiguous_n": int(ambiguous.sum()),
        "mean_dual_asymmetry": float(np.mean(asymmetry)),
        "max_dual_asymmetry": float(np.max(asymmetry)),
        "strict_identity_error_rate": float(np.mean(strict_error)),
        "top2_contains_both_fraction": float(np.mean(both_top2)),
        "entropy_ambiguity_auroc": float(roc_auc_score(ambiguous, entropy)),
        "entropy_ambiguity_auprc": float(average_precision_score(ambiguous, entropy)),
        "top2_margin_ambiguity_auroc": float(roc_auc_score(ambiguous, margin)),
        "top2_margin_ambiguity_auprc": float(average_precision_score(ambiguous, margin)),
    }


def execute_task(spec: dict[str, object]) -> dict[str, object]:
    task_id = str(spec["task_id"])
    task_dir = OUTPUT / "checkpoints" / task_id
    status_path = task_dir / "status.json"
    if status_path.exists() and read_json(status_path).get("status") == "COMPLETED":
        return read_json(status_path)
    task_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    try:
        x, xy, labels = cached_base(str(spec["base_path"]), str(spec["base_side"]))
        scenario = str(spec["scenario"])
        pair = generate_pair(
            x,
            xy,
            labels,
            scenario,
            int(spec["seed"]),
            n=CONFIG["preprocessing"]["semisynthetic_main_n"],
            crop_fraction=float(spec.get("crop_fraction", 0.25)),
            dataset=str(spec["dataset_id"]),
        )
        pair.pair_id = f"{pair.pair_id}_{spec['diagnostic']}"
        audit = run_audit(pair, str(spec["method"]), CONFIG["solver"])
        if scenario == "crop_missing":
            missing = pd.DataFrame(missing_records(pair, audit))
            missing["crop_fraction"] = float(spec["crop_fraction"])
            utility = pd.DataFrame(
                external_utility_records(pair, audit, semisynthetic_losses(pair, audit))
            )
            utility["crop_fraction"] = float(spec["crop_fraction"])
            missing.to_csv(task_dir / "missing.tsv", sep="\t", index=False)
            utility.to_csv(task_dir / "utility.tsv", sep="\t", index=False)
        else:
            record = {
                "dataset": pair.dataset,
                "pair_id": pair.pair_id,
                "scenario": scenario,
                "seed": int(spec["seed"]),
                "method": str(spec["method"]),
                **duplicate_metrics(pair, audit),
            }
            pd.DataFrame([record]).to_csv(task_dir / "duplicate.tsv", sep="\t", index=False)
        converged = bool(
            audit.base.converged
            and all(item.converged for item in audit.deleted.values())
            and all(item.converged for item in audit.endpoint.values())
        )
        payload = status_payload(
            "E5_DIAGNOSTIC_TASK",
            "COMPLETED" if converged else "FAILED_NUMERIC",
            task_id=task_id,
            seconds=time.perf_counter() - started,
            converged=converged,
        )
    except Exception as exc:
        payload = status_payload(
            "E5_DIAGNOSTIC_TASK",
            "FAILED_NUMERIC",
            task_id=task_id,
            seconds=time.perf_counter() - started,
            error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(),
        )
    write_json(status_path, payload)
    return payload


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    bases = select_bases(pd.read_csv(SELECTION, sep="\t"))
    specs = []
    for base_index, row in enumerate(bases.to_dict(orient="records")):
        common = {
            "base_path": str(PROCESSED / f"{row['pair_id']}.npz"),
            "base_side": row["base_side"],
            "dataset_id": f"{row['dataset']}_base{base_index}",
        }
        for seed in CONFIG["semisynthetic"]["seeds"]:
            for method in CONFIG["methods"]["required"]:
                specs.append(
                    {
                        **common,
                        "task_id": f"{common['dataset_id']}__crop10__s{seed}__{method}",
                        "diagnostic": "crop10",
                        "scenario": "crop_missing",
                        "crop_fraction": 0.10,
                        "seed": seed,
                        "method": method,
                    }
                )
                specs.append(
                    {
                        **common,
                        "task_id": f"{common['dataset_id']}__duplicate__s{seed}__{method}",
                        "diagnostic": "duplicate",
                        "scenario": "duplicate_motif",
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
            if status["status"] != "COMPLETED":
                for pending in futures:
                    pending.cancel()
                print(json.dumps(status, ensure_ascii=False, indent=2))
                return 2
            if completed % 25 == 0 or completed == len(futures):
                print(f"progress {completed}/{len(futures)}", flush=True)
    for filename in ["missing.tsv", "utility.tsv", "duplicate.tsv"]:
        paths = list((OUTPUT / "checkpoints").glob(f"*/{filename}"))
        if paths:
            pd.concat([pd.read_csv(path, sep="\t") for path in paths], ignore_index=True).to_csv(
                OUTPUT / filename, sep="\t", index=False
            )
    decision = status_payload(
        "E5_DIAGNOSTICS",
        "COMPLETED",
        tasks=len(statuses),
        bases=len(bases),
        seeds=len(CONFIG["semisynthetic"]["seeds"]),
        methods=len(CONFIG["methods"]["required"]),
        diagnostics=["crop_missing_10pct", "duplicate_motif_identity"],
    )
    write_json(OUTPUT / "E5_DIAGNOSTICS_DECISION.json", decision)
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
