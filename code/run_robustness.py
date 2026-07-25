from __future__ import annotations

import json
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from functools import lru_cache
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd

from validot.benchmark import run_audit
from validot.controls import permutation_controls
from validot.evaluation import (
    exact_combined_risk,
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
SELECTION = ROOT / "01_manifest" / "pair_selection.tsv"
PROCESSED = ROOT / "03_data_processed" / "external_pairs"
OUTPUT = ROOT / "11_E7_robustness"


def parameter_variants(method: str):
    base = deepcopy(CONFIG["solver"])
    variants = [("primary", base)]
    if method in {"row_softmax", "balanced_ot", "uot"}:
        for factor in [0.5, 2.0]:
            setting = deepcopy(base)
            setting["epsilon"] = base["epsilon"] * factor
            variants.append((f"epsilon_{factor:g}x", setting))
    if method == "uot":
        for tau in [1.0, 4.0]:
            setting = deepcopy(base)
            setting["uot_tau"] = tau
            variants.append((f"tau_{tau:g}", setting))
    if method in {"paste_fgw", "paste2_partial_fgw"}:
        for alpha in [0.05, 0.20]:
            setting = deepcopy(base)
            setting["fgw_alpha"] = alpha
            variants.append((f"alpha_{alpha:g}", setting))
    if method == "paste2_partial_fgw":
        for overlap in [0.50, 0.90]:
            setting = deepcopy(base)
            setting["paste2_overlap"] = overlap
            variants.append((f"overlap_{overlap:g}", setting))
    return variants


def task_metrics(pair, method, settings, include_controls: bool = False, control_seed: int = 0):
    audit = run_audit(pair, method, settings)
    solve_results = {
        "base": audit.base,
        **{f"deleted::{key}": value for key, value in audit.deleted.items()},
        **{f"endpoint::{key}": value for key, value in audit.endpoint.items()},
    }
    failed_solves = [name for name, result in solve_results.items() if not result.converged]
    if failed_solves:
        raise RuntimeError(f"non-converged audit solves: {', '.join(failed_solves)}")
    fidelity = pd.DataFrame(internal_fidelity_records(pair, audit))
    losses = semisynthetic_losses(pair, audit)
    utility = pd.DataFrame(external_utility_records(pair, audit, losses))
    missing = pd.DataFrame(missing_records(pair, audit))
    result = {
        "median_spearman": float(fidelity.spearman.median()),
        "median_top_decile": float(fidelity.top_decile_precision.median()),
        "median_spatial_block_same_sign_fraction": float(
            fidelity.spatial_block_same_sign_fraction.median()
        ),
        "exact_top1_nex_aurc": float(
            utility.loc[(utility.witness == "top1_error") & (utility.score == "exact_combined"), "normalized_excess_aurc"].median()
        ),
        "mass_deficit_missing_auroc": float(
            missing.loc[missing.score == "mass_deficit", "auroc"].median()
        ) if len(missing) else float("nan"),
    }
    if include_controls:
        exact = exact_combined_risk(audit)
        for offset, (witness, loss) in enumerate(losses.items()):
            valid = np.isfinite(loss)
            controls = permutation_controls(
                loss[valid],
                exact[valid],
                pair.source_xy[valid],
                repeats=100,
                seed=control_seed + offset,
            )
            prefix = f"control_{witness}__"
            result.update({f"{prefix}{key}": value for key, value in controls.items()})
    return result


@lru_cache(maxsize=4)
def cached_pair(path_string: str):
    return load_pair(Path(path_string))[0]


def execute_task(spec: dict[str, object]) -> dict[str, object]:
    task_id = str(spec["task_id"])
    status_path = OUTPUT / "checkpoints" / task_id / "result.json"
    if status_path.exists() and read_json(status_path).get("status") == "COMPLETED":
        return read_json(status_path)
    status_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    try:
        base = cached_pair(str(spec["base_path"]))
        fold = spec["fold"]
        x = (
            base.source_x
            if fold is None
            else base.source_x[:, int(fold) :: CONFIG["preprocessing"]["heldout_gene_folds"]]
        )
        pair = generate_pair(
            x,
            base.source_xy,
            base.source_labels,
            str(spec["scenario"]),
            int(spec["seed"]),
            n=int(spec["n"]),
            dataset=str(spec["dataset"]),
        )
        record = {
            "dataset": str(spec["dataset"]),
            "scenario": str(spec["scenario"]),
            "seed": int(spec["seed"]),
            "method": str(spec["method"]),
            "variant": str(spec["variant"]),
            "requested_n": int(spec["n"]),
            "actual_n": len(pair.source_x),
            "feature_fold": fold,
            "seconds": 0.0,
            **task_metrics(
                pair,
                str(spec["method"]),
                spec["settings"],
                include_controls=str(spec["variant"]) == "parameter::primary",
                control_seed=int(spec["seed"]) + 1000 * CONFIG["methods"]["required"].index(str(spec["method"])),
            ),
        }
        record["seconds"] = time.perf_counter() - started
        payload = status_payload("E7_TASK", "COMPLETED", task_id=task_id, record=record)
    except Exception as exc:
        payload = status_payload(
            "E7_TASK",
            "FAILED_NUMERIC",
            task_id=task_id,
            dataset=str(spec["dataset"]),
            scenario=str(spec["scenario"]),
            seed=int(spec["seed"]),
            method=str(spec["method"]),
            variant=str(spec["variant"]),
            requested_n=int(spec["n"]),
            feature_fold=spec["fold"],
            seconds=time.perf_counter() - started,
            error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(),
        )
    write_json(status_path, payload)
    return payload


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    selection = pd.read_csv(SELECTION, sep="\t")
    bases = selection.sort_values("pair_id").groupby("dataset", as_index=False).first()
    specs = []
    for row in bases.to_dict(orient="records"):
        for scenario in ["crop_missing", "combined"]:
            for seed in CONFIG["semisynthetic"]["scale_seeds"]:
                for method in CONFIG["methods"]["required"]:
                    tasks = []
                    for name, settings in parameter_variants(method):
                        tasks.append((f"parameter::{name}", 800, None, settings))
                    for n in CONFIG["preprocessing"]["scale_n"]:
                        tasks.append((f"scale::n{n}", int(n), None, deepcopy(CONFIG["solver"])))
                    for fold in range(CONFIG["preprocessing"]["heldout_gene_folds"]):
                        tasks.append((f"feature_fold::{fold}", 800, fold, deepcopy(CONFIG["solver"])))
                    seen = set()
                    for variant, n, fold, settings in tasks:
                        key = (variant, n, fold)
                        if key in seen:
                            continue
                        seen.add(key)
                        task_id = f"{row['dataset']}__{scenario}__s{seed}__{method}__{variant.replace(':','_')}"
                        specs.append(
                            {
                                "task_id": task_id,
                                "base_path": str(PROCESSED / f"{row['pair_id']}.npz"),
                                "dataset": str(row["dataset"]),
                                "scenario": scenario,
                                "seed": int(seed),
                                "method": method,
                                "variant": variant,
                                "n": int(n),
                                "fold": fold,
                                "settings": settings,
                            }
                        )
    records = []
    failures = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(execute_task, spec) for spec in specs]
        for completed, future in enumerate(as_completed(futures), start=1):
            payload = future.result()
            if payload["status"] != "COMPLETED":
                failures.append({key: value for key, value in payload.items() if key != "traceback"})
            else:
                records.append(payload["record"])
            if completed % 25 == 0 or completed == len(futures):
                print(f"progress {completed}/{len(futures)}", flush=True)
    table = pd.DataFrame(records)
    table.to_csv(OUTPUT / "robustness_all.tsv", sep="\t", index=False)
    pd.DataFrame(failures).to_csv(OUTPUT / "robustness_numeric_failures.tsv", sep="\t", index=False)
    primary = table[table.variant == "parameter::primary"].copy()
    comparison = table.merge(
        primary[["dataset", "scenario", "seed", "method", "exact_top1_nex_aurc"]].rename(
            columns={"exact_top1_nex_aurc": "primary_exact_top1_nex_aurc"}
        ),
        on=["dataset", "scenario", "seed", "method"],
        how="left",
    )
    comparison["same_external_direction_as_primary"] = np.sign(comparison.exact_top1_nex_aurc) == np.sign(
        comparison.primary_exact_top1_nex_aurc
    )
    comparison.to_csv(OUTPUT / "robustness_comparison.tsv", sep="\t", index=False)
    decision = status_payload(
        "E7",
        "COMPLETED_WITH_NUMERIC_FAILURES" if failures else "COMPLETED",
        tasks=len(table),
        planned_tasks=len(specs),
        numerical_failures=len(failures),
        technologies=int(table.dataset.nunique()),
        methods=int(table.method.nunique()),
        direction_stability=float(comparison.same_external_direction_as_primary.mean()),
    )
    write_json(OUTPUT / "E7_DECISION.json", decision)
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
