from __future__ import annotations

import json
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd

from validot.benchmark import run_audit
from validot.controls import real_control_records
from validot.evaluation import (
    external_utility_records,
    internal_fidelity_records,
    real_losses,
    risk_scores,
)
from validot.external_data import prepare_pair_from_h5ad
from validot.io import load_pair, save_pair
from validot.metrics import conditional_plan
from validot.semisynthetic import PairedData
from validot.utils import file_hash, read_json, status_payload, write_json


ROOT = Path(__file__).resolve().parents[1]
CONFIG = read_json(ROOT / "00_protocol" / "frozen_config.json")
PAIR_SELECTION = ROOT / "01_manifest" / "pair_selection.tsv"
PROCESSED = ROOT / "03_data_processed" / "external_pairs"
OUTPUT = ROOT / "10_E6_real_external"


def prepare_pair(row: dict[str, object]) -> Path:
    output = PROCESSED / f"{row['pair_id']}.npz"
    if output.exists():
        try:
            pair, extras = load_pair(output)
            required_extras = {"source_library_size", "target_library_size", "source_region_size", "target_region_size"}
            if (
                pair.metadata.get("pair_type") == row["pair_type"]
                and pair.metadata.get("direction") == "forward"
                and required_extras.issubset(extras)
            ):
                return output
            output.unlink()
        except Exception:
            if output.exists():
                output.unlink()
    pair, extras = prepare_pair_from_h5ad(
        Path(str(row["source_path"])),
        Path(str(row["target_path"])),
        str(row["source_label_column"]),
        str(row["target_label_column"]),
        str(row["dataset"]),
        str(row["pair_id"]),
        n_hvg=CONFIG["preprocessing"]["n_hvg"],
        heldout_n=100,
        max_n=CONFIG["preprocessing"]["main_n_max"],
        seed=CONFIG["preprocessing"]["seed"],
    )
    pair.metadata.update(
        {
            "pair_type": row["pair_type"],
            "biological_pair_id": row["pair_id"],
            "direction": "forward",
        }
    )
    save_pair(output, pair, extras)
    return output


def prepare_reverse_pair(forward_path: Path) -> Path:
    pair, extras = load_pair(forward_path)
    output = forward_path.with_name(f"{pair.pair_id}__reverse.npz")
    if output.exists():
        try:
            reverse, reverse_extras = load_pair(output)
            if reverse.metadata.get("direction") == "reverse" and "source_library_size" in reverse_extras:
                return output
        except Exception:
            pass
        output.unlink()
    metadata = dict(pair.metadata)
    metadata["direction"] = "reverse"
    reverse = PairedData(
        dataset=pair.dataset,
        pair_id=f"{pair.pair_id}__reverse",
        source_x=pair.target_x,
        target_x=pair.source_x,
        source_xy=pair.target_xy,
        target_xy=pair.source_xy,
        source_labels=pair.target_labels,
        target_labels=pair.source_labels,
        truth_target=np.full(len(pair.target_x), -1, dtype=int),
        truth_missing=np.zeros(len(pair.target_x), dtype=bool),
        equivalent_targets=[[] for _ in range(len(pair.target_x))],
        metadata=metadata,
    )
    reverse_extras = {
        "source_heldout": extras["target_heldout"],
        "target_heldout": extras["source_heldout"],
        "source_library_size": extras["target_library_size"],
        "target_library_size": extras["source_library_size"],
        "source_region_size": extras["target_region_size"],
        "target_region_size": extras["source_region_size"],
        "source_ids": extras["target_ids"],
        "target_ids": extras["source_ids"],
        "cost_genes": extras["cost_genes"],
        "heldout_genes": extras["heldout_genes"],
    }
    save_pair(output, reverse, reverse_extras)
    return output


def run_one(pair_path: Path, method: str) -> dict[str, object]:
    pair, extras = load_pair(pair_path)
    method_dir = OUTPUT / pair.pair_id / method
    status_path = method_dir / "status.json"
    fidelity_path = method_dir / "internal_fidelity.tsv"
    external_path = method_dir / "external_utility.tsv"
    controls_path = method_dir / "controls.tsv"
    quality_path = method_dir / "alignment_quality.tsv"
    responses_path = method_dir / "row_responses.npz"
    if status_path.exists() and read_json(status_path).get("status") == "COMPLETED":
        return read_json(status_path)
    method_dir.mkdir(parents=True, exist_ok=True)
    write_json(status_path, status_payload("E6_PAIR_METHOD", "RUNNING", pair_id=pair.pair_id, method=method))
    started = time.perf_counter()
    try:
        audit = run_audit(pair, method, CONFIG["solver"])
        fidelity = pd.DataFrame(internal_fidelity_records(pair, audit))
        losses = real_losses(pair, audit, extras["source_heldout"], extras["target_heldout"])
        external = pd.DataFrame(external_utility_records(pair, audit, losses))
        controls = pd.DataFrame(real_control_records(pair, audit, losses, extras, repeats=100))
        quality = pd.DataFrame(
            [
                {
                    "dataset": pair.dataset,
                    "pair_type": pair.metadata.get("pair_type", ""),
                    "biological_pair_id": pair.metadata.get("biological_pair_id", pair.pair_id),
                    "pair_id": pair.pair_id,
                    "direction": pair.metadata.get("direction", "forward"),
                    "method": method,
                    "region_matching_index": float(1.0 - np.mean(losses["label_error"])),
                    "mean_heldout_expression_loss": float(np.mean(losses["heldout_loss"])),
                }
            ]
        )
        fidelity.to_csv(fidelity_path, sep="\t", index=False)
        external.to_csv(external_path, sep="\t", index=False)
        controls.to_csv(controls_path, sep="\t", index=False)
        quality.to_csv(quality_path, sep="\t", index=False)
        conditional, _ = conditional_plan(audit.base.plan)
        predicted = np.argmax(conditional, axis=1)
        scores = risk_scores(pair, audit)
        np.savez_compressed(
            responses_path,
            exact_I_EXPR=audit.exact_response["I_EXPR"],
            exact_I_SPATIAL=audit.exact_response["I_SPATIAL"],
            endpoint_I_EXPR=audit.endpoint_response["I_EXPR"],
            endpoint_I_SPATIAL=audit.endpoint_response["I_SPATIAL"],
            label_error=losses["label_error"],
            heldout_loss=losses["heldout_loss"],
            source_xy=pair.source_xy,
            source_labels=pair.source_labels.astype(str),
            predicted_target_labels=pair.target_labels[predicted].astype(str),
            source_library_size=extras["source_library_size"],
            source_region_size=extras["source_region_size"],
            **{f"risk__{name}": value for name, value in scores.items()},
        )
        converged = bool(
            audit.base.converged
            and all(result.converged for result in audit.deleted.values())
            and all(result.converged for result in audit.endpoint.values())
        )
        status = status_payload(
            "E6_PAIR_METHOD",
            "COMPLETED" if converged else "FAILED_NUMERIC",
            pair_id=pair.pair_id,
            method=method,
            seconds=time.perf_counter() - started,
            converged=converged,
            n_source=len(pair.source_x),
            n_target=len(pair.target_x),
            fidelity_sha256=file_hash(fidelity_path),
            external_sha256=file_hash(external_path),
            controls_sha256=file_hash(controls_path),
            quality_sha256=file_hash(quality_path),
            total_solver_seconds=float(
                audit.base.seconds
                + sum(result.seconds for result in audit.deleted.values())
                + sum(result.seconds for result in audit.endpoint.values())
            ),
            total_solver_iterations=int(
                audit.base.iterations
                + sum(result.iterations for result in audit.deleted.values())
                + sum(result.iterations for result in audit.endpoint.values())
            ),
        )
    except Exception as exc:
        status = status_payload(
            "E6_PAIR_METHOD",
            "FAILED_NUMERIC",
            pair_id=pair.pair_id,
            method=method,
            seconds=time.perf_counter() - started,
            error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(),
        )
    write_json(status_path, status)
    return status


def summarize_external(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    qc_baselines = {"source_boundary_proximity", "source_sparsity", "matched_target_sparsity"}
    direction_averaged = (
        table.groupby(
            ["dataset", "pair_type", "biological_pair_id", "method", "witness", "score"],
            dropna=False,
        )
        .agg(normalized_excess_aurc=("normalized_excess_aurc", "mean"), directions=("direction", "nunique"))
        .reset_index()
    )
    for keys, group in direction_averaged.groupby(
        ["dataset", "pair_type", "biological_pair_id", "method", "witness"]
    ):
        exact_rows = group[group.score == "exact_combined"]
        baselines = group[group.score.isin(qc_baselines)]
        if exact_rows.empty or baselines.empty:
            continue
        exact = float(exact_rows.iloc[0].normalized_excess_aurc)
        best = float(baselines.normalized_excess_aurc.min())
        rows.append(
            {
                "dataset": keys[0],
                "pair_type": keys[1],
                "biological_pair_id": keys[2],
                "method": keys[3],
                "witness": keys[4],
                "directions": int(group.directions.min()),
                "exact_normalized_excess_aurc": exact,
                "best_baseline_normalized_excess_aurc": best,
                "absolute_improvement": best - exact,
                "relative_improvement": 1.0 - exact / max(abs(best), 1e-12),
                "positive": exact < best,
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    if not PAIR_SELECTION.exists():
        raise FileNotFoundError(PAIR_SELECTION)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    selection = pd.read_csv(PAIR_SELECTION, sep="\t")
    pair_paths = []
    for row in selection.to_dict(orient="records"):
        forward_path = prepare_pair(row)
        pair_paths.extend([forward_path, prepare_reverse_pair(forward_path)])
    specs = [
        (pair_path, method)
        for pair_path in pair_paths
        for method in CONFIG["methods"]["required"]
    ]
    statuses = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(run_one, pair_path, method) for pair_path, method in specs]
        for completed, future in enumerate(as_completed(futures), start=1):
            status = future.result()
            statuses.append(status)
            if status["status"] == "FAILED_NUMERIC":
                for pending in futures:
                    pending.cancel()
                pd.DataFrame(statuses).to_json(OUTPUT / "run_statuses.json", orient="records", indent=2)
                print(json.dumps(status, ensure_ascii=False, indent=2))
                return 2
            if completed % 5 == 0 or completed == len(futures):
                print(f"progress {completed}/{len(futures)}", flush=True)
    pd.DataFrame(statuses).to_json(OUTPUT / "run_statuses.json", orient="records", indent=2)
    fidelity_files = list(OUTPUT.glob("*/*/internal_fidelity.tsv"))
    external_files = list(OUTPUT.glob("*/*/external_utility.tsv"))
    controls_files = list(OUTPUT.glob("*/*/controls.tsv"))
    quality_files = list(OUTPUT.glob("*/*/alignment_quality.tsv"))
    fidelity = pd.concat([pd.read_csv(path, sep="\t") for path in fidelity_files], ignore_index=True)
    external = pd.concat([pd.read_csv(path, sep="\t") for path in external_files], ignore_index=True)
    controls = pd.concat([pd.read_csv(path, sep="\t") for path in controls_files], ignore_index=True)
    quality = pd.concat([pd.read_csv(path, sep="\t") for path in quality_files], ignore_index=True)
    fidelity.to_csv(OUTPUT / "all_internal_fidelity.tsv", sep="\t", index=False)
    external.to_csv(OUTPUT / "all_external_utility.tsv", sep="\t", index=False)
    controls.to_csv(OUTPUT / "all_controls.tsv", sep="\t", index=False)
    quality.to_csv(OUTPUT / "all_alignment_quality.tsv", sep="\t", index=False)
    summary = summarize_external(external)
    summary.to_csv(OUTPUT / "pair_method_witness_summary.tsv", sep="\t", index=False)
    decision = status_payload(
        "E6",
        "COMPLETED",
        pair_count=int(selection.pair_id.nunique()),
        technology_count=int(selection.dataset.nunique()),
        directions_per_pair=2,
        method_count=len(CONFIG["methods"]["required"]),
        runs=len(statuses),
        numerical_failures=0,
    )
    write_json(OUTPUT / "E6_DECISION.json", decision)
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
