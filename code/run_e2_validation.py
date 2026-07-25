from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from validot.semisynthetic import generate_pair, validate_truth
from validot.utils import status_payload, write_json


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "06_E2_truth_validation"


def base_data(n: int = 1000, d: int = 48, seed: int = 20260716):
    rng = np.random.default_rng(seed)
    labels = np.asarray([f"R{i % 7}" for i in range(n)])
    centers = rng.normal(size=(7, d))
    x = centers[np.arange(n) % 7] + 0.15 * rng.normal(size=(n, d))
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    radii = 1.0 + 0.15 * np.sin(7 * angles)
    xy = np.column_stack([radii * np.cos(angles), 0.7 * radii * np.sin(angles)])
    return x, xy, labels


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    x, xy, labels = base_data()
    scenarios = ["rigid", "nonrigid", "crop_missing", "duplicate_motif", "batch_noise", "combined"]
    records = []
    for scenario in scenarios:
        for seed in [0, 4, 9]:
            pair = generate_pair(x, xy, labels, scenario, seed, n=800, dataset="E2_toy")
            record = validate_truth(pair)
            record.update(pair.metadata)
            if scenario == "duplicate_motif":
                duplicated = pair.metadata["duplicate_old_indices"]
                start = 800
                symmetry_x = max(
                    float(np.max(np.abs(pair.target_x[old] - pair.target_x[start + offset])))
                    for offset, old in enumerate(duplicated)
                )
                symmetry_xy = max(
                    float(np.max(np.abs(pair.target_xy[old] - pair.target_xy[start + offset])))
                    for offset, old in enumerate(duplicated)
                )
                record["duplicate_expression_max_abs"] = symmetry_x
                record["duplicate_spatial_max_abs"] = symmetry_xy
                record["passed"] = bool(record["passed"] and symmetry_x <= 1e-12 and symmetry_xy <= 1e-12)
            records.append(record)
    table = pd.DataFrame(records)
    table.to_csv(OUTPUT / "truth_validation.tsv", sep="\t", index=False)
    all_pass = bool(table.passed.all())
    decision = status_payload(
        "E2",
        "COMPLETED_GO" if all_pass else "FAILED_TRUTH",
        all_truth_checks_passed=all_pass,
        records=len(table),
        failed=table.loc[~table.passed].to_dict(orient="records"),
    )
    write_json(OUTPUT / "E2_DECISION.json", decision)
    print(json.dumps(decision, ensure_ascii=False, indent=2, default=str))
    return 0 if all_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
