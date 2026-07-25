from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CORRECTION = ROOT / "15_v1_3_correction"
OUTPUT = CORRECTION / "06_reproducibility"
CHECKS: list[dict[str, object]] = []


def check(name: str, condition: bool, detail: str) -> None:
    CHECKS.append({"check": name, "pass": bool(condition), "detail": detail})


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def close(value: float, expected: float, tolerance: float = 1e-6) -> bool:
    return math.isfinite(value) and abs(value - expected) <= tolerance


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)

    config = load_json(ROOT / "00_protocol" / "frozen_config.json")
    check("protocol_version", config["protocol_id"].startswith("valid-ot-v1.3"), config["protocol_id"])
    check(
        "v1.2_snapshot",
        (CORRECTION / "00_v1_2_snapshot" / "frozen_config_v1.2.json").exists(),
        "immutable pre-correction configuration retained",
    )

    semi = load_json(CORRECTION / "01_semisynthetic_rerun" / "V1_3_SEMISYNTHETIC_DECISION.json")
    real = load_json(CORRECTION / "02_real_reanalysis" / "V1_3_REAL_REANALYSIS_DECISION.json")
    stats = load_json(CORRECTION / "03_statistics" / "V1_3_STATISTICS_DECISION.json")
    check("semisynthetic_complete", semi.get("status") == "COMPLETED", json.dumps(semi, sort_keys=True))
    check("real_complete", real.get("status") == "COMPLETED", json.dumps(real, sort_keys=True))
    check("statistics_complete", stats.get("status") == "COMPLETED", json.dumps(stats, sort_keys=True))
    semi_statuses = list((CORRECTION / "01_semisynthetic_rerun" / "checkpoints").rglob("status.json"))
    completed_statuses = sum(load_json(path).get("status") == "COMPLETED" for path in semi_statuses)
    check("semisynthetic_task_count", semi.get("tasks") == 720, f"tasks={semi.get('tasks')}")
    check(
        "semisynthetic_failures",
        len(semi_statuses) == 720 and completed_statuses == 720,
        f"status_files={len(semi_statuses)} completed={completed_statuses}",
    )
    check("real_task_count", real.get("method_tasks") == 42, f"method_tasks={real.get('method_tasks')}")

    fidelity = pd.read_csv(
        CORRECTION / "03_statistics" / "fidelity_gate_pair_level_corrected.tsv", sep="\t"
    )
    fd = fidelity[
        (fidelity.source == "real")
        & (fidelity.proxy == "finite_difference_sensitivity_h001")
    ]
    expected = {
        ("balanced_ot", "I_EXPR"): (0.997363, 0.923333, 0.159301, True),
        ("uot", "I_EXPR"): (0.997103, 0.930000, 0.135084, True),
        ("row_softmax", "I_EXPR"): (0.947350, 0.850000, 1.146828, False),
    }
    for key, values in expected.items():
        row = fd[(fd.method == key[0]) & (fd.intervention == key[1])].iloc[0]
        numerical = all(
            close(float(row[column]), target, 2e-6)
            for column, target in zip(
                ["median_spearman", "median_top_decile_precision", "median_normalized_mae"],
                values[:3],
            )
        )
        decision = bool(row.full_fidelity_pass) == values[3]
        check(f"fidelity_{key[0]}_{key[1]}", numerical and decision, row.to_json())
    spatial_passes = int(fd[fd.intervention == "I_SPATIAL"].full_fidelity_pass.sum())
    check("spatial_fd_gate_passes", spatial_passes == 0, f"passes={spatial_passes}")

    registered = pd.read_csv(
        CORRECTION / "03_statistics" / "registered_v1_2_gate_immutable.tsv", sep="\t"
    )
    registered_passes = int(registered.registered_real_external_gate.fillna(False).sum())
    check("registered_v1.2_gate_immutable", registered_passes == 0, f"passes={registered_passes}")

    external = pd.read_csv(
        CORRECTION / "03_statistics" / "real_external_gate_with_controls.tsv", sep="\t"
    )
    passed = external[external.corrected_external_gate_with_controls.fillna(False)]
    passed_keys = set(zip(passed.method, passed.witness, passed.score))
    expected_keys = {
        ("balanced_ot", "heldout_loss", "finite_difference_I_EXPR_h001"),
        ("uot", "heldout_loss", "exact_I_EXPR"),
        ("uot", "heldout_loss", "finite_difference_I_EXPR_h001"),
    }
    check("corrective_real_gate_with_controls", passed_keys == expected_keys, repr(sorted(passed_keys)))
    exact_combined = external[external.score == "exact_combined"]
    check(
        "exact_combined_gate_with_controls",
        int(exact_combined.corrected_external_gate_with_controls.fillna(False).sum()) == 0,
        "passes=0 expected",
    )

    p1 = [
        CORRECTION / "04_p1_sensitivity" / "endpoint_step" / "ENDPOINT_SENSITIVITY_DECISION.json",
        CORRECTION / "04_p1_sensitivity" / "label_agnostic_sampling" / "LABEL_AGNOSTIC_DECISION.json",
        CORRECTION / "04_p1_sensitivity" / "coordinate_frame" / "COORDINATE_SENSITIVITY_DECISION.json",
        CORRECTION / "04_p1_sensitivity" / "frozen_coupling_baseline" / "FROZEN_COUPLING_BASELINE_DECISION.json",
    ]
    for path in p1:
        payload = load_json(path)
        check(f"p1_{path.parent.name}", payload.get("status") == "COMPLETED", json.dumps(payload, sort_keys=True))

    followup = load_json(CORRECTION / "08_review_followup" / "REVIEW_FOLLOWUP_DECISION.json")
    check(
        "review_followup",
        followup.get("status") == "COMPLETED"
        and followup.get("exact_conditions_median_worse_than_random", 0) > 0,
        json.dumps(followup, sort_keys=True),
    )

    figures = [f"Fig{i}_" for i in range(1, 7)]
    names = [path.name for path in (CORRECTION / "05_figures").glob("*.png")]
    for prefix in figures:
        check(f"figure_{prefix[:-1]}", any(name.startswith(prefix) for name in names), repr(names))

    failed = [item for item in CHECKS if not item["pass"]]
    report = {
        "stage": "V1_3_VERIFICATION",
        "status": "COMPLETED" if not failed else "FAILED",
        "checks": len(CHECKS),
        "failed_checks": len(failed),
        "details": CHECKS,
    }
    with (OUTPUT / "V1_3_VERIFICATION_REPORT.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    table = pd.DataFrame(CHECKS)
    table.to_csv(OUTPUT / "V1_3_VERIFICATION_CHECKS.tsv", sep="\t", index=False)
    print(json.dumps({key: report[key] for key in ["status", "checks", "failed_checks"]}, indent=2))
    if failed:
        raise SystemExit("v1.3 verification failed: " + ", ".join(item["check"] for item in failed))


if __name__ == "__main__":
    main()
