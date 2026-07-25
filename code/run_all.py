from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".runtime" / "Python310" / "python.exe"


STAGES = [
    ("E0", "run_e0.py", ROOT / "01_manifest" / "E0_DECISION.json"),
    ("E1", "run_e1.py", ROOT / "05_E1_solver_validation" / "E1_DECISION.json"),
    ("E2", "run_e2_validation.py", ROOT / "06_E2_truth_validation" / "E2_DECISION.json"),
    ("E3", "run_e3_retrospective.py", ROOT / "07_E3_DLPFC_retrospective" / "E3_DECISION.json"),
    ("PAIR_PREPARATION", "prepare_external_pairs.py", ROOT / "03_data_processed" / "PAIR_PREPARATION_DECISION.json"),
    ("E4_E5", "run_semisynthetic_benchmark.py", ROOT / "08_E4_internal_fidelity" / "E4_E5_DECISION.json"),
    (
        "E5_DIAGNOSTICS",
        "run_semisynthetic_diagnostics.py",
        ROOT / "09_E5_semisynthetic_external" / "diagnostics" / "E5_DIAGNOSTICS_DECISION.json",
    ),
    ("E6", "run_external_benchmark.py", ROOT / "10_E6_real_external" / "E6_DECISION.json"),
    ("E7", "run_robustness.py", ROOT / "11_E7_robustness" / "E7_DECISION.json"),
    ("E8", "run_statistics.py", ROOT / "12_E8_statistics" / "E8_DECISION.json"),
    ("E9_FIGURES", "run_figures.py", ROOT / "13_figures" / "FIGURE_DECISION.json"),
    ("E9_REPORTS", "run_reports.py", ROOT / "14_reports" / "E9_DECISION.json"),
]


def completed(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("status") in {"COMPLETED", "COMPLETED_GO"}
    except Exception:
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the frozen VALID-OT benchmark with checkpoint resume.")
    parser.add_argument("--config", default=str(ROOT / "00_protocol" / "frozen_config.json"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if Path(args.config).resolve() != (ROOT / "00_protocol" / "frozen_config.json").resolve():
        raise ValueError("Only the frozen project config is accepted by this reproducibility entry point")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "04_code")
    for stage, script, decision in STAGES:
        if args.resume and completed(decision):
            print(f"SKIP {stage}: completed checkpoint", flush=True)
            continue
        print(f"RUN {stage}: {script}", flush=True)
        result = subprocess.run([str(PYTHON), str(ROOT / "04_code" / script)], cwd=ROOT, env=env)
        if result.returncode != 0:
            print(f"STOP {stage}: exit code {result.returncode}", file=sys.stderr, flush=True)
            return result.returncode
    print("VALID-OT pipeline completed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
