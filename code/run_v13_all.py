from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "04_code"

STAGES = {
    "semisynthetic": "run_v13_semisynthetic.py",
    "real": "run_v13_real_reanalysis.py",
    "statistics": "run_v13_statistics.py",
    "endpoint": "run_v13_endpoint_sensitivity.py",
    "label": "run_v13_label_agnostic_sensitivity.py",
    "coordinate": "run_v13_coordinate_sensitivity.py",
    "frozen": "run_v13_frozen_coupling_baseline.py",
    "figures": "run_v13_figures.py",
    "review": "run_v13_review_analyses.py",
    "manifest": "build_v13_manifest.py",
    "verify": "verify_v13.py",
}
EXPENSIVE = {"semisynthetic", "real", "endpoint", "label", "coordinate", "frozen"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resume or verify the corrected VALID-OT v1.3 workflow."
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=list(STAGES),
        default=["verify"],
        help="Ordered stages to execute; default is the non-computational verifier.",
    )
    parser.add_argument(
        "--allow-expensive",
        action="store_true",
        help="Required before launching solver-heavy stages; checkpoints are still reused.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    blocked = [stage for stage in args.stages if stage in EXPENSIVE]
    if blocked and not args.allow_expensive:
        raise SystemExit(
            "Refusing solver-heavy stages without --allow-expensive: " + ", ".join(blocked)
        )
    for stage in args.stages:
        script = CODE / STAGES[stage]
        print(f"[VALID-OT v1.3] stage={stage} script={script.name}", flush=True)
        subprocess.run([sys.executable, str(script)], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
