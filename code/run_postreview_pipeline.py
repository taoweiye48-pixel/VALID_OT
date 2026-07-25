"""Checkpointed sequential coordinator for the frozen WP1-WP10 run."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "analysis" / "postreview_wp1_wp10_v1"
RESULTS = ROOT / "results" / "postreview_wp1_wp10_v1"
FREEZE = ANALYSIS / "POSTREVIEW_WP2_WP10_FREEZE.json"
STATUS = ANALYSIS / "PIPELINE_STATUS.json"
LOGS = ANALYSIS / "logs"


def timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_status() -> dict[str, Any]:
    if STATUS.is_file():
        return json.loads(STATUS.read_text(encoding="utf-8"))
    return {"pipeline": "WP1-WP10", "created": timestamp(), "state": "INITIALIZING", "stages": {}}


def update(state: str, stage: str | None = None, **details: Any) -> None:
    payload = load_status()
    payload["state"] = state
    payload["updated"] = timestamp()
    if stage is not None:
        payload["current_stage"] = stage
        payload["stages"].setdefault(stage, {}).update(details)
    write_json(STATUS, payload)


def verify_freeze() -> None:
    manifest = json.loads(FREEZE.read_text(encoding="utf-8"))
    mismatches = []
    for item in manifest["files"]:
        path = ROOT / item["path"]
        observed = sha256(path) if path.is_file() else "MISSING"
        if observed.lower() != item["sha256"].lower():
            mismatches.append({"path": item["path"], "expected": item["sha256"], "observed": observed})
    if mismatches:
        raise RuntimeError(f"frozen-file mismatch: {mismatches}")


def wait_for_wp1() -> None:
    gate_path = RESULTS / "wp1" / "WP1_FULL_GATE.json"
    update("WAITING_FOR_WP1", "WP1", started=load_status()["stages"].get("WP1", {}).get("started", timestamp()))
    while not gate_path.is_file():
        time.sleep(30)
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    passed = bool(gate.get("gate_a_numerical_pass", False))
    update("WP1_COMPLETE" if passed else "STOPPED_GATE_A_WP1", "WP1", finished=timestamp(), gate=gate, passed=passed)
    if not passed:
        raise SystemExit(2)


def run_stage(stage: str, script: str, gate_path: Path, required_gate_key: str | None = None) -> None:
    verify_freeze()
    prior = load_status()["stages"].get(stage, {})
    if prior.get("passed") is True and gate_path.is_file():
        return
    stdout_path = LOGS / f"pipeline_{stage.lower()}_stdout.log"
    stderr_path = LOGS / f"pipeline_{stage.lower()}_stderr.log"
    update("RUNNING", stage, started=timestamp(), command=[sys.executable, str(ROOT / "code" / script), "--run"], stdout=str(stdout_path), stderr=str(stderr_path))
    with stdout_path.open("a", encoding="utf-8") as stdout, stderr_path.open("a", encoding="utf-8") as stderr:
        completed = subprocess.run([sys.executable, str(ROOT / "code" / script), "--run"], cwd=ROOT, stdout=stdout, stderr=stderr, check=False)
    gate = json.loads(gate_path.read_text(encoding="utf-8")) if gate_path.is_file() else {}
    passed = completed.returncode == 0
    if required_gate_key is not None:
        passed = passed and bool(gate.get(required_gate_key, False))
    update("STAGE_COMPLETE" if passed else "STOPPED_STAGE_FAILURE", stage, finished=timestamp(), returncode=completed.returncode, gate=gate, passed=passed)
    if not passed:
        raise SystemExit(completed.returncode or 2)


def main() -> int:
    LOGS.mkdir(parents=True, exist_ok=True)
    try:
        verify_freeze()
        wait_for_wp1()
        run_stage("WP2", "run_postreview_wp2.py", RESULTS / "wp2" / "WP2_GATE.json", "wp2_pass")
        run_stage("WP3", "run_postreview_wp3.py", RESULTS / "wp3" / "WP3_GATE_A_LOCAL_FIDELITY.json", "gate_a_pass")
        run_stage("WP4_WP6", "run_postreview_wp4_wp6.py", RESULTS / "wp4" / "WP4_WP6_GATE_B.json", "gate_b_computational_pass")
        run_stage("WP7", "run_postreview_wp7.py", RESULTS / "wp7" / "WP7_GATE_C_COORDINATE.json", "computational_pass")
        run_stage("WP8", "run_postreview_wp8.py", RESULTS / "wp8" / "WP8_GATE_C_GENE_SPLIT.json", "computational_pass")
        run_stage("WP5_WP9_WP10", "run_postreview_wp5_wp9_wp10.py", RESULTS / "wp10" / "WP5_WP9_GATE_D_AND_WP10_GATE_E_PRIMARY.json", "computational_pass")
        update("COMPLETE", "PIPELINE", finished=timestamp(), passed=True)
        return 0
    except SystemExit:
        raise
    except Exception as exc:
        update("STOPPED_EXCEPTION", "PIPELINE", finished=timestamp(), passed=False, error_type=type(exc).__name__, error=str(exc))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
