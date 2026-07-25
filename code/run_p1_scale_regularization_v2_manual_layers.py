"""Run the frozen spatialLIBD manual cortical-layer P1 v2 extension."""

from __future__ import annotations

import sys
from pathlib import Path

import run_p1_scale_regularization as engine


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "p1_scale_regularization_v2_manual_layers.yaml"
ANALYSIS = ROOT / "analysis" / "p1_scale_regularization_v2_manual_layers"
RESULTS = ROOT / "results" / "p1_scale_regularization_v2_manual_layers"
BUILD = ROOT / "build" / "p1_scale_regularization_v2_manual_layers"
EXPECTED_CONFIG_HASH = "3ce5b5e58766f41ce4b6e6d28ab2ffc1e8f9d4342217533b4cab3aade209786f"
ORIGINAL_COMPUTE_TASK = engine.compute_task


def configure_engine() -> None:
    engine.CONFIG_PATH = CONFIG_PATH
    engine.ANALYSIS = ANALYSIS
    engine.RESULTS = RESULTS
    engine.BUILD = BUILD
    engine.CHECKPOINTS = BUILD / "checkpoints"
    engine.LOGS = ANALYSIS / "logs"
    engine.EXPECTED_CONFIG_HASH = EXPECTED_CONFIG_HASH


def compute_task_manual_layers(task: dict, config: dict) -> dict:
    """Reapply extension globals inside each Windows-spawned worker."""
    configure_engine()
    return ORIGINAL_COMPUTE_TASK(task, config)


def append_freeze_materials() -> None:
    additions = [
        ROOT / "code" / "run_p1_scale_regularization_v2_manual_layers.py",
        ROOT / "code" / "build_p1_v2_manual_layer_config.py",
        ROOT / "code" / "prepare_p1_v2_dlpfc_manual_layers.py",
        ROOT / "code" / "export_p1_v2_dlpfc_manual_layers.R",
        ROOT / "code" / "inspect_p1_v2_dlpfc_manual_layers.R",
        ROOT / "docs" / "P1_AMENDMENT_004_MANUAL_LAYER_EXTENSION.md",
        ROOT / "docs" / "P1_IMPLEMENTATION_CORRECTION_001_GATE_SUMMARY_FIELD_MAPPING.md",
    ]
    manifest = ANALYSIS / "code_manifest.sha256"
    existing = manifest.read_text(encoding="utf-8") if manifest.is_file() else ""
    present = {line.split("  ", 1)[1] for line in existing.splitlines() if "  " in line}
    with manifest.open("a", encoding="utf-8") as handle:
        for path in additions:
            relative = path.relative_to(ROOT).as_posix()
            if relative not in present:
                handle.write(f"{engine.sha256(path)}  {relative}\n")
    amendment = ROOT / "docs" / "P1_AMENDMENT_004_MANUAL_LAYER_EXTENSION.md"
    snapshot = ANALYSIS / "amendment_004_snapshot.md"
    snapshot.write_text(amendment.read_text(encoding="utf-8"), encoding="utf-8")
    (ANALYSIS / "amendment_004_snapshot.sha256").write_text(
        f"{engine.sha256(amendment)}  {snapshot.name}\n", encoding="utf-8"
    )
    correction = ROOT / "docs" / "P1_IMPLEMENTATION_CORRECTION_001_GATE_SUMMARY_FIELD_MAPPING.md"
    correction_snapshot = ANALYSIS / "implementation_correction_001_snapshot.md"
    correction_snapshot.write_text(correction.read_text(encoding="utf-8"), encoding="utf-8")
    (ANALYSIS / "implementation_correction_001_snapshot.sha256").write_text(
        f"{engine.sha256(correction)}  {correction_snapshot.name}\n", encoding="utf-8"
    )


def main() -> int:
    configure_engine()
    engine.compute_task = compute_task_manual_layers
    preparing = "--prepare-freeze" in sys.argv
    status = engine.main()
    if status == 0 and preparing:
        append_freeze_materials()
    return status


if __name__ == "__main__":
    raise SystemExit(main())
