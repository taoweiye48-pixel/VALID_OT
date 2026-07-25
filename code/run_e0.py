from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import psutil

from validot.utils import environment_summary, file_hash, read_json, status_payload, write_json


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "00_protocol" / "frozen_config.json"


def command_output(command: list[str]) -> str:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"ERROR: {exc}"


def main() -> int:
    config = read_json(CONFIG_PATH)
    manifest = ROOT / "01_manifest"
    manifest.mkdir(parents=True, exist_ok=True)

    protocol_hash = file_hash(CONFIG_PATH)
    (ROOT / "00_protocol" / "PROTOCOL_HASH.txt").write_text(protocol_hash + "\n", encoding="utf-8")

    environment = environment_summary()
    environment.update(
        {
            "ram_total_gb": round(psutil.virtual_memory().total / 1024**3, 3),
            "disk_c_free_gb": round(psutil.disk_usage("C:/").free / 1024**3, 3),
            "disk_d_free_gb": round(psutil.disk_usage("D:/").free / 1024**3, 3),
            "nvidia_smi": command_output(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,driver_version",
                    "--format=csv,noheader",
                ]
            ),
        }
    )
    write_json(manifest / "environment_report.json", environment)
    (manifest / "environment_report.txt").write_text(
        json.dumps(environment, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    repo_rows: list[dict[str, str]] = []
    for name in ["paste", "paste2", "3d-OT"]:
        path = ROOT / "04_code" / "vendor" / name
        commit = command_output(["git", "-C", str(path), "rev-parse", "HEAD"])
        repo_rows.append({"name": name, "path": str(path), "commit": commit})
    with (manifest / "code_manifest.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["name", "path", "commit"], delimiter="\t")
        writer.writeheader()
        writer.writerows(repo_rows)

    archive = ROOT / "02_data_raw" / config["external_source"]["archive_name"]
    archive_info: dict[str, object] = {"path": str(archive), "exists": archive.exists()}
    data_ready = False
    if archive.exists():
        archive_info["bytes"] = archive.stat().st_size
        if archive.stat().st_size == config["external_source"]["expected_bytes"]:
            archive_info["md5"] = file_hash(archive, "md5")
            archive_info["md5_match"] = archive_info["md5"] == config["external_source"]["expected_md5"]
            data_ready = bool(archive_info["md5_match"])
        else:
            archive_info["md5_match"] = False
    write_json(manifest / "archive_status.json", archive_info)

    method_imports: dict[str, str] = {}
    for module in ["numpy", "scipy", "pandas", "ot", "anndata", "scanpy", "paste", "paste2"]:
        try:
            loaded = __import__(module)
            method_imports[module] = str(getattr(loaded, "__version__", "installed"))
        except Exception as exc:
            method_imports[module] = f"ERROR: {exc}"
    imports_ok = all(not value.startswith("ERROR") for value in method_imports.values())

    status = "READY_FOR_DATA_INSPECTION" if data_ready and imports_ok else "WAITING_FOR_VERIFIED_ARCHIVE"
    if not imports_ok:
        status = "BLOCKED_BY_ENVIRONMENT"
    decision = status_payload(
        "E0",
        status,
        protocol_hash=protocol_hash,
        archive=archive_info,
        required_imports=method_imports,
        repositories=repo_rows,
    )
    write_json(manifest / "E0_PRECHECK.json", decision)
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    return 0 if imports_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
