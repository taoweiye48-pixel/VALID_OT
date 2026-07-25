from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def file_hash(path: Path, algorithm: str = "sha256", block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def environment_summary() -> dict[str, Any]:
    try:
        pip_freeze = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"], text=True, stderr=subprocess.STDOUT
        ).splitlines()
    except Exception as exc:  # pragma: no cover - diagnostic fallback
        pip_freeze = [f"ERROR: {exc}"]
    return {
        "created_utc": utc_now(),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "packages": pip_freeze,
    }


def status_payload(stage: str, status: str, **kwargs: Any) -> dict[str, Any]:
    return {"stage": stage, "status": status, "updated_utc": utc_now(), **kwargs}
