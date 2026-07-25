from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "15_v1_3_correction" / "06_reproducibility"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def role(path: Path) -> str:
    relative = path.relative_to(ROOT).as_posix()
    if relative.startswith("00_protocol/"):
        return "protocol"
    if relative.startswith("04_code/"):
        return "code"
    if relative.startswith("03_data_processed/"):
        return "processed_input"
    if relative.startswith("15_v1_3_correction/"):
        return "v1.3_evidence"
    return "supporting"


def selected_files() -> list[Path]:
    roots = [
        ROOT / "00_protocol",
        ROOT / "01_manifest",
        ROOT / "03_data_processed" / "external_pairs",
        ROOT / "04_code",
        ROOT / "15_v1_3_correction",
    ]
    excluded_parts = {"__pycache__", ".pytest_cache"}
    excluded_names = {"V1_3_ARTIFACT_MANIFEST.tsv", "V1_3_VERIFICATION_REPORT.json"}
    files: list[Path] = [path for path in [ROOT / "pytest.ini"] if path.exists()]
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.name in excluded_names:
                continue
            if any(part in excluded_parts for part in path.parts):
                continue
            files.append(path)
    return sorted(set(files), key=lambda item: item.relative_to(ROOT).as_posix())


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in selected_files():
        stat = path.stat()
        rows.append(
            {
                "relative_path": path.relative_to(ROOT).as_posix(),
                "role": role(path),
                "bytes": stat.st_size,
                "sha256": sha256(path),
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(OUTPUT / "V1_3_ARTIFACT_MANIFEST.tsv", sep="\t", index=False)
    print(
        f"Manifested {len(table)} files ({table.bytes.sum() / 2**30:.3f} GiB) to "
        f"{OUTPUT / 'V1_3_ARTIFACT_MANIFEST.tsv'}"
    )


if __name__ == "__main__":
    main()
