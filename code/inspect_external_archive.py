from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pandas as pd

from validot.utils import file_hash, read_json, status_payload, write_json


ROOT = Path(__file__).resolve().parents[1]
CONFIG = read_json(ROOT / "00_protocol" / "frozen_config.json")
ARCHIVE = ROOT / "02_data_raw" / CONFIG["external_source"]["archive_name"]
MANIFEST = ROOT / "01_manifest"


def classify(name: str) -> str:
    lower = name.lower()
    if lower.startswith("3dreconstruction/") and lower.endswith(".h5ad"):
        return "Stereo-seq"
    if "starmap" in lower:
        return "STARmap_PLUS"
    if "stereo" in lower or "e15.5" in lower or "e15_5" in lower:
        return "Stereo-seq"
    if "dlpfc" in lower or "15167" in lower:
        return "DLPFC"
    if "breast" in lower:
        return "breast_cancer"
    if "seqfish" in lower or "seq-fish" in lower:
        return "seqFISH"
    return "other"


def main() -> int:
    if not ARCHIVE.exists():
        raise FileNotFoundError(ARCHIVE)
    archive_bytes = ARCHIVE.stat().st_size
    if archive_bytes != CONFIG["external_source"]["expected_bytes"]:
        raise RuntimeError(f"archive bytes {archive_bytes} != expected")
    md5 = file_hash(ARCHIVE, "md5")
    if md5 != CONFIG["external_source"]["expected_md5"]:
        raise RuntimeError(f"archive md5 {md5} != expected")
    records = []
    with zipfile.ZipFile(ARCHIVE) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise RuntimeError(f"corrupt member: {bad}")
        for item in archive.infolist():
            suffix = Path(item.filename).suffix.lower()
            records.append(
                {
                    "member": item.filename,
                    "dataset_class": classify(item.filename),
                    "suffix": suffix,
                    "compressed_bytes": item.compress_size,
                    "uncompressed_bytes": item.file_size,
                    "crc": f"{item.CRC:08x}",
                    "is_dir": item.is_dir(),
                }
            )
    table = pd.DataFrame(records)
    table.to_csv(MANIFEST / "archive_members.tsv", sep="\t", index=False)
    summary = (
        table.loc[~table.is_dir]
        .groupby(["dataset_class", "suffix"], dropna=False)
        .agg(files=("member", "size"), uncompressed_bytes=("uncompressed_bytes", "sum"))
        .reset_index()
        .sort_values(["dataset_class", "uncompressed_bytes"], ascending=[True, False])
    )
    summary.to_csv(MANIFEST / "archive_member_summary.tsv", sep="\t", index=False)
    required = set(CONFIG["external_source"]["required_technologies"])
    present = set(table.loc[table.dataset_class.isin(required), "dataset_class"])
    decision = status_payload(
        "E0_ARCHIVE_INSPECTION",
        "COMPLETED" if required <= present else "BLOCKED_REQUIRED_DATASET_NAMES",
        archive_bytes=archive_bytes,
        archive_md5=md5,
        member_count=len(table),
        required_technologies=sorted(required),
        present_required_technologies=sorted(present),
        h5ad_members=table.loc[table.suffix == ".h5ad", ["member", "dataset_class", "uncompressed_bytes"]].to_dict(orient="records"),
    )
    write_json(MANIFEST / "archive_inspection_decision.json", decision)
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    return 0 if required <= present else 2


if __name__ == "__main__":
    raise SystemExit(main())
