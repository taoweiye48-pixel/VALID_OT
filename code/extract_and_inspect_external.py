from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path
from typing import Any

import anndata as ad
import pandas as pd

from inspect_external_archive import classify
from validot.utils import file_hash, read_json, status_payload, write_json


ROOT = Path(__file__).resolve().parents[1]
CONFIG = read_json(ROOT / "00_protocol" / "frozen_config.json")
ARCHIVE = ROOT / "02_data_raw" / CONFIG["external_source"]["archive_name"]
DESTINATION = ROOT / "02_data_raw" / "external_selected"
MANIFEST = ROOT / "01_manifest"


LABEL_KEYWORDS = [
    "annotation",
    "cell_type",
    "celltype",
    "region",
    "domain",
    "label",
    "cluster",
    "tissue",
    "organ",
]


def safe_destination(member: str) -> Path:
    target = (DESTINATION / member).resolve()
    root = DESTINATION.resolve()
    if root != target and root not in target.parents:
        raise RuntimeError(f"unsafe archive member: {member}")
    return target


def label_candidate_summary(frame: pd.DataFrame) -> list[dict[str, Any]]:
    records = []
    for column in frame.columns:
        series = frame[column]
        non_missing = series.dropna()
        unique = int(non_missing.astype(str).nunique())
        keyword_score = max(
            [len(LABEL_KEYWORDS) - index for index, word in enumerate(LABEL_KEYWORDS) if word in str(column).lower()]
            or [0]
        )
        candidate = bool(2 <= unique <= 100 and keyword_score > 0)
        records.append(
            {
                "column": str(column),
                "dtype": str(series.dtype),
                "non_missing": int(non_missing.size),
                "unique": unique,
                "keyword_score": keyword_score,
                "candidate": candidate,
                "example_values": "|".join(non_missing.astype(str).drop_duplicates().head(12).tolist()),
            }
        )
    return sorted(records, key=lambda row: (row["candidate"], row["keyword_score"], row["non_missing"]), reverse=True)


def main() -> int:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    required = set(CONFIG["external_source"]["required_technologies"])
    extracted = []
    with zipfile.ZipFile(ARCHIVE) as archive:
        members = [
            item
            for item in archive.infolist()
            if not item.is_dir() and classify(item.filename) in required
        ]
        if not members:
            raise RuntimeError("no required technology members found")
        for item in members:
            target = safe_destination(item.filename)
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists() or target.stat().st_size != item.file_size:
                with archive.open(item) as source, target.open("wb") as destination:
                    shutil.copyfileobj(source, destination, length=8 * 1024 * 1024)
            extracted.append(
                {
                    "member": item.filename,
                    "dataset_class": classify(item.filename),
                    "path": str(target),
                    "bytes": target.stat().st_size,
                    "sha256": file_hash(target),
                }
            )
    pd.DataFrame(extracted).to_csv(MANIFEST / "extracted_external_manifest.tsv", sep="\t", index=False)

    h5ad_records = []
    column_records = []
    for item in extracted:
        path = Path(item["path"])
        if path.suffix.lower() != ".h5ad":
            continue
        try:
            data = ad.read_h5ad(path, backed="r")
            candidates = label_candidate_summary(data.obs)
            best = next((candidate for candidate in candidates if candidate["candidate"]), None)
            h5ad_records.append(
                {
                    "dataset_class": item["dataset_class"],
                    "path": str(path),
                    "file_name": path.name,
                    "parent": str(path.parent),
                    "n_obs": int(data.n_obs),
                    "n_vars": int(data.n_vars),
                    "obsm_keys": "|".join(map(str, data.obsm.keys())),
                    "obs_columns": "|".join(map(str, data.obs.columns)),
                    "best_label_candidate": best["column"] if best else "",
                    "best_label_unique": best["unique"] if best else 0,
                    "best_label_non_missing": best["non_missing"] if best else 0,
                    "inspection_error": "",
                }
            )
            for candidate in candidates:
                column_records.append({"path": str(path), **candidate})
            if getattr(data, "file", None) is not None:
                data.file.close()
        except Exception as exc:
            h5ad_records.append(
                {
                    "dataset_class": item["dataset_class"],
                    "path": str(path),
                    "file_name": path.name,
                    "parent": str(path.parent),
                    "n_obs": 0,
                    "n_vars": 0,
                    "obsm_keys": "",
                    "obs_columns": "",
                    "best_label_candidate": "",
                    "best_label_unique": 0,
                    "best_label_non_missing": 0,
                    "inspection_error": f"{type(exc).__name__}: {exc}",
                }
            )
    h5ad_table = pd.DataFrame(h5ad_records)
    h5ad_table.to_csv(MANIFEST / "external_h5ad_metadata.tsv", sep="\t", index=False)
    pd.DataFrame(column_records).to_csv(MANIFEST / "external_obs_columns.tsv", sep="\t", index=False)
    counts = h5ad_table.groupby("dataset_class").size().to_dict() if len(h5ad_table) else {}
    decision = status_payload(
        "E0_EXTERNAL_METADATA",
        "COMPLETED" if required <= set(counts) else "BLOCKED_REQUIRED_H5AD",
        extracted_files=len(extracted),
        h5ad_files=len(h5ad_table),
        h5ad_by_technology=counts,
        inspection_errors=h5ad_table.loc[h5ad_table.inspection_error != ""].to_dict(orient="records") if len(h5ad_table) else [],
    )
    write_json(MANIFEST / "external_metadata_decision.json", decision)
    print(json.dumps(decision, ensure_ascii=False, indent=2, default=str))
    return 0 if required <= set(counts) else 2


if __name__ == "__main__":
    raise SystemExit(main())
