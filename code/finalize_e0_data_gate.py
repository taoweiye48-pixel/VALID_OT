from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import pandas as pd

from validot.utils import file_hash, read_json, status_payload, write_json


ROOT = Path(__file__).resolve().parents[1]
CONFIG = read_json(ROOT / "00_protocol" / "frozen_config.json")
SELECTION_PATH = ROOT / "01_manifest" / "pair_selection.tsv"


def clean_labels(values):
    labels = values.astype(str)
    return labels[~labels.isin(["nan", "None", "NA", "Unknown", "UNKNOWN", "Unlabel"])]


def main() -> int:
    selection = pd.read_csv(SELECTION_PATH, sep="\t")
    records = []
    hash_cache = {}
    def cached_hash(path: Path) -> str:
        key = str(path)
        if key not in hash_cache:
            hash_cache[key] = file_hash(path)
        return hash_cache[key]
    for row in selection.to_dict(orient="records"):
        source_path = Path(row["source_path"])
        target_path = Path(row["target_path"])
        source = ad.read_h5ad(source_path, backed="r")
        target = ad.read_h5ad(target_path, backed="r")
        source_labels = clean_labels(source.obs[row["source_label_column"]])
        target_labels = clean_labels(target.obs[row["target_label_column"]])
        common_labels = sorted(set(source_labels) & set(target_labels))
        record = {
            **row,
            "source_n": int(source.n_obs),
            "target_n": int(target.n_obs),
            "source_labeled_n": int(len(source_labels)),
            "target_labeled_n": int(len(target_labels)),
            "common_label_count": len(common_labels),
            "common_labels": "|".join(common_labels),
            "source_sha256": cached_hash(source_path),
            "target_sha256": cached_hash(target_path),
        }
        record["passed"] = bool(
            record["source_labeled_n"] >= CONFIG["external_source"]["minimum_units_per_slice"]
            and record["target_labeled_n"] >= CONFIG["external_source"]["minimum_units_per_slice"]
            and record["common_label_count"] >= CONFIG["external_source"]["minimum_common_labels"]
        )
        records.append(record)
        source.file.close()
        target.file.close()
    table = pd.DataFrame(records)
    table.to_csv(ROOT / "01_manifest" / "pair_selection_qc.tsv", sep="\t", index=False)
    required = set(CONFIG["external_source"]["required_technologies"])
    all_pass = bool(
        table.passed.all()
        and table.pair_id.nunique() >= CONFIG["external_source"]["minimum_real_pairs_total"]
        and required <= set(table.dataset)
    )
    decision = status_payload(
        "E0",
        "COMPLETED_GO" if all_pass else "BLOCKED_BY_DATA_GATE",
        protocol_hash=file_hash(ROOT / "00_protocol" / "frozen_config.json"),
        selection_sha256=file_hash(SELECTION_PATH),
        pair_count=int(table.pair_id.nunique()),
        technology_count=int(table.dataset.nunique()),
        pair_types=table.groupby("pair_type").size().to_dict(),
        failed_pairs=table.loc[~table.passed].to_dict(orient="records"),
        limitations=[
            "Five Stereo-seq pairs are adjacent developmental stages, not same-age serial sections.",
            "Public labels are external witnesses, not pointwise correspondence truth.",
        ],
    )
    write_json(ROOT / "01_manifest" / "E0_DECISION.json", decision)
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    return 0 if all_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
