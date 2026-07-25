from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from run_external_benchmark import PAIR_SELECTION, PROCESSED, prepare_pair
from validot.utils import file_hash, status_payload, write_json


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    selection = pd.read_csv(PAIR_SELECTION, sep="\t")
    records = []
    for row in selection.to_dict(orient="records"):
        path = prepare_pair(row)
        records.append(
            {
                "dataset": row["dataset"],
                "pair_id": row["pair_id"],
                "pair_type": row["pair_type"],
                "processed_path": str(path),
                "bytes": path.stat().st_size,
                "sha256": file_hash(path),
            }
        )
    table = pd.DataFrame(records)
    table.to_csv(ROOT / "03_data_processed" / "processed_pair_manifest.tsv", sep="\t", index=False)
    decision = status_payload(
        "E0_PAIR_PREPARATION",
        "COMPLETED",
        pairs=len(table),
        total_bytes=int(table.bytes.sum()),
    )
    write_json(ROOT / "03_data_processed" / "PAIR_PREPARATION_DECISION.json", decision)
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
