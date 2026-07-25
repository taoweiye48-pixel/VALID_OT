from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .semisynthetic import PairedData


def save_pair(path: Path, pair: PairedData, extras: dict[str, Any] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "dataset": np.asarray(pair.dataset),
        "pair_id": np.asarray(pair.pair_id),
        "source_x": pair.source_x,
        "target_x": pair.target_x,
        "source_xy": pair.source_xy,
        "target_xy": pair.target_xy,
        "source_labels": pair.source_labels.astype(str),
        "target_labels": pair.target_labels.astype(str),
        "truth_target": pair.truth_target,
        "truth_missing": pair.truth_missing,
        "equivalent_targets_json": np.asarray(json.dumps(pair.equivalent_targets)),
        "metadata_json": np.asarray(json.dumps(pair.metadata)),
    }
    if extras:
        for key, value in extras.items():
            array = np.asarray(value)
            payload[key] = array.astype(str) if array.dtype == object else array
    np.savez_compressed(path, **payload)


def load_pair(path: Path) -> tuple[PairedData, dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as data:
        keys = set(data.files)
        core = {
            "dataset",
            "pair_id",
            "source_x",
            "target_x",
            "source_xy",
            "target_xy",
            "source_labels",
            "target_labels",
            "truth_target",
            "truth_missing",
            "equivalent_targets_json",
            "metadata_json",
        }
        pair = PairedData(
            dataset=str(data["dataset"].item()),
            pair_id=str(data["pair_id"].item()),
            source_x=data["source_x"].copy(),
            target_x=data["target_x"].copy(),
            source_xy=data["source_xy"].copy(),
            target_xy=data["target_xy"].copy(),
            source_labels=data["source_labels"].astype(str),
            target_labels=data["target_labels"].astype(str),
            truth_target=data["truth_target"].copy(),
            truth_missing=data["truth_missing"].astype(bool),
            equivalent_targets=json.loads(str(data["equivalent_targets_json"].item())),
            metadata=json.loads(str(data["metadata_json"].item())),
        )
        extras = {key: data[key].copy() for key in keys - core}
    return pair, extras
