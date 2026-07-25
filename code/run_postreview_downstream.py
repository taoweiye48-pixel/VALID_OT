"""Run the frozen VALID-OT downstream positive-pair and representation tests.

This runner is additive: it reads frozen WP10 outputs and processed HER2ST
pairs, but never changes P0/P1/WP1-WP11 artifacts.  Two analyses are kept
separate:

1. Positive-pair purity at fixed coverage, derived from frozen WP10 risk-
   coverage endpoints.
2. A controlled positive-pair-conditioned representation-transfer stress
   test.  It is deliberately not described as a full AlignDG reproduction.

Patients are the independent units. Directions and spatial folds are repeated
measurements and are aggregated before patient-level inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge

from validot.io import load_pair
from validot.p1 import P1Parameters, mixed_cost, solve_p1
from validot.solvers import cost_components


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = ROOT / "configs" / "postreview_downstream_v1.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(*parts: object) -> int:
    payload = "|".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**32 - 1)


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def dump_tsv(path: Path, table: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    table.to_csv(temporary, sep="\t", index=False)
    temporary.replace(path)


def dataframe_to_markdown(table: pd.DataFrame) -> str:
    """Render a compact Markdown table without pandas' optional tabulate dependency."""
    if table.empty:
        return "_No rows._"
    displayed = table.copy()
    for column in displayed.columns:
        displayed[column] = displayed[column].map(
            lambda value: ""
            if pd.isna(value)
            else (f"{value:.6g}" if isinstance(value, (float, np.floating)) else str(value))
        )
    headers = [str(column).replace("|", "\\|") for column in displayed.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for values in displayed.itertuples(index=False, name=None):
        cells = [str(value).replace("|", "\\|").replace("\n", " ") for value in values]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def normalize_rows(x: np.ndarray) -> np.ndarray:
    values = np.asarray(x, dtype=np.float64)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def holm_adjust(p_values: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(p_values), dtype=float)
    result = np.full_like(values, np.nan)
    finite = np.flatnonzero(np.isfinite(values))
    if not len(finite):
        return result
    ordered = finite[np.argsort(values[finite], kind="mergesort")]
    running = 0.0
    total = len(ordered)
    for rank, index in enumerate(ordered):
        adjusted = min(1.0, (total - rank) * values[index])
        running = max(running, adjusted)
        result[index] = running
    return result


def bootstrap_median_interval(values: np.ndarray, replicates: int, seed: int) -> tuple[float, float]:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if not len(clean):
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    sampled = clean[rng.integers(0, len(clean), size=(replicates, len(clean)))]
    medians = np.median(sampled, axis=1)
    lower, upper = np.quantile(medians, [0.025, 0.975])
    return float(lower), float(upper)


def exact_paired_p(values: np.ndarray) -> float:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    nonzero = clean[clean != 0]
    if len(nonzero) < 2:
        return float("nan")
    # Standard Wilcoxon handling discards exact zero differences. Passing the
    # reduced vector explicitly preserves exact enumeration in SciPy instead
    # of silently switching to a small-sample normal approximation.
    return float(
        wilcoxon(
            nonzero,
            alternative="two-sided",
            zero_method="wilcox",
            method="exact",
        ).pvalue
    )


def summarize_effects(
    table: pd.DataFrame,
    group_column: str,
    effect_column: str,
    config: dict[str, Any],
    family_column: str | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    replicates = int(config["statistics"]["bootstrap_replicates"])
    base_seed = int(config["statistics"]["bootstrap_seed"])
    groupers = [group_column] if family_column is None else [family_column, group_column]
    for keys, frame in table.groupby(groupers, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        values = frame[effect_column].to_numpy(float)
        q25, q75 = np.quantile(values[np.isfinite(values)], [0.25, 0.75])
        low, high = bootstrap_median_interval(values, replicates, stable_seed(base_seed, *keys, effect_column))
        row = {
            group_column: keys[-1],
            "n_independent_units": int(np.isfinite(values).sum()),
            "median_difference": float(np.nanmedian(values)),
            "q25_difference": float(q25),
            "q75_difference": float(q75),
            "bootstrap_median_ci95_low": low,
            "bootstrap_median_ci95_high": high,
            "units_improved": int(np.sum(values > 0)),
            "units_worsened": int(np.sum(values < 0)),
            "units_tied": int(np.sum(values == 0)),
            "wilcoxon_two_sided_p_raw": exact_paired_p(values),
        }
        if family_column is not None:
            row[family_column] = keys[0]
        rows.append(row)
    summary = pd.DataFrame(rows)
    if family_column is None:
        summary["wilcoxon_p_holm"] = holm_adjust(summary["wilcoxon_two_sided_p_raw"])
    else:
        summary["wilcoxon_p_holm"] = np.nan
        for _, indices in summary.groupby(family_column, sort=True).groups.items():
            summary.loc[indices, "wilcoxon_p_holm"] = holm_adjust(
                summary.loc[indices, "wilcoxon_two_sided_p_raw"]
            )
    return summary


def run_positive_pair_quality(config: dict[str, Any], results: Path) -> dict[str, Any]:
    source = ROOT / config["source_results"]
    raw = pd.read_csv(source, sep="\t")
    selected = raw[
        (raw["metric"] == "top1_error_detection")
        & raw["score"].isin(config["positive_pair_quality"]["comparators"])
    ].copy()
    keys = [
        "dataset",
        "biological_pair_id",
        "independent_unit_id",
        "pair_id",
        "direction",
        "method",
    ]
    metrics = [
        "retained_loss_at_80pct_coverage",
        "retained_loss_at_90pct_coverage",
    ]
    maximum = selected[selected["score"] == "low_max_probability"][keys + metrics].rename(
        columns={metric: f"max_probability__{metric}" for metric in metrics}
    )
    margin = selected[selected["score"] == "probability_margin_risk"][keys + metrics].rename(
        columns={metric: f"top2_margin__{metric}" for metric in metrics}
    )
    direction = maximum.merge(margin, on=keys, how="inner", validate="one_to_one")
    for coverage in (80, 90):
        suffix = f"retained_loss_at_{coverage}pct_coverage"
        direction[f"max_probability_precision_at_{coverage}pct"] = 1.0 - direction[f"max_probability__{suffix}"]
        direction[f"top2_margin_precision_at_{coverage}pct"] = 1.0 - direction[f"top2_margin__{suffix}"]
        direction[f"margin_minus_max_precision_at_{coverage}pct"] = (
            direction[f"top2_margin_precision_at_{coverage}pct"]
            - direction[f"max_probability_precision_at_{coverage}pct"]
        )
    dump_tsv(results / "positive_pair_quality_direction.tsv", direction)

    value_columns = [column for column in direction.columns if "precision_at_" in column]
    unit = (
        direction.groupby(["independent_unit_id", "method"], as_index=False)[value_columns]
        .mean()
        .sort_values(["method", "independent_unit_id"])
    )
    dump_tsv(results / "positive_pair_quality_unit.tsv", unit)

    summaries: list[pd.DataFrame] = []
    for coverage in (80, 90):
        effect = f"margin_minus_max_precision_at_{coverage}pct"
        summary = summarize_effects(unit, "method", effect, config)
        summary.insert(0, "coverage", coverage / 100.0)
        for prefix in ("max_probability", "top2_margin"):
            column = f"{prefix}_precision_at_{coverage}pct"
            medians = unit.groupby("method")[column].median()
            q25 = unit.groupby("method")[column].quantile(0.25)
            q75 = unit.groupby("method")[column].quantile(0.75)
            summary[f"{prefix}_median_precision"] = summary["method"].map(medians)
            summary[f"{prefix}_q25_precision"] = summary["method"].map(q25)
            summary[f"{prefix}_q75_precision"] = summary["method"].map(q75)
        summaries.append(summary)
    combined = pd.concat(summaries, ignore_index=True)
    dump_tsv(results / "positive_pair_quality_summary.tsv", combined)
    return {
        "source": str(source),
        "source_sha256": sha256(source),
        "n_direction_method_rows": int(len(direction)),
        "n_independent_units": int(unit["independent_unit_id"].nunique()),
        "n_methods": int(unit["method"].nunique()),
    }


def method_parameters(method: str, config: dict[str, Any]) -> P1Parameters:
    values = config["methods"][method]
    return P1Parameters(
        method=method,
        epsilon=float(values["epsilon"]),
        tau=values.get("tau"),
        max_iter=int(config["solver"]["max_iter"]),
        tolerance=float(config["solver"]["tolerance"]),
    )


def plan_scores(
    pair_path: Path,
    method: str,
    config: dict[str, Any],
    checkpoint_root: Path,
    config_hash: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    identifier = f"{pair_path.stem}__{method}"
    array_path = checkpoint_root / f"{identifier}.npz"
    metadata_path = checkpoint_root / f"{identifier}.json"
    if array_path.is_file() and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("status") == "COMPLETED" and metadata.get("config_sha256") == config_hash:
            with np.load(array_path, allow_pickle=False) as stored:
                arrays = {key: stored[key].copy() for key in stored.files}
            return arrays, metadata

    pair, _ = load_pair(pair_path)
    components = cost_components(pair.source_x, pair.target_x, pair.source_xy, pair.target_xy)
    cost = mixed_cost(components["expression"], components["spatial_cross"], (0.5, 0.5))
    a = np.full(len(pair.source_x), 1.0 / len(pair.source_x))
    b = np.full(len(pair.target_x), 1.0 / len(pair.target_x))
    solved = solve_p1(cost, a, b, method_parameters(method, config))
    if not solved.converged:
        raise RuntimeError(f"baseline solver did not converge: {identifier}")
    mass = solved.plan.sum(axis=1)
    conditional = solved.plan / np.maximum(mass[:, None], 1e-300)
    if conditional.shape[1] > 1:
        top_two = np.partition(conditional, -2, axis=1)[:, -2:]
        best = top_two.max(axis=1)
        second = top_two.min(axis=1)
    else:
        best = conditional[:, 0]
        second = conditional[:, 0]
    arrays = {
        "predicted_target": np.argmax(conditional, axis=1).astype(np.int64),
        "max_probability": best.astype(np.float64),
        "top2_margin": (best - second).astype(np.float64),
        "row_mass": mass.astype(np.float64),
    }
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(array_path, **arrays)
    metadata = {
        "status": "COMPLETED",
        "config_sha256": config_hash,
        "pair_path": str(pair_path),
        "pair_sha256": sha256(pair_path),
        "method": method,
        "converged": bool(solved.converged),
        "iterations": int(solved.iterations),
        "solver_seconds": float(solved.seconds),
        "n_source": int(len(pair.source_x)),
        "n_target": int(len(pair.target_x)),
    }
    dump_json(metadata_path, metadata)
    return arrays, metadata


def spatial_folds(xy: np.ndarray, valid: np.ndarray, n_splits: int, seed: int) -> np.ndarray:
    labels = np.full(len(xy), -1, dtype=int)
    model = KMeans(n_clusters=n_splits, random_state=seed, n_init=20)
    labels[valid] = model.fit_predict(np.asarray(xy, dtype=float)[valid])
    return labels


def select_count(n: int, coverage: float) -> int:
    return min(n, max(2, int(math.ceil(float(coverage) * n))))


def fit_and_evaluate(
    source: np.ndarray,
    target: np.ndarray,
    source_labels: np.ndarray,
    target_labels: np.ndarray,
    truth_target: np.ndarray,
    train_indices: np.ndarray,
    selected_indices: np.ndarray | None,
    selected_targets: np.ndarray | None,
    test_indices: np.ndarray,
    alpha: float,
    identity: bool = False,
) -> dict[str, Any]:
    target_normalized = normalize_rows(target)
    if identity:
        mapped = source[test_indices]
        training_precision = float("nan")
        n_selected = 0
    else:
        if selected_indices is None or selected_targets is None:
            raise ValueError("training pairs are required for a learned mapping")
        model = Ridge(alpha=float(alpha), fit_intercept=True)
        model.fit(source[selected_indices], target[selected_targets])
        mapped = model.predict(source[test_indices])
        training_precision = float(np.mean(selected_targets == truth_target[selected_indices]))
        n_selected = int(len(selected_indices))
    mapped = normalize_rows(mapped)
    similarity = mapped @ target_normalized.T
    true_targets = truth_target[test_indices].astype(int)
    true_scores = similarity[np.arange(len(test_indices)), true_targets]
    ranks = 1 + np.sum(similarity > true_scores[:, None], axis=1)
    predicted = np.argmax(similarity, axis=1)
    return {
        "n_train_available": int(len(train_indices)),
        "n_train_selected": n_selected,
        "n_test": int(len(test_indices)),
        "training_positive_precision": training_precision,
        "top1": float(np.mean(ranks <= 1)),
        "top5": float(np.mean(ranks <= 5)),
        "top10": float(np.mean(ranks <= 10)),
        "mean_reciprocal_rank": float(np.mean(1.0 / ranks)),
        "median_true_rank": float(np.median(ranks)),
        "region_label_accuracy": float(
            np.mean(np.asarray(source_labels)[test_indices] == np.asarray(target_labels)[predicted])
        ),
        "true_pair_cosine": float(np.mean(true_scores)),
    }


def strategy_rows(
    pair: Any,
    extras: dict[str, np.ndarray],
    arrays: dict[str, np.ndarray],
    method: str,
    direction: str,
    config: dict[str, Any],
    smoke: bool,
) -> list[dict[str, Any]]:
    specification = config["representation_transfer"]
    coverage = float(specification["primary_coverage"])
    alpha = float(specification["ridge_alpha"])
    repeats = 1 if smoke else int(specification["random_gate_repeats"])
    feature_source = specification.get("feature_source", "heldout_expression")
    if feature_source == "heldout_expression":
        source = normalize_rows(extras["source_heldout"])
        target = normalize_rows(extras["target_heldout"])
        representation_features = "heldout_expression_100"
    elif feature_source == "cost_features":
        source = normalize_rows(pair.source_x)
        target = normalize_rows(pair.target_x)
        representation_features = "cost_features_500"
    else:
        raise ValueError(f"unsupported representation feature source: {feature_source}")
    valid = np.flatnonzero((~pair.truth_missing) & (pair.truth_target >= 0))
    n_splits = 2 if smoke else 5
    folds = spatial_folds(
        pair.source_xy,
        valid,
        n_splits,
        stable_seed(config["statistics"]["bootstrap_seed"], pair.pair_id, direction),
    )
    rows: list[dict[str, Any]] = []
    predicted = arrays["predicted_target"].astype(int)
    max_probability = arrays["max_probability"]
    margin = arrays["top2_margin"]
    source_ids = extras.get("source_ids", np.arange(len(pair.source_x)).astype(str)).astype(str)
    metadata = {
        "dataset": pair.dataset,
        "pair_id": pair.pair_id,
        "biological_pair_id": pair.metadata.get("biological_pair_id", pair.pair_id.replace("__reverse", "")),
        "independent_unit_id": pair.metadata.get("independent_unit_id", ""),
        "direction": direction,
        "method": method,
        "representation_features": representation_features,
        "model": "multioutput_ridge",
        "ridge_alpha": alpha,
    }
    for fold in range(n_splits):
        test = valid[folds[valid] == fold]
        train = valid[folds[valid] != fold]
        if len(test) < 3 or len(train) < 10:
            raise RuntimeError(f"insufficient spatial fold: {pair.pair_id} {direction} fold={fold}")
        keep = select_count(len(train), coverage)
        max_selected = train[np.argsort(-max_probability[train], kind="mergesort")[:keep]]
        margin_selected = train[np.argsort(-margin[train], kind="mergesort")[:keep]]
        fixed_rng = np.random.default_rng(stable_seed(pair.pair_id, direction, method, fold, "oracle"))
        oracle_selected = np.sort(fixed_rng.choice(train, size=keep, replace=False))

        designs: list[tuple[str, int, np.ndarray | None, np.ndarray | None, bool]] = [
            ("identity_no_training", 0, None, None, True),
            ("all_top1", 0, train, predicted[train], False),
            ("max_probability_gate", 0, max_selected, predicted[max_selected], False),
            ("top2_margin_gate", 0, margin_selected, predicted[margin_selected], False),
            ("oracle_truth", 0, oracle_selected, pair.truth_target[oracle_selected].astype(int), False),
        ]
        for repeat in range(repeats):
            rng = np.random.default_rng(stable_seed(pair.pair_id, direction, method, fold, "random", repeat))
            random_selected = np.sort(rng.choice(train, size=keep, replace=False))
            designs.append(("random_gate", repeat, random_selected, predicted[random_selected], False))

        for strategy, repeat, selected_indices, selected_targets, identity in designs:
            measured = fit_and_evaluate(
                source,
                target,
                pair.source_labels,
                pair.target_labels,
                pair.truth_target,
                train,
                selected_indices,
                selected_targets,
                test,
                alpha,
                identity=identity,
            )
            rows.append(
                {
                    **metadata,
                    "fold": fold,
                    "strategy": strategy,
                    "coverage": 1.0 if strategy == "all_top1" else (0.0 if identity else coverage),
                    "repeat": repeat,
                    "source_ids_sha256": hashlib.sha256("\n".join(source_ids).encode("utf-8")).hexdigest(),
                    **measured,
                }
            )
    return rows


def representation_summary(
    fold_rows: pd.DataFrame,
    config: dict[str, Any],
    results: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics = [
        "training_positive_precision",
        "top1",
        "top5",
        "top10",
        "mean_reciprocal_rank",
        "median_true_rank",
        "region_label_accuracy",
        "true_pair_cosine",
    ]
    repeat_keys = [
        "dataset",
        "pair_id",
        "biological_pair_id",
        "independent_unit_id",
        "direction",
        "method",
        "fold",
        "strategy",
        "coverage",
    ]
    folds = fold_rows.groupby(repeat_keys, as_index=False)[metrics].mean()
    direction_keys = [key for key in repeat_keys if key != "fold"]
    direction = folds.groupby(direction_keys, as_index=False)[metrics].mean()
    unit = (
        direction.groupby(["independent_unit_id", "method", "strategy", "coverage"], as_index=False)[metrics]
        .mean()
        .sort_values(["method", "strategy", "independent_unit_id"])
    )
    descriptive_rows: list[dict[str, Any]] = []
    for keys, frame in unit.groupby(["method", "strategy", "coverage"], sort=True):
        for metric in metrics:
            values = frame[metric].to_numpy(float)
            finite = values[np.isfinite(values)]
            if not len(finite):
                continue
            q25, q75 = np.quantile(finite, [0.25, 0.75])
            descriptive_rows.append({
                "method": keys[0],
                "strategy": keys[1],
                "coverage": keys[2],
                "metric": metric,
                "n_independent_units": int(len(finite)),
                "median": float(np.median(finite)),
                "q25": float(q25),
                "q75": float(q75),
            })
    descriptive = pd.DataFrame(descriptive_rows)

    primary_coverage = float(config["representation_transfer"]["primary_coverage"])
    maximum = unit[
        (unit["strategy"] == "max_probability_gate") & np.isclose(unit["coverage"], primary_coverage)
    ].copy()
    margin = unit[
        (unit["strategy"] == "top2_margin_gate") & np.isclose(unit["coverage"], primary_coverage)
    ].copy()
    merged = maximum.merge(
        margin,
        on=["independent_unit_id", "method"],
        suffixes=("__max_probability", "__top2_margin"),
        validate="one_to_one",
    )
    effect_rows: list[dict[str, Any]] = []
    higher_is_better = {
        "training_positive_precision": True,
        "top1": True,
        "top5": True,
        "top10": True,
        "mean_reciprocal_rank": True,
        "median_true_rank": False,
        "region_label_accuracy": True,
        "true_pair_cosine": True,
    }
    for metric in metrics:
        raw_delta = merged[f"{metric}__top2_margin"] - merged[f"{metric}__max_probability"]
        beneficial = raw_delta if higher_is_better[metric] else -raw_delta
        for index, row in merged.iterrows():
            effect_rows.append({
                "independent_unit_id": row["independent_unit_id"],
                "method": row["method"],
                "metric": metric,
                "max_probability": row[f"{metric}__max_probability"],
                "top2_margin": row[f"{metric}__top2_margin"],
                "raw_margin_minus_max": raw_delta.loc[index],
                "beneficial_margin_effect": beneficial.loc[index],
            })
    effects = pd.DataFrame(effect_rows)
    inferential = summarize_effects(
        effects,
        "method",
        "beneficial_margin_effect",
        config,
        family_column="metric",
    )
    inferential["primary_endpoint"] = inferential["metric"] == config["representation_transfer"]["primary_endpoint"].replace(
        "known-correspondence ", ""
    ).replace(" retrieval on held-out spatial blocks", "")
    # The config wording maps to the top1 column. Keep the mapping explicit.
    inferential["primary_endpoint"] = inferential["metric"] == "top1"
    dump_tsv(results / "representation_fold.tsv", fold_rows)
    dump_tsv(results / "representation_direction.tsv", direction)
    dump_tsv(results / "representation_unit.tsv", unit)
    dump_tsv(results / "representation_descriptive_summary.tsv", descriptive)
    dump_tsv(results / "representation_margin_vs_max_unit.tsv", effects)
    dump_tsv(results / "representation_margin_vs_max_summary.tsv", inferential)
    return direction, unit, descriptive, inferential


def run_representation(
    config: dict[str, Any],
    config_hash: str,
    analysis: Path,
    results: Path,
    smoke: bool,
) -> dict[str, Any]:
    pair_root = Path(config["processed_pair_root"])
    forward_paths = sorted(pair_root.glob("HER2ST_*_CONTROLLED.npz"))
    if smoke:
        forward_paths = forward_paths[:1]
    methods = list(config["methods"])
    if smoke:
        methods = methods[:1]
    rows: list[dict[str, Any]] = []
    solver_metadata: list[dict[str, Any]] = []
    checkpoint_root = analysis / "plan_score_checkpoints"
    for forward in forward_paths:
        for direction in config["directions"]:
            path = forward if direction == "forward" else forward.with_name(forward.stem + "__reverse.npz")
            if not path.is_file():
                raise FileNotFoundError(path)
            pair, extras = load_pair(path)
            for method in methods:
                arrays, solved = plan_scores(path, method, config, checkpoint_root, config_hash)
                solver_metadata.append(solved)
                rows.extend(strategy_rows(pair, extras, arrays, method, direction, config, smoke))
    fold_rows = pd.DataFrame(rows)
    if fold_rows.empty:
        raise RuntimeError("representation experiment produced no rows")
    _, unit, _, inferential = representation_summary(fold_rows, config, results)
    dump_tsv(results / "representation_solver_manifest.tsv", pd.DataFrame(solver_metadata))
    return {
        "n_fold_strategy_rows": int(len(fold_rows)),
        "n_independent_units": int(unit["independent_unit_id"].nunique()),
        "n_methods": int(unit["method"].nunique()),
        "n_directions": int(fold_rows[["pair_id", "direction"]].drop_duplicates().shape[0]),
        "all_solvers_converged": bool(all(row.get("converged", False) for row in solver_metadata)),
        "primary_comparisons": inferential[inferential["primary_endpoint"]].to_dict(orient="records"),
    }


def make_report(config: dict[str, Any], results: Path, status: dict[str, Any]) -> None:
    quality_path = results / "positive_pair_quality_summary.tsv"
    representation_path = results / "representation_margin_vs_max_summary.tsv"
    lines = [
        "# VALID-OT downstream positive-pair and representation analysis",
        "",
        "## Analysis boundary",
        "",
        "This package evaluates whether score choice changes pseudo-positive purity and a controlled",
        "positive-pair-conditioned representation-transfer task. It is not a full AlignDG reproduction",
        "and does not establish general real-section biological utility.",
        "",
        "## Independent unit",
        "",
        "HER2ST patient is the independent unit (n=8 in the full run). Directions, spatial folds and",
        "random-gate repeats are aggregated within patient before inference.",
        "",
    ]
    if quality_path.is_file():
        quality = pd.read_csv(quality_path, sep="\t")
        lines.extend(["## Positive-pair quality", "", dataframe_to_markdown(quality), ""])
    if representation_path.is_file():
        representation = pd.read_csv(representation_path, sep="\t")
        primary = representation[representation["primary_endpoint"].astype(bool)]
        lines.extend(["## Representation transfer: primary endpoint", "", dataframe_to_markdown(primary), ""])
    lines.extend(["## Run status", "", "```json", json.dumps(status, ensure_ascii=False, indent=2), "```", ""])
    (results / "DOWNSTREAM_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--quality-only", action="store_true")
    parser.add_argument("--representation-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.quality_only and args.representation_only:
        raise ValueError("choose at most one of --quality-only and --representation-only")
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config_hash = sha256(config_path)
    suffix = "_smoke" if args.smoke else ""
    analysis = ROOT / (config["analysis_root"] + suffix)
    results = ROOT / (config["results_root"] + suffix)
    analysis.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    status_path = analysis / "DOWNSTREAM_STATUS.json"
    status: dict[str, Any] = {
        "state": "RUNNING",
        "started_local_epoch": time.time(),
        "config_path": str(config_path),
        "config_sha256": config_hash,
        "smoke": bool(args.smoke),
        "python": sys.version,
        "platform": platform.platform(),
    }
    dump_json(status_path, status)
    try:
        if not args.representation_only:
            status["positive_pair_quality"] = run_positive_pair_quality(config, results)
            dump_json(status_path, status)
        if not args.quality_only:
            status["representation_transfer"] = run_representation(
                config, config_hash, analysis, results, args.smoke
            )
            dump_json(status_path, status)
        status["state"] = "COMPLETED"
        status["completed_local_epoch"] = time.time()
        status["elapsed_seconds"] = status["completed_local_epoch"] - status["started_local_epoch"]
        dump_json(status_path, status)
        make_report(config, results, status)
        print(json.dumps(status, ensure_ascii=False, indent=2))
        return 0
    except Exception as error:
        status["state"] = "FAILED"
        status["error"] = repr(error)
        status["traceback"] = traceback.format_exc()
        status["failed_local_epoch"] = time.time()
        dump_json(status_path, status)
        print(status["traceback"], file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
