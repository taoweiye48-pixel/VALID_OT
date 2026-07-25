from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from validot.utils import status_payload, write_json


ROOT = Path(__file__).resolve().parents[1]
STATS = ROOT / "15_v1_3_correction" / "03_statistics"
SENS = ROOT / "15_v1_3_correction" / "04_p1_sensitivity"
OUTPUT = ROOT / "15_v1_3_correction" / "05_figures"
METHOD_COLORS = {"balanced_ot": "#4472C4", "uot": "#ED7D31", "row_softmax": "#70AD47"}


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUTPUT / f"{name}.png", dpi=240, bbox_inches="tight")
    fig.savefig(OUTPUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def design_figure() -> None:
    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.axis("off")
    boxes = [
        (0.02, "Frozen OT model\n+ evidence intervention"),
        (0.27, "Full finite re-run\n model-response reference"),
        (0.52, "Axis 1\n internal fidelity"),
        (0.77, "Axis 2\n external error utility"),
    ]
    for x, text in boxes:
        ax.add_patch(
            plt.Rectangle((x, 0.32), 0.20, 0.38, facecolor="#EAF2F8", edgecolor="#1F4E79", lw=1.5)
        )
        ax.text(x + 0.10, 0.51, text, ha="center", va="center", fontsize=11)
    for x in (0.22, 0.47, 0.72):
        ax.annotate("", xy=(x + 0.045, 0.51), xytext=(x, 0.51), arrowprops={"arrowstyle": "->", "lw": 1.5})
    ax.text(
        0.5,
        0.12,
        "Full re-optimization is an intervention-specific model-response reference, not biological ground truth.",
        ha="center",
        fontsize=10,
        color="#7F6000",
    )
    ax.set_title("VALID-OT v1.3: dual-axis audit design", fontsize=15, weight="bold")
    save(fig, "Fig1_v13_design")


def fidelity_figure() -> None:
    table = pd.read_csv(STATS / "fidelity_gate_pair_level_corrected.tsv", sep="\t")
    table = table[(table.source == "real") & (table.proxy == "finite_difference_sensitivity_h001")].copy()
    table["label"] = table.method + "\n" + table.intervention
    metrics = [
        ("median_spearman", 0.70, "Spearman"),
        ("median_top_decile_precision", 0.60, "Top-decile overlap"),
        ("median_normalized_mae", 0.75, "NMAE (lower is better)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for ax, (column, threshold, title) in zip(axes, metrics):
        colors = [METHOD_COLORS[value] for value in table.method]
        ax.bar(np.arange(len(table)), table[column], color=colors, alpha=0.85)
        ax.axhline(threshold, color="black", ls="--", lw=1, label="frozen threshold")
        ax.set_xticks(np.arange(len(table)), table.label, rotation=45, ha="right", fontsize=8)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Direction-averaged pair median")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Real-slice finite-difference fidelity after pair-level correction", weight="bold")
    save(fig, "Fig2_real_fidelity_corrected")


def external_figure() -> None:
    table = pd.read_csv(STATS / "real_external_gate_with_controls.tsv", sep="\t")
    scores = [
        "exact_I_EXPR",
        "exact_I_SPATIAL",
        "exact_combined",
        "finite_difference_I_EXPR_h001",
        "finite_difference_I_SPATIAL_h001",
        "finite_difference_combined_h001",
    ]
    witnesses = ["heldout_loss", "label_error_shared_closed_set"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), sharey=True)
    for ax, witness in zip(axes, witnesses):
        selected = table[(table.witness == witness) & table.score.isin(scores)].copy()
        selected["label"] = selected.method + " | " + selected.score.str.replace("finite_difference_", "FD_")
        selected = selected.sort_values(["method", "score"])
        colors = [METHOD_COLORS[value] for value in selected.method]
        ax.barh(np.arange(len(selected)), selected.median_nex, color=colors, alpha=0.85)
        ax.axvline(1.0, color="black", ls="--", lw=1, label="random ranking")
        passed = selected.corrected_external_gate_with_controls.to_numpy(bool)
        ax.scatter(
            selected.loc[passed, "median_nex"],
            np.flatnonzero(passed),
            marker="*",
            s=80,
            color="#C00000",
            label="corrected gate + controls",
            zorder=3,
        )
        ax.set_yticks(np.arange(len(selected)), selected.label, fontsize=7)
        ax.invert_yaxis()
        ax.set_title(witness)
        ax.set_xlabel("NEX-AURC (0 oracle; 1 random; lower is better)")
        ax.grid(axis="x", alpha=0.2)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("External utility is intervention- and witness-specific", weight="bold")
    save(fig, "Fig3_real_external_intervention_specific")


def relation_figure() -> None:
    table = pd.read_csv(STATS / "fidelity_utility_matched_units.tsv", sep="\t")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True, sharey=True)
    for ax, source in zip(axes, ["semisynthetic", "real"]):
        selected = table[table.source == source]
        for intervention, marker in [("I_EXPR", "o"), ("I_SPATIAL", "s")]:
            part = selected[selected.intervention == intervention]
            ax.scatter(
                part.spearman,
                part.external_utility,
                s=22,
                alpha=0.45,
                marker=marker,
                label=intervention,
            )
        ax.axhline(-1.0, color="black", ls="--", lw=1, label="random utility")
        ax.axvline(0.70, color="#808080", ls=":", lw=1, label="fidelity threshold")
        ax.set_title(source)
        ax.set_xlabel("Internal Spearman fidelity")
        ax.grid(alpha=0.15)
    axes[0].set_ylabel("External utility = -NEX-AURC (0 oracle; -1 random)")
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle("Internal fidelity and external utility require separate evaluation", weight="bold")
    save(fig, "Fig4_matched_dual_axis")


def sensitivity_figure() -> None:
    endpoint = pd.read_csv(SENS / "endpoint_step" / "endpoint_step_summary.tsv", sep="\t")
    coordinate = pd.read_csv(SENS / "coordinate_frame" / "coordinate_comparison.tsv", sep="\t")
    sampling = pd.read_csv(
        SENS / "label_agnostic_sampling" / "sampling_mode_comparison.tsv", sep="\t"
    )
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for (method, intervention), group in endpoint.groupby(["method", "intervention"]):
        axes[0].plot(
            group.endpoint_step,
            group.median_normalized_mae,
            marker="o" if intervention == "I_EXPR" else "s",
            color=METHOD_COLORS[method],
            ls="-" if intervention == "I_EXPR" else "--",
            label=f"{method}:{intervention}",
        )
    axes[0].axhline(0.75, color="black", ls=":", lw=1)
    axes[0].set_xlabel("finite-difference step h")
    axes[0].set_ylabel("median NMAE")
    axes[0].set_title("Endpoint-step sensitivity")
    heldout = coordinate[
        (coordinate.witness == "heldout_loss")
        & coordinate.score.isin(["exact_combined", "exact_I_EXPR", "finite_difference_I_EXPR_h001"])
    ]
    coord_summary = heldout.groupby("coordinate_variant").delta_nex_from_baseline.apply(
        lambda x: float(np.nanmax(np.abs(x)))
    )
    axes[1].bar(np.arange(len(coord_summary)), coord_summary.values, color="#5B9BD5")
    axes[1].set_xticks(np.arange(len(coord_summary)), coord_summary.index, rotation=45, ha="right", fontsize=8)
    axes[1].set_ylabel("max |delta NEX| from baseline")
    axes[1].set_title("Coordinate-frame sensitivity")
    estimable = sampling.same_direction_relative_to_random.dropna()
    axes[2].bar([0], [estimable.mean()], color="#A5A5A5", width=0.55)
    axes[2].set_xticks([0], ["all estimable scores"])
    axes[2].set_ylim(0, 1)
    axes[2].set_ylabel("direction stability vs random")
    axes[2].set_title("Label-agnostic resampling")
    axes[0].legend(frameon=False, fontsize=6, ncol=2)
    fig.suptitle("P1 sensitivity analyses", weight="bold")
    save(fig, "Fig5_p1_sensitivities")


def validity_figure() -> None:
    cards = pd.read_csv(STATS / "validity_cards_intervention_witness.tsv", sep="\t")
    cards = cards[
        (cards.source == "real")
        & (cards.proxy == "finite_difference_sensitivity_h001")
        & cards.witness.isin(["heldout_loss", "label_error_shared_closed_set"])
    ].copy()
    cards["row"] = cards.method + ":" + cards.intervention
    pivot = cards.pivot_table(index="row", columns="witness", values="quadrant", aggfunc="first")
    mapping = {"A": 3, "B": 2, "C": 1, "D": 0}
    matrix = pivot.apply(lambda column: column.map(mapping)).to_numpy(float)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    image = ax.imshow(matrix, vmin=0, vmax=3, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(np.arange(len(pivot.columns)), pivot.columns, rotation=20, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)), pivot.index)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, pivot.iloc[i, j], ha="center", va="center", weight="bold")
    ax.set_title("Intervention × witness validity cards (A best, D neither axis passes)")
    fig.colorbar(image, ax=ax, ticks=[0, 1, 2, 3], label="D / C / B / A")
    save(fig, "Fig6_validity_cards_corrected")


def robustness_figure() -> None:
    table = pd.read_csv(STATS / "robustness_summary_corrected.tsv", sep="\t")
    summary = table.groupby("method").direction_stability_relative_to_random.mean()
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.bar(
        np.arange(len(summary)),
        summary.values,
        color=[METHOD_COLORS[index] for index in summary.index],
    )
    ax.set_xticks(np.arange(len(summary)), summary.index)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Mean direction stability relative to random")
    ax.set_title("Robustness is high but not 100%")
    ax.grid(axis="y", alpha=0.2)
    save(fig, "FigS1_robustness_corrected")


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    design_figure()
    fidelity_figure()
    external_figure()
    relation_figure()
    sensitivity_figure()
    validity_figure()
    robustness_figure()
    decision = status_payload(
        "V1_3_FIGURES",
        "COMPLETED",
        main_figures=6,
        supplementary_figures=1,
        utility_reference="0 oracle; 1 NEX random; -1 signed-utility random",
        figure4="intervention matched; real and semisynthetic separated",
    )
    write_json(OUTPUT / "FIGURE_DECISION.json", decision)
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
