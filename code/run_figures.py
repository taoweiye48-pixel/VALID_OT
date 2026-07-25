from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from validot.utils import status_payload, write_json


ROOT = Path(__file__).resolve().parents[1]
STATS = ROOT / "12_E8_statistics"
OUTPUT = ROOT / "13_figures"
METHOD_ORDER = ["row_softmax", "balanced_ot", "uot", "paste_fgw", "paste2_partial_fgw"]


def save(fig, name: str) -> None:
    fig.savefig(OUTPUT / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def figure_1() -> None:
    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.axis("off")
    boxes = [
        (0.03, 0.55, "Frozen spatial OT\nbase plan P"),
        (0.26, 0.55, "Delete evidence group\nI_EXPR or I_SPATIAL"),
        (0.50, 0.55, "Exact re-solve\nmodel response reference"),
        (0.74, 0.70, "Internal fidelity\nrank / top-decile / amplitude"),
        (0.74, 0.30, "External validity\nerror / missing / held-out loss"),
    ]
    for x, y, label in boxes:
        ax.text(
            x,
            y,
            label,
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=11,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#eef4ff", edgecolor="#315b8a", lw=1.5),
        )
    for start, end in [((0.19, 0.55), (0.25, 0.55)), ((0.43, 0.55), (0.49, 0.55))]:
        ax.annotate("", xy=end, xytext=start, xycoords="axes fraction", arrowprops=dict(arrowstyle="->", lw=1.8))
    ax.annotate("", xy=(0.73, 0.70), xytext=(0.67, 0.58), xycoords="axes fraction", arrowprops=dict(arrowstyle="->", lw=1.8))
    ax.annotate("", xy=(0.73, 0.30), xytext=(0.67, 0.52), xycoords="axes fraction", arrowprops=dict(arrowstyle="->", lw=1.8))
    ax.text(0.91, 0.50, "2-axis\nvalidity card", transform=ax.transAxes, ha="center", va="center", fontsize=12, weight="bold")
    ax.annotate("", xy=(0.87, 0.53), xytext=(0.85, 0.68), xycoords="axes fraction", arrowprops=dict(arrowstyle="->"))
    ax.annotate("", xy=(0.87, 0.47), xytext=(0.85, 0.32), xycoords="axes fraction", arrowprops=dict(arrowstyle="->"))
    ax.set_title("VALID-OT: separate model fidelity from external validity", fontsize=15, weight="bold")
    save(fig, "Fig1_design")


def figure_2() -> None:
    validation = pd.read_csv(ROOT / "05_E1_solver_validation" / "solver_validation.tsv", sep="\t")
    runtime = pd.read_csv(ROOT / "01_manifest" / "runtime_probe.tsv", sep="\t")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    sns.barplot(data=validation, x="method", y="repeat_relative_l1", order=METHOD_ORDER, ax=axes[0], color="#4776b4")
    axes[0].axhline(1e-8, color="crimson", ls="--", label="registered limit")
    axes[0].set_yscale("symlog", linthresh=1e-14)
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].set_title("Repeated-solve consistency")
    axes[0].legend(frameon=False)
    sns.lineplot(data=runtime, x="n", y="five_solve_seconds", hue="method", hue_order=METHOD_ORDER, marker="o", ax=axes[1])
    axes[1].set_yscale("log")
    axes[1].set_title("Five-solve audit runtime probe")
    axes[1].set_ylabel("seconds (log scale)")
    axes[1].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    save(fig, "Fig2_numeric_runtime")


def figure_3() -> None:
    table = pd.read_csv(STATS / "fidelity_summary.tsv", sep="\t")
    table = table[table.source == "semisynthetic"]
    pivot = table.pivot_table(index=["method", "intervention"], columns="proxy", values="median_spearman")
    pivot = pivot.reindex(pd.MultiIndex.from_product([METHOD_ORDER, ["I_EXPR", "I_SPATIAL"]]))
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.heatmap(pivot, cmap="vlag", center=0, vmin=-1, vmax=1, annot=True, fmt=".2f", ax=ax, cbar_kws={"label": "median Spearman"})
    ax.set_title("Semisynthetic internal fidelity")
    ax.set_xlabel("local explanation / proxy")
    ax.set_ylabel("method and intervention")
    fig.tight_layout()
    save(fig, "Fig3_semisynthetic_fidelity")


def figure_4() -> None:
    table = pd.read_csv(STATS / "fidelity_utility_units.tsv", sep="\t")
    table = table[table.source == "semisynthetic"]
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=table, x="spearman", y="external_utility", hue="method", hue_order=METHOD_ORDER, style="score", alpha=0.65, ax=ax)
    ax.axvline(0.70, color="black", ls="--", lw=1)
    ax.axhline(0, color="black", ls=":", lw=1)
    ax.set_title("Fidelity and external utility are separate axes")
    ax.set_ylabel("external utility (-normalized excess AURC)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7, frameon=False)
    fig.tight_layout()
    save(fig, "Fig4_fidelity_utility")


def figure_5() -> None:
    table = pd.read_csv(STATS / "utility_gain_units.tsv", sep="\t")
    table = table[(table.source == "real") & (table.score == "exact_combined")]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, witness in zip(axes, sorted(table.witness.unique())):
        subset = table[table.witness == witness]
        sns.stripplot(
            data=subset,
            x="method",
            y="absolute_gain_over_best_qc",
            hue="technology",
            order=METHOD_ORDER,
            dodge=True,
            jitter=0.12,
            ax=ax,
        )
        ax.axhline(0, color="black", lw=1, ls="--")
        ax.tick_params(axis="x", rotation=30)
        ax.set_title(witness)
        ax.set_xlabel("")
        ax.legend(fontsize=8, frameon=False)
    axes[0].set_ylabel("gain over best frozen QC baseline")
    fig.suptitle("Real-data external validity of exact response")
    fig.tight_layout()
    save(fig, "Fig5_real_external_validity")


def figure_6() -> None:
    cards = pd.read_csv(STATS / "validity_cards.tsv", sep="\t")
    cards["card"] = cards.full_fidelity_pass.astype(int) + 2 * cards.external_gate_pass.astype(int)
    pivot = cards.pivot_table(index=["source", "method"], columns="proxy", values="card", aggfunc="max")
    colors = ["#d9d9d9", "#5aa469", "#e6a04b", "#4472c4"]
    from matplotlib.colors import ListedColormap

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(pivot, cmap=ListedColormap(colors), vmin=0, vmax=3, cbar=False, linewidths=0.5, linecolor="white", ax=ax)
    for y, index in enumerate(pivot.index):
        for x, column in enumerate(pivot.columns):
            value = pivot.loc[index, column]
            if np.isfinite(value):
                label = {0: "D", 1: "B", 2: "C", 3: "A"}[int(value)]
                ax.text(x + 0.5, y + 0.5, label, ha="center", va="center", weight="bold")
    ax.set_title("Two-axis validity cards (A: both, B: fidelity only, C: utility only, D: neither)")
    ax.set_xlabel("explanation / proxy")
    ax.set_ylabel("data source and OT method")
    fig.tight_layout()
    save(fig, "Fig6_validity_cards")


def supplements() -> None:
    robustness_path = STATS / "robustness_summary.tsv"
    if robustness_path.exists():
        table = pd.read_csv(robustness_path, sep="\t")
        fig, ax = plt.subplots(figsize=(12, 5))
        sns.stripplot(data=table, x="variant", y="same_direction_fraction", hue="method", ax=ax)
        ax.axhline(2 / 3, color="black", ls="--")
        ax.tick_params(axis="x", rotation=60)
        ax.set_title("Robustness direction stability")
        fig.tight_layout()
        save(fig, "FigS1_robustness")
    duplicate_path = ROOT / "09_E5_semisynthetic_external" / "diagnostics" / "duplicate.tsv"
    if duplicate_path.exists():
        table = pd.read_csv(duplicate_path, sep="\t")
        fig, ax = plt.subplots(figsize=(9, 5))
        melted = table.melt(
            id_vars=["method"],
            value_vars=["entropy_ambiguity_auroc", "top2_margin_ambiguity_auroc"],
            var_name="diagnostic",
            value_name="AUROC",
        )
        sns.boxplot(data=melted, x="method", y="AUROC", hue="diagnostic", order=METHOD_ORDER, ax=ax)
        ax.axhline(0.5, color="black", ls="--")
        ax.tick_params(axis="x", rotation=30)
        ax.set_title("Duplicate-motif ambiguity diagnostics")
        fig.tight_layout()
        save(fig, "FigS2_duplicate_motif")
    m5_path = ROOT / "10_E6_real_external" / "optional_3dot" / "pair_direction_averaged_gain.tsv"
    if m5_path.exists():
        table = pd.read_csv(m5_path, sep="\t")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.stripplot(
            data=table,
            x="witness",
            y="absolute_gain",
            hue="dataset",
            dodge=True,
            jitter=0.12,
            ax=ax,
        )
        ax.axhline(0, color="black", ls="--", lw=1)
        ax.set_ylabel("gain over best frozen QC baseline")
        ax.set_title("Post-hoc exploratory 3d-OT transport-head validity")
        ax.legend(frameon=False)
        fig.tight_layout()
        save(fig, "FigS3_3dot_transport_head")


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")
    figure_1()
    figure_2()
    figure_3()
    figure_4()
    figure_5()
    figure_6()
    supplements()
    decision = status_payload(
        "E9_FIGURES",
        "COMPLETED",
        png_count=len(list(OUTPUT.glob("*.png"))),
        pdf_count=len(list(OUTPUT.glob("*.pdf"))),
    )
    write_json(OUTPUT / "FIGURE_DECISION.json", decision)
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
