"""Create Figure 5: local fidelity and witness-specific utility.

Panel a compares the held-out-expression utility of the same finite-response
score, its high-accuracy local reference and the exact endpoint response.
Panel b audits registered practical scores on separate internal-fidelity and
external-utility axes.  All summaries use frozen independent-unit outputs.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = HERE / "data"
OUT = ROOT / "figures"

WP5_FILE = DATA / "wp5_same_score_external_utility_unit.tsv"
WP9_INTERNAL_FILE = DATA / "wp9_real_score_internal_audit_unit.tsv"
WP9_EXTERNAL_FILE = DATA / "wp9_real_score_external_audit_unit.tsv"
SCORE_REGISTRY_FILE = DATA / "wp9_score_registry.tsv"

INK = "#25343B"
MUTED = "#68767D"
HAIR = "#AAB4B9"
GRID = "#E2E8EA"
PAPER = "#FFFFFF"
NEGATIVE_PALE = "#FAF1EF"
POSITIVE_PALE = "#EDF5F2"

METHOD_ORDER = ["balanced_ot", "uot", "row_softmax"]
METHOD_LABEL = {
    "balanced_ot": "Balanced OT",
    "uot": "UOT",
    "row_softmax": "Row-softmax",
}
METHOD_SHORT = {
    "balanced_ot": "Balanced",
    "uot": "UOT",
    "row_softmax": "Row-softmax",
}
METHOD_COLOR = {
    "balanced_ot": "#447CB2",
    "uot": "#2F857C",
    "row_softmax": "#BC7A27",
}

COHORT_ORDER = [
    "spatialDLPFC",
    "HER2ST controlled",
    "Legacy replication",
    "Manual-layer donors",
]
COHORT_MAP = {
    "primary_expansion": "spatialDLPFC",
    "manual_truth_controlled": "HER2ST controlled",
    "legacy_replication": "Legacy replication",
    "manual_layer_truth": "Manual-layer donors",
}
COHORT_N = {
    "spatialDLPFC": 10,
    "HER2ST controlled": 8,
    "Legacy replication": 3,
    "Manual-layer donors": 3,
}

GAIN_SCORES = {
    "finite_response_h001": "finite_gain",
    "local_reference": "local_gain",
    "endpoint_response": "endpoint_gain",
}
EXPECTED_FINITE_MEDIANS = {
    ("spatialDLPFC", "balanced_ot"): 0.446,
    ("spatialDLPFC", "uot"): 0.664,
    ("spatialDLPFC", "row_softmax"): -0.557,
    ("HER2ST controlled", "balanced_ot"): 0.653,
    ("HER2ST controlled", "uot"): 0.829,
    ("HER2ST controlled", "row_softmax"): -0.106,
    ("Legacy replication", "balanced_ot"): 0.065,
    ("Legacy replication", "uot"): 0.261,
    ("Legacy replication", "row_softmax"): -0.490,
    ("Manual-layer donors", "balanced_ot"): 0.377,
    ("Manual-layer donors", "uot"): 0.635,
    ("Manual-layer donors", "row_softmax"): -0.388,
}

PRIMARY_SCORES = [
    "assigned_raw_cost",
    "barycentric_displacement",
    "conditional_entropy",
    "finite_response_h001",
    "low_max_probability",
    "probability_margin_risk",
    "source_boundary_proximity",
    "transported_mass_deficit",
]
SCORE_SHORT = {
    "assigned_raw_cost": "cost",
    "barycentric_displacement": "bary.",
    "conditional_entropy": "entropy",
    "finite_response_h001": r"$s(0.01)$",
    "low_max_probability": "low-p",
    "probability_margin_risk": "margin",
    "source_boundary_proximity": "boundary",
    "transported_mass_deficit": "mass",
}
SCORE_MARKER = {
    "assigned_raw_cost": "s",
    "barycentric_displacement": "D",
    "conditional_entropy": "^",
    "finite_response_h001": "o",
    "low_max_probability": "v",
    "probability_margin_risk": "P",
    "source_boundary_proximity": "h",
    "transported_mass_deficit": "X",
}

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7,
        "axes.linewidth": 0.75,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "xtick.major.width": 0.65,
        "ytick.major.width": 0.65,
        "xtick.major.size": 2.6,
        "ytick.major.size": 2.6,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "mathtext.fontset": "dejavusans",
    }
)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and validate frozen WP5 and WP9 unit-level results."""
    wp5 = pd.read_csv(WP5_FILE, sep="\t")
    internal = pd.read_csv(WP9_INTERNAL_FILE, sep="\t")
    external = pd.read_csv(WP9_EXTERNAL_FILE, sep="\t")
    registry = pd.read_csv(SCORE_REGISTRY_FILE, sep="\t")

    heldout = wp5[wp5["witness"].eq("heldout_expression_loss")].copy()
    heldout["cohort"] = heldout["cohort_role"].map(COHORT_MAP)
    if heldout["cohort"].isna().any():
        raise ValueError("WP5 contains an unmapped held-out-expression cohort")

    gain = heldout.pivot(
        index=["independent_unit_id", "cohort", "method"],
        columns="score",
        values="normalized_excess_aurc",
    ).reset_index()
    required = {"source_boundary_proximity", *GAIN_SCORES}
    if not required.issubset(gain.columns):
        raise ValueError(f"WP5 is missing required scores: {required - set(gain.columns)}")
    if len(gain) != 72 or gain["independent_unit_id"].nunique() != 24:
        raise ValueError("WP5 panel requires 72 unit-method rows from 24 independent units")
    expected_counts = {
        (cohort, method): COHORT_N[cohort]
        for cohort in COHORT_ORDER
        for method in METHOD_ORDER
    }
    if gain.groupby(["cohort", "method"]).size().to_dict() != expected_counts:
        raise ValueError("WP5 cohort-method unit counts do not match the frozen design")
    for source, target in GAIN_SCORES.items():
        gain[target] = gain["source_boundary_proximity"] - gain[source]

    observed = gain.groupby(["cohort", "method"])["finite_gain"].median().to_dict()
    for key, expected in EXPECTED_FINITE_MEDIANS.items():
        if round(float(observed[key]), 3) != expected:
            raise ValueError(f"Frozen WP5 median mismatch for {key}: {observed[key]}")
    max_local_difference = np.max(np.abs(gain["finite_gain"] - gain["local_gain"]))
    if max_local_difference > 0.004:
        raise ValueError("Finite response and local-reference external utility diverged unexpectedly")

    registered_primary = registry.loc[registry["primary"].astype(bool), "score"].tolist()
    if set(registered_primary) != set(PRIMARY_SCORES):
        raise ValueError("WP9 primary-score registry does not match the frozen score set")

    internal = internal[
        internal["branch"].eq("main21")
        & internal["reference"].eq("local_reference")
        & internal["score"].isin(PRIMARY_SCORES)
    ][["independent_unit_id", "method", "score", "spearman"]].copy()
    external = external[
        external["branch"].eq("main21")
        & external["witness"].eq("heldout_expression_loss")
        & external["score"].isin(PRIMARY_SCORES)
    ][
        [
            "independent_unit_id",
            "method",
            "score",
            "normalized_excess_aurc",
        ]
    ].copy()
    dual = internal.merge(
        external,
        on=["independent_unit_id", "method", "score"],
        how="inner",
        validate="one_to_one",
    )
    if len(dual) != 21 * 3 * len(PRIMARY_SCORES):
        raise ValueError("WP9 dual-axis panel does not contain 21 units per method-score")
    if dual[["spearman", "normalized_excess_aurc"]].isna().any().any():
        raise ValueError("WP9 dual-axis panel contains missing values")

    summary = dual.groupby(["method", "score"]).agg(
        fidelity=("spearman", "median"),
        utility=("normalized_excess_aurc", "median"),
    )
    locked = {
        ("row_softmax", "assigned_raw_cost"): (-0.346, 0.256),
        ("row_softmax", "finite_response_h001"): (1.000, 1.355),
        ("uot", "transported_mass_deficit"): (0.520, 0.321),
    }
    for key, expected in locked.items():
        row = summary.loc[key]
        observed_pair = (round(float(row["fidelity"]), 3), round(float(row["utility"]), 3))
        if observed_pair != expected:
            raise ValueError(f"Frozen WP9 summary mismatch for {key}: {observed_pair}")
    return gain, dual


def style_axis(ax: plt.Axes, grid_axis: str = "x") -> None:
    ax.spines["left"].set_color(HAIR)
    ax.spines["bottom"].set_color(HAIR)
    ax.tick_params(colors=MUTED, labelsize=5.3, pad=1.5)
    ax.set_axisbelow(True)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.55)


def panel_heading(fig: plt.Figure, ax: plt.Axes, letter: str, title: str) -> None:
    box = ax.get_position()
    y = box.y1 + 0.045
    fig.text(box.x0 - 0.041, y, letter, fontsize=8.4, fontweight="bold", color=INK)
    fig.text(box.x0, y, title, fontsize=8.0, fontweight="bold", color=INK)


def row_layout() -> tuple[dict[tuple[str, str], float], list[float], list[str]]:
    positions: dict[tuple[str, str], float] = {}
    ticks: list[float] = []
    labels: list[str] = []
    bases = [11.0, 8.1, 5.2, 2.3]
    for cohort, base in zip(COHORT_ORDER, bases):
        for index, method in enumerate(METHOD_ORDER):
            y = base - 0.65 * index
            positions[(cohort, method)] = y
            ticks.append(y)
            # Method identity is encoded once in the shared legend. Repeating
            # the same labels for every cohort adds density without evidence.
            labels.append("")
    return positions, ticks, labels


def draw_same_score_panel(ax: plt.Axes, gain: pd.DataFrame) -> None:
    """Draw cohort-stratified absolute NEX gains for three response scores."""
    positions, ticks, labels = row_layout()
    style_axis(ax, "x")
    ax.axvspan(-1.05, 0.0, color=NEGATIVE_PALE, zorder=0)
    ax.axvspan(0.0, 1.10, color=POSITIVE_PALE, zorder=0)
    ax.axvline(0.0, color=MUTED, linewidth=0.75, zorder=1)

    for cohort_index, cohort in enumerate(COHORT_ORDER):
        base = positions[(cohort, "balanced_ot")]
        ax.text(
            -1.01,
            base + 0.42,
            cohort,
            ha="left",
            va="bottom",
            fontsize=5.4,
            fontweight="bold",
            color=INK,
        )
        if cohort_index > 0:
            ax.axhline(base + 0.78, color=GRID, linewidth=0.55, zorder=0)

        for method in METHOD_ORDER:
            subset = gain[gain["cohort"].eq(cohort) & gain["method"].eq(method)].sort_values(
                "independent_unit_id"
            )
            y = positions[(cohort, method)]
            colour = METHOD_COLOR[method]
            finite = subset["finite_gain"].to_numpy(float)
            local = subset["local_gain"].to_numpy(float)
            endpoint = subset["endpoint_gain"].to_numpy(float)

            point_y = y - 0.09 + np.linspace(-0.055, 0.055, len(finite))
            ax.scatter(
                finite,
                point_y,
                s=8,
                color=colour,
                alpha=0.38,
                edgecolors="none",
                zorder=2,
            )
            if len(finite) >= 8:
                lo, hi = np.quantile(finite, [0.25, 0.75])
            else:
                lo, hi = np.min(finite), np.max(finite)
            median_finite = float(np.median(finite))
            ax.plot([lo, hi], [y - 0.09, y - 0.09], color=colour, linewidth=2.2, alpha=0.50, zorder=3)
            ax.scatter(
                [median_finite],
                [y - 0.09],
                marker="o",
                s=34,
                facecolor=colour,
                edgecolor=PAPER,
                linewidth=0.65,
                zorder=5,
            )
            ax.scatter(
                [np.median(local)],
                [y + 0.10],
                marker="D",
                s=25,
                facecolor=PAPER,
                edgecolor=colour,
                linewidth=1.0,
                zorder=5,
            )
            ax.scatter(
                [np.median(endpoint)],
                [y + 0.01],
                marker="^",
                s=28,
                facecolor=PAPER,
                edgecolor=colour,
                linewidth=1.0,
                zorder=5,
            )

    ax.set_xlim(-1.05, 1.10)
    ax.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax.set_ylim(0.45, 11.65)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=5.4, color=MUTED)
    ax.tick_params(axis="y", length=2.4, pad=2.0)
    ax.set_xlabel(r"held-out-expression gain versus fixed-QC  ($\Delta$NEX)", fontsize=6.2, color=INK, labelpad=5)


def draw_dual_axis_panel(axes: list[plt.Axes], dual: pd.DataFrame) -> None:
    """Plot registered score medians on separate fidelity and utility axes."""
    summary = (
        dual.groupby(["method", "score"], as_index=False)
        .agg(
            fidelity=("spearman", "median"),
            utility=("normalized_excess_aurc", "median"),
        )
    )
    for index, (ax, method) in enumerate(zip(axes, METHOD_ORDER)):
        style_axis(ax, "both")
        ax.axhline(1.00, color=MUTED, linewidth=0.70, linestyle=(0, (2.4, 2.0)), zorder=1)
        colour = METHOD_COLOR[method]
        panel = summary[summary["method"].eq(method)].set_index("score")
        for score in PRIMARY_SCORES:
            row = panel.loc[score]
            x = float(row["fidelity"])
            y = float(row["utility"])
            ax.scatter(
                [x],
                [y],
                marker=SCORE_MARKER[score],
                s=31 if score == "finite_response_h001" else 25,
                facecolor=colour,
                edgecolor=PAPER,
                linewidth=0.55,
                zorder=4,
            )
        ax.set_xlim(-0.42, 1.08)
        ax.set_ylim(0.12, 1.43)
        ax.set_xticks([-0.3, 0.0, 0.3, 0.7, 1.0])
        ax.set_yticks([0.2, 0.6, 1.0, 1.4])
        ax.set_title(METHOD_LABEL[method], fontsize=6.3, fontweight="bold", color=colour, pad=5)
        ax.set_xlabel(r"local-fidelity Spearman  $\rho$", fontsize=5.8, color=INK, labelpad=4)
        if index == 0:
            ax.set_ylabel("held-out-expression NEX-AURC", fontsize=5.7, color=INK, labelpad=4)
        else:
            ax.set_yticklabels([])
            ax.tick_params(axis="y", length=0)


def build_figure(gain: pd.DataFrame, dual: pd.DataFrame) -> plt.Figure:
    fig = plt.figure(figsize=(7.0, 5.25), facecolor=PAPER)
    outer = fig.add_gridspec(
        2,
        1,
        height_ratios=[1.08, 1.0],
        left=0.125,
        right=0.975,
        top=0.86,
        bottom=0.135,
        hspace=0.43,
    )
    ax_a = fig.add_subplot(outer[0, 0])
    lower = outer[1, 0].subgridspec(1, 3, wspace=0.18)
    axes_b = [fig.add_subplot(lower[0, idx]) for idx in range(3)]

    draw_same_score_panel(ax_a, gain)
    draw_dual_axis_panel(axes_b, dual)

    panel_heading(fig, ax_a, "a", r"Held-out-expression $\Delta$NEX across cohorts")
    panel_heading(fig, axes_b[0], "b", "Local-fidelity Spearman versus held-out-expression NEX-AURC")

    method_handles = [
        Line2D([0], [0], marker="o", color="none",
               markerfacecolor=METHOD_COLOR[method], markeredgecolor=PAPER,
               markeredgewidth=0.5, markersize=4.2,
               label=METHOD_LABEL[method])
        for method in METHOD_ORDER
    ]
    method_legend = fig.legend(
        handles=method_handles,
        loc="upper left",
        bbox_to_anchor=(0.125, 0.955),
        ncol=3,
        fontsize=5.4,
        handletextpad=0.35,
        columnspacing=0.9,
    )
    fig.add_artist(method_legend)

    response_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=INK,
               markeredgecolor=PAPER, markeredgewidth=0.5, markersize=4.2,
               label=r"finite response  $s(0.01)$"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor=PAPER,
               markeredgecolor=INK, markeredgewidth=0.9, markersize=4.0,
               label=r"local reference  $r^{L}$"),
        Line2D([0], [0], marker="^", color="none", markerfacecolor=PAPER,
               markeredgecolor=INK, markeredgewidth=0.9, markersize=4.2,
               label=r"endpoint response  $r^{E}$"),
    ]
    fig.legend(
        handles=response_handles,
        loc="upper right",
        bbox_to_anchor=(0.972, 0.955),
        ncol=3,
        fontsize=5.4,
        handletextpad=0.35,
        columnspacing=0.9,
    )

    audit_score_handles = [
        Line2D(
            [0],
            [0],
            marker=SCORE_MARKER[score],
            color="none",
            markerfacecolor=INK,
            markeredgecolor=PAPER,
            markeredgewidth=0.5,
            markersize=3.8,
            label=SCORE_SHORT[score],
        )
        for score in PRIMARY_SCORES
    ]
    fig.legend(
        handles=audit_score_handles,
        loc="lower center",
        bbox_to_anchor=(0.55, 0.012),
        ncol=8,
        fontsize=5.0,
        handlelength=0.8,
        handletextpad=0.25,
        columnspacing=0.55,
    )
    return fig


def save_figure(fig: plt.Figure) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    stem = OUT / "fig5_heldout_expression_utility"
    fig.savefig(stem.with_suffix(".svg"), facecolor=PAPER)
    fig.savefig(stem.with_suffix(".png"), dpi=300, facecolor=PAPER)
    fig.savefig(
        stem.with_suffix(".tiff"),
        dpi=600,
        facecolor=PAPER,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    try:
        fig.savefig(stem.with_suffix(".pdf"), facecolor=PAPER)
    except PermissionError:
        fig.savefig(OUT / "fig5_heldout_expression_utility_latest.pdf", facecolor=PAPER)
    fig.savefig(HERE / "fig5_final_width_preview.png", dpi=300, facecolor=PAPER)


def main() -> None:
    gain, dual = load_data()
    figure = build_figure(gain, dual)
    save_figure(figure)
    plt.close(figure)


if __name__ == "__main__":
    main()
