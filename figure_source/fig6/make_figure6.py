"""Create Figure 6: controlled utility, selection consequence and witness boundary.

The main panel shows the patient-level gain in retained candidate-positive-pair
precision when top-two probability margin replaces maximum coupling probability
at fixed coverage. Supporting panels preserve the registered controlled-error
diagnostics and the three-donor anatomical-witness transfer boundary. Cohorts
and independent units are never pooled.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = HERE / "data"
OUT = ROOT / "figures"

PAIR_UNIT_FILE = DATA / "positive_pair_quality_unit.tsv"
PAIR_SUMMARY_FILE = DATA / "positive_pair_quality_summary.tsv"
CORR_FILE = DATA / "wp10_her2st_correspondence_truth_direction.tsv"
CROP_FILE = DATA / "wp10_crop_missingness_utility_direction.tsv"
MANUAL_FILE = DATA / "p1_manual_layer_validation.csv"

INK = "#25343B"
MUTED = "#68767D"
HAIR = "#AAB4B9"
GRID = "#E2E8EA"
PAPER = "#FFFFFF"
ZERO = "#5F6C72"

METHOD_ORDER = ["balanced_ot", "uot", "row_softmax"]
METHOD_LABEL = {
    "balanced_ot": "Balanced OT",
    "uot": "UOT",
    "row_softmax": "Row-softmax",
}
METHOD_ROW = {"balanced_ot": "Balanced", "uot": "UOT", "row_softmax": "Row-softmax"}
METHOD_COLOR = {
    "balanced_ot": "#447CB2",
    "uot": "#2F857C",
    "row_softmax": "#BC7A27",
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
SCORE_SHORT = ["cost", "bary.", "entropy", r"$s(0.01)$", "low-p", "margin", "boundary", "mass"]

DONOR_ORDER = ["Br5292", "Br5595", "Br8100"]
DONOR_MARKER = {"Br5292": "o", "Br5595": "s", "Br8100": "^"}
WITNESS_ORDER = ["heldout_loss", "label_error_shared_closed_set"]
WITNESS_LABEL = ["Held-out\nexpression", "Manual\nlayer"]

EXPECTED_PAIR = {
    (0.8, "balanced_ot"): (0.12731668009669628, 0.09451219512195119, 0.15047393364928913),
    (0.8, "uot"): (0.11043360433604338, 0.08040935672514615, 0.127906976744186),
    (0.8, "row_softmax"): (0.06305595408895265, 0.03801169590643272, 0.12015503875968991),
    (0.9, "balanced_ot"): (0.05928361639637339, 0.027173913043478326, 0.06896551724137928),
    (0.9, "uot"): (0.04927792451222632, 0.04285714285714276, 0.06910569105691056),
    (0.9, "row_softmax"): (0.030652070442646367, 0.015789473684210464, 0.05689655172413799),
}
EXPECTED_MANUAL = {
    "balanced_ot": (-0.126822, 0.474172),
    "uot": (-0.176433, 0.750189),
    "row_softmax": (-0.065955, -0.782942),
}

AUROC_CMAP = LinearSegmentedColormap.from_list(
    "diagnostic_auroc",
    [(0.0, "#E8C3BE"), (0.5, "#F7F7F4"), (1.0, "#2F857C")],
)

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


def _read_tsv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, sep="\t")


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load frozen outputs and enforce independent-unit aggregation."""
    pair_unit = _read_tsv(PAIR_UNIT_FILE)
    pair_summary = _read_tsv(PAIR_SUMMARY_FILE)
    correspondence = _read_tsv(CORR_FILE)
    crop = _read_tsv(CROP_FILE)
    manual_source = pd.read_csv(MANUAL_FILE)

    if len(pair_unit) != 8 * 3 or pair_unit.duplicated(["independent_unit_id", "method"]).any():
        raise ValueError("Fixed-budget panel requires 24 unique patient-method rows")
    if set(pair_unit["method"]) != set(METHOD_ORDER):
        raise ValueError("Fixed-budget method set changed")
    if len(pair_summary) != 2 * 3 or pair_summary.duplicated(["coverage", "method"]).any():
        raise ValueError("Fixed-budget summary requires six coverage-method rows")

    for key, expected in EXPECTED_PAIR.items():
        coverage, method = key
        row = pair_summary[
            np.isclose(pair_summary["coverage"], coverage)
            & pair_summary["method"].eq(method)
        ].iloc[0]
        observed = (
            float(row["median_difference"]),
            float(row["bootstrap_median_ci95_low"]),
            float(row["bootstrap_median_ci95_high"]),
        )
        if not np.allclose(observed, expected, atol=5e-12):
            raise ValueError(f"Locked fixed-budget summary changed for {key}")
        if int(row["n_independent_units"]) != 8 or int(row["units_improved"]) != 8:
            raise ValueError(f"Independent-unit count changed for {key}")
        if not np.isclose(float(row["wilcoxon_p_holm"]), 0.0234375, atol=5e-12):
            raise ValueError(f"Holm-adjusted P value changed for {key}")

    top1 = correspondence[
        correspondence["metric"].eq("top1_error_detection")
        & correspondence["score"].isin(PRIMARY_SCORES)
    ].copy()
    top1_unit = (
        top1.groupby(["independent_unit_id", "method", "score"], as_index=False)["auroc"]
        .mean()
    )
    crop_unit = (
        crop[crop["score"].isin(PRIMARY_SCORES)]
        .groupby(["independent_unit_id", "method", "score"], as_index=False)["auroc"]
        .mean()
    )
    expected_diag = 8 * 3 * len(PRIMARY_SCORES)
    if len(top1_unit) != expected_diag or len(crop_unit) != expected_diag:
        raise ValueError(
            f"Expected {expected_diag} rows per controlled diagnostic; "
            f"found top1={len(top1_unit)}, crop={len(crop_unit)}"
        )

    manual = manual_source[
        manual_source["arm"].eq("R")
        & manual_source["score"].eq("exact_I_EXPR")
        & manual_source["grid_role"].eq("baseline")
        & np.isclose(manual_source["epsilon"], 0.25)
        & manual_source["witness"].isin(WITNESS_ORDER)
        & manual_source["method"].isin(METHOD_ORDER)
        & (
            manual_source["method"].ne("uot")
            | np.isclose(manual_source["tau"].fillna(-1), 2.0)
        )
    ].copy()
    manual["donor"] = manual["independent_unit_id"].str.split("::").str[-1]
    if len(manual) != 18 or manual.duplicated(["donor", "method", "witness"]).any():
        raise ValueError("Manual-layer grid must contain 18 unique donor-method-witness rows")

    for method in METHOD_ORDER:
        manual_med = float(
            manual.loc[
                manual["method"].eq(method)
                & manual["witness"].eq("label_error_shared_closed_set"),
                "relative_fixed_qc_gain",
            ].median()
        )
        heldout_med = float(
            manual.loc[
                manual["method"].eq(method) & manual["witness"].eq("heldout_loss"),
                "relative_fixed_qc_gain",
            ].median()
        )
        if not np.allclose((manual_med, heldout_med), EXPECTED_MANUAL[method], atol=5e-7):
            raise ValueError(f"Locked witness medians changed for {method}")

    return pair_unit, pair_summary, top1_unit, crop_unit, manual


def style_axis(ax: plt.Axes, grid_axis: str = "x") -> None:
    ax.spines["left"].set_color(HAIR)
    ax.spines["bottom"].set_color(HAIR)
    ax.tick_params(colors=MUTED, labelsize=5.3, pad=1.5)
    ax.set_axisbelow(True)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.55)


def panel_heading(fig: plt.Figure, ax: plt.Axes, letter: str, title: str) -> None:
    box = ax.get_position()
    y = box.y1 + 0.043
    fig.text(box.x0 - 0.040, y, letter, fontsize=8.4, fontweight="bold", color=INK)
    fig.text(box.x0, y, title, fontsize=7.8, fontweight="bold", color=INK)


def draw_fixed_budget(
    ax: plt.Axes,
    unit: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    """Forest-style patient effects with frozen bootstrap median intervals."""
    base_y = {"balanced_ot": 2.0, "uot": 1.0, "row_softmax": 0.0}
    coverage_spec = {
        0.8: {"offset": 0.13, "marker": "o", "filled": True, "label": "80% primary"},
        0.9: {"offset": -0.13, "marker": "^", "filled": False, "label": "90% sensitivity"},
    }
    jitter = np.linspace(-0.045, 0.045, 8)

    ax.axvline(0.0, color=ZERO, lw=0.8, ls=(0, (3, 2)), zorder=1)
    for method in METHOD_ORDER:
        colour = METHOD_COLOR[method]
        rows = unit[unit["method"].eq(method)].sort_values("independent_unit_id")
        for coverage, spec in coverage_spec.items():
            suffix = "80pct" if np.isclose(coverage, 0.8) else "90pct"
            values = rows[f"margin_minus_max_precision_at_{suffix}"].to_numpy(float)
            y = base_y[method] + spec["offset"]
            face = colour if spec["filled"] else PAPER
            ax.scatter(
                values,
                y + jitter,
                s=12,
                marker=spec["marker"],
                facecolor=face,
                edgecolor=colour,
                linewidth=0.55,
                alpha=0.55,
                zorder=3,
            )
            row = summary[
                np.isclose(summary["coverage"], coverage)
                & summary["method"].eq(method)
            ].iloc[0]
            median = float(row["median_difference"])
            lo = float(row["bootstrap_median_ci95_low"])
            hi = float(row["bootstrap_median_ci95_high"])
            ax.plot([lo, hi], [y, y], color=colour, lw=2.0, solid_capstyle="round", zorder=4)
            ax.scatter(
                [median],
                [y],
                s=34,
                marker=spec["marker"],
                facecolor=face,
                edgecolor=colour,
                linewidth=1.0,
                zorder=5,
            )

    ax.set_xlim(-0.008, 0.182)
    ax.set_ylim(-0.42, 2.42)
    ax.set_xticks([0.00, 0.05, 0.10, 0.15])
    ax.set_yticks([2.0, 1.0, 0.0], [METHOD_LABEL[m] for m in METHOD_ORDER])
    for tick, method in zip(ax.get_yticklabels(), METHOD_ORDER):
        tick.set_color(METHOD_COLOR[method])
        tick.set_fontweight("bold")
    ax.set_xlabel(
        r"precision gain: top-2 margin $-$ maximum probability",
        fontsize=5.8,
        color=INK,
        labelpad=3,
    )
    style_axis(ax, "x")
    coverage_handles = [
        Line2D(
            [0], [0], marker=coverage_spec[c]["marker"], linestyle="none",
            markersize=4.3, markerfacecolor=INK if coverage_spec[c]["filled"] else PAPER,
            markeredgecolor=INK, markeredgewidth=0.8, label=coverage_spec[c]["label"],
        )
        for c in (0.8, 0.9)
    ]
    ax.legend(
        handles=coverage_handles,
        loc="upper left",
        bbox_to_anchor=(0.0, 1.02),
        ncol=2,
        fontsize=5.1,
        handletextpad=0.35,
        columnspacing=0.8,
        borderaxespad=0,
    )


def diagnostic_matrix(unit_data: pd.DataFrame) -> np.ndarray:
    summary = unit_data.groupby(["method", "score"])["auroc"].median()
    return np.array(
        [[summary.loc[(method, score)] for score in PRIMARY_SCORES] for method in METHOD_ORDER],
        dtype=float,
    )


def draw_heatmap(
    ax: plt.Axes,
    matrix: np.ndarray,
    title: str,
    show_x: bool,
    show_y: bool,
) -> mpl.image.AxesImage:
    norm = TwoSlopeNorm(vmin=0.0, vcenter=0.5, vmax=1.0)
    image = ax.imshow(matrix, cmap=AUROC_CMAP, norm=norm, aspect="auto", interpolation="none")
    ax.set_title(title, fontsize=5.9, fontweight="bold", color=INK, pad=3.0)
    ax.set_yticks(range(3))
    if show_y:
        ax.set_yticklabels([METHOD_ROW[m] for m in METHOD_ORDER], fontsize=5.0)
        for tick, method in zip(ax.get_yticklabels(), METHOD_ORDER):
            tick.set_color(METHOD_COLOR[method])
    else:
        ax.set_yticklabels([])
    ax.set_xticks(range(len(PRIMARY_SCORES)))
    if show_x:
        ax.set_xticklabels(SCORE_SHORT, rotation=42, ha="right", rotation_mode="anchor", fontsize=5.0)
    else:
        ax.set_xticklabels([])
    ax.tick_params(length=0, pad=1.5)
    for row in range(matrix.shape[0]):
        best = int(np.nanargmax(matrix[row]))
        ax.add_patch(
            mpl.patches.Rectangle(
                (best - 0.5, row - 0.5), 1, 1,
                fill=False, edgecolor=INK, linewidth=0.75, zorder=4,
            )
        )
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            text_colour = PAPER if value >= 0.78 or value <= 0.18 else INK
            ax.text(
                col, row, f"{value:.2f}", ha="center", va="center",
                fontsize=5.0, color=text_colour,
                fontweight="bold" if col == best else "normal",
            )
    for spine in ax.spines.values():
        spine.set_visible(False)
    return image


def draw_witness_pair(ax: plt.Axes, data: pd.DataFrame, method: str, show_y: bool) -> None:
    subset = data[data["method"].eq(method)].copy()
    colour = METHOD_COLOR[method]
    ax.axhspan(-1.05, 0.0, color="#FAF1EF", zorder=0)
    ax.axhspan(0.0, 0.90, color="#EDF5F2", zorder=0)
    ax.axhline(0.0, color=ZERO, lw=0.70, zorder=1)
    for donor in DONOR_ORDER:
        donor_rows = subset[subset["donor"].eq(donor)].set_index("witness")
        values = [float(donor_rows.loc[w, "relative_fixed_qc_gain"]) for w in WITNESS_ORDER]
        ax.plot([0, 1], values, color=HAIR, lw=0.75, zorder=2)
        ax.scatter(
            [0, 1], values, marker=DONOR_MARKER[donor], s=22,
            facecolor=PAPER, edgecolor=colour, linewidth=0.8, zorder=4,
        )
    medians = [
        float(subset.loc[subset["witness"].eq(w), "relative_fixed_qc_gain"].median())
        for w in WITNESS_ORDER
    ]
    ax.plot([0, 1], medians, color=colour, lw=1.9, zorder=3)
    ax.scatter(
        [0, 1], medians, marker="D", s=28, facecolor=colour,
        edgecolor=PAPER, linewidth=0.55, zorder=5,
    )
    ax.set_xlim(-0.22, 1.22)
    ax.set_ylim(-1.05, 0.90)
    ax.set_xticks([0, 1], WITNESS_LABEL)
    ax.set_yticks([-1.0, -0.5, 0.0, 0.5])
    ax.set_title(METHOD_LABEL[method], fontsize=6.0, fontweight="bold", color=colour, pad=4)
    if show_y:
        ax.set_ylabel(r"gain versus fixed-QC  ($\Delta$NEX)", fontsize=5.7, color=INK, labelpad=3)
    else:
        ax.set_yticklabels([])
        ax.tick_params(axis="y", length=0)
    style_axis(ax, "y")


def build_figure() -> plt.Figure:
    pair_unit, pair_summary, top1_unit, crop_unit, manual = load_data()
    fig = plt.figure(figsize=(7.0, 5.15), facecolor=PAPER)
    outer = fig.add_gridspec(
        2, 1,
        height_ratios=[1.02, 0.86],
        left=0.095,
        right=0.975,
        top=0.885,
        bottom=0.105,
        hspace=0.53,
    )

    upper = outer[0].subgridspec(1, 2, width_ratios=[5.3, 6.7], wspace=0.32)
    ax_a = fig.add_subplot(upper[0, 0])
    draw_fixed_budget(ax_a, pair_unit, pair_summary)

    heat_grid = upper[0, 1].subgridspec(
        2, 2, width_ratios=[24.0, 0.85], hspace=0.32, wspace=0.16
    )
    ax_b1 = fig.add_subplot(heat_grid[0, 0])
    ax_b2 = fig.add_subplot(heat_grid[1, 0])
    cax_b = fig.add_subplot(heat_grid[:, 1])
    heat_image = draw_heatmap(
        ax_b1, diagnostic_matrix(top1_unit), "top-1 mismatch", False, True
    )
    draw_heatmap(ax_b2, diagnostic_matrix(crop_unit), "crop missingness", True, False)
    colourbar = fig.colorbar(heat_image, cax=cax_b, orientation="vertical")
    colourbar.set_ticks([0.0, 0.5, 1.0])
    colourbar.set_ticklabels(["0", "0.5", "1.0"])
    colourbar.ax.tick_params(labelsize=5.0, colors=MUTED, width=0.55, length=2.2, pad=1.6)
    colourbar.set_label("AUROC", fontsize=5.4, color=INK, labelpad=2.5)
    colourbar.outline.set_edgecolor(HAIR)
    colourbar.outline.set_linewidth(0.55)

    lower = outer[1].subgridspec(1, 3, wspace=0.22)
    axes_c = [fig.add_subplot(lower[0, i]) for i in range(3)]
    for index, (ax, method) in enumerate(zip(axes_c, METHOD_ORDER)):
        draw_witness_pair(ax, manual, method, show_y=index == 0)

    panel_heading(fig, ax_a, "a", "Fixed-budget candidate-pair precision")
    panel_heading(fig, ax_b1, "b", "Controlled diagnostic-score AUROC")
    panel_heading(fig, axes_c[0], "c", r"Held-out-expression versus manual-layer $\Delta$NEX")

    donor_handles = [
        Line2D(
            [0], [0], marker=DONOR_MARKER[d], linestyle="none", markersize=3.8,
            markerfacecolor=PAPER, markeredgecolor=INK, markeredgewidth=0.7, label=d,
        )
        for d in DONOR_ORDER
    ]
    fig.legend(
        handles=donor_handles,
        loc="upper right",
        bbox_to_anchor=(0.975, 0.467),
        ncol=3,
        fontsize=5.0,
        handletextpad=0.3,
        columnspacing=0.8,
    )
    return fig


def save_figure(fig: plt.Figure) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    stem = OUT / "fig6_manual_layer_negative_transfer"
    fig.savefig(stem.with_suffix(".svg"), facecolor=PAPER)
    fig.savefig(stem.with_suffix(".png"), dpi=300, facecolor=PAPER)
    fig.savefig(
        stem.with_suffix(".tiff"), dpi=600, facecolor=PAPER,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    fig.savefig(HERE / "fig6_final_width_preview.png", dpi=300, facecolor=PAPER)
    fig.savefig(stem.with_suffix(".pdf"), facecolor=PAPER)


if __name__ == "__main__":
    figure = build_figure()
    save_figure(figure)
    plt.close(figure)
    print(f"Saved Figure 6 to {OUT}")
