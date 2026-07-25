"""Create Figure 2: numerical validation of local-response fidelity.

The figure uses frozen WP1--WP3 result tables copied into ``data``.  It
separates reference validation (multi-step convergence and independent
derivative checks) from the biological-unit summary at h=0.01.  No endpoint
transportability result is used in this figure.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import LogFormatterMathtext, NullLocator


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = HERE / "data"
OUT = ROOT / "figures"

STEP_FILE = DATA / "wp1_full_step_convergence_unit.tsv"
REFERENCE_FILE = DATA / "wp1_full_local_reference_summary.tsv"
IMPLICIT_FILE = DATA / "wp2_derivative_cross_validation.tsv"
UOT_INDEPENDENT_FILE = DATA / "uot_independent_derivative_conditions.tsv"
UNIT_FILE = DATA / "wp3_local_fidelity_unit.tsv"
FAMILY_FILE = DATA / "wp3_local_fidelity_family.tsv"

INK = "#25343B"
MUTED = "#68767D"
HAIR = "#AAB4B9"
PAPER = "#FFFFFF"
PANEL_BG = "#FBFCFD"
GATE = "#B6534E"
EXPR = "#447CB2"
SPATIAL = "#80669A"

METHOD_ORDER = ["balanced_ot", "uot", "row_softmax"]
METHOD_LABEL = {
    "balanced_ot": "Balanced OT",
    "uot": "UOT",
    "row_softmax": "Row-softmax",
}
METHOD_SHORT = {
    "balanced_ot": "Balanced",
    "uot": "UOT",
    "row_softmax": "Softmax",
}
METHOD_COLOR = {
    "balanced_ot": "#447CB2",
    "uot": "#2F857C",
    "row_softmax": "#BC7A27",
}
METHOD_PALE = {
    "balanced_ot": "#EDF3F8",
    "uot": "#ECF5F3",
    "row_softmax": "#FBF3E8",
}
CHANNEL_COLOR = {"I_EXPR": EXPR, "I_SPATIAL": SPATIAL}
CHANNEL_LABEL = {"I_EXPR": "expr", "I_SPATIAL": "spatial"}
ARM_STYLE = {
    "R": {
        "linestyle": "-",
        "marker": "o",
        "alpha": 0.62,
        "linewidth": 1.00,
        "markersize": 2.8,
        "filled": True,
        "zorder": 3,
    },
    "N": {
        "linestyle": (0, (3.2, 2.0)),
        "marker": "^",
        "alpha": 1.00,
        "linewidth": 1.35,
        "markersize": 3.35,
        "filled": False,
        "zorder": 4,
    },
}

STEP_GATE = 0.05
LOCAL_THRESHOLDS = {
    "vector_relative_l1_median": 0.10,
    "h001_rmae": 0.10,
    "neighborhood_error_median": 0.15,
}

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7,
        "axes.linewidth": 0.75,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.major.width": 0.65,
        "ytick.major.width": 0.65,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "mathtext.fontset": "dejavusans",
    }
)


def as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().eq("true")


def load_data() -> tuple[pd.DataFrame, ...]:
    step = pd.read_csv(STEP_FILE, sep="\t")
    reference = pd.read_csv(REFERENCE_FILE, sep="\t")
    implicit = pd.read_csv(IMPLICIT_FILE, sep="\t")
    uot_independent = pd.read_csv(UOT_INDEPENDENT_FILE, sep="\t")
    unit = pd.read_csv(UNIT_FILE, sep="\t")
    family = pd.read_csv(FAMILY_FILE, sep="\t")

    unit["gate_pass"] = as_bool(unit["gate_pass"])
    family["family_gate_pass"] = as_bool(family["family_gate_pass"])
    implicit["converged"] = as_bool(implicit["converged"])
    uot_independent["validation_pass"] = as_bool(
        uot_independent["validation_pass"]
    )

    if len(step) != 3000:
        raise ValueError(f"WP1 step table must contain 3000 rows; found {len(step)}")
    if len(reference) != 600:
        raise ValueError(
            f"WP1 reference table must contain 600 rows; found {len(reference)}"
        )
    analytic = reference[
        reference["method"].eq("row_softmax")
        & reference["analytic_relative_l1_median"].notna()
    ].copy()
    if len(analytic) != 200:
        raise ValueError(f"Expected 200 analytic Row-softmax conditions; found {len(analytic)}")
    analytic_pass = (
        (analytic["analytic_relative_l1_median"] <= 0.01)
        & (analytic["analytic_relative_l1_q90"] <= 0.05)
    )
    if not analytic_pass.all():
        raise ValueError("All 200 analytic Row-softmax conditions must pass both frozen error gates")
    if len(implicit) != 104 or not implicit["converged"].all():
        raise ValueError("Balanced-OT implicit validation must contain 104 converged rows")
    implicit_pass = (
        (implicit["global_plan_relative_l1"] <= 0.01)
        & (implicit["row_relative_l1_median"] <= 0.01)
        & (implicit["row_relative_l1_q90"] <= 0.05)
        & (implicit["row_direction_cosine_median"] >= 0.999)
    )
    if not implicit_pass.all():
        raise ValueError("All 104 Balanced-OT implicit conditions must pass the frozen gates")
    if len(uot_independent) != 48:
        raise ValueError(
            "Independent UOT validation must contain 48 arm-channel conditions"
        )
    if uot_independent["independent_unit_id"].nunique() != 4:
        raise ValueError("Independent UOT validation must contain four fixed units")
    if not uot_independent["validation_pass"].all():
        raise ValueError("All independent UOT validation conditions must pass")
    if len(unit) != 252 or unit["independent_unit_id"].nunique() != 21:
        raise ValueError("WP3 unit table must contain 252 rows from 21 independent units")
    if len(family) != 12 or not family["family_gate_pass"].all():
        raise ValueError("WP3 family table must contain 12 passing families")
    if not unit["gate_pass"].all():
        raise ValueError("All 252 unit-family local-fidelity gates must pass")
    if set(step["method"]) != set(METHOD_ORDER):
        raise ValueError("Unexpected method set in WP1 step table")

    plotted = {
        "step": (step, ["h_small", "relative_l1_median"]),
        "analytic": (analytic, ["analytic_relative_l1_median"]),
        "implicit": (implicit, ["row_relative_l1_median"]),
        "uot_independent": (
            uot_independent,
            ["row_relative_l1_median"],
        ),
        "unit": (
            unit,
            [
                "vector_relative_l1_median",
                "h001_rmae",
                "neighborhood_error_median",
            ],
        ),
        "family": (
            family,
            [
                "h001_spearman",
                "h001_top_overlap",
                "direction_cosine_median",
            ],
        ),
    }
    for name, (frame, columns) in plotted.items():
        if frame[columns].isna().any().any():
            raise ValueError(f"{name} contains missing plotted values")
        if (frame[columns] <= 0).any().any() and name != "family":
            raise ValueError(f"{name} contains a non-positive value used on a log axis")
    return step, analytic, implicit, uot_independent, unit, family


def style_axis(ax: plt.Axes, grid_axis: str | None = None) -> None:
    ax.set_facecolor(PANEL_BG)
    ax.spines["left"].set_color(HAIR)
    ax.spines["bottom"].set_color(HAIR)
    ax.tick_params(
        axis="both",
        which="both",
        color=HAIR,
        labelcolor=MUTED,
        labelsize=5.2,
        pad=1.5,
    )
    if grid_axis:
        ax.grid(axis=grid_axis, color="#E4E9EB", lw=0.45, zorder=0)
    ax.set_axisbelow(True)


def step_summary(step: pd.DataFrame) -> pd.DataFrame:
    keys = ["method", "arm", "intervention", "h_small"]
    return (
        step.groupby(keys, dropna=False)["relative_l1_median"]
        .agg(
            median="median",
            q25=lambda x: x.quantile(0.25),
            q75=lambda x: x.quantile(0.75),
            conditions="size",
        )
        .reset_index()
    )


def draw_panel_a(fig: plt.Figure, axes: list[plt.Axes], step: pd.DataFrame) -> None:
    summary = step_summary(step)
    steps = np.array(sorted(summary["h_small"].unique()))

    for index, (ax, method) in enumerate(zip(axes, METHOD_ORDER)):
        style_axis(ax, "y")
        for arm in ["R", "N"]:
            for intervention in ["I_EXPR", "I_SPATIAL"]:
                rows = summary[
                    summary["method"].eq(method)
                    & summary["arm"].eq(arm)
                    & summary["intervention"].eq(intervention)
                ].sort_values("h_small")
                if len(rows) != 5 or not rows["conditions"].eq(50).all():
                    raise ValueError(
                        f"{method}/{arm}/{intervention} must have five steps and 50 conditions per step"
                    )
                color = CHANNEL_COLOR[intervention]
                style = ARM_STYLE[arm]
                x = rows["h_small"].to_numpy(float)
                y = rows["median"].to_numpy(float)
                q25 = rows["q25"].to_numpy(float)
                q75 = rows["q75"].to_numpy(float)
                ax.fill_between(x, q25, q75, color=color, alpha=0.055, lw=0, zorder=1)
                ax.plot(
                    x,
                    y,
                    color=color,
                    lw=style["linewidth"],
                    linestyle=style["linestyle"],
                    marker=style["marker"],
                    markersize=style["markersize"],
                    markerfacecolor=color if style["filled"] else PAPER,
                    markeredgewidth=0.35 if style["filled"] else 0.70,
                    markeredgecolor=PAPER if style["filled"] else color,
                    alpha=style["alpha"],
                    zorder=style["zorder"],
                )

        ax.axhline(STEP_GATE, color=GATE, lw=0.75, linestyle=(0, (3, 2)), zorder=2)
        # The tested steps double successively, so base 2 gives equal visual
        # spacing.  Automatic logarithmic minor ticks are suppressed because
        # they add no tested step and create a false impression of irregular
        # sampling.
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.xaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_minor_locator(NullLocator())
        ax.set_xlim(5.0e-4, 1.2e-2)
        ax.set_ylim(2.0e-5, 8.0e-2)
        ax.set_xticks(steps)
        ax.set_xticklabels(
            ["6.25e−4", "1.25e−3", "2.5e−3", "5e−3", "1e−2"],
            rotation=34,
            ha="right",
            fontsize=5.0,
        )
        ax.set_title(
            METHOD_LABEL[method],
            fontsize=6.6,
            fontweight="bold",
            color=METHOD_COLOR[method],
            pad=4,
        )
        ax.set_xlabel(r"smaller step  $h_s$", fontsize=5.4, labelpad=1)
        if index == 0:
            ax.set_ylabel("adjacent-step relative L1", fontsize=5.4, labelpad=2)
            ax.yaxis.set_major_formatter(LogFormatterMathtext())
        else:
            ax.set_yticklabels([])
        ax.text(
            0.98,
            STEP_GATE,
            "smallest-step gate 0.05",
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="bottom",
            fontsize=5.0,
            color=GATE,
        )

    fig.text(0.018, 0.965, "a", fontsize=8.5, fontweight="bold", color=INK, va="center")
    fig.text(
        0.050,
        0.965,
        "Multi-step convergence establishes a stable local reference",
        fontsize=8.2,
        fontweight="bold",
        color=INK,
        va="center",
    )
    handles = []
    labels = []
    for arm in ["R", "N"]:
        for intervention in ["I_EXPR", "I_SPATIAL"]:
            handles.append(
                Line2D(
                    [0],
                    [0],
                    color=CHANNEL_COLOR[intervention],
                    lw=ARM_STYLE[arm]["linewidth"],
                    linestyle=ARM_STYLE[arm]["linestyle"],
                    marker=ARM_STYLE[arm]["marker"],
                    markersize=3.2 if arm == "R" else 3.7,
                    markerfacecolor=(
                        CHANNEL_COLOR[intervention] if ARM_STYLE[arm]["filled"] else PAPER
                    ),
                    markeredgecolor=(
                        PAPER if ARM_STYLE[arm]["filled"] else CHANNEL_COLOR[intervention]
                    ),
                    markeredgewidth=0.45 if arm == "R" else 0.75,
                    alpha=ARM_STYLE[arm]["alpha"],
                )
            )
            labels.append(f"Arm {arm} — {CHANNEL_LABEL[intervention]}")
    fig.legend(
        handles,
        labels,
        ncol=4,
        loc="upper right",
        bbox_to_anchor=(0.987, 0.979),
        fontsize=5.1,
        handlelength=1.8,
        handletextpad=0.35,
        columnspacing=0.8,
    )


def deterministic_jitter(n: int, amplitude: float) -> np.ndarray:
    index = np.arange(n, dtype=float)
    return amplitude * np.sin(index * 2.399963229728653)


def draw_panel_b(
    ax: plt.Axes,
    analytic: pd.DataFrame,
    implicit: pd.DataFrame,
    uot_independent: pd.DataFrame,
) -> None:
    style_axis(ax, "y")
    groups = [
        (
            "Row-softmax\nanalytic",
            analytic["analytic_relative_l1_median"].to_numpy(float),
            METHOD_COLOR["row_softmax"],
        ),
        (
            "Balanced OT\nimplicit",
            implicit["row_relative_l1_median"].to_numpy(float),
            METHOD_COLOR["balanced_ot"],
        ),
        (
            "UOT\nindependent",
            uot_independent["row_relative_l1_median"].to_numpy(float),
            METHOD_COLOR["uot"],
        ),
    ]
    for x, (label, values, color) in enumerate(groups):
        jitter = deterministic_jitter(len(values), 0.15)
        ax.scatter(
            np.full(len(values), x) + jitter,
            values,
            s=10,
            facecolor=color,
            edgecolor=PAPER,
            linewidth=0.30,
            alpha=0.50,
            zorder=3,
        )
        median = float(np.median(values))
        ax.plot(
            [x - 0.18, x + 0.18],
            [median, median],
            color=INK,
            lw=1.65,
            solid_capstyle="round",
            zorder=5,
        )
        ax.text(
            x,
            0.91,
            f"{len(values)}/{len(values)} pass",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=5.0,
            color=color,
            fontweight="bold",
        )

    ax.set_yscale("log")
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_ylim(1.0e-9, 1.0e-6)
    ax.set_xlim(-0.45, 2.45)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels([g[0] for g in groups], fontsize=5.3)
    ax.set_ylabel("per-condition row-plan relative L1", fontsize=5.4, labelpad=2)
    ax.yaxis.set_major_formatter(LogFormatterMathtext())
    ax.text(
        0.98,
        0.98,
        r"prespecified gate $=10^{-2}$ (off-scale)",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=5.0,
        color=GATE,
    )
    ax.text(
        -0.18,
        1.14,
        "b",
        transform=ax.transAxes,
        fontsize=8.5,
        fontweight="bold",
        color=INK,
        va="center",
    )
    ax.text(
        0.0,
        1.14,
        "Independent derivative checks agree",
        transform=ax.transAxes,
        fontsize=7.4,
        fontweight="bold",
        color=INK,
        va="center",
    )


def family_order() -> list[tuple[str, str, str]]:
    return [
        (method, arm, intervention)
        for method in METHOD_ORDER
        for arm, intervention in [
            ("R", "I_EXPR"),
            ("R", "I_SPATIAL"),
            ("N", "I_EXPR"),
            ("N", "I_SPATIAL"),
        ]
    ]


def draw_local_metric_axis(
    ax: plt.Axes,
    unit: pd.DataFrame,
    family: pd.DataFrame,
    metric: str,
    title: str,
    threshold: float,
    show_ylabels: bool,
) -> None:
    style_axis(ax, "x")
    order = family_order()
    positions = np.array([0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13], dtype=float)

    for method_index, method in enumerate(METHOD_ORDER):
        start = positions[method_index * 4] - 0.45
        end = positions[method_index * 4 + 3] + 0.45
        ax.axhspan(start, end, color=METHOD_PALE[method], alpha=0.72, zorder=0)

    for position, (method, arm, intervention) in zip(positions, order):
        rows = unit[
            unit["method"].eq(method)
            & unit["arm"].eq(arm)
            & unit["intervention"].eq(intervention)
        ].sort_values("independent_unit_id")
        fam = family[
            family["method"].eq(method)
            & family["arm"].eq(arm)
            & family["intervention"].eq(intervention)
        ]
        if len(rows) != 21 or len(fam) != 1:
            raise ValueError(f"Expected 21 units and one family row for {method}/{arm}/{intervention}")
        values = rows[metric].to_numpy(float)
        y = position + deterministic_jitter(len(values), 0.13)
        color = METHOD_COLOR[method]
        marker = ARM_STYLE[arm]["marker"]
        ax.scatter(
            values,
            y,
            s=8,
            marker=marker,
            facecolor=color,
            edgecolor=PAPER,
            linewidth=0.25,
            alpha=0.42,
            zorder=3,
        )
        ax.scatter(
            float(fam.iloc[0][metric]),
            position,
            s=25,
            marker="D",
            facecolor=color,
            edgecolor=INK,
            linewidth=0.55,
            zorder=5,
        )

    ax.axvline(threshold, color=GATE, lw=0.75, linestyle=(0, (3, 2)), zorder=2)
    ax.set_xscale("log")
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlim(1.0e-4, 2.2e-1)
    ax.set_ylim(13.65, -0.65)
    ax.set_yticks(positions)
    if show_ylabels:
        labels = [f"{a} — {CHANNEL_LABEL[i]}" for _, a, i in order]
        ax.set_yticklabels(labels, fontsize=5.0)
    else:
        ax.set_yticklabels([])
        ax.tick_params(axis="y", length=0)
    ax.xaxis.set_major_formatter(LogFormatterMathtext())
    ax.set_title(title, fontsize=5.8, fontweight="bold", color=INK, pad=4)
    ax.text(
        threshold,
        -0.40,
        f"gate = {threshold:.2f}",
        ha="right",
        va="bottom",
        fontsize=5.0,
        color=GATE,
    )


def draw_panel_c(
    fig: plt.Figure,
    axes: list[plt.Axes],
    unit: pd.DataFrame,
    family: pd.DataFrame,
) -> None:
    specs = [
        ("vector_relative_l1_median", "Row-plan rel. L1", 0.10),
        ("h001_rmae", "Scalar rMAE", 0.10),
        ("neighborhood_error_median", "Neighborhood error", 0.15),
    ]
    for index, (ax, (metric, title, threshold)) in enumerate(zip(axes, specs)):
        draw_local_metric_axis(
            ax,
            unit,
            family,
            metric,
            title,
            threshold,
            show_ylabels=index == 0,
        )

    axes[0].text(
        -0.19,
        1.14,
        "c",
        transform=axes[0].transAxes,
        fontsize=8.5,
        fontweight="bold",
        color=INK,
        va="center",
    )
    axes[0].text(
        0.0,
        1.14,
        r"$h=0.01$ passes all 12 local-fidelity families",
        transform=axes[0].transAxes,
        fontsize=7.4,
        fontweight="bold",
        color=INK,
        va="center",
    )
    median_handle = Line2D(
        [0],
        [0],
        marker="D",
        linestyle="none",
        markerfacecolor="#8A969C",
        markeredgecolor=INK,
        markeredgewidth=0.55,
        markersize=4.2,
        label="family median",
    )
    unit_handle = Line2D(
        [0],
        [0],
        marker="o",
        linestyle="none",
        markerfacecolor="#8A969C",
        markeredgecolor=PAPER,
        markersize=3.2,
        alpha=0.5,
        label="independent unit",
    )
    method_handles = [
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="none",
            markerfacecolor=METHOD_COLOR[method],
            markeredgecolor=METHOD_COLOR[method],
            markersize=3.8,
            label=METHOD_LABEL[method],
        )
        for method in METHOD_ORDER
    ]
    axes[0].legend(
        handles=method_handles,
        loc="lower left",
        bbox_to_anchor=(0.0, -0.235),
        ncol=3,
        fontsize=5.0,
        handletextpad=0.25,
        columnspacing=0.75,
        borderaxespad=0,
    )
    axes[-1].legend(
        handles=[unit_handle, median_handle],
        loc="lower right",
        bbox_to_anchor=(1.0, -0.235),
        ncol=2,
        fontsize=5.0,
        handletextpad=0.3,
        columnspacing=0.8,
        borderaxespad=0,
    )


def save_figure(fig: plt.Figure) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    stem = OUT / "fig2_internal_fidelity"
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
        fig.savefig(
            OUT / "fig2_internal_fidelity_latest.pdf",
            facecolor=PAPER,
        )


def main() -> None:
    step, analytic, implicit, uot_independent, unit, family = load_data()

    fig = plt.figure(figsize=(7.0, 5.55), facecolor=PAPER)
    outer = fig.add_gridspec(
        2,
        1,
        height_ratios=[1.0, 1.16],
        left=0.072,
        right=0.985,
        top=0.865,
        bottom=0.105,
        hspace=0.47,
    )
    top_grid = outer[0, 0].subgridspec(1, 3, wspace=0.24)
    axes_a = [fig.add_subplot(top_grid[0, index]) for index in range(3)]

    bottom = outer[1, 0].subgridspec(1, 2, width_ratios=[0.78, 1.82], wspace=0.33)
    ax_b = fig.add_subplot(bottom[0, 0])
    c_grid = bottom[0, 1].subgridspec(1, 3, wspace=0.17)
    axes_c = [fig.add_subplot(c_grid[0, index]) for index in range(3)]

    draw_panel_a(fig, axes_a, step)
    draw_panel_b(ax_b, analytic, implicit, uot_independent)
    draw_panel_c(fig, axes_c, unit, family)
    save_figure(fig)
    plt.close(fig)


if __name__ == "__main__":
    main()
