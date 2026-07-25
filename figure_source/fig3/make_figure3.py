"""Create Figure 3: local-to-endpoint transportability and path dynamics.

The figure reads the frozen WP4 independent-unit table directly.  It reports
the second module of finite-intervention local-explanation fidelity: whether a
high-accuracy local response can be transported to the complete intervention
endpoint.  It does not reuse the old Arm-N failure framing and does not treat
the 21 independent units as pooled spot-level replicates.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = HERE / "data"
OUT = ROOT / "figures"
SOURCE_FILE = DATA / "wp4_path_geometry_unit.tsv"

INK = "#25343B"
MUTED = "#68767D"
HAIR = "#AAB4B9"
GRID = "#DCE3E6"
PAPER = "#FFFFFF"
PANEL_BG = "#FCFDFD"
REFERENCE = "#8A969C"

METHOD_ORDER = ["balanced_ot", "uot", "row_softmax"]
METHOD_LABEL = {
    "balanced_ot": "Balanced OT",
    "uot": "UOT",
    "row_softmax": "Row-softmax",
}
METHOD_COLOR = {
    "balanced_ot": "#447CB2",
    "uot": "#2F857C",
    "row_softmax": "#BC7A27",
}
METHOD_OFFSET = {
    "balanced_ot": 0.18,
    "uot": 0.00,
    "row_softmax": -0.18,
}

CONDITIONS = [
    ("I_EXPR", "R", "R — expr", 3.0),
    ("I_SPATIAL", "R", "R — spatial", 2.0),
    ("I_EXPR", "N", "N — expr", 1.0),
    ("I_SPATIAL", "N", "N — spatial", 0.0),
]

METRICS = {
    "rho": {
        "column": "local_to_endpoint_spearman",
        "title": "Endpoint rank transportability",
        "xlabel": r"Spearman $\rho$  (higher is better)",
        "xlim": (0.15, 1.015),
        "xticks": [0.2, 0.4, 0.6, 0.8, 1.0],
    },
    "rmae": {
        "column": "local_to_endpoint_rmae",
        "title": "Endpoint magnitude error",
        "xlabel": "rMAE  (lower is better)",
        "xlim": (0.0, 0.45),
        "xticks": [0.0, 0.1, 0.2, 0.3, 0.4],
    },
    "eta": {
        "column": "path_eta_median",
        "title": "Paths remain nearly direct",
        "xlabel": r"path directness $\eta$",
        "xlim": (0.93, 1.005),
        "xticks": [0.94, 0.96, 0.98, 1.00],
        "reference": 1.0,
    },
    "kappa": {
        "column": "path_kappa_median",
        "title": "Path speed changes by condition",
        "xlabel": r"late/early speed ratio $\kappa$",
        "xlim": (0.60, 2.25),
        "xticks": [0.625, 1.0, 2.0],
        "xticklabels": ["0.625", "1", "2"],
        "reference": 1.0,
        "log2": True,
    },
}


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7,
        "axes.linewidth": 0.75,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.facecolor": PANEL_BG,
        "xtick.major.width": 0.65,
        "ytick.major.width": 0.65,
        "xtick.major.size": 2.5,
        "ytick.major.size": 0,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "mathtext.fontset": "dejavusans",
    }
)


def load_data() -> pd.DataFrame:
    data = pd.read_csv(SOURCE_FILE, sep="\t")
    required = {
        "independent_unit_id",
        "method",
        "arm",
        "intervention",
        "local_to_endpoint_spearman",
        "local_to_endpoint_rmae",
        "path_eta_median",
        "path_kappa_median",
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"WP4 source table is missing columns: {sorted(missing)}")

    if len(data) != 252:
        raise ValueError(f"WP4 unit table must contain 252 rows; found {len(data)}")
    if data["independent_unit_id"].nunique() != 21:
        raise ValueError("WP4 unit table must contain 21 independent units")
    if set(data["method"]) != set(METHOD_ORDER):
        raise ValueError("Unexpected method set in WP4 unit table")
    if set(data["arm"]) != {"R", "N"}:
        raise ValueError("WP4 unit table must contain Arm R and Arm N")
    if set(data["intervention"]) != {"I_EXPR", "I_SPATIAL"}:
        raise ValueError("WP4 unit table must contain expression and spatial interventions")

    group_sizes = data.groupby(["method", "arm", "intervention"]).size()
    if len(group_sizes) != 12 or not (group_sizes == 21).all():
        raise ValueError("Each of the 12 method-arm-channel families must contain 21 units")

    numeric_columns = [spec["column"] for spec in METRICS.values()]
    if data[numeric_columns].isna().any().any():
        raise ValueError("All plotted WP4 values must be estimable")
    if not data["local_to_endpoint_spearman"].between(-1, 1).all():
        raise ValueError("Spearman values fall outside [-1, 1]")
    if not data["local_to_endpoint_rmae"].ge(0).all():
        raise ValueError("rMAE values must be non-negative")
    if not data["path_eta_median"].between(0, 1).all():
        raise ValueError("Path directness eta must lie in [0, 1]")
    if not data["path_kappa_median"].gt(0).all():
        raise ValueError("Path speed ratio kappa must be strictly positive")
    return data


def style_axis(ax: plt.Axes) -> None:
    ax.spines["left"].set_color(HAIR)
    ax.spines["bottom"].set_color(HAIR)
    ax.tick_params(axis="x", colors=MUTED, labelsize=6.2)
    ax.tick_params(axis="y", colors=INK, labelsize=6.3)
    ax.grid(axis="x", color=GRID, linewidth=0.55, alpha=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.axhline(1.5, color=GRID, linewidth=0.75, zorder=0)


def panel_label(ax: plt.Axes, letter: str, title: str) -> None:
    ax.text(
        -0.13,
        1.105,
        letter,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.2,
        fontweight="bold",
        color=INK,
        clip_on=False,
    )
    ax.set_title(
        title,
        loc="left",
        fontsize=7.5,
        fontweight="bold",
        color=INK,
        pad=9,
    )


def draw_metric_panel(
    ax: plt.Axes,
    data: pd.DataFrame,
    metric_key: str,
    letter: str,
    show_ylabels: bool,
) -> None:
    spec = METRICS[metric_key]
    column = spec["column"]
    style_axis(ax)

    if spec.get("log2", False):
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_locator(FixedLocator(spec["xticks"]))
        ax.xaxis.set_major_formatter(FixedFormatter(spec["xticklabels"]))
        ax.xaxis.set_minor_locator(NullLocator())
    else:
        ax.set_xticks(spec["xticks"])

    for intervention, arm, _label, base_y in CONDITIONS:
        for method in METHOD_ORDER:
            subset = data[
                data["intervention"].eq(intervention)
                & data["arm"].eq(arm)
                & data["method"].eq(method)
            ]
            values = subset[column].to_numpy(dtype=float)
            y = base_y + METHOD_OFFSET[method]
            colour = METHOD_COLOR[method]

            ax.scatter(
                values,
                np.full(values.shape, y),
                s=7.5,
                c=colour,
                edgecolors="none",
                alpha=0.28,
                zorder=2,
                clip_on=True,
            )
            q25, median, q75 = np.quantile(values, [0.25, 0.50, 0.75])
            ax.plot(
                [q25, q75],
                [y, y],
                color=colour,
                linewidth=2.3,
                solid_capstyle="round",
                zorder=3,
            )
            ax.scatter(
                [median],
                [y],
                s=24,
                c=colour,
                edgecolors=PAPER,
                linewidths=0.65,
                zorder=4,
            )

    if "reference" in spec:
        ax.axvline(
            spec["reference"],
            color=REFERENCE,
            linestyle=(0, (3, 2)),
            linewidth=0.9,
            zorder=1,
        )

    ax.set_xlim(*spec["xlim"])
    ax.set_ylim(-0.48, 3.48)
    ax.set_yticks([item[3] for item in CONDITIONS])
    if show_ylabels:
        ax.set_yticklabels([item[2] for item in CONDITIONS])
        ax.tick_params(axis="y", pad=4)
    else:
        ax.set_yticklabels([])
    ax.set_xlabel(spec["xlabel"], fontsize=6.7, color=INK, labelpad=5)
    panel_label(ax, letter, spec["title"])


def build_figure(data: pd.DataFrame) -> plt.Figure:
    fig = plt.figure(figsize=(7.0, 5.0), facecolor=PAPER)
    grid = fig.add_gridspec(
        2,
        2,
        left=0.135,
        right=0.985,
        top=0.91,
        bottom=0.135,
        wspace=0.30,
        hspace=0.48,
    )
    axes = [
        fig.add_subplot(grid[0, 0]),
        fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[1, 0]),
        fig.add_subplot(grid[1, 1]),
    ]

    draw_metric_panel(axes[0], data, "rho", "a", show_ylabels=True)
    draw_metric_panel(axes[1], data, "rmae", "b", show_ylabels=False)
    draw_metric_panel(axes[2], data, "eta", "c", show_ylabels=True)
    draw_metric_panel(axes[3], data, "kappa", "d", show_ylabels=False)

    method_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="-",
            color=METHOD_COLOR[method],
            markerfacecolor=METHOD_COLOR[method],
            markeredgecolor=PAPER,
            markeredgewidth=0.5,
            linewidth=2.0,
            markersize=4.2,
            label=METHOD_LABEL[method],
        )
        for method in METHOD_ORDER
    ]
    fig.legend(
        handles=method_handles,
        loc="lower center",
        bbox_to_anchor=(0.55, 0.02),
        ncol=3,
        fontsize=5.8,
        handlelength=1.9,
        handletextpad=0.4,
        columnspacing=1.25,
    )
    return fig


def save_figure(fig: plt.Figure) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    stem = OUT / "fig3_scale_equivalence"
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
            OUT / "fig3_scale_equivalence_latest.pdf",
            facecolor=PAPER,
        )

    fig.savefig(
        HERE / "fig3_final_width_preview.png",
        dpi=300,
        facecolor=PAPER,
    )


def main() -> None:
    data = load_data()
    figure = build_figure(data)
    save_figure(figure)
    plt.close(figure)


if __name__ == "__main__":
    main()
