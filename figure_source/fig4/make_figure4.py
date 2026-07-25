"""Create Figure 4: WP11 objective-composition and scale decomposition.

The figure reads frozen WP11 unit-level outputs.  It replaces the obsolete
spatial-channel-failure figure with a model-internal decomposition of the
Arm-R/Arm-N difference into cost composition, deletion/compensation and
objective-scale contributions.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = HERE / "data"
OUT = ROOT / "figures"

REGISTRY_FILE = DATA / "wp11_condition_registry.tsv"
SURFACE_FILE = DATA / "wp11_alpha_beta_surface_unit.tsv"
FACTORIAL_FILE = DATA / "wp11_factorial_contrasts_unit.tsv"

INK = "#25343B"
MUTED = "#68767D"
HAIR = "#AAB4B9"
GRID = "#DCE3E6"
PAPER = "#FFFFFF"
PANEL_BG = "#FCFDFD"
ZERO = "#89969C"

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

U_GRID = [0.5, 0.75, 1.0]
V_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]

FACTOR_ROWS = [
    ("expression", "delta_removal", "expr — removal", 7.0),
    ("expression", "delta_compensation", "expr — compensation", 6.0),
    ("expression", "delta_interaction", "expr — interaction", 5.0),
    ("expression", "delta_joint", "expr — joint", 4.0),
    ("spatial", "delta_removal", "spatial — removal", 3.0),
    ("spatial", "delta_compensation", "spatial — compensation", 2.0),
    ("spatial", "delta_interaction", "spatial — interaction", 1.0),
    ("spatial", "delta_joint", "spatial — joint", 0.0),
]

SURFACE_CMAP = LinearSegmentedColormap.from_list(
    "validot_surface",
    ["#F7FAFB", "#D6E6EA", "#87B7C1", "#2F6F7B", "#173F4A"],
)


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
        "ytick.major.size": 2.5,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "mathtext.fontset": "dejavusans",
    }
)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    registry = pd.read_csv(REGISTRY_FILE, sep="\t")
    surface = pd.read_csv(SURFACE_FILE, sep="\t")
    factorial = pd.read_csv(FACTORIAL_FILE, sep="\t")

    if len(registry) != 34:
        raise ValueError(f"WP11 registry must contain 34 conditions; found {len(registry)}")
    if len(surface) != 2142:
        raise ValueError(f"WP11 surface table must contain 2142 rows; found {len(surface)}")
    if len(factorial) != 2016:
        raise ValueError(f"WP11 factorial table must contain 2016 rows; found {len(factorial)}")
    if surface["independent_unit_id"].nunique() != 21:
        raise ValueError("WP11 surface table must contain 21 independent units")
    if factorial["independent_unit_id"].nunique() != 21:
        raise ValueError("WP11 factorial table must contain 21 independent units")
    if set(surface["method"]) != set(METHOD_ORDER):
        raise ValueError("Unexpected method set in WP11 surface table")

    surface_sizes = surface.groupby(["method", "condition_id"]).size()
    if len(surface_sizes) != 102 or not (surface_sizes == 21).all():
        raise ValueError("Every WP11 method-condition group must contain 21 units")

    grid = surface[surface["condition_family"].eq("grid")].copy()
    expected_grid_rows = 21 * 3 * 3 * 5 * 2
    if len(grid) != expected_grid_rows:
        raise ValueError(
            f"Registered 3x5x2 WP11 grid must contain {expected_grid_rows} rows; found {len(grid)}"
        )
    if set(np.round(grid["u"].astype(float), 8)) != set(U_GRID):
        raise ValueError("Unexpected u grid in WP11 surface table")
    if set(np.round(grid["v"].astype(float), 8)) != set(V_GRID):
        raise ValueError("Unexpected v grid in WP11 surface table")
    if set(grid["regularization_regime"]) != {"fixed", "coregularized"}:
        raise ValueError("WP11 grid must contain fixed and coregularized regimes")

    panel_a = grid[grid["regularization_regime"].eq("fixed")]
    if len(panel_a) != 945 or panel_a["endpoint_response_mean"].isna().any():
        raise ValueError("Panel-a fixed grid must contain 945 estimable rows")

    panel_b = factorial[
        factorial["metric"].eq("endpoint_response_mean")
        & factorial["regularization_regime"].eq("fixed")
    ]
    if len(panel_b) != 126:
        raise ValueError("Panel-b factorial subset must contain 126 rows")
    factor_columns = [item[1] for item in FACTOR_ROWS]
    if panel_b[factor_columns].isna().any().any():
        raise ValueError("All plotted WP11 factorial contrasts must be estimable")

    panel_c = grid[
        grid["v"].isin([0.0, 1.0])
        & grid["u"].isin(U_GRID)
    ]
    if len(panel_c) != 756 or panel_c["endpoint_response_mean"].isna().any():
        raise ValueError("Panel-c pure-composition grid must contain 756 estimable rows")

    coreg = panel_c[panel_c["regularization_regime"].eq("coregularized")]
    max_coreg_range = (
        coreg.groupby(["independent_unit_id", "method", "v"])["endpoint_response_mean"]
        .agg(lambda values: float(values.max() - values.min()))
        .max()
    )
    if max_coreg_range > 1.0e-12:
        raise ValueError("Co-regularized scale control is not invariant within tolerance")
    if not panel_a["endpoint_response_mean"].ge(0).all():
        raise ValueError("Endpoint response must be non-negative")
    return grid, panel_b, panel_c


def panel_heading(fig: plt.Figure, anchor: plt.Axes, letter: str, title: str) -> None:
    position = anchor.get_position()
    y = position.y1 + 0.047
    fig.text(
        position.x0 - 0.045,
        y,
        letter,
        ha="left",
        va="bottom",
        fontsize=8.2,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        position.x0,
        y,
        title,
        ha="left",
        va="bottom",
        fontsize=7.5,
        fontweight="bold",
        color=INK,
    )


def style_axis(ax: plt.Axes, grid_axis: str = "x") -> None:
    ax.spines["left"].set_color(HAIR)
    ax.spines["bottom"].set_color(HAIR)
    ax.tick_params(colors=MUTED, labelsize=6.0)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.55, alpha=0.8, zorder=0)
    ax.set_axisbelow(True)


def draw_surface_panel(
    fig: plt.Figure,
    axes: list[plt.Axes],
    cax: plt.Axes,
    grid: pd.DataFrame,
) -> None:
    image = None
    for index, (ax, method) in enumerate(zip(axes, METHOD_ORDER)):
        subset = grid[
            grid["method"].eq(method)
            & grid["regularization_regime"].eq("fixed")
        ]
        medians = (
            subset.groupby(["u", "v"], as_index=False)["endpoint_response_mean"]
            .median()
            .pivot(index="u", columns="v", values="endpoint_response_mean")
            .reindex(index=U_GRID, columns=V_GRID)
        )
        if medians.isna().any().any():
            raise ValueError(f"Incomplete fixed response surface for {method}")

        image = ax.imshow(
            medians.to_numpy(float),
            origin="lower",
            aspect="auto",
            vmin=0.0,
            vmax=0.50,
            cmap=SURFACE_CMAP,
            interpolation="nearest",
            zorder=1,
        )
        ax.set_xticks(np.arange(len(V_GRID)))
        ax.set_xticklabels(["0", "0.25", "0.50", "0.75", "1"], fontsize=5.7)
        ax.set_yticks(np.arange(len(U_GRID)))
        if index == 0:
            ax.set_yticklabels(["0.50", "0.75", "1.00"], fontsize=5.7)
            ax.set_ylabel(r"total scale  $u$", fontsize=6.3, color=INK, labelpad=4)
        else:
            ax.set_yticklabels([])
            ax.tick_params(axis="y", length=0)
        if index == 1:
            ax.set_xlabel(r"spatial-cost fraction  $v$", fontsize=6.3, color=INK, labelpad=4)
        ax.set_title(
            METHOD_LABEL[method],
            fontsize=6.5,
            fontweight="bold",
            color=METHOD_COLOR[method],
            pad=5,
        )
        ax.set_xticks(np.arange(-0.5, len(V_GRID), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(U_GRID), 1), minor=True)
        ax.grid(which="minor", color=PAPER, linewidth=0.85)
        ax.tick_params(which="minor", bottom=False, left=False)
        for spine in ax.spines.values():
            spine.set_visible(False)

        endpoint_marks = [
            (2, 2, "D"),
            (0, 0, "o"),
            (4, 0, "o"),
            (0, 2, "^"),
            (4, 2, "^"),
        ]
        for x, y, marker in endpoint_marks:
            ax.scatter(
                x,
                y,
                marker=marker,
                s=29,
                facecolor=PAPER,
                edgecolor=INK,
                linewidth=0.75,
                zorder=4,
            )

    if image is None:
        raise RuntimeError("Surface panel was not drawn")
    colourbar = fig.colorbar(image, cax=cax)
    colourbar.set_ticks([0.0, 0.25, 0.50])
    colourbar.ax.tick_params(labelsize=5.5, colors=MUTED, width=0.6, length=2)
    colourbar.outline.set_linewidth(0.6)
    colourbar.outline.set_edgecolor(HAIR)
    colourbar.set_label(
        "response",
        fontsize=5.8,
        color=INK,
        rotation=270,
        labelpad=6,
    )

    endpoint_handles = [
        Line2D([0], [0], marker="D", linestyle="none", markerfacecolor=PAPER,
               markeredgecolor=INK, markersize=4.2, label="baseline"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=PAPER,
               markeredgecolor=INK, markersize=4.2, label="Arm R endpoints"),
        Line2D([0], [0], marker="^", linestyle="none", markerfacecolor=PAPER,
               markeredgecolor=INK, markersize=4.4, label="Arm N endpoints"),
    ]
    position = axes[1].get_position()
    fig.legend(
        handles=endpoint_handles,
        loc="lower center",
        bbox_to_anchor=(position.x0 + position.width / 2, position.y1 + 0.018),
        ncol=3,
        fontsize=5.3,
        handletextpad=0.35,
        columnspacing=0.9,
    )


def draw_factorial_panel(ax: plt.Axes, factorial: pd.DataFrame) -> None:
    style_axis(ax, "x")
    ax.axvline(0.0, color=ZERO, linewidth=0.8, linestyle=(0, (3, 2)), zorder=1)
    ax.axhline(3.5, color=GRID, linewidth=0.75, zorder=0)

    for channel, column, _label, base_y in FACTOR_ROWS:
        for method in METHOD_ORDER:
            subset = factorial[
                factorial["channel"].eq(channel)
                & factorial["method"].eq(method)
            ]
            values = subset[column].to_numpy(float)
            if len(values) != 21:
                raise ValueError(f"Expected 21 values for {channel}/{column}/{method}")
            y = base_y + METHOD_OFFSET[method]
            colour = METHOD_COLOR[method]
            ax.scatter(
                values,
                np.full(values.shape, y),
                s=7.0,
                c=colour,
                edgecolors="none",
                alpha=0.25,
                zorder=2,
            )
            q25, median, q75 = np.quantile(values, [0.25, 0.50, 0.75])
            ax.plot(
                [q25, q75],
                [y, y],
                color=colour,
                linewidth=2.25,
                solid_capstyle="round",
                zorder=3,
            )
            ax.scatter(
                [median],
                [y],
                s=22,
                c=colour,
                edgecolors=PAPER,
                linewidths=0.6,
                zorder=4,
            )

    ax.set_xlim(-0.13, 0.52)
    ax.set_xticks([-0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    ax.set_ylim(-0.5, 7.5)
    ax.set_yticks([item[3] for item in FACTOR_ROWS])
    ax.set_yticklabels([item[2] for item in FACTOR_ROWS], fontsize=5.7, color=INK)
    ax.tick_params(axis="y", length=0, pad=4)
    ax.set_xlabel(r"change in endpoint mean response  $\Delta$", fontsize=6.3, color=INK, labelpad=5)


def summarize_scale_difference(
    panel: pd.DataFrame,
    method: str,
    v: float,
) -> pd.DataFrame:
    subset = panel[
        panel["method"].eq(method)
        & np.isclose(panel["v"].astype(float), v)
    ]
    rows: list[dict[str, float]] = []
    for u in U_GRID:
        current = subset[np.isclose(subset["u"].astype(float), u)]
        paired = current.pivot(
            index="independent_unit_id",
            columns="regularization_regime",
            values="endpoint_response_mean",
        )
        if len(paired) != 21 or set(paired.columns) != {"fixed", "coregularized"}:
            raise ValueError(
                f"Expected 21 paired scale-control values for {method}/v={v}/u={u}"
            )
        values = (paired["fixed"] - paired["coregularized"]).to_numpy(float)
        if np.isclose(u, 1.0) and np.max(np.abs(values)) > 1.0e-12:
            raise ValueError(f"Shared u=1 endpoint is not equivalent for {method}/v={v}")
        q25, median, q75 = np.quantile(values, [0.25, 0.50, 0.75])
        rows.append({"u": u, "q25": q25, "median": median, "q75": q75})
    return pd.DataFrame(rows)


def draw_scale_panel(axes: list[plt.Axes], panel: pd.DataFrame) -> None:
    for index, (ax, v, title) in enumerate(
        zip(axes, [0.0, 1.0], [r"Pure expression  ($v=0$)", r"Pure spatial  ($v=1$)"])
    ):
        style_axis(ax, "y")
        ax.axhline(0.0, color=MUTED, linewidth=0.75, linestyle=(0, (2.2, 2.2)), zorder=1)
        for method in METHOD_ORDER:
            colour = METHOD_COLOR[method]
            summary = summarize_scale_difference(panel, method, v)
            yerr = np.vstack(
                [
                    summary["median"].to_numpy() - summary["q25"].to_numpy(),
                    summary["q75"].to_numpy() - summary["median"].to_numpy(),
                ]
            )
            ax.errorbar(
                summary["u"],
                summary["median"],
                yerr=yerr,
                color=colour,
                linestyle="-",
                linewidth=1.25,
                marker="o",
                markersize=3.3,
                markerfacecolor=colour,
                markeredgecolor=PAPER,
                markeredgewidth=0.45,
                elinewidth=0.70,
                capsize=1.6,
                alpha=0.98,
                zorder=3,
            )
        ax.set_xlim(0.47, 1.03)
        ax.set_xticks(U_GRID)
        ax.set_xticklabels(["0.50", "0.75", "1.00"])
        ax.set_ylim(-0.205, 0.035)
        ax.set_yticks([-0.20, -0.10, 0.00])
        if index == 0:
            ax.set_ylabel(
                "scale contribution\nfixed − co-regularized",
                fontsize=5.8,
                color=INK,
                labelpad=3,
            )
        else:
            ax.set_yticklabels([])
            ax.tick_params(axis="y", length=0)
        ax.set_xlabel(r"total scale  $u$", fontsize=6.1, color=INK, labelpad=4)
        ax.set_title(title, fontsize=6.1, fontweight="bold", color=INK, pad=5)


def build_figure(
    grid: pd.DataFrame,
    factorial: pd.DataFrame,
    panel_c: pd.DataFrame,
) -> plt.Figure:
    fig = plt.figure(figsize=(7.0, 5.35), facecolor=PAPER)
    outer = fig.add_gridspec(
        2,
        1,
        height_ratios=[0.92, 1.22],
        left=0.13,
        right=0.94,
        top=0.84,
        bottom=0.13,
        hspace=0.52,
    )

    top = outer[0, 0].subgridspec(1, 4, width_ratios=[1, 1, 1, 0.055], wspace=0.18)
    axes_a = [fig.add_subplot(top[0, idx]) for idx in range(3)]
    cax = fig.add_subplot(top[0, 3])

    bottom = outer[1, 0].subgridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.31)
    ax_b = fig.add_subplot(bottom[0, 0])
    right = bottom[0, 1].subgridspec(1, 2, wspace=0.22)
    axes_c = [fig.add_subplot(right[0, idx]) for idx in range(2)]

    draw_surface_panel(fig, axes_a, cax, grid)
    draw_factorial_panel(ax_b, factorial)
    draw_scale_panel(axes_c, panel_c)

    panel_heading(fig, axes_a[0], "a", "Fixed-regularization response surface")
    panel_heading(fig, ax_b, "b", "Factorial response components")
    panel_heading(fig, axes_c[0], "c", "Fixed-minus-co-regularized response")

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
            linewidth=1.8,
            markersize=3.8,
            label=METHOD_LABEL[method],
        )
        for method in METHOD_ORDER
    ]
    fig.legend(
        handles=method_handles,
        loc="lower left",
        bbox_to_anchor=(0.10, 0.018),
        ncol=3,
        fontsize=5.4,
        handlelength=1.8,
        handletextpad=0.35,
        columnspacing=0.85,
    )
    return fig


def save_figure(fig: plt.Figure) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    stem = OUT / "fig4_spatial_failures"
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
            OUT / "fig4_spatial_failures_latest.pdf",
            facecolor=PAPER,
        )
    fig.savefig(
        HERE / "fig4_final_width_preview.png",
        dpi=300,
        facecolor=PAPER,
    )


def main() -> None:
    grid, factorial, panel_c = load_data()
    figure = build_figure(grid, factorial, panel_c)
    save_figure(figure)
    plt.close(figure)


if __name__ == "__main__":
    main()
