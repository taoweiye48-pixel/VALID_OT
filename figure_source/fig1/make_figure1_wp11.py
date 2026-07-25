"""Create a WP11-linked revision of the VALID-OT framework schematic."""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

import make_figure1_modular as base


ARM_R = "#3F73A5"
ARM_N = "#80658D"

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "mathtext.fontset": "dejavusans",
    }
)


def data_arrow(ax, start, end, color, lw=2.2, shrink_b=2.35):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=9,
        linewidth=lw,
        color=color,
        shrinkA=0,
        shrinkB=shrink_b,
        clip_on=False,
        zorder=4,
    )
    ax.add_patch(patch)
    return patch


def draw_panel_b_wp11(ax):
    ax.set_axis_off()
    base.panel_label(ax, "b")
    ax.text(
        0.055,
        1.02,
        "Intervention paths in coefficient space",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.8,
        fontweight="bold",
        color=base.INK,
    )

    # The same (u, v) plane will be reused for the quantitative WP11 response
    # surfaces. This panel defines paths only and contains no WP11 result.
    uv = ax.inset_axes([0.055, 0.155, 0.535, 0.715])
    uv.set_xlim(-0.025, 0.55)
    uv.set_ylim(0.45, 1.05)
    uv.set_xticks([0.0, 0.5])
    uv.set_xticklabels(["0", "0.5"], fontsize=5.2)
    uv.set_yticks([0.5, 1.0])
    uv.set_yticklabels(["0.5", "1.0"], fontsize=5.2)
    uv.tick_params(length=2.2, width=0.6, color=base.MUTED, pad=1.5)
    uv.spines["left"].set_color(base.HAIR)
    uv.spines["bottom"].set_color(base.HAIR)
    uv.spines["top"].set_visible(False)
    uv.spines["right"].set_visible(False)
    uv.set_xlabel(r"deleted-channel coefficient  $u$", fontsize=5.5, labelpad=1.5)
    uv.set_ylabel(r"retained-channel coefficient  $v$", fontsize=5.5, labelpad=2)
    uv.set_facecolor("#FBFCFD")

    baseline = (0.5, 0.5)
    endpoint_r = (0.0, 0.5)
    endpoint_n = (0.0, 1.0)
    # The arrow tips meet the outer boundary of the s=20 endpoint markers.
    # A point-based shrink is direction-independent and avoids both overlap
    # and a visible gap at final journal size.
    data_arrow(uv, baseline, endpoint_r, ARM_R)
    data_arrow(uv, baseline, endpoint_n, ARM_N)

    uv.scatter(
        [baseline[0]],
        [baseline[1]],
        s=24,
        facecolor=base.PAPER,
        edgecolor=base.INK,
        linewidth=0.8,
        zorder=6,
        clip_on=False,
    )
    uv.scatter(
        [endpoint_r[0]],
        [endpoint_r[1]],
        s=20,
        facecolor=ARM_R,
        edgecolor=ARM_R,
        linewidth=0.4,
        zorder=6,
        clip_on=False,
    )
    uv.scatter(
        [endpoint_n[0]],
        [endpoint_n[1]],
        s=20,
        facecolor=ARM_N,
        edgecolor=ARM_N,
        linewidth=0.4,
        zorder=6,
        clip_on=False,
    )

    uv.text(
        0.495,
        0.475,
        "baseline",
        ha="right",
        va="top",
        fontsize=5.0,
        color=base.MUTED,
    )
    uv.text(
        0.018,
        0.475,
        "R endpoint",
        ha="left",
        va="top",
        fontsize=5.0,
        color=ARM_R,
    )
    uv.text(
        0.018,
        1.012,
        "N endpoint",
        ha="left",
        va="bottom",
        fontsize=5.0,
        color=ARM_N,
    )

    uv.text(
        0.300,
        0.220,
        "Arm R",
        ha="center",
        va="center",
        transform=uv.transAxes,
        fontsize=5.6,
        fontweight="bold",
        color=ARM_R,
    )
    uv.text(
        0.330,
        0.760,
        "Arm N",
        ha="center",
        va="center",
        transform=uv.transAxes,
        fontsize=5.6,
        fontweight="bold",
        color=ARM_N,
    )

    # Arm S is deliberately subordinate: it changes the overall objective
    # scale and is not a path on the (u, v) plane.
    base.rounded(
        ax,
        0.660,
        0.155,
        0.315,
        0.705,
        "#FBFCFC",
        base.SOLVE,
        lw=0.75,
        radius=0.015,
        z=1,
    )
    ax.text(
        0.8175,
        0.805,
        "Arm S · scale control",
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=6.5,
        fontweight="bold",
        color=base.INK,
    )
    ax.text(
        0.8175,
        0.690,
        r"$\lambda=0.5$",
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=6.3,
        fontweight="bold",
        color=base.SOLVE,
    )
    ax.text(
        0.8175,
        0.600,
        r"$C^{S}=\lambda C^{N}(1)=C^{R}(1)$",
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=5.5,
        color=base.INK,
    )

    chips = [
        (0.690, "cost"),
        (0.785, r"$\epsilon$"),
        (0.880, "$\\tau$\n(UOT)"),
    ]
    for xx, label in chips:
        base.rounded(
            ax,
            xx,
            0.405,
            0.076,
            0.105,
            base.SOLVE_PALE,
            base.SOLVE,
            lw=0.6,
            radius=0.010,
            z=2,
        )
        ax.text(
            xx + 0.038,
            0.457,
            label,
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=5.0,
            color=base.INK,
        )
    ax.text(
        0.8175,
        0.360,
        r"all scales $\times\lambda$",
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=5.0,
        color=base.SOLVE,
    )
    base.rounded(
        ax,
        0.705,
        0.205,
        0.225,
        0.095,
        base.SOLVE_PALE,
        base.SOLVE,
        lw=0.7,
        radius=0.012,
        z=2,
    )
    ax.text(
        0.8175,
        0.252,
        r"$P^{S}=P^{N}(1)$",
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=6.3,
        fontweight="bold",
        color=base.SOLVE,
    )


def main():
    base.OUT.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(7.0, 5.58), facecolor=base.PAPER)
    gs = fig.add_gridspec(
        2,
        1,
        height_ratios=[0.98, 1.07],
        left=0.035,
        right=0.985,
        top=0.965,
        bottom=0.055,
        hspace=0.0,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    lower = gs[1, 0].subgridspec(1, 2, width_ratios=[1.58, 1.0], wspace=0.14)
    ax_b = fig.add_subplot(lower[0, 0])
    ax_c = fig.add_subplot(lower[0, 1])

    base.draw_panel_a(ax_a)
    draw_panel_b_wp11(ax_b)
    base.draw_panel_c(ax_c)

    stem = base.OUT / "fig1_audit_design_wp11_coordinates"
    fig.savefig(stem.with_suffix(".svg"), facecolor=base.PAPER)
    fig.savefig(stem.with_suffix(".png"), dpi=300, facecolor=base.PAPER)
    fig.savefig(
        stem.with_suffix(".tiff"),
        dpi=600,
        facecolor=base.PAPER,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    try:
        fig.savefig(stem.with_suffix(".pdf"), facecolor=base.PAPER)
    except PermissionError:
        fig.savefig(
            base.OUT / "fig1_audit_design_wp11_coordinates_latest.pdf",
            facecolor=base.PAPER,
        )
    plt.close(fig)


if __name__ == "__main__":
    main()
