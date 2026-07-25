"""Create the revised VALID-OT finite-intervention audit schematic."""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "figures"

INK = "#26343B"
MUTED = "#66747C"
HAIR = "#8C969B"
PAPER = "#FFFFFF"
PALE = "#F5F7F8"
EXPR = "#3F73A5"
EXPR_PALE = "#E8F0F6"
SPATIAL = "#766391"
SPATIAL_PALE = "#F0ECF5"
INTERVENTION = "#B65948"
INTERVENTION_PALE = "#F8ECE8"
SOLVE = "#377A72"
SOLVE_PALE = "#E7F2F0"
INTERNAL = "#3F73A5"
INTERNAL_PALE = "#E9F1F7"
EXTERNAL = "#9D681A"
EXTERNAL_PALE = "#FAF1E4"
BOUNDARY = "#A74A48"
BOUNDARY_PALE = "#F9ECEC"


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7,
        "axes.linewidth": 0.75,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "mathtext.fontset": "dejavusans",
    }
)


def rounded(ax, x, y, w, h, fc, ec=HAIR, lw=0.8, radius=0.018, z=1):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.008,rounding_size={radius}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        transform=ax.transAxes,
        clip_on=False,
        zorder=z,
    )
    ax.add_patch(patch)
    return patch


def arrow(ax, x0, y0, x1, y1, color=MUTED, lw=1.0, style="-|>", ms=8, z=4):
    arr = FancyArrowPatch(
        (x0, y0),
        (x1, y1),
        arrowstyle=style,
        mutation_scale=ms,
        linewidth=lw,
        color=color,
        transform=ax.transAxes,
        shrinkA=0,
        shrinkB=0,
        clip_on=False,
        zorder=z,
    )
    ax.add_patch(arr)
    return arr


def panel_label(ax, letter):
    ax.text(
        -0.012,
        1.02,
        letter,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        color=INK,
    )


def matrix_icon(ax, x, y, w, h, color, pattern):
    rows, cols = pattern.shape
    gap = min(w / cols, h / rows) * 0.10
    cw = (w - gap * (cols - 1)) / cols
    ch = (h - gap * (rows - 1)) / rows
    for r in range(rows):
        for c in range(cols):
            alpha = 0.12 + 0.74 * float(pattern[r, c])
            ax.add_patch(
                Rectangle(
                    (x + c * (cw + gap), y + (rows - 1 - r) * (ch + gap)),
                    cw,
                    ch,
                    transform=ax.transAxes,
                    facecolor=mpl.colors.to_rgba(color, alpha),
                    edgecolor="none",
                    zorder=3,
                )
            )


def draw_slice(ax, x, y, w, h, color, variant):
    # Fixed schematic points: visual vocabulary only, not experimental data.
    base = np.array(
        [
            [0.12, 0.24], [0.20, 0.68], [0.28, 0.40], [0.34, 0.77],
            [0.39, 0.20], [0.45, 0.53], [0.52, 0.31], [0.58, 0.72],
            [0.62, 0.47], [0.68, 0.16], [0.74, 0.58], [0.81, 0.34],
            [0.87, 0.76], [0.22, 0.16], [0.48, 0.84], [0.70, 0.83],
            [0.32, 0.56], [0.84, 0.51],
        ]
    )
    if variant == 8:
        base = np.column_stack((1.0 - base[:, 0], base[:, 1] * 0.86 + 0.07))
    rounded(ax, x, y, w, h, "#FCFCFC", color, lw=0.65, radius=0.010, z=2)
    pts = np.column_stack((x + base[:, 0] * w, y + base[:, 1] * h))
    ax.scatter(
        pts[:, 0],
        pts[:, 1],
        s=4.5,
        facecolor=mpl.colors.to_rgba(color, 0.70),
        edgecolor=PAPER,
        linewidth=0.25,
        transform=ax.transAxes,
        zorder=4,
    )


def draw_panel_a(ax):
    ax.set_axis_off()
    panel_label(ax, "a")
    ax.text(
        0.040,
        1.018,
        r"Finite-intervention audit instance  $\mathcal{A}=(I,r^{\ast},s,w)$",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.6,
        fontweight="bold",
        color=INK,
    )
    # Paired spatial sections
    ax.text(0.0775, 0.79, "Paired spatial\nsections", ha="center", va="top",
            transform=ax.transAxes, fontsize=7.0, fontweight="bold", color=INK)
    ax.plot([0.032, 0.0775], [0.690, 0.690], color=EXPR, lw=1.1,
            transform=ax.transAxes, clip_on=False)
    ax.plot([0.0775, 0.123], [0.690, 0.690], color=SPATIAL, lw=1.1,
            transform=ax.transAxes, clip_on=False)
    draw_slice(ax, 0.032, 0.48, 0.076, 0.115, EXPR, 4)
    draw_slice(ax, 0.047, 0.38, 0.076, 0.115, SPATIAL, 8)
    ax.text(0.0775, 0.285, "expression + coordinates", ha="center", va="center",
            transform=ax.transAxes, fontsize=5.6, color=MUTED)

    # Two frozen cost channels
    ax.text(0.2535, 0.79, "Frozen cost\nconstruction", ha="center", va="top",
            transform=ax.transAxes, fontsize=7.0, fontweight="bold", color=INK)
    ax.plot([0.200, 0.2535], [0.690, 0.690], color=EXPR, lw=1.1,
            transform=ax.transAxes, clip_on=False)
    ax.plot([0.2535, 0.307], [0.690, 0.690], color=SPATIAL, lw=1.1,
            transform=ax.transAxes, clip_on=False)
    expr_pattern = np.array([[0.2, 0.7, 0.4, 0.9], [0.8, 0.3, 0.6, 0.2], [0.4, 0.9, 0.2, 0.6]])
    spatial_pattern = np.array([[0.9, 0.6, 0.3, 0.1], [0.6, 0.8, 0.5, 0.2], [0.3, 0.5, 0.7, 0.9]])
    matrix_icon(ax, 0.194, 0.51, 0.047, 0.105, EXPR, expr_pattern)
    matrix_icon(ax, 0.265, 0.51, 0.047, 0.105, SPATIAL, spatial_pattern)
    ax.text(0.2175, 0.475, r"$C_{\rm expr}$", ha="center", va="top",
            transform=ax.transAxes, fontsize=6.3, color=EXPR, fontweight="bold")
    ax.text(0.2885, 0.475, r"$C_{\rm spatial}$", ha="center", va="top",
            transform=ax.transAxes, fontsize=6.3, color=SPATIAL, fontweight="bold")
    ax.text(0.2535, 0.335, r"$C^{0}=0.5C_{\rm expr}+0.5C_{\rm spatial}$",
            ha="center", va="center", transform=ax.transAxes, fontsize=6.1, color=INK)
    # Finite intervention
    ax.text(0.4315, 0.79, "Cost-channel\nintervention", ha="center", va="top",
            transform=ax.transAxes, fontsize=7.0, fontweight="bold", color=INK)
    ax.plot([0.389, 0.474], [0.690, 0.690], color=INTERVENTION, lw=1.1,
            transform=ax.transAxes, clip_on=False)
    ax.text(0.4315, 0.605, r"$I^{A,k}(t)$", ha="center", va="center",
            transform=ax.transAxes, fontsize=9.0, color=INTERVENTION, fontweight="bold")
    arrow(ax, 0.389, 0.495, 0.474, 0.495, color=INTERVENTION, lw=1.3, ms=7)
    ax.text(0.389, 0.452, "0", ha="center", va="top", transform=ax.transAxes,
            fontsize=5.7, color=MUTED)
    ax.text(0.474, 0.452, "1", ha="center", va="top", transform=ax.transAxes,
            fontsize=5.7, color=MUTED)
    ax.text(0.4315, 0.37, r"$A\in\{R,N\}$", ha="center", va="center",
            transform=ax.transAxes, fontsize=6.1, color=INK)
    ax.text(0.4315, 0.305, r"$k\in\{\mathrm{expr,spatial}\}$", ha="center",
            va="center", transform=ax.transAxes, fontsize=5.6, color=MUTED)

    # Complete reoptimization.  Keep the title, rule and first content row on
    # the same horizontal grid used by the three modules to the left.
    ax.text(0.6215, 0.79, "Complete reoptimization", ha="center", va="top",
            transform=ax.transAxes, fontsize=7.0, fontweight="bold", color=INK)
    ax.plot([0.554, 0.689], [0.690, 0.690], color=SOLVE, lw=1.1,
            transform=ax.transAxes, clip_on=False)
    solve_rows = [
        (0.577, r"baseline  $P^{0}$"),
        (0.437, r"local  $P^{A,k}(h)$   $(h=0.01)$"),
        (0.297, r"endpoint  $P^{A,k}(1)$"),
    ]
    for yy, label in solve_rows:
        rounded(ax, 0.554, yy - 0.047, 0.135, 0.085, SOLVE_PALE, SOLVE, lw=0.65, radius=0.010, z=2)
        ax.text(0.6215, yy, label, ha="center", va="center", transform=ax.transAxes,
                fontsize=5.9, color=INK)
    # Response quantities: the finite-step score, a high-accuracy local
    # reference, and the fully reoptimized finite endpoint are distinct.
    ax.text(0.8115, 0.79, "Response quantities", ha="center", va="top",
            transform=ax.transAxes, fontsize=7.0, fontweight="bold", color=INK)
    ax.plot([0.753, 0.792], [0.690, 0.690], color=INTERNAL, lw=1.1,
            transform=ax.transAxes, clip_on=False)
    ax.plot([0.792, 0.831], [0.690, 0.690], color=SOLVE, lw=1.1,
            transform=ax.transAxes, clip_on=False)
    ax.plot([0.831, 0.870], [0.690, 0.690], color=INTERVENTION, lw=1.1,
            transform=ax.transAxes, clip_on=False)
    response_specs = [
        (0.497, INTERNAL_PALE, INTERNAL, r"finite-step  $s_i(0.01)$", r"$D_i(P^0,P(h))/h$"),
        (0.342, SOLVE_PALE, SOLVE, r"local reference  $r_i^{L}$", "high-accuracy local slope"),
        (0.187, INTERVENTION_PALE, INTERVENTION, r"endpoint  $r_i^{E}$", r"$D_i(P^0,P(1))$"),
    ]
    for yy, fc, ec, label, definition in response_specs:
        rounded(ax, 0.753, yy, 0.117, 0.118, fc, ec, lw=0.7, radius=0.010, z=2)
        ax.text(0.8115, yy + 0.078, label, ha="center", va="center",
                transform=ax.transAxes, fontsize=5.5, fontweight="bold", color=ec)
        ax.text(0.8115, yy + 0.032, definition, ha="center", va="center",
                transform=ax.transAxes, fontsize=5.0, color=INK)
    # Audit tuple
    rounded(ax, 0.910, 0.16, 0.084, 0.70, "#FBFBFB", INK, lw=0.85, radius=0.014)
    ax.text(0.952, 0.810, "Audit\ntuple", ha="center", va="top",
            transform=ax.transAxes, fontsize=6.8, fontweight="bold", color=INK)
    tuple_items = [
        (0.635, r"$I$", INTERVENTION_PALE, INTERVENTION),
        (0.505, r"$r^{\ast}$", INTERVENTION_PALE, INTERVENTION),
        (0.375, r"$s$", INTERNAL_PALE, INTERNAL),
        (0.245, r"$w$", EXTERNAL_PALE, EXTERNAL),
    ]
    for yy, text, fc, ec in tuple_items:
        rounded(ax, 0.930, yy - 0.037, 0.044, 0.070, fc, ec, lw=0.65, radius=0.012, z=2)
        ax.text(0.952, yy, text, ha="center", va="center", transform=ax.transAxes,
                fontsize=7.0, color=ec, fontweight="bold")
    # Main-flow arrows.
    for x0, x1 in [(0.145, 0.177), (0.330, 0.363), (0.500, 0.534), (0.709, 0.744), (0.879, 0.914)]:
        arrow(ax, x0, 0.535, x1, 0.535, color=MUTED, lw=1.0, ms=7)


def style_path_axis(ax, title, show_y):
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.04, 1.04)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["0", "1"], fontsize=5.6)
    ax.set_yticks([0, 0.5, 1.0])
    if show_y:
        ax.set_yticklabels(["0", "0.5", "1.0"], fontsize=5.6)
        ax.set_ylabel("channel coefficient", fontsize=5.8, labelpad=2)
    else:
        ax.set_yticklabels([])
    ax.tick_params(length=2.2, width=0.6, color=MUTED, pad=1.5)
    ax.spines["left"].set_color(HAIR)
    ax.spines["bottom"].set_color(HAIR)
    ax.set_xlabel("intervention path  $t$", fontsize=5.8, labelpad=1)
    ax.set_title(title, loc="left", fontsize=7.0, fontweight="bold", color=INK, pad=4)


def draw_panel_b(ax):
    ax.set_axis_off()
    panel_label(ax, "b")
    ax.text(
        0.055,
        1.02,
        "Two intervention paths plus an objective-scale control",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.8,
        fontweight="bold",
        color=INK,
    )

    ax_r = ax.inset_axes([0.035, 0.27, 0.235, 0.60])
    ax_n = ax.inset_axes([0.335, 0.27, 0.235, 0.60])
    t = np.array([0.0, 1.0])

    style_path_axis(ax_r, "Arm R · fixed channel", True)
    ax_r.plot(t, [0.5, 0.0], color=INTERVENTION, lw=1.8, marker="o", ms=3.0,
              clip_on=False, zorder=4)
    ax_r.plot(t, [0.5, 0.5], color=INTERNAL, lw=1.8, marker="o", ms=3.0,
              clip_on=False, zorder=4)
    ax_r.text(0.52, 0.06, "deleted", ha="left", va="bottom", fontsize=5.3, color=INTERVENTION)
    ax_r.text(0.62, 0.58, "retained", ha="left", va="bottom", fontsize=5.3, color=INTERNAL)

    style_path_axis(ax_n, "Arm N · renormalized", False)
    ax_n.plot(t, [0.5, 0.0], color=INTERVENTION, lw=1.8, marker="o", ms=3.0,
              clip_on=False, zorder=4)
    ax_n.plot(t, [0.5, 1.0], color=INTERNAL, lw=1.8, marker="o", ms=3.0,
              clip_on=False, zorder=4)
    ax_n.text(0.71, 0.16, "deleted", ha="left", va="bottom", fontsize=5.3, color=INTERVENTION)
    ax_n.text(0.71, 0.75, "retained", ha="left", va="bottom", fontsize=5.3, color=INTERNAL)

    ax.text(
        0.152,
        0.100,
        "$C^{R}(t)=0.5(1-t)C_{\\rm del}$\n$+\\,0.5C_{\\rm keep}$",
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=5.0,
        color=INK,
        linespacing=1.15,
    )
    ax.text(
        0.452,
        0.100,
        "$C^{N}(t)=0.5(1-t)C_{\\rm del}$\n$+\\,0.5(1+t)C_{\\rm keep}$",
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=5.0,
        color=INK,
        linespacing=1.15,
    )

    # Arm-S scaling control.
    # Match the visible inter-arm gaps. Arm R ends at 0.270 and Arm N starts
    # at 0.335 (gap 0.065); Arm N ends at 0.570, so the leftmost Arm-S chip
    # starts at 0.635. The rightmost chip ends at 0.925.
    arm_s_x = 0.7800
    ax.text(arm_s_x, 0.895, "Arm S · scale control", ha="center", va="baseline",
            transform=ax.transAxes, fontsize=7.0, fontweight="bold", color=INK)
    ax.text(arm_s_x, 0.735, r"$C^{S}=0.5C^{N}(1)=C^{R}(1)$",
            ha="center", va="center", transform=ax.transAxes, fontsize=7.0, color=INK)

    chips = [
        (0.635, "cost ×0.5"),
        (0.737, r"$\epsilon$ ×0.5"),
        (0.839, "$\\tau$ ×0.5\n(UOT)"),
    ]
    for xx, label in chips:
        rounded(ax, xx, 0.535, 0.086, 0.105, SOLVE_PALE, SOLVE, lw=0.65, radius=0.010, z=2)
        ax.text(xx + 0.043, 0.587, label, ha="center", va="center",
                transform=ax.transAxes, fontsize=5.2, color=INK)

    rounded(ax, arm_s_x - 0.1185, 0.335, 0.237, 0.105, SOLVE_PALE, SOLVE,
            lw=0.75, radius=0.012, z=2)
    ax.text(arm_s_x, 0.387, r"$P^{S}=P^{N}(1)$",
            ha="center", va="center", transform=ax.transAxes, fontsize=7.0,
            color=SOLVE, fontweight="bold")


def draw_panel_c(ax):
    ax.set_axis_off()
    panel_label(ax, "c")
    ax.text(
        0.085,
        1.02,
        "Fidelity diagnostics and separate utility",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.3,
        fontweight="bold",
        color=INK,
    )

    # One umbrella concept with two diagnostic modules. The end-to-end readout
    # is reported separately and is not drawn as an algebraic composition.
    ax.text(0.500, 0.905, "Finite-intervention local fidelity", ha="center", va="center",
            transform=ax.transAxes, fontsize=6.6, fontweight="bold", color=INTERNAL)
    ax.plot([0.055, 0.945], [0.865, 0.865], color=INTERNAL, lw=0.9,
            transform=ax.transAxes, clip_on=False)

    modules = [
        (0.045, "Diagnostic 1", "Local response", r"$s(0.01)\;\leftrightarrow\;r^{L}$",
         r"$\rho$ · overlap · rMAE", INTERNAL_PALE, INTERNAL),
        (0.520, "Diagnostic 2", "Endpoint transport", r"$r^{L}\;\leftrightarrow\;r^{E}$",
         r"$\eta$ · $\kappa$ · calibration", SOLVE_PALE, SOLVE),
    ]
    for xx, kicker, title, formula, metrics, fc, ec in modules:
        rounded(ax, xx, 0.575, 0.430, 0.225, fc, fc, lw=0.0, radius=0.014, z=1)
        ax.text(xx + 0.215, 0.755, kicker, ha="center", va="center",
                transform=ax.transAxes, fontsize=5.0, fontweight="bold", color=ec)
        ax.text(xx + 0.215, 0.711, title, ha="center", va="center",
                transform=ax.transAxes, fontsize=5.8, fontweight="bold", color=ec)
        ax.text(xx + 0.215, 0.657, formula, ha="center", va="center",
                transform=ax.transAxes, fontsize=6.2, color=INK)
        ax.text(xx + 0.215, 0.606, metrics, ha="center", va="center",
                transform=ax.transAxes, fontsize=5.0, color=MUTED)

    end_box = rounded(ax, 0.170, 0.385, 0.660, 0.125, "#FBFCFD", HAIR,
                      lw=0.7, radius=0.012, z=1)
    end_box.set_linestyle((0, (2.0, 2.0)))
    ax.text(0.500, 0.472, "End-to-end readout", ha="center", va="center",
            transform=ax.transAxes, fontsize=5.1, fontweight="bold", color=INK)
    ax.text(0.500, 0.425, r"$s(0.01)\;\leftrightarrow\;r^{E}$",
            ha="center", va="center", transform=ax.transAxes, fontsize=6.4, color=INK)

    ax.plot([0.100, 0.900], [0.315, 0.315], color=HAIR, lw=0.65,
            transform=ax.transAxes, clip_on=False)
    rounded(ax, 0.115, 0.075, 0.770, 0.165, EXTERNAL_PALE, EXTERNAL,
            lw=0.65, radius=0.014, z=1)
    ax.text(0.500, 0.205, "External utility · separate evidence axis",
            ha="center", va="center", transform=ax.transAxes, fontsize=5.7,
            fontweight="bold", color=EXTERNAL)
    ax.text(0.500, 0.151, r"response score $\longrightarrow w$", ha="center", va="center",
            transform=ax.transAxes, fontsize=6.2, color=INK)
    ax.text(0.500, 0.105, r"NEX–AURC  ·  $\Delta$ fixed-QC", ha="center", va="center",
            transform=ax.transAxes, fontsize=5.0, color=MUTED)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(7.0, 5.58), facecolor=PAPER)
    gs = fig.add_gridspec(
        2,
        1,
        height_ratios=[0.98, 1.07],
        left=0.035,
        right=0.985,
        top=0.965,
        bottom=0.055,
        hspace=0.19,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    lower = gs[1, 0].subgridspec(1, 2, width_ratios=[1.58, 1.0], wspace=0.14)
    ax_b = fig.add_subplot(lower[0, 0])
    ax_c = fig.add_subplot(lower[0, 1])

    draw_panel_a(ax_a)
    draw_panel_b(ax_b)
    draw_panel_c(ax_c)

    # Use a new revision stem so a PDF viewer caching or locking an earlier
    # export cannot leave a mixed set of old and new vector/raster files.
    stem = OUT / "fig1_audit_design_local_fidelity"
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor=PAPER)
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor=PAPER)
    fig.savefig(
        stem.with_suffix(".tiff"),
        dpi=600,
        bbox_inches="tight",
        facecolor=PAPER,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    try:
        fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor=PAPER)
    except PermissionError:
        # Illustrator can lock an open PDF on Windows. Preserve the requested
        # revision under an unambiguous fallback name instead of leaving stale
        # raster/vector companions.
        fig.savefig(OUT / "fig1_audit_design_local_fidelity_latest.pdf", bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)


if __name__ == "__main__":
    main()
