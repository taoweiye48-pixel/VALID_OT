"""Create non-reusable Supplementary Figures S1-S8 and S10.

All panels read frozen independent-unit tables copied into ``data``.  S9 is a
previously audited representative tissue-context figure and is intentionally
reused rather than regenerated here.
"""

from __future__ import annotations

from pathlib import Path
import string
import io
import shutil

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import fitz
from PIL import Image


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = HERE / "data"
OUT = ROOT / "figures" / "supplementary"

INK = "#25343B"
MUTED = "#68767D"
HAIR = "#AAB4B9"
GRID = "#E2E8EA"
PAPER = "#FFFFFF"
RED = "#B6534E"
METHODS = ["balanced_ot", "uot", "row_softmax"]
MLABEL = {"balanced_ot": "Balanced OT", "uot": "UOT", "row_softmax": "Row-softmax"}
MCOLOR = {"balanced_ot": "#447CB2", "uot": "#2F857C", "row_softmax": "#BC7A27"}
ARMS = ["R", "N"]
INTERVENTIONS = ["I_EXPR", "I_SPATIAL"]
ILABEL = {"I_EXPR": "expression", "I_SPATIAL": "spatial"}

SCORES = [
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


def read(name: str) -> pd.DataFrame:
    path = DATA / name
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, sep="\t" if path.suffix == ".tsv" else ",")


def finite_values(series: pd.Series, context: str) -> np.ndarray:
    """Return a numeric vector and fail rather than silently exclude rows."""
    values = series.to_numpy(float)
    if len(values) == 0 or not np.isfinite(values).all():
        raise ValueError(f"Missing or non-finite plotted values in {context}")
    return values


def default_rows(data: pd.DataFrame) -> pd.DataFrame:
    ok = np.isclose(data["epsilon"], 0.25)
    if "tau" in data:
        ok &= data["method"].ne("uot") | np.isclose(data["tau"].fillna(-1), 2.0)
    return data[ok].copy()


def style(ax: plt.Axes, grid: str = "y") -> None:
    ax.spines["left"].set_color(HAIR)
    ax.spines["bottom"].set_color(HAIR)
    ax.tick_params(colors=MUTED, labelsize=5.4, pad=1.5)
    ax.grid(axis=grid, color=GRID, lw=0.55)
    ax.set_axisbelow(True)


def panel(ax: plt.Axes, letter: str, title: str) -> None:
    ax.text(-0.12, 1.08, letter, transform=ax.transAxes, fontsize=8.3,
            fontweight="bold", color=INK, va="bottom")
    ax.text(0.0, 1.08, title, transform=ax.transAxes, fontsize=7.6,
            fontweight="bold", color=INK, va="bottom")


def method_legend(fig: plt.Figure, y: float = 0.985) -> None:
    handles = [
        Line2D([0], [0], color=MCOLOR[m], marker="o", lw=1.5, ms=3.8,
               markeredgecolor=PAPER, markeredgewidth=0.4, label=MLABEL[m])
        for m in METHODS
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.08, y),
               ncol=3, fontsize=5.4, handlelength=1.05, handletextpad=0.60,
               columnspacing=0.9)


def export(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    base = OUT / stem
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight", facecolor=PAPER)
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", facecolor=PAPER)
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor=PAPER)
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight", facecolor=PAPER,
                pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


def supplement_s1() -> None:
    data = default_rows(read("wp1_full_step_convergence_unit.tsv"))
    # The frozen table retains pair-directions. Aggregate those records within
    # each biological independent unit before forming the displayed summaries.
    data = (
        data.groupby(
            ["independent_unit_id", "method", "arm", "intervention", "h_large", "h_small"],
            as_index=False,
        )[["relative_l1_median", "relative_l1_n_estimable"]]
        .median()
    )
    expected = 21 * 3 * 2 * 2 * 5
    if len(data) != expected:
        raise ValueError(f"S1 default grid expected {expected} rows; found {len(data)}")
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.65), sharey=True)
    markers = {"R": "o", "N": "^"}
    dashes = {"R": "-", "N": "--"}
    for ax, method in zip(axes, METHODS):
        sub = data[data["method"].eq(method)]
        for arm in ARMS:
            for intervention in INTERVENTIONS:
                rows = sub[sub["arm"].eq(arm) & sub["intervention"].eq(intervention)]
                med = rows.groupby("h_small")["relative_l1_median"].median().sort_index()
                q1 = rows.groupby("h_small")["relative_l1_median"].quantile(0.25).sort_index()
                q3 = rows.groupby("h_small")["relative_l1_median"].quantile(0.75).sort_index()
                colour = MCOLOR[method]
                alpha = 1.0 if intervention == "I_EXPR" else 0.58
                ax.fill_between(med.index, q1, q3, color=colour, alpha=0.08 * alpha, lw=0)
                ax.plot(med.index, med, dashes[arm], color=colour, alpha=alpha,
                        marker=markers[arm], ms=3.0, lw=1.35)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(MLABEL[method], fontsize=6.4, fontweight="bold", color=MCOLOR[method], pad=5)
        ax.set_xlabel("smaller step  $h$", fontsize=5.8, color=INK)
        style(ax, "both")
    axes[0].set_ylabel("adjacent-step relative L1", fontsize=5.8, color=INK)
    panel(axes[0], "a", "Multi-step numerical convergence")
    handles = [
        Line2D([0], [0], color=INK, lw=1.2, ls="-", marker="o", ms=3, label="Arm R"),
        Line2D([0], [0], color=INK, lw=1.2, ls="--", marker="^", ms=3, label="Arm N"),
        Line2D([0], [0], color=INK, lw=1.2, alpha=1.0, label="expression"),
        Line2D([0], [0], color=INK, lw=1.2, alpha=0.50, label="spatial"),
    ]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.98, 0.99),
               ncol=4, fontsize=5.1, handletextpad=0.3, columnspacing=0.7)
    fig.subplots_adjust(left=0.10, right=0.98, top=0.76, bottom=0.19, wspace=0.18)
    export(fig, "figS1_multistep_convergence")


def supplement_s2() -> None:
    data = read("wp2_derivative_cross_validation.tsv")
    if set(data["method"]) != {"balanced_ot"} or not data["converged"].all():
        raise ValueError("S2 must contain converged Balanced-OT implicit validations only")
    data = (
        data.groupby(["independent_unit_id", "arm", "intervention"], as_index=False)[
            ["global_plan_relative_l1", "row_relative_l1_median", "row_direction_cosine_median"]
        ]
        .median()
    )
    if len(data) != 9 * 4:
        raise ValueError(f"S2 expected 36 independent-unit conditions; found {len(data)}")
    labels = [f"{a}–{ILABEL[i]}" for a in ARMS for i in INTERVENTIONS]
    groups = [(a, i) for a in ARMS for i in INTERVENTIONS]
    metrics = [
        ("global_plan_relative_l1", "global plan relative L1", True),
        ("row_relative_l1_median", "row derivative relative L1", True),
        ("row_direction_cosine_median", r"$|1-$row direction cosine$|$", True),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.65))
    for k, (ax, (metric, ylabel, log)) in enumerate(zip(axes, metrics)):
        for idx, (arm, intervention) in enumerate(groups):
            vals = finite_values(
                data[data["arm"].eq(arm) & data["intervention"].eq(intervention)][metric],
                f"S2/{arm}/{intervention}/{metric}",
            )
            if metric == "row_direction_cosine_median":
                vals = np.maximum(np.abs(1.0 - vals), 1e-16)
            x = idx + np.linspace(-0.10, 0.10, len(vals))
            ax.scatter(x, vals, s=9, color="#447CB2", alpha=0.38, edgecolors="none")
            ax.hlines(np.median(vals), idx - 0.20, idx + 0.20, color=INK, lw=1.7)
        if log:
            ax.set_yscale("log")
        ax.set_xticks(range(4), labels, rotation=35, ha="right", fontsize=5.0)
        ax.set_ylabel(ylabel, fontsize=5.7, color=INK)
        style(ax, "y")
        panel(ax, string.ascii_lowercase[k], ["Plan-level agreement", "Row-level agreement", "Direction agreement"][k])
    fig.subplots_adjust(left=0.09, right=0.98, top=0.77, bottom=0.25, wspace=0.34)
    export(fig, "figS2_balanced_implicit_validation")


def _summary_matrix(data: pd.DataFrame, metric: str) -> np.ndarray:
    rows = [(m, a, i) for m in METHODS for a in ARMS for i in INTERVENTIONS]
    return np.array([data[(data["method"].eq(m)) & (data["arm"].eq(a)) &
                          (data["intervention"].eq(i))][metric].median()
                     for m, a, i in rows]).reshape(3, 4)


def heatmap(ax: plt.Axes, matrix: np.ndarray, rowlabels: list[str], collabels: list[str],
            title: str, cmap, norm, fmt: str = ".2f") -> mpl.image.AxesImage:
    im = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto", interpolation="none")
    ax.set_xticks(range(len(collabels)), collabels, rotation=35, ha="right", fontsize=5.0)
    ax.set_yticks(range(len(rowlabels)), rowlabels, fontsize=5.0)
    ax.tick_params(length=0, pad=2)
    for r in range(matrix.shape[0]):
        for c in range(matrix.shape[1]):
            value = matrix[r, c]
            rgba = im.cmap(im.norm(value))
            luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            ax.text(c, r, format(value, fmt), ha="center", va="center", fontsize=5.0,
                    color=PAPER if luminance < 0.48 else INK)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, fontsize=6.2, fontweight="bold", color=INK, pad=5)
    return im


def supplement_s3() -> None:
    data = read("wp4_endpoint_transportability_pair.tsv")
    data = (
        data.groupby(["independent_unit_id", "method", "arm", "intervention"], as_index=False)[
            ["h001_to_endpoint_spearman", "h001_to_endpoint_rmae"]
        ]
        .median()
    )
    if len(data) != 21 * 3 * 2 * 2:
        raise ValueError(f"S3 expected 252 independent-unit conditions; found {len(data)}")
    cols = [f"{a}–{'expr' if i == 'I_EXPR' else 'spatial'}" for a in ARMS for i in INTERVENTIONS]
    rows = [MLABEL[m] for m in METHODS]
    rho = _summary_matrix(data, "h001_to_endpoint_spearman")
    rmae = _summary_matrix(data, "h001_to_endpoint_rmae")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.65))
    im1 = heatmap(axes[0], rho, rows, cols, "Rank transportability", "Blues", Normalize(0, 1))
    im2 = heatmap(axes[1], rmae, rows, cols, "Magnitude transportability", "Oranges", Normalize(0, max(1.0, np.nanmax(rmae))))
    panel(axes[0], "a", r"$s(0.01)$ versus endpoint response")
    for ax, im, label in [(axes[0], im1, "Spearman"), (axes[1], im2, "relative MAE")]:
        cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
        cb.ax.tick_params(labelsize=5.0, length=2)
        cb.set_label(label, fontsize=5.0)
    fig.subplots_adjust(left=0.10, right=0.98, top=0.74, bottom=0.25, wspace=0.35)
    export(fig, "figS3_endpoint_transportability")


def supplement_s4() -> None:
    data = read("wp6_uot_mass_shape_unit.tsv")
    conditions = [(a, i) for a in ARMS for i in INTERVENTIONS]
    rowlabels = [f"{a}–{'expr' if i == 'I_EXPR' else 'spatial'}" for a, i in conditions]
    rho = np.array([[data[data["arm"].eq(a) & data["intervention"].eq(i)][f"{part}_local_to_endpoint_spearman"].median()
                     for part in ["mass", "shape"]] for a, i in conditions])
    rmae = np.array([[data[data["arm"].eq(a) & data["intervention"].eq(i)][f"{part}_local_to_endpoint_rmae"].median()
                      for part in ["mass", "shape"]] for a, i in conditions])
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.75))
    im1 = heatmap(axes[0], rho, rowlabels, ["mass", "shape"], "Rank transfer", "Blues", Normalize(0, 1))
    im2 = heatmap(axes[1], rmae, rowlabels, ["mass", "shape"], "Magnitude transfer", "Oranges", Normalize(0, max(1.0, np.nanmax(rmae))))
    panel(axes[0], "a", "UOT mass–shape decomposition")
    for ax, im, label in [(axes[0], im1, "Spearman"), (axes[1], im2, "relative MAE")]:
        cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
        cb.ax.tick_params(labelsize=5.0, length=2)
        cb.set_label(label, fontsize=5.0)
    fig.subplots_adjust(left=0.12, right=0.98, top=0.73, bottom=0.20, wspace=0.36)
    export(fig, "figS4_uot_mass_shape")


def supplement_s5() -> None:
    data = read("wp7_coordinate_frame_sensitivity_unit.tsv")
    data = data[data["score"].eq("finite_response_h001")]
    variants = ["baseline", "label_free_rigid", "her2_oracle"]
    vlabel = ["baseline", "label-free\nrigid", "HER2\noracle"]
    metrics = [
        ("spatial_cost_pearson_vs_baseline", "spatial-cost correlation", (-0.05, 1.05)),
        ("local_fidelity_spearman", "local-fidelity Spearman", (-0.45, 1.05)),
        ("heldout_normalized_excess_aurc", "held-out NEX-AURC", (0.0, 1.6)),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.8))
    for k, (ax, (metric, ylabel, ylim)) in enumerate(zip(axes, metrics)):
        if k == 0:
            med = []
            q1 = []
            q3 = []
            for variant in variants:
                vals = finite_values(
                    data[data["variant"].eq(variant)][metric],
                    f"S5/shared/{variant}/{metric}",
                )
                med.append(np.median(vals)); q1.append(np.quantile(vals, .25)); q3.append(np.quantile(vals, .75))
            x = np.arange(3)
            ax.fill_between(x, q1, q3, color=MUTED, alpha=0.12, lw=0)
            ax.plot(x, med, color=INK, marker="o", ms=3.2, lw=1.4)
        else:
            offsets = {"balanced_ot": -0.055, "uot": 0.0, "row_softmax": 0.055}
            for method in METHODS:
                med = []
                q1 = []
                q3 = []
                for variant in variants:
                    vals = finite_values(
                        data[data["method"].eq(method) & data["variant"].eq(variant)][metric],
                        f"S5/{method}/{variant}/{metric}",
                    )
                    med.append(np.median(vals)); q1.append(np.quantile(vals, .25)); q3.append(np.quantile(vals, .75))
                x = np.arange(3, dtype=float) + offsets[method]
                ax.fill_between(x, q1, q3, color=MCOLOR[method], alpha=0.09, lw=0)
                ax.plot(x, med, color=MCOLOR[method], marker="o", ms=3.2, lw=1.4)
        ax.set_xticks(range(3), vlabel)
        ax.set_ylim(*ylim)
        ax.set_ylabel(ylabel, fontsize=5.6, color=INK)
        style(ax, "y")
        panel(ax, string.ascii_lowercase[k], ["Cost construction", "Finite-step fidelity", "External utility"][k])
    method_legend(fig, 1.0)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.70, bottom=0.20, wspace=0.35)
    export(fig, "figS5_coordinate_robustness")


def supplement_s6() -> None:
    data = read("wp8_heldout_split_robustness_unit.tsv")
    data = data[data["score"].eq("finite_response_h001")]
    splits = ["historical", "random_20260720", "random_20260721", "random_20260722",
              "random_20260723", "random_20260724", "source_only"]
    labels = ["historical", "random 1", "random 2", "random 3", "random 4", "random 5", "source-only"]
    fig, ax = plt.subplots(figsize=(7.0, 2.85))
    x = np.arange(len(splits))
    for method in METHODS:
        med, q1, q3 = [], [], []
        for split in splits:
            vals = data[data["method"].eq(method) & data["split"].eq(split)]["heldout_normalized_excess_aurc"].to_numpy(float)
            med.append(np.median(vals)); q1.append(np.quantile(vals, .25)); q3.append(np.quantile(vals, .75))
        ax.fill_between(x, q1, q3, color=MCOLOR[method], alpha=0.10, lw=0)
        ax.plot(x, med, color=MCOLOR[method], marker="o", ms=3.2, lw=1.5, label=MLABEL[method])
    ax.axhline(1.0, color=RED, ls=(0, (3, 2)), lw=0.8)
    ax.set_xticks(x, labels, rotation=25, ha="right")
    ax.set_ylabel("held-out-expression NEX-AURC", fontsize=5.8, color=INK)
    style(ax, "y")
    panel(ax, "a", "Held-out-expression utility across seven gene splits")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.02), ncol=3, fontsize=5.2)
    fig.subplots_adjust(left=0.10, right=0.98, top=0.75, bottom=0.27)
    export(fig, "figS6_gene_split_robustness")


def supplement_s7() -> None:
    internal = read("wp9_real_score_internal_audit_unit.tsv")
    external = read("wp9_real_score_external_audit_unit.tsv")
    internal = internal[
        internal["branch"].eq("main21")
        & internal["reference"].eq("local_reference")
        & internal["score"].isin(SCORES)
    ]
    external = external[
        external["branch"].eq("main21")
        & external["witness"].eq("heldout_expression_loss")
        & external["score"].isin(SCORES)
    ]
    def matrix(d: pd.DataFrame, metric: str) -> np.ndarray:
        s = d.groupby(["method", "score"])[metric].median()
        return np.array([[s.loc[(m, score)] for score in SCORES] for m in METHODS])
    rho = matrix(internal, "spearman")
    nex = matrix(external, "normalized_excess_aurc")
    rows = [MLABEL[m] for m in METHODS]
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 3.95))
    im1 = heatmap(axes[0], rho, rows, SCORE_SHORT, "Internal fidelity", "RdBu", TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1))
    im2 = heatmap(axes[1], nex, rows, SCORE_SHORT, "Held-out-expression utility", "YlGnBu_r", Normalize(0, max(1.5, np.nanmax(nex))))
    panel(axes[0], "a", "Registered practical-score audit")
    for ax, im, label in [(axes[0], im1, "Spearman"), (axes[1], im2, "NEX-AURC")]:
        cb = fig.colorbar(im, ax=ax, fraction=0.018, pad=0.02)
        cb.ax.tick_params(labelsize=5.0, length=2)
        cb.set_label(label, fontsize=5.0)
    fig.subplots_adjust(left=0.13, right=0.98, top=0.81, bottom=0.18, hspace=0.58)
    export(fig, "figS7_practical_score_audit")


def supplement_s8() -> None:
    data = read("wp11_alpha_beta_surface_unit.tsv")
    data = data[data["condition_family"].eq("grid")]
    uvals = [0.5, 0.75, 1.0]
    vvals = [0.0, 0.25, 0.5, 0.75, 1.0]
    matrices: dict[tuple[str, str], np.ndarray] = {}
    vmax = 0.0
    for regime in ["fixed", "coregularized"]:
        for method in METHODS:
            matrix = np.empty((len(vvals), len(uvals)))
            for r, v in enumerate(vvals):
                for c, u in enumerate(uvals):
                    vals = data[
                        data["regularization_regime"].eq(regime)
                        & data["method"].eq(method)
                        & np.isclose(data["u"], u)
                        & np.isclose(data["v"], v)
                    ]["endpoint_response_mean"].to_numpy(float)
                    if len(vals) != 21:
                        raise ValueError(f"S8 missing surface cell {regime}/{method}/{u}/{v}")
                    matrix[r, c] = np.median(vals)
            matrices[(regime, method)] = matrix
            vmax = max(vmax, float(np.nanmax(matrix)))
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.25), sharex=True, sharey=True)
    im = None
    for r, regime in enumerate(["fixed", "coregularized"]):
        for c, method in enumerate(METHODS):
            ax = axes[r, c]
            im = ax.imshow(matrices[(regime, method)], origin="lower", aspect="auto",
                           cmap="viridis", norm=Normalize(0, vmax), interpolation="none")
            ax.set_xticks(range(len(uvals)), [f"{x:g}" for x in uvals])
            ax.set_yticks(range(len(vvals)), [f"{x:g}" for x in vvals])
            ax.tick_params(length=0, labelsize=5.0)
            if r == 0:
                ax.set_title(MLABEL[method], fontsize=6.1, fontweight="bold", color=MCOLOR[method], pad=4)
            if c == 0:
                ax.set_ylabel(f"{regime}\nretained coefficient  $v$", fontsize=5.5, color=INK)
            if r == 1:
                ax.set_xlabel("deleted coefficient  $u$", fontsize=5.5, color=INK)
            for spine in ax.spines.values():
                spine.set_visible(False)
    panel(axes[0, 0], "a", "WP11 two-dimensional endpoint-response surfaces")
    # Reserve a dedicated right margin before adding the shared colour bar.
    # Adding an automatic colour bar before ``subplots_adjust`` makes the later
    # layout update move the heat maps underneath it.
    fig.subplots_adjust(
        left=0.11,
        right=0.88,
        top=0.82,
        bottom=0.13,
        hspace=0.22,
        wspace=0.16,
    )
    colourbar_ax = fig.add_axes([0.905, 0.24, 0.014, 0.45])
    cb = fig.colorbar(im, cax=colourbar_ax)
    cb.ax.tick_params(labelsize=5.0, length=2)
    cb.set_label("median endpoint response", fontsize=5.0)
    export(fig, "figS8_wp11_response_surfaces")


def supplement_s10() -> None:
    data = read("p1_internal_unit_level.csv")
    data = data[data["score"].eq("matched_finite_difference") & data["intervention"].eq("I_EXPR")]
    eps = data[data["grid_role"].isin(["epsilon_scan", "baseline"])]
    tau = data[data["method"].eq("uot") & data["grid_role"].isin(["tau_scan", "baseline"])]
    equivalence = read("p1_scale_equivalence.csv")
    if len(equivalence) != 372 or not equivalence["equivalence_pass"].all():
        raise ValueError("S10 requires the locked 372/372 Arm-S equivalence controls")
    fig = plt.figure(figsize=(7.0, 3.15))
    gs = fig.add_gridspec(1, 3, left=0.09, right=0.98, top=0.76, bottom=0.20, wspace=0.38)
    ax1, ax2, ax3 = [fig.add_subplot(gs[0, i]) for i in range(3)]
    for method in METHODS:
        sub = eps[eps["method"].eq(method)]
        for arm, ls, marker in [("R", "-", "o"), ("N", "--", "^")]:
            med = sub[sub["arm"].eq(arm)].groupby("epsilon")["nmae"].median().sort_index()
            ax1.plot(med.index, med, ls=ls, marker=marker, ms=3, lw=1.25,
                     color=MCOLOR[method])
    ax1.axhline(0.75, color=RED, ls=(0, (3, 2)), lw=0.75)
    ax1.set_xlabel(r"$\epsilon$", fontsize=5.8)
    ax1.set_ylabel("median NMAE", fontsize=5.8)
    ax1.set_yscale("log")
    style(ax1, "both")
    panel(ax1, "a", r"Historical $\epsilon$ sensitivity")

    for arm, ls, marker in [("R", "-", "o"), ("N", "--", "^")]:
        med = tau[tau["arm"].eq(arm)].groupby("tau")["nmae"].median().sort_index()
        ax2.plot(med.index, med, ls=ls, marker=marker, ms=3.2, lw=1.4,
                 color=MCOLOR["uot"], label=f"Arm {arm}")
    ax2.axhline(0.75, color=RED, ls=(0, (3, 2)), lw=0.75)
    ax2.set_xlabel(r"UOT $\tau$", fontsize=5.8)
    ax2.set_ylabel("median NMAE", fontsize=5.8)
    ax2.set_yscale("log")
    style(ax2, "both")
    ax2.legend(fontsize=5.0, loc="upper right")
    panel(ax2, "b", r"Historical UOT $\tau$ sensitivity")

    metrics = [
        ("max |plan difference|", float(equivalence["max_absolute_plan_difference"].max())),
        ("normalized L1", float(equivalence["normalized_l1_plan_difference"].max())),
        ("response NMAE", float(equivalence["row_response_nmae"].max())),
        ("min Spearman", float(equivalence["row_response_spearman"].min())),
    ]
    y = np.arange(len(metrics))[::-1]
    ax3.set_xlim(0, 1)
    ax3.set_ylim(-0.6, len(metrics) - 0.4)
    for yy, (label, value) in zip(y, metrics):
        ax3.add_patch(mpl.patches.Rectangle((0.0, yy - 0.30), 1.0, 0.60,
                                            facecolor="#EDF5F2", edgecolor="#2F857C", lw=0.65))
        ax3.text(0.03, yy, label, va="center", ha="left", fontsize=5.0, color=INK)
        ax3.text(0.97, yy, f"{value:.1f}  PASS", va="center", ha="right",
                 fontsize=5.0, fontweight="bold", color="#2F857C")
    ax3.axis("off")
    panel(ax3, "c", "Arm-S implementation control (372/372)")
    method_legend(fig, 1.0)
    fig.subplots_adjust(top=0.76)
    export(fig, "figS10_parameter_sensitivity_and_armS")


def reuse_s9() -> None:
    """Re-export the audited tissue-context figure without changing content."""
    source = HERE / "reused" / "figS9_tissue_context_source.pdf"
    if not source.exists():
        raise FileNotFoundError(source)
    OUT.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, OUT / "figS9_tissue_context.pdf")
    document = fitz.open(source)
    page = document[0]
    svg = page.get_svg_image(matrix=fitz.Matrix(1, 1), text_as_path=False)
    (OUT / "figS9_tissue_context.svg").write_text(svg, encoding="utf-8")
    pix300 = page.get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72), alpha=False)
    pix300.save(OUT / "figS9_tissue_context.png")
    pix600 = page.get_pixmap(matrix=fitz.Matrix(600 / 72, 600 / 72), alpha=False)
    image = Image.open(io.BytesIO(pix600.tobytes("png")))
    image.save(OUT / "figS9_tissue_context.tiff", compression="tiff_lzw", dpi=(600, 600))
    document.close()


def main() -> None:
    supplement_s1()
    supplement_s2()
    supplement_s3()
    supplement_s4()
    supplement_s5()
    supplement_s6()
    supplement_s7()
    supplement_s8()
    reuse_s9()
    supplement_s10()
    print(f"Saved supplementary figures to {OUT}")


if __name__ == "__main__":
    main()
