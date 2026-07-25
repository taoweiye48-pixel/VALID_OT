#!/usr/bin/env python3
"""Create Figure S11: reproducible HER2ST controlled-target generator."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = (
    ROOT
    / "supplementary_source_data"
    / "her2st"
    / "p1_v2_her2st_manual_truth_manifest.csv"
)
OUT = ROOT / "figures" / "supplementary"

INK = "#1D2B34"
GREY = "#687780"
LIGHT_GREY = "#F5F7F8"
BLUE = "#447CB2"
BLUE_LIGHT = "#EAF1F7"
TEAL = "#2F857C"
TEAL_LIGHT = "#E8F3F1"
OCHRE = "#A66B18"
OCHRE_LIGHT = "#FBF2E5"
PURPLE = "#80638B"
PURPLE_LIGHT = "#F1ECF3"
RED = "#B6534E"
RED_LIGHT = "#F8ECEB"


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    lines: list[str],
    edge: str,
    face: str,
    title_color: str | None = None,
    body_size: float = 6.2,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.008,rounding_size=0.012",
        linewidth=1.0,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h - 0.038,
        title,
        ha="center",
        va="top",
        fontsize=7.5,
        fontweight="bold",
        color=title_color or edge,
    )
    ax.text(
        x + w / 2,
        y + h * 0.41,
        "\n".join(lines),
        ha="center",
        va="center",
        fontsize=body_size,
        color=INK,
        linespacing=1.35,
    )


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str = GREY,
    style: str = "-|>",
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=9,
            linewidth=1.1,
            color=color,
            shrinkA=1,
            shrinkB=1,
        )
    )


def main() -> None:
    configure_style()
    manifest = pd.read_csv(MANIFEST)
    if len(manifest) != 8:
        raise ValueError(f"Expected eight HER2ST sections, found {len(manifest)}")
    if not manifest["truth_valid"].astype(bool).all():
        raise ValueError("The frozen truth registry contains an invalid section")
    if not (manifest["truth_label_agreement"] == 1.0).all():
        raise ValueError("Retained truth-label agreement is not 1.0")
    crop_fraction = (
        manifest["missing_source_spots"].sum() / manifest["source_spots"].sum()
    )
    if not 0.19 <= crop_fraction <= 0.21:
        raise ValueError(f"Unexpected aggregate crop fraction: {crop_fraction:.3f}")

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.015,
        0.965,
        "Controlled HER2ST target construction and truth registration",
        ha="left",
        va="top",
        fontsize=10.0,
        fontweight="bold",
        color=INK,
    )
    ax.text(
        0.015,
        0.91,
        "8 measured sections · deterministic patient-specific seeds",
        ha="left",
        va="top",
        fontsize=7.1,
        color=GREY,
    )

    # Main construction path.
    y, h = 0.57, 0.24
    boxes = [
        (
            0.02,
            0.14,
            "Measured section",
            ["counts", "coordinates", "pathology labels"],
            BLUE,
            BLUE_LIGHT,
        ),
        (
            0.205,
            0.16,
            "Frozen features",
            ["library normalize + log1p", "500 cost / 100 held-out", "row L2 normalize"],
            PURPLE,
            PURPLE_LIGHT,
        ),
        (
            0.405,
            0.13,
            "Register truth",
            [r"$j_i^{true}=i$", r"$m_i=0$"],
            TEAL,
            TEAL_LIGHT,
        ),
        (
            0.575,
            0.21,
            "Perturb target",
            ["1  rigid transform", "2  non-rigid warp", "3  expression noise", "4  20% spatial crop"],
            OCHRE,
            OCHRE_LIGHT,
        ),
        (
            0.825,
            0.15,
            "Save pair",
            ["source and target", "forward + reverse", "manifest + SHA-256"],
            BLUE,
            LIGHT_GREY,
        ),
    ]
    for x, w, title, lines, edge, face in boxes:
        box(ax, x, y, w, h, title, lines, edge, face)
    for left, right in zip(boxes[:-1], boxes[1:]):
        arrow(
            ax,
            (left[0] + left[1] + 0.007, y + h / 2),
            (right[0] - 0.007, y + h / 2),
        )

    # Ordered truth update and evaluation eligibility.
    ax.text(
        0.02,
        0.48,
        "Truth update precedes model fitting and score evaluation",
        ha="left",
        va="center",
        fontsize=8.2,
        fontweight="bold",
        color=INK,
    )
    arrow(ax, (0.68, y - 0.005), (0.68, 0.45), color=OCHRE)

    box(
        ax,
        0.08,
        0.12,
        0.25,
        0.22,
        "Retained target spots",
        [r"old-to-new index map", r"$j_i^{true}\leftarrow\mathrm{reindex}(i)$", r"$m_i=0$"],
        TEAL,
        TEAL_LIGHT,
    )
    box(
        ax,
        0.375,
        0.12,
        0.21,
        0.22,
        "Cropped source spots",
        [r"$j_i^{true}=-1$", r"$m_i=1$"],
        RED,
        RED_LIGHT,
    )
    box(
        ax,
        0.66,
        0.12,
        0.30,
        0.22,
        "Evaluation eligibility",
        [
            r"correspondence: $j_i^{true}\geq0$",
            r"missingness: all source spots",
            r"fixed budget: correspondence + row mass $>10^{-12}$",
        ],
        BLUE,
        BLUE_LIGHT,
    )
    arrow(ax, (0.68, 0.45), (0.205, 0.35), color=OCHRE)
    arrow(ax, (0.68, 0.45), (0.48, 0.35), color=OCHRE)
    arrow(ax, (0.33, 0.24), (0.65, 0.255), color=GREY)
    arrow(ax, (0.595, 0.20), (0.65, 0.205), color=GREY)

    ax.text(
        0.02,
        0.035,
        (
            f"Manifest check: {manifest['source_spots'].sum():,} source spots; "
            f"{manifest['target_spots'].sum():,} retained targets; "
            f"{manifest['missing_source_spots'].sum():,} cropped "
            f"({100 * crop_fraction:.1f}%); retained label agreement = 1.000."
        ),
        ha="left",
        va="bottom",
        fontsize=6.9,
        color=GREY,
    )

    OUT.mkdir(parents=True, exist_ok=True)
    stem = OUT / "figS11_her2st_controlled_generator"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(
        stem.with_suffix(".tiff"),
        dpi=600,
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)
    print(f"Wrote {stem}.pdf/.svg/.png/.tiff")


if __name__ == "__main__":
    main()
