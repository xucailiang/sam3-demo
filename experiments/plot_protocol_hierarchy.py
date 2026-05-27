#!/usr/bin/env python3
"""Generate Figure 1: protocol hierarchy and deployability framework."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


SCRIPT_DIR = Path(__file__).resolve().parent
VIS_DIR = SCRIPT_DIR / "visualizations"


LEVELS = [
    {
        "title": "Deployable Protocol",
        "side_title": "Deployable",
        "methods": 'Text Prompt ("crack")',
        "dependency": "No GT prompt, no GT alignment",
        "meaning": "Similar surface-crack screening\nLow-label review\nAnnotation support",
        "color": "#009E73",
    },
    {
        "title": "GT-Derived Diagnostic Protocol",
        "side_title": "GT-Derived\nDiagnostic",
        "methods": "Box Prompt / Point Prompt",
        "dependency": "GT used only to generate prompts",
        "meaning": "Prompt-mode diagnosis\nAnnotation workflow analysis",
        "color": "#E69F00",
    },
    {
        "title": "GT-Assisted Fusion Diagnostic Protocol",
        "side_title": "GT-Assisted Fusion\nDiagnostic",
        "methods": "AMPF / Mean Fusion / Ablations",
        "dependency": "GT prompt generation + GT instance alignment",
        "meaning": "Fusion potential analysis\nResearch diagnostic only",
        "color": "#D55E00",
    },
]


def add_box(ax: plt.Axes, x: float, y: float, w: float, h: float, spec: dict) -> None:
    """Draw one rounded protocol-level box."""
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        linewidth=1.4,
        edgecolor=spec["color"],
        facecolor=spec["color"],
        alpha=0.13,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h * 0.72,
        spec["title"],
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
        color="#222222",
    )
    ax.text(
        x + w / 2,
        y + h * 0.48,
        spec["methods"],
        ha="center",
        va="center",
        fontsize=11,
        color="#222222",
    )
    ax.text(
        x + w / 2,
        y + h * 0.25,
        spec["dependency"],
        ha="center",
        va="center",
        fontsize=9,
        color="#444444",
    )


def add_side_panel(ax: plt.Axes, x: float, y: float, w: float, h: float) -> None:
    """Draw the practical-meaning panel."""
    panel = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.016,rounding_size=0.02",
        linewidth=1.0,
        edgecolor="#666666",
        facecolor="#F7F7F7",
    )
    ax.add_patch(panel)
    ax.text(
        x + w / 2,
        y + h - 0.08,
        "Practical Meaning",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
        color="#222222",
    )

    row_h = (h - 0.16) / 3
    for i, spec in enumerate(LEVELS):
        row_y = y + h - 0.16 - (i + 1) * row_h
        ax.plot([x + 0.04, x + w - 0.04], [row_y + row_h, row_y + row_h], color="#DDDDDD", lw=0.8)
        ax.scatter(x + 0.07, row_y + row_h * 0.55, s=90, color=spec["color"], zorder=3)
        ax.text(
            x + 0.11,
            row_y + row_h * 0.72,
            spec["side_title"],
            ha="left",
            va="center",
            fontsize=8.9,
            fontweight="bold",
            color="#222222",
            linespacing=1.05,
        )
        ax.text(
            x + 0.11,
            row_y + row_h * 0.38,
            spec["meaning"],
            ha="left",
            va="center",
            fontsize=8.4,
            color="#444444",
            linespacing=1.15,
        )


def build_figure() -> plt.Figure:
    """Build the protocol hierarchy figure."""
    fig, ax = plt.subplots(figsize=(11.5, 6.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5,
        0.96,
        "Protocol Hierarchy and Deployability Framework",
        ha="center",
        va="center",
        fontsize=17,
        fontweight="bold",
        color="#111111",
    )

    left_x = 0.08
    box_w = 0.52
    box_h = 0.17
    ys = [0.69, 0.43, 0.17]

    for spec, y in zip(LEVELS, ys):
        add_box(ax, left_x, y, box_w, box_h, spec)

    for y0, y1 in [(ys[0], ys[1]), (ys[1], ys[2])]:
        arrow = FancyArrowPatch(
            (left_x + box_w / 2, y0),
            (left_x + box_w / 2, y1 + box_h),
            arrowstyle="-|>",
            mutation_scale=18,
            linewidth=1.2,
            color="#666666",
        )
        ax.add_patch(arrow)

    ax.text(
        left_x + box_w / 2,
        0.125,
        "More GT assistance, less direct deployability",
        ha="center",
        va="center",
        fontsize=10.5,
        color="#555555",
        fontstyle="italic",
    )

    add_side_panel(ax, 0.64, 0.17, 0.31, 0.69)

    ax.text(
        0.5,
        0.045,
        "GT-assisted methods are diagnostic in this study and require automatic, interactive, or prior-based replacements before deployment.",
        ha="center",
        va="center",
        fontsize=9.5,
        color="#555555",
    )
    return fig


def main() -> None:
    VIS_DIR.mkdir(parents=True, exist_ok=True)
    fig = build_figure()
    for ext in ("png", "pdf"):
        out = VIS_DIR / f"fig_protocol_hierarchy.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(out)
    plt.close(fig)


if __name__ == "__main__":
    main()
