#!/usr/bin/env python3
"""Generate Figure 3: precision-recall operating points."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
VIS_DIR = SCRIPT_DIR / "visualizations"


METHOD_ORDER = [
    "text",
    "box",
    "point",
    "ampf",
    "ablation_fusion_mean",
    "unet",
    "deeplabv3plus",
    "yolov8seg",
]

METHOD_LABELS = {
    "text": "Text",
    "box": "Box",
    "point": "Point",
    "ampf": "AMPF",
    "ablation_fusion_mean": "Mean Fusion",
    "unet": "U-Net",
    "deeplabv3plus": "DeepLabV3+",
    "yolov8seg": "YOLOv8-seg",
}

METHOD_COLORS = {
    "text": "#0072B2",
    "box": "#E69F00",
    "point": "#CC79A7",
    "ampf": "#D55E00",
    "ablation_fusion_mean": "#009E73",
    "unet": "#666666",
    "deeplabv3plus": "#222222",
    "yolov8seg": "#999999",
}

METHOD_MARKERS = {
    "text": "o",
    "box": "s",
    "point": "^",
    "ampf": "D",
    "ablation_fusion_mean": "P",
    "unet": "X",
    "deeplabv3plus": "v",
    "yolov8seg": "*",
}

DATASET_LABELS = {
    "crackforest": "CrackForest",
    "deepcrack": "DeepCrack",
}


def add_f1_contours(ax: plt.Axes) -> None:
    """Add light iso-F1 contours to a precision-recall plot."""
    recall = np.linspace(0.35, 1.0, 240)
    precision = np.linspace(0.25, 1.0, 240)
    r_grid, p_grid = np.meshgrid(recall, precision)
    f1 = 2 * p_grid * r_grid / (p_grid + r_grid + 1e-12)
    contours = ax.contour(
        r_grid,
        p_grid,
        f1,
        levels=[0.5, 0.6, 0.7, 0.8],
        colors="#BBBBBB",
        linewidths=0.7,
        linestyles="dashed",
        alpha=0.85,
    )
    ax.clabel(contours, inline=True, fontsize=7, fmt="F1 %.1f")


def plot_dataset(ax: plt.Axes, summary: pd.DataFrame, dataset: str) -> None:
    """Plot PR operating points for one dataset."""
    sub = summary[summary["dataset"] == dataset].set_index("method")
    add_f1_contours(ax)

    for method in METHOD_ORDER:
        if method not in sub.index:
            continue
        row = sub.loc[method]
        ax.scatter(
            row["recall"],
            row["precision"],
            s=105 if method != "yolov8seg" else 140,
            marker=METHOD_MARKERS[method],
            color=METHOD_COLORS[method],
            edgecolor="white",
            linewidth=0.9,
            zorder=4,
            label=METHOD_LABELS[method],
        )

        dx, dy = {
            "text": (0.008, -0.018),
            "box": (0.008, 0.012),
            "point": (0.008, -0.020),
            "ampf": (0.008, 0.011),
            "ablation_fusion_mean": (0.008, -0.020),
            "unet": (-0.065, 0.012),
            "deeplabv3plus": (0.008, 0.012),
            "yolov8seg": (0.008, -0.022),
        }[method]
        ax.text(
            row["recall"] + dx,
            row["precision"] + dy,
            METHOD_LABELS[method],
            fontsize=8.2,
            color="#222222",
        )

    chain = ["text", "ampf", "ablation_fusion_mean"]
    for start, end in zip(chain, chain[1:]):
        if start not in sub.index or end not in sub.index:
            continue
        start_row = sub.loc[start]
        end_row = sub.loc[end]
        ax.annotate(
            "",
            xy=(end_row["recall"], end_row["precision"]),
            xytext=(start_row["recall"], start_row["precision"]),
            arrowprops={
                "arrowstyle": "->",
                "color": "#555555",
                "lw": 1.1,
                "shrinkA": 8,
                "shrinkB": 8,
            },
            zorder=3,
        )

    ax.set_title(DATASET_LABELS[dataset], fontsize=13, fontweight="bold")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0.40, 1.0)
    ax.set_ylim(0.30, 0.88)
    ax.grid(color="#DDDDDD", linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)


def main() -> None:
    path = RESULTS_DIR / "summary_stats_v2.csv"
    summary = pd.read_csv(path)

    VIS_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.8), sharex=True, sharey=True)
    for ax, dataset in zip(axes, ["crackforest", "deepcrack"]):
        plot_dataset(ax, summary, dataset)

    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=8, frameon=False, bbox_to_anchor=(0.5, 0.965), fontsize=8.5)
    fig.suptitle("Precision-Recall Operating Points", fontsize=16, fontweight="bold", y=1.03)
    fig.text(
        0.5,
        0.012,
        "Arrows show the descriptive movement from Text to GT-assisted AMPF and Mean Fusion; they do not indicate a deployable processing chain.",
        ha="center",
        va="bottom",
        fontsize=9.2,
        color="#444444",
    )
    fig.tight_layout(rect=(0, 0.045, 1, 0.9))

    for ext in ("png", "pdf"):
        out = VIS_DIR / f"fig_precision_recall_operating_points.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(out)
    plt.close(fig)


if __name__ == "__main__":
    main()
