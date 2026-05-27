#!/usr/bin/env python3
"""Generate Figure 2: per-image foreground IoU distributions."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


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
]

METHOD_LABELS = {
    "text": "Text\n(auto)",
    "box": "Box\n(GT prompt)",
    "point": "Point\n(GT prompt)",
    "ampf": "AMPF\n(GT assisted)",
    "ablation_fusion_mean": "Mean Fusion\n(GT assisted)",
    "unet": "U-Net\n(supervised)",
    "deeplabv3plus": "DeepLabV3+\n(supervised)",
}

METHOD_COLORS = {
    "text": "#0072B2",
    "box": "#E69F00",
    "point": "#CC79A7",
    "ampf": "#D55E00",
    "ablation_fusion_mean": "#009E73",
    "unet": "#666666",
    "deeplabv3plus": "#222222",
}

DATASET_LABELS = {
    "crackforest": "CrackForest (n=24)",
    "deepcrack": "DeepCrack (n=237)",
}


def paired_pvalue(df: pd.DataFrame, dataset: str, method_a: str, method_b: str) -> float | None:
    """Return paired Wilcoxon p-value for two methods on one dataset."""
    sub = df[(df["dataset"] == dataset) & (df["method"].isin([method_a, method_b]))]
    pivot = sub.pivot_table(index="image_id", columns="method", values="iou", aggfunc="first")
    if method_a not in pivot.columns or method_b not in pivot.columns:
        return None
    paired = pivot[[method_a, method_b]].dropna()
    if paired.empty:
        return None
    diff = paired[method_a] - paired[method_b]
    if np.allclose(diff.to_numpy(), 0.0):
        return 1.0
    return float(wilcoxon(paired[method_a], paired[method_b]).pvalue)


def p_label(pvalue: float | None) -> str:
    """Format a p-value for compact figure annotations."""
    if pvalue is None:
        return "p=n/a"
    if pvalue < 1e-3:
        return f"p={pvalue:.1e}"
    return f"p={pvalue:.3f}"


def add_bracket(ax: plt.Axes, x1: float, x2: float, y: float, text: str) -> None:
    """Draw one significance bracket."""
    h = 0.018
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color="#333333", lw=0.8)
    ax.text((x1 + x2) / 2, y + h + 0.006, text, ha="center", va="bottom", fontsize=7.5)


def plot_dataset(ax: plt.Axes, df: pd.DataFrame, dataset: str) -> None:
    """Plot all selected method distributions for one dataset."""
    rng = np.random.default_rng(42)
    positions = np.arange(1, len(METHOD_ORDER) + 1)
    values: list[np.ndarray] = []

    for method in METHOD_ORDER:
        method_values = (
            df[(df["dataset"] == dataset) & (df["method"] == method)]["iou"]
            .dropna()
            .to_numpy()
        )
        values.append(method_values)

    parts = ax.violinplot(values, positions=positions, widths=0.72, showextrema=False)
    for body, method in zip(parts["bodies"], METHOD_ORDER):
        body.set_facecolor(METHOD_COLORS[method])
        body.set_edgecolor("#333333")
        body.set_alpha(0.28)
        body.set_linewidth(0.7)

    box = ax.boxplot(
        values,
        positions=positions,
        widths=0.28,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#111111", "linewidth": 1.2},
        boxprops={"linewidth": 0.8},
        whiskerprops={"linewidth": 0.8},
        capprops={"linewidth": 0.8},
    )
    for patch, method in zip(box["boxes"], METHOD_ORDER):
        patch.set_facecolor(METHOD_COLORS[method])
        patch.set_alpha(0.45)
        patch.set_edgecolor("#333333")

    for pos, method, method_values in zip(positions, METHOD_ORDER, values):
        if len(method_values) == 0:
            continue
        jitter = rng.uniform(-0.12, 0.12, size=len(method_values))
        point_size = 15 if dataset == "crackforest" else 5.5
        alpha = 0.78 if dataset == "crackforest" else 0.35
        ax.scatter(
            np.full(len(method_values), pos) + jitter,
            method_values,
            s=point_size,
            color=METHOD_COLORS[method],
            edgecolors="none",
            alpha=alpha,
            zorder=3,
        )

    comparisons = [
        ("ablation_fusion_mean", "text", 1.015),
        ("ampf", "text", 1.075),
        ("text", "unet", 1.135),
    ]
    for method_a, method_b, y in comparisons:
        x1 = METHOD_ORDER.index(method_a) + 1
        x2 = METHOD_ORDER.index(method_b) + 1
        pvalue = paired_pvalue(df, dataset, method_a, method_b)
        add_bracket(ax, min(x1, x2), max(x1, x2), y, p_label(pvalue))

    ax.set_title(DATASET_LABELS[dataset], fontsize=13, fontweight="bold")
    ax.set_xticks(positions)
    ax.set_xticklabels([METHOD_LABELS[m] for m in METHOD_ORDER], fontsize=8)
    ax.set_ylim(0, 1.19)
    ax.set_ylabel("Per-image foreground IoU")
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.6, alpha=0.85)
    ax.set_axisbelow(True)


def validate_methods(df: pd.DataFrame, methods: Iterable[str]) -> None:
    """Fail early if an expected method is missing."""
    missing = sorted(set(methods) - set(df["method"].unique()))
    if missing:
        raise ValueError(f"Missing methods in all_results_v2.csv: {missing}")


def main() -> None:
    path = RESULTS_DIR / "all_results_v2.csv"
    df = pd.read_csv(path)
    validate_methods(df, METHOD_ORDER)

    VIS_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 6.2), sharey=True)
    for ax, dataset in zip(axes, ["crackforest", "deepcrack"]):
        plot_dataset(ax, df, dataset)

    fig.suptitle("Per-Image IoU Distributions by Protocol", fontsize=16, fontweight="bold", y=0.99)
    fig.text(
        0.5,
        0.012,
        "AMPF and Mean Fusion are GT-assisted diagnostic protocols; p-values are paired descriptive comparisons, not deployability claims.",
        ha="center",
        va="bottom",
        fontsize=9.2,
        color="#444444",
    )
    fig.tight_layout(rect=(0, 0.045, 1, 0.955))

    for ext in ("png", "pdf"):
        out = VIS_DIR / f"fig_iou_distribution.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(out)
    plt.close(fig)


if __name__ == "__main__":
    main()
