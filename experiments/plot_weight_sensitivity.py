#!/usr/bin/env python3
"""Plot AMPF weight sensitivity heatmap (Fig. S2).

Generates a triangular heatmap over valid (alpha, beta) pairs where gamma >= 0.
The current paper configuration (α=0.4, β=0.35, γ=0.25) is marked.

Expected input:  experiments/results/ampf_weight_sensitivity.csv
Expected output: experiments/visualizations/fig_ampf_weight_sensitivity_heatmap.pdf/png
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", "/tmp/sam3-demo-matplotlib")


def main() -> None:
    csv_path = SCRIPT_DIR / "results" / "ampf_weight_sensitivity.csv"
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found. Run run_weight_sensitivity.py first.")
        return

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} weight combos from {csv_path}")

    # Pivot to heatmap data
    alphas = sorted(df["alpha"].unique())
    betas = sorted(df["beta"].unique())

    heatmap = np.full((len(betas), len(alphas)), np.nan)
    for _, row in df.iterrows():
        ai = alphas.index(row["alpha"])
        bi = betas.index(row["beta"])
        heatmap[bi, ai] = row["mean_iou"]

    fig, ax = plt.subplots(figsize=(8, 6.5))

    # Mask invalid combos not present in the CSV.
    masked = np.ma.masked_invalid(heatmap)

    im = ax.imshow(masked, origin="lower", cmap="viridis", aspect="auto",
                    extent=[alphas[0] - 0.05, alphas[-1] + 0.05,
                            betas[0] - 0.05, betas[-1] + 0.05])

    cbar = fig.colorbar(im, ax=ax, label="Mean foreground IoU (DeepCrack)", shrink=0.82)
    cbar.ax.tick_params(labelsize=9)

    # Mark current paper configuration
    ax.plot(0.4, 0.35, "r*", markersize=18, markeredgecolor="white", markeredgewidth=0.8,
            label="Current (α=0.4, β=0.35, γ=0.25)")
    ax.legend(fontsize=10, loc="upper right",
              framealpha=0.9, edgecolor="gray", facecolor="white")

    # Annotate best point
    if len(df) > 0:
        best = df.loc[df["mean_iou"].idxmax()]
        ax.plot(best["alpha"], best["beta"], "wo", markersize=10, markeredgecolor="black",
                markeredgewidth=0.6)
        ax.annotate(f"Best ({best['alpha']:.1f},{best['beta']:.1f},{best['gamma']:.1f})\nIoU={best['mean_iou']:.4f}",
                    xy=(best["alpha"], best["beta"]),
                    xytext=(best["alpha"] + 0.08, best["beta"] - 0.08),
                    fontsize=8, ha="left",
                    arrowprops=dict(arrowstyle="->", color="white", lw=1.2),
                    bbox={"facecolor": "white", "alpha": 0.85, "pad": 3, "edgecolor": "gray"})

    ax.set_xlabel("α (detection weight)", fontsize=12)
    ax.set_ylabel("β (stability weight)", fontsize=12)
    ax.set_title("Figure S2 — AMPF Weight Sensitivity (DeepCrack)", fontsize=13)

    # Add gamma annotation on the diagonal
    gamma_examples = []
    for a, b in [(0.1, 0.1), (0.2, 0.5), (0.5, 0.2), (0.6, 0.1)]:
        g = 1.0 - a - b
        if g > 0:
            gamma_examples.append(f"γ={g:.1f}")
    ax.text(0.75, 0.15, f"γ = 1 - α - β\nValid region: γ ≥ 0",
            transform=ax.transAxes, fontsize=8, va="bottom", ha="left",
            bbox={"facecolor": "white", "alpha": 0.8, "pad": 3, "edgecolor": "lightgray"})

    fig.tight_layout()

    # Add n_images note
    n_images = df["n_images"].iloc[0] if "n_images" in df.columns else "?"
    fig.text(0.5, 0.02, f"Based on {n_images} DeepCrack test images per weight combination. "
             f"Triangular region: α + β + γ = 1, with γ ≥ 0 boundary rows included.",
             ha="center", fontsize=8, style="italic")

    for fmt in ("png", "pdf"):
        out = SCRIPT_DIR / "visualizations" / f"fig_ampf_weight_sensitivity_heatmap.{fmt}"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out), dpi=300, bbox_inches="tight")
        print(f"Saved {out}")

    plt.close(fig)

    # Print summary
    print(f"\nIoU range: {df['mean_iou'].min():.4f} - {df['mean_iou'].max():.4f}")
    current = df[(df["alpha"] == 0.4) & (df["beta"] == 0.35)]
    if len(current) > 0:
        print(f"Current config IoU: {current.iloc[0]['mean_iou']:.4f}")
        better = (df["mean_iou"] > current.iloc[0]["mean_iou"]).sum()
        total_valid = len(df)
        print(f"Combos better than current: {better}/{total_valid} ({100*better/total_valid:.1f}%)")


if __name__ == "__main__":
    main()
