#!/usr/bin/env python3
"""Plot calibration reliability diagram from candidate-level confidence data.

Generates Fig. S1: Reliability diagram with ECE for Text, Box, Point candidates.

Expected input:  experiments/results/candidate_confidence.csv
Expected output: experiments/visualizations/fig_calibration_reliability.pdf/png
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


def compute_ece(confidences: np.ndarray, accuracies: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if in_bin.sum() == 0:
            continue
        bin_conf = confidences[in_bin].mean()
        bin_acc = accuracies[in_bin].mean()
        ece += (in_bin.sum() / len(confidences)) * abs(bin_acc - bin_conf)
    return ece


def reliability_curve(
    confidences: np.ndarray, ious: np.ndarray, n_bins: int = 10
) -> tuple[np.ndarray, np.ndarray]:
    """Return (mean_confidence_per_bin, mean_iou_per_bin)."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
    mean_confs = []
    mean_ious = []
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if in_bin.sum() == 0:
            mean_confs.append(bin_centers[i])
            mean_ious.append(np.nan)
        else:
            mean_confs.append(confidences[in_bin].mean())
            mean_ious.append(ious[in_bin].mean())
    return np.array(mean_confs), np.array(mean_ious)


def main() -> None:
    csv_path = SCRIPT_DIR / "results" / "candidate_confidence.csv"
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found. Run run_candidate_logging.py first.")
        return

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} candidates from {csv_path}")

    # Color palette (consistent with other figures)
    colors = {"text": "#1f77b4", "box": "#ff7f0e", "point": "#9467bd"}
    labels = {"text": "Text", "box": "Box", "point": "Point"}

    datasets = [
        dataset
        for dataset in ["deepcrack", "crackforest"]
        if not df[df["dataset"] == dataset].empty
    ]
    if not datasets:
        raise ValueError("No calibration records found for DeepCrack or CrackForest.")

    fig, axes = plt.subplots(
        1,
        len(datasets),
        figsize=(7 * len(datasets), 5.5),
        squeeze=False,
    )
    axes = axes[0]
    n_bins = 10

    for ax_idx, dataset in enumerate(datasets):
        ax = axes[ax_idx]
        ds_df = df[df["dataset"] == dataset]

        ece_text = ""
        for mode in ["text", "box", "point"]:
            mode_df = ds_df[ds_df["prompt_mode"] == mode]
            if len(mode_df) < 10:
                continue

            # Use composite_confidence for prediction, candidate_iou for actual quality
            confs = mode_df["composite_confidence"].values
            ious = mode_df["candidate_iou"].values

            # Clip to [0, 1]
            confs = np.clip(confs, 0, 1)

            mean_conf, mean_iou = reliability_curve(confs, ious, n_bins)
            ece = compute_ece(confs, ious, n_bins)

            valid = ~np.isnan(mean_iou)
            ax.plot(mean_conf[valid], mean_iou[valid], "o-", color=colors[mode],
                    label=f"{labels[mode]} (ECE={ece:.3f})", markersize=5, linewidth=1.5)
            ece_text += f"  {labels[mode]} ECE={ece:.4f}\n"

        # Perfect calibration diagonal
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.4, label="Perfect calibration")

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Mean predicted confidence", fontsize=11)
        ax.set_ylabel("Mean candidate IoU", fontsize=11)
        ax.set_title(f"{dataset} (n={ds_df['image_id'].nunique()} images)", fontsize=12)
        ax.legend(fontsize=9, loc="lower right")
        ax.grid(True, alpha=0.3)

        # ECE annotation
        ax.text(0.02, 0.98, ece_text.strip(), transform=ax.transAxes,
                va="top", fontsize=8, family="monospace",
                bbox={"facecolor": "white", "alpha": 0.8, "pad": 4})

    fig.suptitle("Figure S1 — Confidence Calibration Reliability Diagram", fontsize=14, y=1.01)
    fig.tight_layout()

    for fmt in ("png", "pdf"):
        out = SCRIPT_DIR / "visualizations" / f"fig_calibration_reliability.{fmt}"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out), dpi=300, bbox_inches="tight")
        print(f"Saved {out}")

    plt.close(fig)


if __name__ == "__main__":
    main()
