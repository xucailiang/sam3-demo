#!/usr/bin/env python3
"""Plot inference time vs accuracy scatter (Fig. S3).

Combines SAM3 runtime profiling data with supervised baseline timing from
literature-level estimates, plotting mean IoU vs ms/image.

Expected input:  experiments/results/runtime_profiling.csv
Expected output: experiments/visualizations/fig_runtime_accuracy_scatter.pdf/png
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
    csv_path = SCRIPT_DIR / "results" / "runtime_profiling.csv"
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found. Run run_profiling.py first.")
        return

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} methods from {csv_path}")

    # Color by protocol type (consistent palette)
    protocol_colors = {
        "deployable": "#2ca02c",           # green
        "gt-derived diagnostic": "#ff7f0e",  # orange
        "gt-assisted diagnostic": "#d62728", # red
        "supervised": "#7f7f7f",             # gray
    }
    protocol_markers = {
        "deployable": "o",
        "gt-derived diagnostic": "s",
        "gt-assisted diagnostic": "D",
        "supervised": "^",
    }

    fig, ax = plt.subplots(figsize=(9, 6))

    for _, row in df.iterrows():
        ptype = row.get("protocol_type", "unknown")
        color = protocol_colors.get(ptype, "#333333")
        marker = protocol_markers.get(ptype, "o")

        ax.errorbar(
            row["mean_ms_per_image"], row["mean_iou"],
            xerr=row.get("std_ms_per_image", 0),
            fmt=marker, color=color, markersize=10,
            capsize=4, capthick=1.2, elinewidth=1.0,
            label=ptype if ptype not in [ax.get_legend_handles_labels()[1]] else "",
        )

        # Method name label
        label = row["method"]
        ax.annotate(
            label, (row["mean_ms_per_image"], row["mean_iou"]),
            textcoords="offset points", xytext=(8, 4),
            fontsize=9, ha="left",
            bbox={"facecolor": "white", "alpha": 0.7, "pad": 1, "edgecolor": "none"},
        )

    ax.set_xlabel("Mean inference time (ms / image)", fontsize=12)
    ax.set_ylabel("Mean foreground IoU (DeepCrack)", fontsize=12)
    ax.set_title("Figure S3 — Inference Time vs Accuracy (DeepCrack)", fontsize=13)

    # Legend with unique protocol types
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    unique_labels = []
    unique_handles = []
    for k, v in by_label.items():
        if k not in unique_labels:
            unique_labels.append(k)
            unique_handles.append(v)
    ax.legend(unique_handles, unique_labels, fontsize=9, loc="lower right")

    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")  # Log scale for time typically makes sense

    fig.tight_layout()

    for fmt in ("png", "pdf"):
        out = SCRIPT_DIR / "visualizations" / f"fig_runtime_accuracy_scatter.{fmt}"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out), dpi=300, bbox_inches="tight")
        print(f"Saved {out}")

    plt.close(fig)


if __name__ == "__main__":
    main()
