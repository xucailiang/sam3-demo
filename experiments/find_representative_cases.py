#!/usr/bin/env python3
"""Find representative cases from all_results_v2.csv for qualitative figure.

Selection criteria (following PAPER_FIGURE_OPTIMIZATION_PLAN.md Sec 3.1):
  Row 1 — Text Over-segmentation: text low precision + high recall
  Row 2 — Point Local Recovery:   point IoU << text IoU
  Row 3 — AMPF False Negative:     ampf recall < text recall
  Row 4 — Mean Fusion Complementary: fusion_mean IoU > text IoU
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent

# Load
df = pd.read_csv(SCRIPT_DIR / "results" / "all_results_v2.csv")

# Check available columns
METHOD_COL = "method" if "method" in df.columns else "model"

# Pivot to per-image × per-method metrics
# We need: text, point, ampf, ablation_fusion_mean
metrics_cols = ["iou", "dice", "precision", "recall", "f1"]

pivots = {}
for metric in metrics_cols:
    pv = df.pivot_table(
        index=["dataset", "image_id"],
        columns=METHOD_COL,
        values=metric,
    )
    pivots[metric] = pv

# Build a combined DataFrame with all needed metrics
combined_rows = []
for (ds, img_id), _ in pivots["iou"].iterrows():
    row = {"dataset": ds, "image_id": img_id}
    for metric in metrics_cols:
        for method in ["text", "box", "point", "ampf", "ablation_fusion_mean"]:
            if method in pivots[metric].columns:
                val = pivots[metric].loc[(ds, img_id), method]
                if not pd.isna(val):
                    row[f"{method}_{metric}"] = val
    combined_rows.append(row)

cases = pd.DataFrame(combined_rows)

# ── Row 1: Text Over-segmentation ──────────────────────────────────────────
# Text precision is LOW while recall is HIGH (high FP rate)
# Focus on DeepCrack (more samples, more reliable)
dc = cases[cases["dataset"] == "deepcrack"].copy()
dc_has_text = dc.dropna(subset=["text_precision", "text_recall"])

# Rank by: lowest precision among high-recall samples
dc_has_text["over_seg_score"] = (
    dc_has_text["text_recall"] - dc_has_text["text_precision"]
)
over_seg = dc_has_text.sort_values("over_seg_score", ascending=False)

print("=" * 72)
print("Row 1 candidates — Text Over-segmentation (high recall, low precision)")
print("=" * 72)
for _, r in over_seg.head(5).iterrows():
    print(
        f"  {r['dataset']}:{r['image_id']}  "
        f"P={r['text_precision']:.3f}  R={r['text_recall']:.3f}  "
        f"IoU={r['text_iou']:.3f}"
    )

# ── Row 2: Point Local Recovery ────────────────────────────────────────────
# Point IoU significantly lower than text IoU
dc_has_both = dc.dropna(subset=["text_iou", "point_iou"]).copy()
dc_has_both["point_text_diff"] = dc_has_both["text_iou"] - dc_has_both["point_iou"]
point_local = dc_has_both.sort_values("point_text_diff", ascending=False)

print()
print("=" * 72)
print("Row 2 candidates — Point Local Recovery (Point IoU << Text IoU)")
print("=" * 72)
for _, r in point_local.head(5).iterrows():
    print(
        f"  {r['dataset']}:{r['image_id']}  "
        f"Text_IoU={r['text_iou']:.3f}  Point_IoU={r['point_iou']:.3f}  "
        f"Δ={r['point_text_diff']:.3f}"
    )

# ── Row 3: AMPF False Negative ─────────────────────────────────────────────
# AMPF recall is lower than Text recall
dc_has_ampf = dc.dropna(subset=["text_recall", "ampf_recall"]).copy()
dc_has_ampf["recall_drop"] = dc_has_ampf["text_recall"] - dc_has_ampf["ampf_recall"]
ampf_fn = dc_has_ampf.sort_values("recall_drop", ascending=False)

print()
print("=" * 72)
print("Row 3 candidates — AMPF False Negative (AMPF recall < Text recall)")
print("=" * 72)
for _, r in ampf_fn.head(5).iterrows():
    print(
        f"  {r['dataset']}:{r['image_id']}  "
        f"Text_R={r['text_recall']:.3f}  AMPF_R={r['ampf_recall']:.3f}  "
        f"ΔR={r['recall_drop']:.3f}  Text_IoU={r['text_iou']:.3f}  AMPF_IoU={r['ampf_iou']:.3f}"
    )

# ── Row 4: Mean Fusion Complementary Recovery ──────────────────────────────
# fusion_mean IoU > text IoU (multi-prompt fusion helps)
dc_has_fm = dc.dropna(subset=["text_iou", "ablation_fusion_mean_iou"]).copy()
dc_has_fm["fusion_gain"] = (
    dc_has_fm["ablation_fusion_mean_iou"] - dc_has_fm["text_iou"]
)
fusion_gain = dc_has_fm.sort_values("fusion_gain", ascending=False)

print()
print("=" * 72)
print("Row 4 candidates — Mean Fusion Complementary (fusion_mean IoU > text IoU)")
print("=" * 72)
for _, r in fusion_gain.head(5).iterrows():
    print(
        f"  {r['dataset']}:{r['image_id']}  "
        f"Text_IoU={r['text_iou']:.3f}  MeanFusion_IoU={r['ablation_fusion_mean_iou']:.3f}  "
        f"Gain={r['fusion_gain']:.3f}"
    )

# ── Also check CrackForest ─────────────────────────────────────────────────
print()
print("=" * 72)
print("CrackForest candidates (small test set, n=24 — interpret with caution)")
print("=" * 72)
cf = cases[cases["dataset"] == "crackforest"].copy()

cf_both = cf.dropna(subset=["text_iou", "point_iou"])
if len(cf_both) > 0:
    cf_both["diff"] = cf_both["text_iou"] - cf_both["point_iou"]
    top = cf_both.sort_values("diff", ascending=False).head(3)
    for _, r in top.iterrows():
        print(
            f"  {r['dataset']}:{r['image_id']}  "
            f"Text_IoU={r['text_iou']:.3f}  Point_IoU={r['point_iou']:.3f}"
        )

print()
print("=" * 72)
print("RECOMMENDED CASES for --case arguments:")
print("=" * 72)

# Pick the best representative for each row, preferring DeepCrack
r1 = over_seg.iloc[0]
r2 = point_local.iloc[0]
r3 = ampf_fn.iloc[0]
r4 = fusion_gain.iloc[0]

# Parse integer index from image_id
def parse_idx(img_id: str) -> int:
    parts = img_id.split("_")
    return int(parts[-1])

print(f"  --case {r1['dataset']}:{parse_idx(r1['image_id'])}  # Row 1: Text Over-segmentation")
print(f"  --case {r2['dataset']}:{parse_idx(r2['image_id'])}  # Row 2: Point Local Recovery")
print(f"  --case {r3['dataset']}:{parse_idx(r3['image_id'])}  # Row 3: AMPF False Negative")
print(f"  --case {r4['dataset']}:{parse_idx(r4['image_id'])}  # Row 4: Mean Fusion Complementary")
