#!/usr/bin/env python3
"""Rebuild paper results, tables, and visualizations with v2 point protocol.

Merges unchanged old results (text, box, supervised) with new v2 results
from the nearest-foreground point protocol re-run. Outputs:

  - experiments/results/all_results_v2.csv       merged per-image results
  - experiments/results/summary_stats_v2.csv      per-method summary
  - experiments/results/paper_tables_v2.tex       LaTeX tables
  - experiments/visualizations/                   comparison charts
"""

import sys
from pathlib import Path
from copy import deepcopy

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = _SCRIPT_DIR / "results"
OLD_RESULTS_DIR = RESULTS_DIR / "old"
VIS_DIR = _SCRIPT_DIR / "visualizations"


# ---------------------------------------------------------------------------
# Step 1: Load and merge
# ---------------------------------------------------------------------------

def load_old_single(method: str, dataset: str) -> pd.DataFrame:
    """Load old unchanged single-mode results."""
    if dataset == "crackforest":
        path = RESULTS_DIR / f"crackforest_{method}.csv"
    else:
        path = RESULTS_DIR / f"deepcrack_full_{method}.csv"
        if not path.exists():
            path = RESULTS_DIR / f"deepcrack_{method}.csv"
    if path.exists():
        df = pd.read_csv(path)
        # Normalise method name
        df["method"] = method
        return df
    return pd.DataFrame()


def load_v2(method_suffix: str, dataset: str) -> pd.DataFrame:
    """Load v2 point-dependent results."""
    path = RESULTS_DIR / f"{dataset}_{method_suffix}_v2.csv"
    if path.exists():
        df = pd.read_csv(path)
        # Strip _v2 suffix for clean method names
        df["method"] = df["method"].str.replace("_v2", "", regex=False)
        return df
    return pd.DataFrame()


def load_baselines(dataset: str) -> pd.DataFrame:
    """Load supervised baseline results from results/ or results/old/.

    Prefers files whose row count matches the expected test-set size so that
    corrupt / partial files are skipped.  For CrackForest the canonical
    non-'supervised_' filenames also preserve the original manuscript values.
    """
    expected_rows = {"crackforest": 24, "deepcrack": 237}

    frames = []
    method_files = {
        "unet": [
            f"{dataset}_unet.csv",
            f"{dataset}_supervised_unet.csv",
        ],
        "deeplabv3plus": [
            f"{dataset}_deeplabv3plus.csv",
            f"{dataset}_deeplabv3p.csv",
            f"{dataset}_supervised_deeplabv3plus.csv",
            f"{dataset}_supervised_deeplabv3p.csv",
        ],
        "yolov8seg": [
            f"{dataset}_yolov8seg.csv",
            f"{dataset}_supervised_yolov8seg.csv",
        ],
    }
    for method, paths in method_files.items():
        best_df = None
        for p in paths:
            for base in (RESULTS_DIR, OLD_RESULTS_DIR):
                candidate = base / p
                if not candidate.exists():
                    continue
                df = pd.read_csv(candidate)
                if len(df) >= expected_rows.get(dataset, 1):
                    best_df = df  # first complete match wins
                    break
            if best_df is not None:
                break
        if best_df is not None:
            best_df["method"] = best_df["method"].replace({
                "deeplabv3p": "deeplabv3plus",
            })
            best_df["method"] = method
            frames.append(best_df)
    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame()


def merge_all() -> pd.DataFrame:
    """Assemble the complete v2 results dataset."""
    all_frames = []

    for dataset in ["crackforest", "deepcrack"]:
        # Unchanged: text, box
        for method in ["text", "box"]:
            df = load_old_single(method, dataset)
            if not df.empty:
                all_frames.append(df)

        # v2 point-dependent
        for method in [
            "point", "ampf",
            "ablation_no_detection", "ablation_no_stability",
            "ablation_no_boundary", "ablation_no_alignment",
            "ablation_fusion_max", "ablation_fusion_mean",
        ]:
            df = load_v2(method, dataset)
            if not df.empty:
                all_frames.append(df)

        # v2 combos (internal names may be reversed, e.g. "point+text")
        for combo in ["text+point", "box+point"]:
            df = load_v2(f"combo_{combo}", dataset)
            if not df.empty:
                all_frames.append(df)

        # Old combos not involving point (text+box unchanged)
        for combo in ["text+box"]:
            path = None
            for candidate in [
                RESULTS_DIR / f"{dataset}_full_combo_{combo}.csv",
                OLD_RESULTS_DIR / f"{dataset}_full_combo_{combo}.csv",
                RESULTS_DIR / f"{dataset}_combo_{combo}.csv",
                OLD_RESULTS_DIR / f"{dataset}_combo_{combo}.csv",
            ]:
                if candidate.exists():
                    path = candidate
                    break
            if path:
                df = pd.read_csv(path)
                all_frames.append(df)

        # Supervised baselines (unchanged)
        df = load_baselines(dataset)
        if not df.empty:
            all_frames.append(df)

    combined = pd.concat(all_frames, ignore_index=True)

    # Normalise combo method names: ensure consistent order (text > box > point)
    combo_renames = {
        "box+text": "text+box",
        "point+text": "text+point",
    }
    combined["method"] = combined["method"].replace(combo_renames)

    # Normalise protocol column
    if "protocol" not in combined.columns:
        combined["protocol"] = None
    known_protocols = {
        "text": "automatic_text_prompt",
        "box": "oracle_gt_prompt",
        "point": "oracle_gt_prompt",
        "ampf": "oracle_gt_prompt_gt_alignment",
        "ablation_no_detection": "oracle_gt_prompt_gt_alignment",
        "ablation_no_stability": "oracle_gt_prompt_gt_alignment",
        "ablation_no_boundary": "oracle_gt_prompt_gt_alignment",
        "ablation_no_alignment": "oracle_gt_prompt_gt_alignment",
        "ablation_fusion_max": "oracle_gt_prompt_gt_alignment",
        "ablation_fusion_mean": "oracle_gt_prompt_gt_alignment",
        "text+box": "oracle_gt_prompt_gt_alignment",
        "text+point": "oracle_gt_prompt_gt_alignment",
        "box+point": "oracle_gt_prompt_gt_alignment",
        "point+text": "oracle_gt_prompt_gt_alignment",
        "unet": "supervised_train_val_test",
        "deeplabv3plus": "supervised_train_val_test",
        "yolov8seg": "supervised_train_val_test",
    }
    for method, proto in known_protocols.items():
        combined.loc[combined["method"] == method, "protocol"] = proto
    # Also handle "box+text" = "text+box" equivalence
    combined.loc[combined["method"] == "box+text", "protocol"] = "oracle_gt_prompt_gt_alignment"

    return combined


# ---------------------------------------------------------------------------
# Step 2: Summary statistics
# ---------------------------------------------------------------------------

def compute_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-method per-dataset mean metrics."""
    metrics = ["iou", "dice", "precision", "recall"]
    if "f1" in df.columns:
        metrics.append("f1")
    group_cols = ["method", "dataset"]
    if "protocol" in df.columns:
        group_cols.append("protocol")
    return df.groupby(group_cols, dropna=False)[metrics].mean().reset_index()


# ---------------------------------------------------------------------------
# Step 3: LaTeX tables
# ---------------------------------------------------------------------------

TABLE_METHOD_ORDER = [
    "text", "box", "point",
    "ampf",
    "ablation_no_detection", "ablation_no_stability", "ablation_no_boundary",
    "ablation_no_alignment", "ablation_fusion_max", "ablation_fusion_mean",
    "text+box", "text+point", "box+point",
    "unet", "deeplabv3plus", "yolov8seg",
]

TABLE_METHOD_LABELS = {
    "text": "Text",
    "box": "Box",
    "point": "Point (nearest-fg)",
    "ampf": "AMPF",
    "ablation_no_detection": "w/o $S_{det}$",
    "ablation_no_stability": "w/o $S_{stab}$",
    "ablation_no_boundary": "w/o $S_{bnd}$",
    "ablation_no_alignment": "w/o alignment",
    "ablation_fusion_max": "fusion = max",
    "ablation_fusion_mean": "fusion = mean",
    "text+box": "Text + Box",
    "text+point": "Text + Point",
    "box+point": "Box + Point",
    "unet": "U-Net",
    "deeplabv3plus": "DeepLabV3+",
    "yolov8seg": "YOLOv8-seg",
}

TABLE_PROTOCOL_LABELS = {
    "text": "automatic text prompt",
    "box": "GT-derived box",
    "point": "GT-derived nearest-fg point",
    "ampf": "GT-assisted prompt + alignment",
}


def build_latex_tables(summary: pd.DataFrame) -> str:
    """Generate paper LaTeX tables (4 tables)."""
    lines = []

    def _get_metrics(summary: pd.DataFrame, ds: str, methods: list) -> pd.DataFrame:
        sub = summary[summary["dataset"] == ds].set_index("method")
        ordered = [m for m in methods if m in sub.index]
        return sub.loc[ordered]

    # Table 1: Main comparison
    methods_t1 = ["text", "box", "point", "ampf", "text+box", "text+point", "box+point"]
    sub_cf = _get_metrics(summary, "crackforest", methods_t1)
    sub_dc = _get_metrics(summary, "deepcrack", methods_t1)

    lines.append("% Table 1: Main comparison (nearest-foreground point protocol)")
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\caption{Foreground IoU and Dice on CrackForest and DeepCrack.}")
    ncol = 5
    cols = "l" + "r" * (ncol - 1)
    lines.append(r"\begin{tabular}{" + cols + "}")
    lines.append(r"\toprule")
    lines.append(r"Method & CrackForest IoU & CrackForest Dice & DeepCrack IoU & DeepCrack Dice \\")
    lines.append(r"\midrule")
    for m in methods_t1:
        label = TABLE_METHOD_LABELS.get(m, m)
        if m in sub_cf.index and m in sub_dc.index:
            cf_iou = sub_cf.loc[m, "iou"]
            cf_dice = sub_cf.loc[m, "dice"]
            dc_iou = sub_dc.loc[m, "iou"]
            dc_dice = sub_dc.loc[m, "dice"]
            # Bold best
            bf_cf = lambda v, col: f"\\textbf{{{v:.4f}}}" if v == sub_cf[col].max() else f"{v:.4f}"
            bf_dc = lambda v, col: f"\\textbf{{{v:.4f}}}" if v == sub_dc[col].max() else f"{v:.4f}"
            lines.append(
                f"{label} & {bf_cf(cf_iou, 'iou')} & {bf_cf(cf_dice, 'dice')} & "
                f"{bf_dc(dc_iou, 'iou')} & {bf_dc(dc_dice, 'dice')} \\\\"
            )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")

    # Table 2: Ablation
    methods_t2 = [
        "ampf", "ablation_no_detection", "ablation_no_stability",
        "ablation_no_boundary", "ablation_no_alignment",
        "ablation_fusion_max", "ablation_fusion_mean",
    ]
    sub_cf2 = _get_metrics(summary, "crackforest", methods_t2)
    sub_dc2 = _get_metrics(summary, "deepcrack", methods_t2)

    lines.append("% Table 2: Ablation study")
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\caption{Ablation study of AMPF components and fusion rules.}")
    lines.append(r"\begin{tabular}{lrrrr}")
    lines.append(r"\toprule")
    lines.append(r"Configuration & CrackForest IoU & CrackForest Dice & DeepCrack IoU & DeepCrack Dice \\")
    lines.append(r"\midrule")
    for m in methods_t2:
        label = TABLE_METHOD_LABELS.get(m, m)
        if m in sub_cf2.index and m in sub_dc2.index:
            cf_iou = sub_cf2.loc[m, "iou"]
            cf_dice = sub_cf2.loc[m, "dice"]
            dc_iou = sub_dc2.loc[m, "iou"]
            dc_dice = sub_dc2.loc[m, "dice"]
            bf_cf = lambda v, col: f"\\textbf{{{v:.4f}}}" if v == sub_cf2[col].max() else f"{v:.4f}"
            bf_dc = lambda v, col: f"\\textbf{{{v:.4f}}}" if v == sub_dc2[col].max() else f"{v:.4f}"
            lines.append(
                f"{label} & {bf_cf(cf_iou, 'iou')} & {bf_cf(cf_dice, 'dice')} & "
                f"{bf_dc(dc_iou, 'iou')} & {bf_dc(dc_dice, 'dice')} \\\\"
            )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")

    # Table 3: Protocol-stratified supervised comparison
    methods_t3 = ["text", "ampf", "ablation_fusion_mean", "unet", "deeplabv3plus", "yolov8seg"]
    sub_cf3 = _get_metrics(summary, "crackforest", methods_t3)
    sub_dc3 = _get_metrics(summary, "deepcrack", methods_t3)

    protocol_labels_t3 = {
        "text": "automatic text prompt",
        "ampf": "GT-assisted fusion diagnostic",
        "ablation_fusion_mean": "GT-assisted fusion diagnostic",
        "unet": "supervised train/val/test",
        "deeplabv3plus": "supervised train/val/test",
        "yolov8seg": "supervised train/val/test",
    }

    lines.append("% Table 3: Protocol-stratified comparison with supervised baselines")
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\caption{Protocol-stratified comparison with supervised baselines. Bold values should be interpreted within protocol context only and do not indicate deployable superiority across automatic, GT-assisted, and supervised protocols.}")
    lines.append(r"\begin{tabular}{llrrrr}")
    lines.append(r"\toprule")
    lines.append(r"Method & Protocol & CrackForest IoU & CrackForest Dice & DeepCrack IoU & DeepCrack Dice \\")
    lines.append(r"\midrule")
    for m in methods_t3:
        label = TABLE_METHOD_LABELS.get(m, m)
        if m in sub_cf3.index and m in sub_dc3.index:
            cf_iou = sub_cf3.loc[m, "iou"]
            cf_dice = sub_cf3.loc[m, "dice"]
            dc_iou = sub_dc3.loc[m, "iou"]
            dc_dice = sub_dc3.loc[m, "dice"]
            bf_cf = lambda v, col: f"\\textbf{{{v:.4f}}}" if m != "ablation_fusion_mean" and v == sub_cf3[col].drop(index=["ablation_fusion_mean"], errors="ignore").max() else f"{v:.4f}"
            bf_dc = lambda v, col: f"\\textbf{{{v:.4f}}}" if m != "ablation_fusion_mean" and v == sub_dc3[col].drop(index=["ablation_fusion_mean"], errors="ignore").max() else f"{v:.4f}"
            protocol = protocol_labels_t3.get(m, "")
            lines.append(
                f"{label} & {protocol} & {bf_cf(cf_iou, 'iou')} & {bf_cf(cf_dice, 'dice')} & "
                f"{bf_dc(dc_iou, 'iou')} & {bf_dc(dc_dice, 'dice')} \\\\"
            )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")

    # Table 4: DeepCrack full detail
    methods_t4 = [
        "text", "box", "point", "ampf",
        "ablation_no_detection", "ablation_no_stability", "ablation_no_boundary",
        "ablation_no_alignment", "ablation_fusion_max", "ablation_fusion_mean",
        "text+box", "text+point", "box+point",
        "unet", "deeplabv3plus", "yolov8seg",
    ]
    sub_dc4 = _get_metrics(summary, "deepcrack", methods_t4)

    lines.append("% Table 4: DeepCrack full results (nearest-foreground point)")
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\caption{Complete DeepCrack results.}")
    lines.append(r"\begin{tabular}{lrrrrr}")
    lines.append(r"\toprule")
    lines.append(r"Method & IoU & Dice & Precision & Recall & F1 \\")
    lines.append(r"\midrule")
    for m in methods_t4:
        if m in sub_dc4.index:
            label = TABLE_METHOD_LABELS.get(m, m)
            row = sub_dc4.loc[m]
            lines.append(
                f"{label} & {row['iou']:.4f} & {row['dice']:.4f} & "
                f"{row['precision']:.4f} & {row['recall']:.4f} & "
                f"{row.get('f1', row['dice']):.4f} \\\\"
            )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Step 4: Visualizations
# ---------------------------------------------------------------------------

METRIC_NAMES = {"iou": "IoU", "dice": "Dice", "precision": "Precision", "recall": "Recall"}
COLORS = plt.cm.tab10.colors


def _plot_comparison(summary: pd.DataFrame, ds: str, methods: list, title: str, filename: str):
    """Bar chart comparing methods on one dataset."""
    sub = summary[summary["dataset"] == ds].set_index("method")
    ordered = [m for m in methods if m in sub.index]
    if not ordered:
        return

    labels = [TABLE_METHOD_LABELS.get(m, m) for m in ordered]
    metrics_to_plot = ["iou", "dice", "precision", "recall"]
    x = np.arange(len(ordered))
    width = 0.2

    fig, ax = plt.subplots(figsize=(max(10, len(ordered) * 1.2), 5))
    for i, metric in enumerate(metrics_to_plot):
        values = [sub.loc[m, metric] if metric in sub.columns else 0 for m in ordered]
        bars = ax.bar(x + i * width, values, width, label=METRIC_NAMES[metric], color=COLORS[i])
        for bar in bars:
            if bar.get_height() > 0.02:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    fig.savefig(VIS_DIR / filename, dpi=150)
    plt.close(fig)


def generate_visualizations(summary: pd.DataFrame):
    """Generate all paper figures."""
    VIS_DIR.mkdir(parents=True, exist_ok=True)

    # Figure 1: Main comparison
    methods_main = ["text", "box", "point", "ampf", "text+box", "text+point", "box+point"]
    _plot_comparison(
        summary, "crackforest", methods_main,
        "CrackForest: Prompt Modes and AMPF (nearest-fg point)",
        "fig_main_comparison_crackforest.png",
    )
    _plot_comparison(
        summary, "deepcrack", methods_main,
        "DeepCrack: Prompt Modes and AMPF (nearest-fg point)",
        "fig_main_comparison_deepcrack.png",
    )

    # Figure 2: Ablation
    methods_ablation = [
        "ampf", "ablation_no_detection", "ablation_no_stability",
        "ablation_no_boundary", "ablation_no_alignment",
        "ablation_fusion_max", "ablation_fusion_mean",
    ]
    for ds in ["crackforest", "deepcrack"]:
        _plot_comparison(
            summary, ds, methods_ablation,
            f"{ds}: Ablation Study (nearest-fg point)",
            f"fig_ablation_{ds}.png",
        )

    # Figure 3: Supervised comparison
    methods_sup = ["text", "ampf", "ablation_fusion_mean", "unet", "deeplabv3plus", "yolov8seg"]
    for ds in ["crackforest", "deepcrack"]:
        _plot_comparison(
            summary, ds, methods_sup,
            f"{ds}: SAM3 vs. Supervised Baselines (nearest-fg point)",
            f"fig_supervised_{ds}.png",
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Rebuilding paper materials with v2 (nearest-foreground) point protocol")
    print("=" * 60)

    # 1. Merge
    print("\n[1/4] Merging old and v2 results...")
    combined = merge_all()
    print(f"  Merged: {len(combined)} rows, {combined['method'].nunique()} methods")

    # 2. Summary
    print("\n[2/4] Computing summary statistics...")
    summary = compute_summary(combined)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_csv(RESULTS_DIR / "all_results_v2.csv", index=False)
    summary.to_csv(RESULTS_DIR / "summary_stats_v2.csv", index=False)
    print(f"  all_results_v2.csv: {len(combined)} rows")
    print(f"  summary_stats_v2.csv: {len(summary)} rows")

    # 3. Tables
    print("\n[3/4] Generating LaTeX tables...")
    latex = build_latex_tables(summary)
    (RESULTS_DIR / "paper_tables_v2.tex").write_text(latex)
    print("  paper_tables_v2.tex written")

    # 4. Visualizations
    print("\n[4/4] Generating visualizations...")
    generate_visualizations(summary)
    import os
    pngs = sorted(f for f in os.listdir(VIS_DIR) if f.endswith(".png"))
    for png in pngs:
        print(f"  {VIS_DIR / png}")

    # Print summary to console
    metrics = ["iou", "dice", "precision", "recall"]
    pivoted = summary[summary["method"].isin([
        "text", "box", "point", "ampf", "ablation_fusion_mean",
        "ablation_no_alignment", "unet",
    ])].pivot_table(
        index="method", columns="dataset", values=metrics, aggfunc="first"
    )
    print("\n" + "=" * 60)
    print("Key Results Preview (foreground IoU)")
    print("=" * 60)
    for method in ["text", "box", "point", "ampf", "ablation_fusion_mean", "ablation_no_alignment", "unet"]:
        if method in summary["method"].values:
            for ds in ["crackforest", "deepcrack"]:
                row = summary[(summary["method"] == method) & (summary["dataset"] == ds)]
                if not row.empty:
                    print(f"  {method:25s} {ds:15s} IoU={row['iou'].values[0]:.4f}  Dice={row['dice'].values[0]:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
