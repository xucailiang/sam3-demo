#!/usr/bin/env python3
"""Re-run point-dependent experiments with nearest-foreground point protocol.

This script re-runs only the experiment groups affected by the point prompt
generation fix (centroid -> nearest-foreground). It reuses the existing
``ExperimentRunner`` from ``run_experiments.py`` for all inference logic.

Groups re-run (×2 datasets):
  - point single mode
  - ampf full pipeline (uses text+box+point)
  - all 6 ablations (all use AMPF)
  - combo text+point
  - combo box+point

Groups NOT re-run:
  - text / box single mode (no point involved)
  - combo text+box (no point involved)
  - supervised baselines (no point involved)

Usage:
    python experiments/rerun_point_experiments.py
    python experiments/rerun_point_experiments.py --dataset crackforest
    python experiments/rerun_point_experiments.py --dataset deepcrack
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path
from typing import Dict

import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT / "backend"))

from app.services.model_manager import SAM3ModelManager
from app.services.segmentation_service import SegmentationService
from app.services.ampf_engine import AMPFConfig

from data_loader import CrackDataset
from run_experiments import ExperimentRunner

MODEL_PATH = "/home/justin/llm_models/facebook/sam3/sam3.pt"
DATASETS_DIR = _SCRIPT_DIR / "datasets"
RESULTS_DIR = _SCRIPT_DIR / "results"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("rerun_point")


def load_datasets(dataset_filter: str | None = None) -> Dict[str, CrackDataset]:
    """Load full test splits for point-dependent re-run."""
    datasets: Dict[str, CrackDataset] = {}

    if dataset_filter is None or dataset_filter == "crackforest":
        cf_root = DATASETS_DIR / "CrackForest-dataset"
        if cf_root.exists():
            cf = CrackDataset("crackforest", str(cf_root))
            datasets["crackforest"] = cf.get_split("test")
            logger.info("CrackForest test: %d images", len(datasets["crackforest"]))

    if dataset_filter is None or dataset_filter == "deepcrack":
        dc_root = DATASETS_DIR / "DeepCrack"
        if dc_root.exists():
            dc = CrackDataset("deepcrack", str(dc_root))
            datasets["deepcrack"] = dc.get_split("test")
            logger.info("DeepCrack test: %d images", len(datasets["deepcrack"]))

    return datasets


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-run point-dependent experiments with nearest-foreground points"
    )
    parser.add_argument(
        "--dataset", type=str, default=None,
        choices=["crackforest", "deepcrack"],
        help="Run on a single dataset (default: both)",
    )
    args = parser.parse_args()

    # Load datasets
    datasets = load_datasets(dataset_filter=args.dataset)
    if not datasets:
        logger.error("No datasets found.")
        raise SystemExit(1)

    # Initialize SAM3
    logger.info("Initializing SAM3 with model: %s", MODEL_PATH)
    SAM3ModelManager.reset_instance()
    mm = SAM3ModelManager(model_path=MODEL_PATH)
    seg = SegmentationService(model_manager=mm)
    config = AMPFConfig(stability_perturbations=5)
    runner = ExperimentRunner(seg_service=seg, config=config)

    t0 = time.time()
    all_frames: list[pd.DataFrame] = []

    for ds_name, eval_ds in datasets.items():
        logger.info("=" * 60)
        logger.info("Dataset: %s (%d test images)", ds_name, len(eval_ds))
        logger.info("=" * 60)

        # --- Point single mode ---
        logger.info("[1/10] point single mode")
        df = await runner.run_single_mode(eval_ds, "point")
        df["method"] = df["method"].replace("point", "point_v2")
        runner.save_results(df, f"{ds_name}_point_v2.csv")
        runner.print_summary(df)
        all_frames.append(df)

        # --- AMPF full ---
        logger.info("[2/10] ampf full")
        df = await runner.run_ampf(eval_ds)
        df["method"] = df["method"].replace("ampf", "ampf_v2")
        runner.save_results(df, f"{ds_name}_ampf_v2.csv")
        runner.print_summary(df)
        all_frames.append(df)

        # --- 6 ablations ---
        for i, abl in enumerate(runner.ABLATION_TYPES, start=3):
            logger.info("[%d/10] ablation %s", i, abl)
            df = await runner.run_ablation(eval_ds, abl)
            df["method"] = df["method"].replace(
                f"ablation_{abl}", f"ablation_{abl}_v2"
            )
            runner.save_results(df, f"{ds_name}_ablation_{abl}_v2.csv")
            runner.print_summary(df)
            all_frames.append(df)

        # --- Point-involving combos (skip text+box) ---
        for j, modes in enumerate((["text", "point"], ["box", "point"]), start=9):
            combo_name = "+".join(modes)
            logger.info("[%d/10] combo %s", j, combo_name)
            df = await runner.run_mode_combination(eval_ds, modes)
            df["method"] = df["method"].replace(
                f"combo_{combo_name}", f"combo_{combo_name}_v2"
            )
            runner.save_results(df, f"{ds_name}_combo_{combo_name}_v2.csv")
            runner.print_summary(df)
            all_frames.append(df)

    # Consolidated output
    combined = pd.concat(all_frames, ignore_index=True)
    runner.save_results(combined, "point_experiments_v2.csv")

    elapsed = time.time() - t0
    logger.info("=" * 60)
    logger.info("Re-run complete in %.1f min", elapsed / 60)

    # Print summary
    metrics = ["iou", "dice", "precision", "recall", "f1"]
    summary = combined.groupby(["method", "dataset"])[metrics].mean()
    print("\n" + "=" * 72)
    print("  Point v2 Experiment Summary (nearest-foreground points)")
    print("=" * 72)
    print(summary.to_string(float_format="{:.4f}".format))
    print("=" * 72)


if __name__ == "__main__":
    asyncio.run(main())
