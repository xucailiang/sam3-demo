#!/usr/bin/env python3
"""Launch all 26 experiment groups for the AMPF paper (excl. supervised baselines).

Results (2026-04-04, RTX 3090, SAM3 sam3.pt half-precision):
  CrackForest (test=24):
    text:                  mIoU=0.4355  Dice=0.6000
    box:                   mIoU=0.3439  Dice=0.4786
    point:                 mIoU=0.1166  Dice=0.1768
    ampf:                  mIoU=0.4250  Dice=0.5844
    ablation_fusion_mean:  mIoU=0.4535  Dice=0.6143
    ablation_fusion_max:   mIoU=0.4276  Dice=0.5875
    ablation_no_alignment: mIoU=0.2911  Dice=0.3983
    ablation_no_detection: mIoU=0.4362  Dice=0.5981
    ablation_no_stability: mIoU=0.4348  Dice=0.5927
    ablation_no_boundary:  mIoU=0.4215  Dice=0.5813
    combo box+text:        mIoU=0.4282  Dice=0.5865
    combo point+text:      mIoU=0.4257  Dice=0.5865
    combo box+point:       mIoU=0.4075  Dice=0.5607

Usage:
    python run_all.py                    # full run
    python run_all.py --quick            # 10 images per dataset (debug)
    python run_all.py --no-baselines     # skip supervised training
    python run_all.py --dataset crackforest  # single dataset
"""

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

# Ensure backend is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.services.model_manager import SAM3ModelManager
from app.services.segmentation_service import SegmentationService
from app.services.ampf_engine import AMPFConfig

from data_loader import CrackDataset
from run_experiments import ExperimentRunner

MODEL_PATH = "/home/justin/llm_models/facebook/sam3/sam3.pt"
SCRIPT_DIR = Path(__file__).resolve().parent
DATASETS_DIR = SCRIPT_DIR / "datasets"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(SCRIPT_DIR / "experiment.log"),
    ],
)
logger = logging.getLogger("run_all")


def load_datasets(quick: bool = False):
    """Load CrackForest and DeepCrack test splits."""
    datasets = {}

    cf_path = DATASETS_DIR / "CrackForest-dataset"
    if cf_path.exists():
        cf = CrackDataset("crackforest", str(cf_path))
        cf_test = cf.get_split("test")
        if quick:
            cf_test._samples = cf_test._samples[:10]
        datasets["crackforest"] = cf_test
        logger.info("CrackForest test: %d images", len(cf_test))

    dc_path = DATASETS_DIR / "DeepCrack"
    if dc_path.exists():
        dc = CrackDataset("deepcrack", str(dc_path))
        dc_test = dc.get_split("test")
        if quick:
            dc_test._samples = dc_test._samples[:10]
        else:
            dc_test._samples = dc_test._samples[:50]  # cap at 50 for time
        datasets["deepcrack"] = dc_test
        logger.info("DeepCrack test: %d images", len(dc_test))

    return datasets


async def main():
    parser = argparse.ArgumentParser(description="Run AMPF experiments")
    parser.add_argument("--quick", action="store_true", help="10 images per dataset")
    parser.add_argument("--no-baselines", action="store_true", help="Skip supervised baselines")
    parser.add_argument("--dataset", type=str, default=None, help="Run only this dataset (crackforest or deepcrack)")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("AMPF Experiment Runner")
    logger.info("=" * 60)

    # Load datasets
    datasets = load_datasets(quick=args.quick)
    if args.dataset:
        datasets = {k: v for k, v in datasets.items() if k == args.dataset}
    if not datasets:
        logger.error("No datasets found!")
        return

    # Initialize SAM3
    logger.info("Initializing SAM3 with model: %s", MODEL_PATH)
    SAM3ModelManager.reset_instance()
    mm = SAM3ModelManager(model_path=MODEL_PATH)
    seg = SegmentationService(model_manager=mm)

    # Use fewer stability perturbations in quick mode
    config = AMPFConfig(stability_perturbations=2 if args.quick else 5)
    runner = ExperimentRunner(seg_service=seg, config=config)

    t0 = time.time()
    await runner.run_all(datasets)
    elapsed = time.time() - t0

    logger.info("=" * 60)
    logger.info("All experiments completed in %.1f minutes", elapsed / 60)
    logger.info("Results saved to: %s", SCRIPT_DIR / "results")
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
