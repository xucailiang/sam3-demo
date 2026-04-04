#!/usr/bin/env python3
"""Run core experiments on full DeepCrack test set (237 images).

Results (2026-04-04, RTX 3090, SAM3 sam3.pt half-precision):
  DeepCrack (test=237):
    text:                  mIoU=0.6658  Dice=0.7866
    box:                   mIoU=0.5408  Dice=0.6441
    point:                 mIoU=0.3129  Dice=0.3860
    ampf:                  mIoU=0.6624  Dice=0.7840
    ablation_fusion_mean:  mIoU=0.6791  Dice=0.7985
    ablation_fusion_max:   mIoU=0.6664  Dice=0.7866
    ablation_no_alignment: mIoU=0.4439  Dice=0.5236
"""

import asyncio
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.services.model_manager import SAM3ModelManager
from app.services.segmentation_service import SegmentationService
from app.services.ampf_engine import AMPFConfig

from data_loader import CrackDataset
from run_experiments import ExperimentRunner

MODEL_PATH = "/home/justin/llm_models/facebook/sam3/sam3.pt"
SCRIPT_DIR = Path(__file__).resolve().parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(SCRIPT_DIR / "deepcrack_full.log"),
    ],
)
logger = logging.getLogger("dc_full")


async def main():
    logger.info("=== DeepCrack Full Test Set (237 images) ===")

    dc = CrackDataset("deepcrack", str(SCRIPT_DIR / "datasets" / "DeepCrack"))
    dc_test = dc.get_split("test")
    logger.info("DeepCrack test: %d images", len(dc_test))

    SAM3ModelManager.reset_instance()
    mm = SAM3ModelManager(model_path=MODEL_PATH)
    seg = SegmentationService(model_manager=mm)
    config = AMPFConfig()
    runner = ExperimentRunner(seg_service=seg, config=config)

    t0 = time.time()

    # 1. Single mode baselines
    for mode in ("text", "box", "point"):
        logger.info("Running %s...", mode)
        df = await runner.run_single_mode(dc_test, mode)
        runner.save_results(df, f"deepcrack_full_{mode}.csv")
        runner.print_summary(df)

    # 2. AMPF
    logger.info("Running AMPF...")
    df = await runner.run_ampf(dc_test)
    runner.save_results(df, "deepcrack_full_ampf.csv")
    runner.print_summary(df)

    # 3. Key ablations only
    for abl in ("no_alignment", "fusion_max", "fusion_mean"):
        logger.info("Running ablation %s...", abl)
        df = await runner.run_ablation(dc_test, abl)
        runner.save_results(df, f"deepcrack_full_ablation_{abl}.csv")
        runner.print_summary(df)

    elapsed = time.time() - t0
    logger.info("Done in %.1f minutes", elapsed / 60)


if __name__ == "__main__":
    asyncio.run(main())
