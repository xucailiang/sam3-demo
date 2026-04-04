#!/usr/bin/env python3
"""补跑 DeepCrack 237张全集上缺失的 6 组实验."""

import asyncio
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

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
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("dc_missing")


async def main():
    dc = CrackDataset("deepcrack", str(SCRIPT_DIR / "datasets" / "DeepCrack"))
    dc_test = dc.get_split("test")
    logger.info("DeepCrack test: %d images", len(dc_test))

    SAM3ModelManager.reset_instance()
    mm = SAM3ModelManager(model_path=MODEL_PATH)
    seg = SegmentationService(model_manager=mm)
    runner = ExperimentRunner(seg_service=seg, config=AMPFConfig())

    t0 = time.time()

    # 3 missing ablations
    for abl in ("no_detection", "no_stability", "no_boundary"):
        logger.info("Running ablation %s...", abl)
        df = await runner.run_ablation(dc_test, abl)
        runner.save_results(df, f"deepcrack_full_ablation_{abl}.csv")
        logger.info("  mIoU=%.4f", df["iou"].mean())

    # 3 missing combos
    for combo in (["text", "box"], ["text", "point"], ["box", "point"]):
        name = "+".join(combo)
        logger.info("Running combo %s...", name)
        df = await runner.run_mode_combination(dc_test, combo)
        runner.save_results(df, f"deepcrack_full_combo_{name}.csv")
        logger.info("  mIoU=%.4f", df["iou"].mean())

    logger.info("Done in %.1f minutes", (time.time() - t0) / 60)


if __name__ == "__main__":
    asyncio.run(main())
