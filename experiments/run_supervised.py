#!/usr/bin/env python3
"""Train and evaluate U-Net and DeepLabV3+ supervised baselines.

Results (2026-04-04, RTX 3090, 50 epochs, Adam lr=1e-4, BCE+Dice loss):
  CrackForest (train=94, test=24):
    U-Net:      mIoU=0.4619  Dice=0.6252  P=0.6555  R=0.6245  F1=0.6252
    DeepLabV3+: mIoU=0.4327  Dice=0.5900  P=0.5918  R=0.6672  F1=0.5900
  DeepCrack (train=300, test=237):
    U-Net:      mIoU=0.6562  Dice=0.7792  P=0.8232  R=0.7791  F1=0.7792
    DeepLabV3+: mIoU=0.6618  Dice=0.7818  P=0.8128  R=0.7905  F1=0.7818
"""

import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_loader import CrackDataset
from run_baselines import BaselineTrainer
from run_experiments import ExperimentRunner

SCRIPT_DIR = Path(__file__).resolve().parent
DATASETS_DIR = SCRIPT_DIR / "datasets"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("supervised")


def main():
    trainer = BaselineTrainer()
    logger.info("Device: %s", trainer.device)

    datasets = {}
    cf = CrackDataset("crackforest", str(DATASETS_DIR / "CrackForest-dataset"))
    datasets["crackforest"] = cf
    dc = CrackDataset("deepcrack", str(DATASETS_DIR / "DeepCrack"))
    datasets["deepcrack"] = dc

    results_dir = SCRIPT_DIR / "results"
    results_dir.mkdir(exist_ok=True)

    for ds_name, ds in datasets.items():
        train_ds = ds.get_split("train")
        test_ds = ds.get_split("test")
        logger.info("=== %s: train=%d, test=%d ===", ds_name, len(train_ds), len(test_ds))

        for model_name, train_fn in [("unet", trainer.train_unet), ("deeplabv3p", trainer.train_deeplabv3plus)]:
            logger.info("Training %s on %s...", model_name, ds_name)
            t0 = time.time()
            model = train_fn(ds, epochs=100, lr=1e-4)
            train_time = time.time() - t0
            logger.info("%s trained in %.1fs", model_name, train_time)

            logger.info("Evaluating %s on %s test set...", model_name, ds_name)
            df = trainer.evaluate_model(model, ds, model_name)
            df["dataset"] = ds_name
            out = results_dir / f"{ds_name}_supervised_{model_name}.csv"
            df.to_csv(out, index=False)
            logger.info("Saved %s (%d rows)", out.name, len(df))
            if not df.empty and "iou" in df.columns:
                logger.info("  mIoU=%.4f  Dice=%.4f  F1=%.4f",
                    df["iou"].mean(), df["dice"].mean(), df["f1"].mean())
            else:
                logger.warning("  Empty or malformed results for %s/%s", model_name, ds_name)
                logger.warning("  Columns: %s", list(df.columns))

    logger.info("Done!")


if __name__ == "__main__":
    main()
