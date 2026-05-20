"""Small SAM3 inference smoke test for the crack experiments.

This script runs a few samples through selected methods and writes a CSV.
It is meant to verify environment/model compatibility before launching the
full experiment matrix in ``run_experiments.py``.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "backend"))
sys.path.insert(0, str(SCRIPT_DIR))

from app.services.ampf_engine import AMPFConfig
from app.services.model_manager import initialize_model_manager
from app.services.segmentation_service import SegmentationService
from data_loader import CrackDataset
from run_experiments import ExperimentRunner

LOGGER = logging.getLogger(__name__)

DATASET_ROOTS = {
    "crackforest": SCRIPT_DIR / "datasets" / "CrackForest-dataset",
    "deepcrack": SCRIPT_DIR / "datasets" / "DeepCrack",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a small SAM3 crack-segmentation smoke test."
    )
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASET_ROOTS),
        default="crackforest",
        help="Dataset to sample from.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1,
        help="Number of samples to run.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["text", "box", "point", "ampf"],
        default=["text", "box", "point"],
        help="Methods to evaluate.",
    )
    parser.add_argument(
        "--model-path",
        default=str(PROJECT_ROOT / "models" / "sam3" / "sam3.pt"),
        help="Path to local SAM3 model file.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Inference device passed to the model manager.",
    )
    parser.add_argument(
        "--output",
        default=str(SCRIPT_DIR / "results" / "smoke_results.csv"),
        help="CSV output path.",
    )
    return parser.parse_args()


def limited_dataset(dataset: CrackDataset, limit: int) -> CrackDataset:
    new_ds = object.__new__(CrackDataset)
    new_ds.dataset_name = dataset.dataset_name
    new_ds.root_dir = dataset.root_dir
    new_ds._samples = dataset._samples[: max(limit, 0)]
    return new_ds


async def run_smoke(args: argparse.Namespace) -> pd.DataFrame:
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")
    os.environ["SAM3_MODEL_PATH"] = args.model_path

    initialize_model_manager(
        model_path=args.model_path,
        device=args.device,
        preload=False,
    )
    service = SegmentationService()

    dataset = CrackDataset(args.dataset, str(DATASET_ROOTS[args.dataset]))
    dataset = limited_dataset(dataset, args.limit)

    # Keep AMPF smoke checks affordable on CPU. This does not represent the
    # paper's final AMPF configuration.
    config = AMPFConfig(
        stability_perturbations=1,
        text_synonyms=["crack"],
    )
    runner = ExperimentRunner(service, config)

    frames: list[pd.DataFrame] = []
    for method in args.methods:
        LOGGER.info("Running %s on %s sample(s)", method, len(dataset))
        if method in {"text", "box", "point"}:
            frames.append(await runner.run_single_mode(dataset, method))
        elif method == "ampf":
            frames.append(await runner.run_ampf(dataset))

    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(result.to_string(index=False))
    print(f"\nSaved: {output}")
    return result


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    args = parse_args()
    asyncio.run(run_smoke(args))


if __name__ == "__main__":
    main()
