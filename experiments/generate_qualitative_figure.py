#!/usr/bin/env python3
"""Generate qualitative SAM3 crack-segmentation comparison panels.

The paper tables report per-image metrics but do not cache predicted masks.
This script re-runs a small set of representative test images and saves a
multi-column overlay figure for qualitative analysis.

Expected output:
    experiments/visualizations/fig_qualitative_prompt_examples.png
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/sam3-demo-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(PROJECT_ROOT / "backend"))
sys.path.insert(0, str(SCRIPT_DIR))

from app.services.ampf_engine import AMPFConfig, AMPFEngine
from app.services.model_manager import SAM3ModelManager
from app.services.segmentation_service import SegmentationService
from data_loader import CrackDataset
from prompt_generator import PromptGenerator
from run_experiments import ExperimentRunner


def overlay_mask(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int],
    alpha: float = 0.45,
) -> np.ndarray:
    """Overlay a binary mask on an RGB image."""
    out = image.copy().astype(np.float32)
    binary = mask > 127
    tint = np.zeros_like(out)
    tint[:, :] = color
    out[binary] = (1 - alpha) * out[binary] + alpha * tint[binary]
    return np.clip(out, 0, 255).astype(np.uint8)


def parse_case(case: str) -> tuple[str, int]:
    """Parse case strings like 'deepcrack:0' or 'crackforest:12'."""
    if ":" not in case:
        raise argparse.ArgumentTypeError("Cases must use dataset:index format.")
    dataset, idx_text = case.split(":", 1)
    dataset = dataset.strip().lower()
    if dataset not in {"crackforest", "deepcrack"}:
        raise argparse.ArgumentTypeError("Dataset must be crackforest or deepcrack.")
    return dataset, int(idx_text)


async def predict_case(
    runner: ExperimentRunner,
    engine: AMPFEngine,
    image: np.ndarray,
    gt_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    """Run the prompt modes needed for the qualitative panel."""
    prompts = PromptGenerator.generate_all_prompts(gt_mask)
    masks: dict[str, np.ndarray] = {
        "Ground truth": gt_mask,
        "Text": await runner._run_single_mode_inference(image, gt_mask, prompts, "text"),
        "Box": await runner._run_single_mode_inference(image, gt_mask, prompts, "box"),
        "Point": await runner._run_single_mode_inference(image, gt_mask, prompts, "point"),
        "AMPF": await engine.run_pipeline(image, gt_mask, prompts),
        "Mean fusion": await runner._run_ablation_image(
            image, gt_mask, prompts, "fusion_mean"
        ),
    }
    return masks


async def build_figure(args: argparse.Namespace) -> None:
    datasets_root = Path(args.datasets_dir)
    cases = [parse_case(case) for case in args.case]

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"SAM3 model not found: {model_path}")

    os.environ["SAM3_MODEL_PATH"] = str(model_path)
    SAM3ModelManager.reset_instance()
    model_manager = SAM3ModelManager(model_path=str(model_path))
    seg_service = SegmentationService(model_manager=model_manager)
    cfg = AMPFConfig(random_seed=args.seed)
    runner = ExperimentRunner(seg_service, cfg)
    engine = AMPFEngine(seg_service, cfg)

    columns = ["Image", "Ground truth", "Text", "Box", "Point", "AMPF", "Mean fusion"]
    fig, axes = plt.subplots(
        len(cases),
        len(columns),
        figsize=(3.0 * len(columns), 2.6 * len(cases)),
        squeeze=False,
    )

    colors = {
        "Ground truth": (0, 220, 80),
        "Text": (255, 60, 60),
        "Box": (255, 170, 0),
        "Point": (60, 140, 255),
        "AMPF": (200, 80, 255),
        "Mean fusion": (0, 200, 220),
    }

    for row, (dataset_name, idx) in enumerate(cases):
        root = datasets_root / (
            "CrackForest-dataset" if dataset_name == "crackforest" else "DeepCrack"
        )
        dataset = CrackDataset(dataset_name, str(root)).get_split("test")
        if idx < 0 or idx >= len(dataset):
            raise IndexError(f"{dataset_name}:{idx} is outside test split.")

        image, gt_mask = dataset[idx]
        masks = await predict_case(runner, engine, image, gt_mask)
        image_label = f"{dataset_name}:{idx}"

        for col, title in enumerate(columns):
            ax = axes[row, col]
            ax.axis("off")
            if row == 0:
                ax.set_title(title, fontsize=10)
            if title == "Image":
                ax.imshow(image)
                ax.text(
                    0.02,
                    0.96,
                    image_label,
                    transform=ax.transAxes,
                    va="top",
                    ha="left",
                    fontsize=9,
                    color="white",
                    bbox={"facecolor": "black", "alpha": 0.55, "pad": 2},
                )
            else:
                ax.imshow(overlay_mask(image, masks[title], colors[title]))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved qualitative figure to {out}")


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets-dir",
        default=str(SCRIPT_DIR / "datasets"),
        help="Directory containing CrackForest-dataset/ and DeepCrack/.",
    )
    parser.add_argument(
        "--model",
        default=str(PROJECT_ROOT / "models" / "sam3" / "sam3.pt"),
        help="Path to sam3.pt.",
    )
    parser.add_argument(
        "--case",
        action="append",
        default=["crackforest:0", "deepcrack:0", "deepcrack:25"],
        help="Representative case in dataset:index format; repeatable.",
    )
    parser.add_argument(
        "--output",
        default=str(SCRIPT_DIR / "visualizations" / "fig_qualitative_prompt_examples.png"),
        help="Output PNG path.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    asyncio.run(build_figure(args))


if __name__ == "__main__":
    main()
