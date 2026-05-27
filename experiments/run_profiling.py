#!/usr/bin/env python3
"""Runtime profiling for SAM3 prompt modes and supervised baselines (Fig. S3).

Measures per-image inference time for each method on a fixed hardware config.
Records mean and std ms/image along with accuracy metrics.

Expected output:
  experiments/results/runtime_profiling.csv
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "backend"))
sys.path.insert(0, str(SCRIPT_DIR))

from app.services.ampf_engine import AMPFConfig, AMPFEngine
from app.services.model_manager import SAM3ModelManager
from app.services.segmentation_service import SegmentationService
from data_loader import CrackDataset
from evaluation import evaluate_single
from prompt_generator import PromptGenerator
from run_experiments import ExperimentRunner


async def time_text(seg_service, image, prompts) -> tuple[float, np.ndarray]:
    t0 = time.perf_counter()
    result = await seg_service.segment_with_text(image, [prompts["text"]], 0.25)
    if not result.masks:
        return time.perf_counter() - t0, np.zeros(image.shape[:2], dtype=np.uint8)
    combined = np.zeros(image.shape[:2], dtype=np.uint8)
    for md in result.masks:
        mask = AMPFEngine._decode_mask_base64(md.mask_base64)
        combined = np.maximum(combined, mask)
    return time.perf_counter() - t0, combined


async def time_box(seg_service, image, prompts) -> tuple[float, np.ndarray]:
    if prompts.get("box") is None:
        return 0.0, np.zeros(image.shape[:2], dtype=np.uint8)
    t0 = time.perf_counter()
    result = await seg_service.segment_with_boxes(image, [prompts["box"]])
    if not result.masks:
        return time.perf_counter() - t0, np.zeros(image.shape[:2], dtype=np.uint8)
    combined = np.zeros(image.shape[:2], dtype=np.uint8)
    for md in result.masks:
        mask = AMPFEngine._decode_mask_base64(md.mask_base64)
        combined = np.maximum(combined, mask)
    return time.perf_counter() - t0, combined


async def time_point(seg_service, image, prompts) -> tuple[float, np.ndarray]:
    points = prompts.get("points", [])
    labels = prompts.get("labels", [])
    if not points:
        return 0.0, np.zeros(image.shape[:2], dtype=np.uint8)
    t0 = time.perf_counter()
    combined = np.zeros(image.shape[:2], dtype=np.uint8)
    for pt, lbl in zip(points, labels):
        result = await seg_service.segment_with_points(image, [pt], [lbl])
        if result.masks:
            best_md = max(result.masks, key=lambda m: m.score)
            mask = AMPFEngine._decode_mask_base64(best_md.mask_base64)
            combined = np.maximum(combined, mask)
    return time.perf_counter() - t0, combined


async def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets-dir", default=str(SCRIPT_DIR / "datasets"))
    parser.add_argument("--model", default="/home/justin/llm_models/facebook/sam3/sam3.pt")
    parser.add_argument("--n-images", type=int, default=20,
                        help="Number of DeepCrack test images per method")
    parser.add_argument("--warmup", type=int, default=2,
                        help="Warmup images before timing")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Init
    os.environ["SAM3_MODEL_PATH"] = args.model
    SAM3ModelManager.reset_instance()
    model_manager = SAM3ModelManager(model_path=args.model)
    seg_service = SegmentationService(model_manager=model_manager)
    cfg = AMPFConfig(random_seed=args.seed)
    engine = AMPFEngine(seg_service, cfg)
    runner = ExperimentRunner(seg_service, cfg)

    # Load DeepCrack
    root = Path(args.datasets_dir) / "DeepCrack"
    dataset = CrackDataset("deepcrack", str(root)).get_split("test")
    n_total = args.warmup + args.n_images
    print(f"DeepCrack test: {len(dataset)} images, profiling {n_total} (warmup={args.warmup})")

    rows: List[dict] = []

    methods = [
        ("text", time_text),
        ("box", time_box),
        ("point", time_point),
    ]

    for method_name, timing_fn in methods:
        print(f"\nProfiling {method_name}...")
        times = []
        ious = []

        for idx in range(n_total):
            image, gt_mask = dataset[idx]
            prompts = PromptGenerator.generate_all_prompts(gt_mask)

            elapsed, pred = await timing_fn(seg_service, image, prompts)
            metrics = evaluate_single(pred, gt_mask)

            if idx >= args.warmup:
                times.append(elapsed * 1000)  # ms
                ious.append(metrics["iou"])

        mean_ms = np.mean(times)
        std_ms = np.std(times)
        mean_iou = np.mean(ious)

        protocol = "automatic_text_prompt" if method_name == "text" else "oracle_gt_prompt"
        rows.append({
            "method": method_name,
            "dataset": "deepcrack",
            "protocol": protocol,
            "protocol_type": "deployable" if method_name == "text" else "gt-derived diagnostic",
            "n_images": len(times),
            "total_time_s": sum(times) / 1000,
            "mean_ms_per_image": mean_ms,
            "std_ms_per_image": std_ms,
            "mean_iou": mean_iou,
        })
        print(f"  {method_name}: {mean_ms:.1f} ± {std_ms:.1f} ms/image, IoU={mean_iou:.4f}")

    # AMPF (full pipeline with stability)
    print("\nProfiling AMPF...")
    ampf_times = []
    ampf_ious = []
    for idx in range(n_total):
        image, gt_mask = dataset[idx]
        prompts = PromptGenerator.generate_all_prompts(gt_mask)
        t0 = time.perf_counter()
        pred = await engine.run_pipeline(image, gt_mask, prompts)
        elapsed = (time.perf_counter() - t0) * 1000
        metrics = evaluate_single(pred, gt_mask)
        if idx >= args.warmup:
            ampf_times.append(elapsed)
            ampf_ious.append(metrics["iou"])

    rows.append({
        "method": "ampf",
        "dataset": "deepcrack",
        "protocol": "oracle_gt_prompt_gt_alignment",
        "protocol_type": "gt-assisted diagnostic",
        "n_images": len(ampf_times),
        "total_time_s": sum(ampf_times) / 1000,
        "mean_ms_per_image": np.mean(ampf_times),
        "std_ms_per_image": np.std(ampf_times),
        "mean_iou": np.mean(ampf_ious),
    })
    print(f"  AMPF: {np.mean(ampf_times):.1f} ± {np.std(ampf_times):.1f} ms/image, IoU={np.mean(ampf_ious):.4f}")

    # Mean Fusion
    print("\nProfiling Mean Fusion...")
    mf_times = []
    mf_ious = []
    for idx in range(n_total):
        image, gt_mask = dataset[idx]
        prompts = PromptGenerator.generate_all_prompts(gt_mask)
        t0 = time.perf_counter()
        pred = await runner._run_ablation_image(image, gt_mask, prompts, "fusion_mean")
        elapsed = (time.perf_counter() - t0) * 1000
        metrics = evaluate_single(pred, gt_mask)
        if idx >= args.warmup:
            mf_times.append(elapsed)
            mf_ious.append(metrics["iou"])

    rows.append({
        "method": "ablation_fusion_mean",
        "dataset": "deepcrack",
        "protocol": "oracle_gt_prompt_gt_alignment",
        "protocol_type": "gt-assisted diagnostic",
        "n_images": len(mf_times),
        "total_time_s": sum(mf_times) / 1000,
        "mean_ms_per_image": np.mean(mf_times),
        "std_ms_per_image": np.std(mf_times),
        "mean_iou": np.mean(mf_ious),
    })
    print(f"  Mean Fusion: {np.mean(mf_times):.1f} ± {np.std(mf_times):.1f} ms/image, IoU={np.mean(mf_ious):.4f}")

    # Save
    df = pd.DataFrame(rows)
    out_path = SCRIPT_DIR / "results" / "runtime_profiling.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved runtime profiling to {out_path}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    asyncio.run(main())
