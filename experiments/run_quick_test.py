#!/usr/bin/env python3
"""Quick smoke test: run SAM3 on one CrackForest image with all 3 prompt modes.

Results (2026-04-02, first image crackforest_0000):
  Text mode:  4 masks, best score=0.933
  Box mode:   1 mask,  score=0.980
  Point mode: 1 mask,  score=0.692
  AMPF:       IoU=0.5665  Dice=0.7233  P=0.5836  R=0.9508  F1=0.7233
"""

import asyncio
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.services.model_manager import SAM3ModelManager
from app.services.segmentation_service import SegmentationService
from app.services.ampf_engine import AMPFEngine, AMPFConfig
from data_loader import CrackDataset
from prompt_generator import PromptGenerator
from evaluation import evaluate_single

MODEL_PATH = "/home/justin/llm_models/facebook/sam3/sam3.pt"


async def main():
    print("=" * 60)
    print("AMPF Quick Smoke Test")
    print("=" * 60)

    # Load dataset
    ds = CrackDataset("crackforest", str(Path(__file__).resolve().parent / "datasets" / "CrackForest-dataset"))
    print(f"CrackForest loaded: {len(ds)} samples")
    image, gt_mask = ds[0]
    print(f"Image: {image.shape}, GT mask unique: {np.unique(gt_mask)}")

    # Generate prompts from GT
    prompts = PromptGenerator.generate_all_prompts(gt_mask)
    print(f"Text prompt: {prompts['text']}")
    print(f"Box prompt: {prompts['box']}")
    print(f"Point prompts: {len(prompts['points'])} points")

    # Initialize SAM3
    print("\nLoading SAM3 model...")
    t0 = time.time()
    SAM3ModelManager.reset_instance()
    mm = SAM3ModelManager(model_path=MODEL_PATH)
    seg = SegmentationService(model_manager=mm)
    print(f"Model manager ready in {time.time() - t0:.1f}s")

    # --- Test Text Mode ---
    print("\n--- Text Mode ---")
    t0 = time.time()
    text_result = await seg.segment_with_text(image, ["crack"], confidence=0.25)
    print(f"  Masks: {text_result.count}, Time: {text_result.processing_time_ms:.0f}ms")
    if text_result.masks:
        for i, m in enumerate(text_result.masks):
            print(f"  Mask {i}: score={m.score:.3f}, area={m.area}")

    # --- Test Box Mode ---
    print("\n--- Box Mode ---")
    if prompts["box"]:
        box_result = await seg.segment_with_boxes(image, [prompts["box"]])
        print(f"  Masks: {box_result.count}, Time: {box_result.processing_time_ms:.0f}ms")
        if box_result.masks:
            for i, m in enumerate(box_result.masks):
                print(f"  Mask {i}: score={m.score:.3f}, area={m.area}")
    else:
        print("  Skipped (empty GT mask)")

    # --- Test Point Mode ---
    print("\n--- Point Mode ---")
    if prompts["points"]:
        pt = prompts["points"][0]
        lbl = prompts["labels"][0]
        point_result = await seg.segment_with_points(image, [pt], [lbl])
        print(f"  Masks: {point_result.count}, Time: {point_result.processing_time_ms:.0f}ms")
        if point_result.masks:
            for i, m in enumerate(point_result.masks):
                print(f"  Mask {i}: score={m.score:.3f}, area={m.area}")
    else:
        print("  Skipped (no points)")

    # --- Test AMPF Pipeline ---
    print("\n--- AMPF Full Pipeline ---")
    config = AMPFConfig(stability_perturbations=2)  # fewer perturbations for speed
    engine = AMPFEngine(segmentation_service=seg, config=config)
    t0 = time.time()
    ampf_mask = await engine.run_pipeline(image, gt_mask, prompts)
    ampf_time = time.time() - t0
    print(f"  AMPF mask shape: {ampf_mask.shape}, unique: {np.unique(ampf_mask)}")
    print(f"  AMPF time: {ampf_time:.1f}s")

    # Evaluate
    metrics = evaluate_single(ampf_mask, gt_mask)
    print(f"\n--- AMPF Metrics ---")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    print("\n" + "=" * 60)
    print("Smoke test complete!")


if __name__ == "__main__":
    asyncio.run(main())
