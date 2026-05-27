#!/usr/bin/env python3
"""Candidate-level confidence logging for calibration analysis (Fig. S1).

Runs AMPF Stage 1 (independent prompting) on each test image and saves
per-candidate scores before fusion:
  - detection_score (SAM3 raw score)
  - stability_score (consistency under prompt perturbation)
  - boundary_score (image-gradient at mask boundary)
  - composite_confidence (weighted combination)
  - candidate_iou (vs ground truth)

The output CSV supports a reliability diagram and ECE computation,
addressing the reviewer request for calibration evidence.

Expected output:
  experiments/results/candidate_confidence.csv
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

from app.services.ampf_engine import AMPFConfig, AMPFEngine, Candidate
from app.services.model_manager import SAM3ModelManager
from app.services.segmentation_service import SegmentationService
from data_loader import CrackDataset
from evaluation import evaluate_single
from prompt_generator import PromptGenerator


async def collect_candidate_data(
    engine: AMPFEngine,
    image: np.ndarray,
    gt_mask: np.ndarray,
    prompts: dict,
    image_id: str,
    dataset_name: str,
) -> List[dict]:
    """Run all prompt modes and collect per-candidate calibration data."""
    rows: List[dict] = []

    # ── Text mode ──
    text_candidates = await engine.run_text_mode(
        image, prompts.get("text", "crack"), engine.config.sam3_confidence
    )
    for i, cand in enumerate(text_candidates):
        s_stab = await engine.compute_stability_score(image, cand)
        s_bound = engine.compute_boundary_score(cand.mask, image)
        s_comp = (
            engine.config.alpha * cand.detection_score
            + engine.config.beta * s_stab
            + engine.config.gamma * s_bound
        )
        cand_iou = evaluate_single(cand.mask, gt_mask)["iou"]
        rows.append({
            "dataset": dataset_name,
            "image_id": image_id,
            "method": "text",
            "candidate_id": i,
            "prompt_mode": "text",
            "detection_score": cand.detection_score,
            "stability_score": s_stab,
            "boundary_score": s_bound,
            "composite_confidence": s_comp,
            "candidate_iou": cand_iou,
        })

    # ── Box mode ──
    box_candidates = await engine.run_box_mode(image, prompts["box"]) if prompts.get("box") else []
    for i, cand in enumerate(box_candidates):
        s_stab = await engine.compute_stability_score(image, cand)
        s_bound = engine.compute_boundary_score(cand.mask, image)
        s_comp = (
            engine.config.alpha * cand.detection_score
            + engine.config.beta * s_stab
            + engine.config.gamma * s_bound
        )
        cand_iou = evaluate_single(cand.mask, gt_mask)["iou"]
        rows.append({
            "dataset": dataset_name,
            "image_id": image_id,
            "method": "box",
            "candidate_id": i,
            "prompt_mode": "box",
            "detection_score": cand.detection_score,
            "stability_score": s_stab,
            "boundary_score": s_bound,
            "composite_confidence": s_comp,
            "candidate_iou": cand_iou,
        })

    # ── Point mode ──
    for i, (pt, lbl) in enumerate(zip(prompts.get("points", []), prompts.get("labels", []))):
        cand = await engine.run_point_mode(image, tuple(pt), lbl)
        if cand is None:
            continue
        s_stab = await engine.compute_stability_score(image, cand)
        s_bound = engine.compute_boundary_score(cand.mask, image)
        s_comp = (
            engine.config.alpha * cand.detection_score
            + engine.config.beta * s_stab
            + engine.config.gamma * s_bound
        )
        cand_iou = evaluate_single(cand.mask, gt_mask)["iou"]
        rows.append({
            "dataset": dataset_name,
            "image_id": image_id,
            "method": "point",
            "candidate_id": i,
            "prompt_mode": "point",
            "detection_score": cand.detection_score,
            "stability_score": s_stab,
            "boundary_score": s_bound,
            "composite_confidence": s_comp,
            "candidate_iou": cand_iou,
        })

    return rows


async def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets-dir", default=str(SCRIPT_DIR / "datasets"))
    parser.add_argument("--model", default="/home/justin/llm_models/facebook/sam3/sam3.pt")
    parser.add_argument("--datasets", nargs="+", default=["deepcrack", "crackforest"])
    parser.add_argument("--max-images", type=int, default=0,
                        help="Limit images per dataset (0 = all)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Init
    os.environ["SAM3_MODEL_PATH"] = args.model
    SAM3ModelManager.reset_instance()
    model_manager = SAM3ModelManager(model_path=args.model)
    seg_service = SegmentationService(model_manager=model_manager)
    cfg = AMPFConfig(random_seed=args.seed)
    engine = AMPFEngine(seg_service, cfg)

    all_rows: List[dict] = []

    for ds_name in args.datasets:
        root_map = {
            "crackforest": "CrackForest-dataset",
            "deepcrack": "DeepCrack",
        }
        root = Path(args.datasets_dir) / root_map[ds_name]
        dataset = CrackDataset(ds_name, str(root)).get_split("test")
        n = len(dataset) if args.max_images <= 0 else min(args.max_images, len(dataset))
        print(f"[{ds_name}] Processing {n}/{len(dataset)} test images...")

        t0 = time.time()
        for idx in range(n):
            image, gt_mask = dataset[idx]
            image_id = f"{ds_name}_{idx:04d}"
            prompts = PromptGenerator.generate_all_prompts(gt_mask)

            try:
                rows = await collect_candidate_data(
                    engine, image, gt_mask, prompts, image_id, ds_name
                )
                all_rows.extend(rows)
            except Exception:
                print(f"  FAILED: {image_id}")
                continue

            if (idx + 1) % 20 == 0:
                elapsed = time.time() - t0
                rate = elapsed / (idx + 1)
                remaining = rate * (n - idx - 1)
                print(f"  [{ds_name}] {idx+1}/{n}  "
                      f"elapsed={elapsed:.0f}s  rem={remaining:.0f}s  "
                      f"candidates={len(all_rows)}")

        print(f"[{ds_name}] Done in {time.time()-t0:.0f}s, {len(all_rows)} total candidates")

    # Save
    df = pd.DataFrame(all_rows)
    out_path = SCRIPT_DIR / "results" / "candidate_confidence.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df)} candidate records to {out_path}")

    # Quick summary
    for ds in df["dataset"].unique():
        sub = df[df["dataset"] == ds]
        print(f"\n{ds}: {len(sub)} candidates from {sub['image_id'].nunique()} images")
        for mode in ["text", "box", "point"]:
            mode_df = sub[sub["prompt_mode"] == mode]
            if len(mode_df) > 0:
                print(f"  {mode}: {len(mode_df)} candidates, "
                      f"mean det={mode_df['detection_score'].mean():.3f}, "
                      f"mean comp_conf={mode_df['composite_confidence'].mean():.3f}, "
                      f"mean cand_IoU={mode_df['candidate_iou'].mean():.4f}")


if __name__ == "__main__":
    asyncio.run(main())
