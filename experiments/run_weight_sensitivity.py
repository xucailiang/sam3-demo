#!/usr/bin/env python3
"""AMPF hyperparameter weight sensitivity grid search (Fig. S2) — CACHED VERSION.

Key optimization: SAM3 inference and stability/boundary computation only run
ONCE per image. The weight grid search only varies the fusion weights,
recomputing composite_confidence and re-fusing from cached candidate data.

Following PAPER_FIGURE_OPTIMIZATION_PLAN.md Sec 3.3:
  - alpha ∈ {0.1, 0.2, ..., 0.7}
  - beta  ∈ {0.1, 0.2, ..., 0.7}
  - gamma = 1 - alpha - beta, retain gamma >= 0 boundary combinations

Expected outputs:
  experiments/results/ampf_weight_sensitivity_cache/  (per-image candidate caches)
  experiments/results/ampf_weight_sensitivity.csv
"""

from __future__ import annotations

import asyncio
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "backend"))
sys.path.insert(0, str(SCRIPT_DIR))

from app.services.ampf_engine import AMPFConfig, AMPFEngine, Candidate, AlignedInstance
from app.services.model_manager import SAM3ModelManager
from app.services.segmentation_service import SegmentationService
from data_loader import CrackDataset
from evaluation import evaluate_single
from prompt_generator import PromptGenerator


# ---------------------------------------------------------------------------
# Phase 1: Cache all expensive SAM3 inference results
# ---------------------------------------------------------------------------

async def cache_image_data(
    engine: AMPFEngine,
    image: np.ndarray,
    gt_mask: np.ndarray,
    prompts: dict,
    image_id: str,
    cache_dir: Path,
) -> str:
    """Run SAM3 inference once per image and cache all candidate data + alignment.

    Returns the cache file path.
    """
    h, w = gt_mask.shape[:2]

    # Stage 1: Independent Prompting
    text_cands = await engine.run_text_mode(
        image, prompts.get("text", "crack"), engine.config.sam3_confidence
    )
    box_cands: List[Candidate] = []
    if prompts.get("box") is not None:
        box_cands = await engine.run_box_mode(image, prompts["box"])
    point_cands: List[Candidate] = []
    for pt, lbl in zip(prompts.get("points", []), prompts.get("labels", [])):
        c = await engine.run_point_mode(image, tuple(pt), lbl)
        if c is not None:
            point_cands.append(c)

    # Compute per-candidate scores
    all_cands_data = []
    score_by_obj_id = {}
    for mode, cands in [("text", text_cands), ("box", box_cands), ("point", point_cands)]:
        for cand_idx, c in enumerate(cands):
            s_stab = await engine.compute_stability_score(image, c)
            s_bound = engine.compute_boundary_score(c.mask, image)
            candidate_id = f"{mode}_{cand_idx:03d}"
            cand_record = {
                "candidate_id": candidate_id,
                "mode": mode,
                "mask": c.mask.copy(),
                "detection_score": c.detection_score,
                "stability_score": s_stab,
                "boundary_score": s_bound,
            }
            all_cands_data.append(cand_record)
            score_by_obj_id[id(c)] = cand_record

    # Stage 2: Alignment
    gt_instances = engine.extract_gt_instances(gt_mask)
    aligned = engine.align_candidates(gt_instances, text_cands, box_cands, point_cands)

    # Serialize aligned structure
    aligned_data = []
    for ai in aligned:
        inst_data = {"gt_mask": ai.gt_mask.copy()}
        for role, cand in [("text", ai.text_candidate), ("box", ai.box_candidate),
                           ("point", ai.point_candidate)]:
            if cand is not None:
                cand_scores = score_by_obj_id.get(id(cand), {})
                inst_data[role] = {
                    "candidate_id": cand_scores.get("candidate_id"),
                    "mask": cand.mask.copy(),
                    "detection_score": cand.detection_score,
                    "stability_score": cand_scores.get("stability_score"),
                    "boundary_score": cand_scores.get("boundary_score"),
                    "mode": cand.mode,
                }
            else:
                inst_data[role] = None
        aligned_data.append(inst_data)

    cache = {
        "image_id": image_id,
        "all_candidates": all_cands_data,
        "aligned_instances": aligned_data,
        "image_shape": (h, w),
    }

    cache_path = cache_dir / f"{image_id}.pkl"
    with open(cache_path, "wb") as f:
        pickle.dump(cache, f)

    return str(cache_path)


# ---------------------------------------------------------------------------
# Phase 2: Fuse with different weights
# ---------------------------------------------------------------------------

def find_cached_scores(
    cache: dict,
    mode: str,
    detection_score: float,
    mask: np.ndarray,
    candidate_id: Optional[str] = None,
) -> tuple[float, float]:
    """Return cached stability/boundary scores for one aligned candidate.

    New caches store a stable candidate_id. Older caches do not, so the fallback
    requires mask equality as well as mode and detection score to avoid selecting
    the wrong candidate when SAM3 emits duplicate confidence values.
    """
    if candidate_id is not None:
        for ac in cache["all_candidates"]:
            if ac.get("candidate_id") == candidate_id:
                return ac.get("stability_score", 0.0), ac.get("boundary_score", 0.0)

    for ac in cache["all_candidates"]:
        if ac["mode"] != mode:
            continue
        if ac["detection_score"] != detection_score:
            continue
        if np.array_equal(ac["mask"], mask):
            return ac.get("stability_score", 0.0), ac.get("boundary_score", 0.0)

    return 0.0, 0.0


def reconstruct_gt_from_cache(cache: dict) -> np.ndarray:
    """Reconstruct the image-level GT mask from cached aligned instances."""
    h, w = cache["image_shape"]
    gt_mask = np.zeros((h, w), dtype=np.uint8)
    for ai_data in cache.get("aligned_instances", []):
        if ai_data.get("gt_mask") is not None:
            gt_mask = np.maximum(gt_mask, ai_data["gt_mask"].astype(np.uint8))
    return gt_mask


def fuse_with_weights(
    cache: dict,
    alpha: float,
    beta: float,
    gamma: float,
    confidence_threshold: float = 0.5,
    morph_kernel_size: int = 3,
) -> np.ndarray:
    """Re-fuse cached candidates with new weight configuration.

    Only arithmetic operations, no SAM3 inference.
    """
    h, w = cache["image_shape"]

    fused_masks: List[np.ndarray] = []

    for ai_data in cache["aligned_instances"]:
        inst_cands: List[dict] = []

        for role in ["text", "box", "point"]:
            if ai_data.get(role) is not None:
                cd = ai_data[role]
                inst_cands.append(cd)

        if not inst_cands:
            continue

        # Compute composite confidence for each candidate using cached scores
        scored: List[Tuple[np.ndarray, float]] = []
        for cd in inst_cands:
            mask = cd["mask"]
            det_score = cd["detection_score"]
            stab_score = cd.get("stability_score")
            bound_score = cd.get("boundary_score")
            if stab_score is None or bound_score is None:
                stab_score, bound_score = find_cached_scores(
                    cache,
                    cd["mode"],
                    det_score,
                    mask,
                    cd.get("candidate_id"),
                )
            comp_conf = alpha * det_score + beta * stab_score + gamma * bound_score
            scored.append((mask, comp_conf))

        # Filter by threshold
        above = [(m, c) for m, c in scored if c >= confidence_threshold]
        if not above:
            above = [max(scored, key=lambda x: x[1])]

        # Weighted pixel average
        weighted_sum = np.zeros((h, w), dtype=np.float64)
        weight_total = 0.0
        for m, conf in above:
            weighted_sum += (m.astype(np.float64) / 255.0) * conf
            weight_total += conf

        avg = weighted_sum / weight_total if weight_total > 0 else weighted_sum
        binary = (avg >= 0.5).astype(np.uint8) * 255

        ks = morph_kernel_size
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        fused_masks.append(binary)

    if not fused_masks:
        # Fallback: pick best detection_score candidate
        best_mask = None
        best_score = -1
        for ac in cache["all_candidates"]:
            if ac["detection_score"] > best_score:
                best_score = ac["detection_score"]
                best_mask = ac["mask"]
        return best_mask if best_mask is not None else np.zeros((h, w), dtype=np.uint8)

    combined = np.zeros((h, w), dtype=np.uint8)
    for fm in fused_masks:
        combined = np.maximum(combined, fm)
    return combined


def generate_weight_combos() -> List[Tuple[float, float, float]]:
    alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    betas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    combos = []
    for a in alphas:
        for b in betas:
            g = 1.0 - a - b
            if g >= -1e-12:
                g = max(0.0, g)
                combos.append((a, b, g))
    return combos


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets-dir", default=str(SCRIPT_DIR / "datasets"))
    parser.add_argument("--model", default=str(PROJECT_ROOT / "models" / "sam3" / "sam3.pt"))
    parser.add_argument("--max-images", type=int, default=30,
                        help="DeepCrack test subset size (0=all 237)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-caching", action="store_true",
                        help="Skip Phase 1 if caches already exist")
    args = parser.parse_args()

    cache_dir = SCRIPT_DIR / "results" / "ampf_weight_sensitivity_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    image_data = []
    engine = None

    if args.skip_caching:
        cache_paths = sorted(cache_dir.glob("deepcrack_*.pkl"))
        if args.max_images > 0:
            cache_paths = cache_paths[:args.max_images]
        if not cache_paths:
            raise FileNotFoundError(
                f"No cached DeepCrack pickle files found in {cache_dir}. "
                "Run without --skip-caching first."
            )
        for cache_path in cache_paths:
            with open(cache_path, "rb") as f:
                cache = pickle.load(f)
            image_id = cache.get("image_id", cache_path.stem)
            gt_mask = reconstruct_gt_from_cache(cache)
            image_data.append((None, gt_mask, None, image_id))
        print(f"DeepCrack cached subset: {len(image_data)} images")
    else:
        # Init model only when Phase 1 caching is requested.
        os.environ["SAM3_MODEL_PATH"] = args.model
        SAM3ModelManager.reset_instance()
        model_manager = SAM3ModelManager(model_path=args.model)
        seg_service = SegmentationService(model_manager=model_manager)
        cfg = AMPFConfig(random_seed=args.seed)
        engine = AMPFEngine(seg_service, cfg)

        # Load dataset
        root = Path(args.datasets_dir) / "DeepCrack"
        full_dataset = CrackDataset("deepcrack", str(root)).get_split("test")
        n_images = min(args.max_images, len(full_dataset)) if args.max_images > 0 else len(full_dataset)
        print(f"DeepCrack test subset: {n_images} images")

        # Pre-load images
        for idx in range(n_images):
            image, gt_mask = full_dataset[idx]
            prompts = PromptGenerator.generate_all_prompts(gt_mask)
            image_data.append((image, gt_mask, prompts, f"deepcrack_{idx:04d}"))

    # ── Phase 1: Cache ──
    if not args.skip_caching:
        print("\n=== Phase 1: Caching SAM3 inference results ===")
        t0 = time.time()
        for i, (image, gt_mask, prompts, image_id) in enumerate(image_data):
            try:
                assert engine is not None
                cache_path = await cache_image_data(
                    engine, image, gt_mask, prompts, image_id, cache_dir
                )
                if (i + 1) % 10 == 0:
                    elapsed = time.time() - t0
                    rate = elapsed / (i + 1)
                    remaining = rate * (n_images - i - 1)
                    print(f"  Cached {i+1}/{n_images}  "
                          f"elapsed={elapsed:.0f}s  est_remaining={remaining:.0f}s")
            except Exception as e:
                print(f"  FAILED caching {image_id}: {e}")
        print(f"Phase 1 done: {time.time()-t0:.0f}s")

    # ── Phase 2: Grid search ──
    combos = generate_weight_combos()
    print(f"\n=== Phase 2: Grid search over {len(combos)} weight combos ===")

    # Load all caches
    caches: Dict[str, dict] = {}
    for img_id in [d[3] for d in image_data]:
        cache_path = cache_dir / f"{img_id}.pkl"
        if cache_path.exists():
            with open(cache_path, "rb") as f:
                caches[img_id] = pickle.load(f)

    print(f"Loaded {len(caches)} cached images")

    all_rows: List[dict] = []
    t_start = time.time()

    for ci, (alpha, beta, gamma) in enumerate(combos):
        t_combo = time.time()
        combo_metrics: List[dict] = []

        for image, gt_mask, prompts, image_id in image_data:
            if image_id not in caches:
                continue
            try:
                pred = fuse_with_weights(caches[image_id], alpha, beta, gamma)
                metrics = evaluate_single(pred, gt_mask)
                metrics["image_id"] = image_id
                metrics["alpha"] = alpha
                metrics["beta"] = beta
                metrics["gamma"] = gamma
                combo_metrics.append(metrics)
            except Exception:
                continue

        if combo_metrics:
            mean_metrics = {
                "alpha": alpha,
                "beta": beta,
                "gamma": gamma,
                "dataset": "deepcrack",
                "n_images": len(combo_metrics),
                "mean_iou": np.mean([m["iou"] for m in combo_metrics]),
                "mean_dice": np.mean([m["dice"] for m in combo_metrics]),
                "mean_precision": np.mean([m["precision"] for m in combo_metrics]),
                "mean_recall": np.mean([m["recall"] for m in combo_metrics]),
                "mean_f1": np.mean([m["f1"] for m in combo_metrics]),
            }
            all_rows.append(mean_metrics)
        else:
            mean_metrics = {"mean_iou": float("nan")}

        elapsed = time.time() - t_combo
        total_elapsed = time.time() - t_start
        remaining_combos = len(combos) - ci - 1
        est_rem = elapsed * remaining_combos
        print(f"  [{ci+1}/{len(combos)}] α={alpha:.1f} β={beta:.1f} γ={gamma:.1f}  "
              f"IoU={mean_metrics['mean_iou']:.4f}  "
              f"combo={elapsed:.1f}s  "
              f"total={total_elapsed:.0f}s  rem_est={est_rem:.0f}s")

    # Save results
    df = pd.DataFrame(all_rows)
    out_path = SCRIPT_DIR / "results" / "ampf_weight_sensitivity.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df)} weight combos to {out_path}")
    print(f"Total Phase 2 time: {time.time()-t_start:.0f}s")

    # Summary
    if len(df) > 0:
        best = df.loc[df["mean_iou"].idxmax()]
        print(f"\nBest: α={best['alpha']:.1f} β={best['beta']:.1f} γ={best['gamma']:.1f} "
              f"IoU={best['mean_iou']:.4f}")
        current = df[(df["alpha"] == 0.4) & (df["beta"] == 0.35)]
        if len(current) > 0:
            print(f"Current (0.4,0.35,0.25): IoU={current.iloc[0]['mean_iou']:.4f}")
            better = (df["mean_iou"] > current.iloc[0]["mean_iou"]).sum()
            print(f"Combos better than current: {better}/{len(df)} "
                  f"({100*better/len(df):.1f}%)")


if __name__ == "__main__":
    asyncio.run(main())
