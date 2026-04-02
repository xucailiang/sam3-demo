"""Experiment orchestration for AMPF crack segmentation evaluation.

Runs all 32 experiment groups across CrackForest and DeepCrack datasets:
  - 6 single-mode baselines (text/box/point × 2 datasets)
  - 2 AMPF full pipeline
  - 12 ablation studies (6 ablation types × 2 datasets)
  - 6 mode combinations (3 pairs × 2 datasets)
  - 6 supervised baselines (U-Net, DeepLabV3+, YOLOv8-seg × 2 datasets)
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Add backend to sys.path so imports from app.* work
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT / "backend"))

from app.services.ampf_engine import AMPFConfig, AMPFEngine, Candidate
from app.services.segmentation_service import SegmentationService

from data_loader import CrackDataset
from evaluation import evaluate_single
from prompt_generator import PromptGenerator
from run_baselines import run_all_baselines

logger = logging.getLogger(__name__)

# Output directories
RESULTS_DIR = _SCRIPT_DIR / "results"
VIS_DIR = _SCRIPT_DIR / "visualizations"


# ---------------------------------------------------------------------------
# ExperimentRunner — core framework (Task 7.1)
# ---------------------------------------------------------------------------

class ExperimentRunner:
    """Orchestrates all AMPF experiment groups."""

    def __init__(
        self,
        seg_service: SegmentationService,
        config: Optional[AMPFConfig] = None,
    ) -> None:
        """Initialize with segmentation service and AMPF config.

        Args:
            seg_service: Existing SegmentationService instance for SAM3 inference.
            config: AMPF configuration; uses defaults if None.
        """
        self.seg_service = seg_service
        self.config = config or AMPFConfig()
        self.engine = AMPFEngine(seg_service, self.config)

    # ------------------------------------------------------------------
    # Single-mode baseline (Req 8.1)
    # ------------------------------------------------------------------

    async def run_single_mode(
        self, dataset: CrackDataset, mode: str
    ) -> pd.DataFrame:
        """Run a single-mode baseline experiment (text / box / point).

        For each image the raw SAM3 service is called directly (not through
        AMPFEngine.run_pipeline) so we measure the unmodified single-prompt
        performance.

        Args:
            dataset: CrackDataset instance.
            mode: One of "text", "box", "point".

        Returns:
            DataFrame with columns: image_id, method, dataset, iou, dice,
            precision, recall, f1.
        """
        rows: List[Dict[str, Any]] = []

        for idx in range(len(dataset)):
            image_id = f"{dataset.dataset_name}_{idx:04d}"
            try:
                image, gt_mask = dataset[idx]
                prompts = PromptGenerator.generate_all_prompts(gt_mask)
                pred_mask = await self._run_single_mode_inference(
                    image, gt_mask, prompts, mode
                )
                metrics = evaluate_single(pred_mask, gt_mask)
                rows.append(
                    {
                        "image_id": image_id,
                        "method": mode,
                        "dataset": dataset.dataset_name,
                        **metrics,
                    }
                )
            except Exception:
                logger.exception(
                    "Single-mode '%s' failed on image %s", mode, image_id
                )

        return pd.DataFrame(rows)

    async def _run_single_mode_inference(
        self,
        image: np.ndarray,
        gt_mask: np.ndarray,
        prompts: dict,
        mode: str,
    ) -> np.ndarray:
        """Call SAM3 for a single prompt mode and return the best binary mask."""
        h, w = gt_mask.shape[:2]

        if mode == "text":
            result = await self.seg_service.segment_with_text(
                image,
                [prompts["text"]],
                self.config.sam3_confidence,
            )
            if not result.masks:
                return np.zeros((h, w), dtype=np.uint8)
            # Combine all returned masks via logical OR
            combined = np.zeros((h, w), dtype=np.uint8)
            for md in result.masks:
                mask = AMPFEngine._decode_mask_base64(md.mask_base64)
                combined = np.maximum(combined, mask)
            return combined

        elif mode == "box":
            bbox = prompts.get("box")
            if bbox is None:
                return np.zeros((h, w), dtype=np.uint8)
            result = await self.seg_service.segment_with_boxes(image, [bbox])
            if not result.masks:
                return np.zeros((h, w), dtype=np.uint8)
            combined = np.zeros((h, w), dtype=np.uint8)
            for md in result.masks:
                mask = AMPFEngine._decode_mask_base64(md.mask_base64)
                combined = np.maximum(combined, mask)
            return combined

        elif mode == "point":
            points = prompts.get("points", [])
            labels = prompts.get("labels", [])
            if not points:
                return np.zeros((h, w), dtype=np.uint8)
            combined = np.zeros((h, w), dtype=np.uint8)
            for pt, lbl in zip(points, labels):
                result = await self.seg_service.segment_with_points(
                    image, [pt], [lbl]
                )
                if result.masks:
                    best_md = max(result.masks, key=lambda m: m.score)
                    mask = AMPFEngine._decode_mask_base64(best_md.mask_base64)
                    combined = np.maximum(combined, mask)
            return combined

        else:
            raise ValueError(f"Unknown mode '{mode}'. Expected text/box/point.")

    # ------------------------------------------------------------------
    # Save / print helpers (Req 8.5, 8.6)
    # ------------------------------------------------------------------

    @staticmethod
    def save_results(df: pd.DataFrame, filename: str) -> None:
        """Save per-image metrics to CSV.

        Columns: image_id, method, dataset, iou, dice, precision, recall, f1.
        File is written to ``experiments/results/<filename>``.
        """
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        path = RESULTS_DIR / filename
        df.to_csv(path, index=False)
        logger.info("Results saved to %s (%d rows)", path, len(df))

    @staticmethod
    def print_summary(df: pd.DataFrame) -> None:
        """Print mean metrics summary table to stdout."""
        if df.empty:
            print("[Summary] No results to summarise.")
            return

        metrics = ["iou", "dice", "precision", "recall", "f1"]
        summary = df.groupby(["method", "dataset"])[metrics].mean()
        print("\n" + "=" * 72)
        print("  Experiment Summary (mean metrics)")
        print("=" * 72)
        print(summary.to_string(float_format="{:.4f}".format))
        print("=" * 72 + "\n")


    # ------------------------------------------------------------------
    # AMPF full pipeline (Req 8.2) — Task 7.2
    # ------------------------------------------------------------------

    async def run_ampf(self, dataset: CrackDataset) -> pd.DataFrame:
        """Run the full AMPF 3-stage pipeline on every image in *dataset*.

        Returns:
            DataFrame with per-image metrics (same schema as run_single_mode).
        """
        rows: List[Dict[str, Any]] = []
        engine = AMPFEngine(self.seg_service, self.config)

        for idx in range(len(dataset)):
            image_id = f"{dataset.dataset_name}_{idx:04d}"
            try:
                image, gt_mask = dataset[idx]
                prompts = PromptGenerator.generate_all_prompts(gt_mask)
                pred_mask = await engine.run_pipeline(image, gt_mask, prompts)
                metrics = evaluate_single(pred_mask, gt_mask)
                rows.append(
                    {
                        "image_id": image_id,
                        "method": "ampf",
                        "dataset": dataset.dataset_name,
                        **metrics,
                    }
                )
            except Exception:
                logger.exception("AMPF failed on image %s", image_id)

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Ablation experiments (Req 8.3) — Task 7.2
    # ------------------------------------------------------------------

    ABLATION_TYPES = (
        "no_detection",   # α = 0
        "no_stability",   # β = 0
        "no_boundary",    # γ = 0
        "no_alignment",   # skip Stage 2 alignment
        "fusion_max",     # replace weighted average with element-wise max
        "fusion_mean",    # replace weighted average with unweighted mean
    )

    async def run_ablation(
        self, dataset: CrackDataset, ablation: str
    ) -> pd.DataFrame:
        """Run AMPF with one component removed or an alternative fusion strategy.

        Supported *ablation* values:
          - ``no_detection``  — set α = 0 (renormalise β, γ)
          - ``no_stability``  — set β = 0 (renormalise α, γ)
          - ``no_boundary``   — set γ = 0 (renormalise α, β)
          - ``no_alignment``  — skip instance-level alignment (Stage 2)
          - ``fusion_max``    — replace weighted pixel average with element-wise max
          - ``fusion_mean``   — replace weighted pixel average with unweighted mean
        """
        if ablation not in self.ABLATION_TYPES:
            raise ValueError(
                f"Unknown ablation '{ablation}'. "
                f"Supported: {self.ABLATION_TYPES}"
            )

        rows: List[Dict[str, Any]] = []
        method_name = f"ablation_{ablation}"

        for idx in range(len(dataset)):
            image_id = f"{dataset.dataset_name}_{idx:04d}"
            try:
                image, gt_mask = dataset[idx]
                prompts = PromptGenerator.generate_all_prompts(gt_mask)
                pred_mask = await self._run_ablation_image(
                    image, gt_mask, prompts, ablation
                )
                metrics = evaluate_single(pred_mask, gt_mask)
                rows.append(
                    {
                        "image_id": image_id,
                        "method": method_name,
                        "dataset": dataset.dataset_name,
                        **metrics,
                    }
                )
            except Exception:
                logger.exception(
                    "Ablation '%s' failed on image %s", ablation, image_id
                )

        return pd.DataFrame(rows)

    async def _run_ablation_image(
        self,
        image: np.ndarray,
        gt_mask: np.ndarray,
        prompts: dict,
        ablation: str,
    ) -> np.ndarray:
        """Execute one ablation variant for a single image."""
        cfg = deepcopy(self.config)

        # --- weight ablations: zero out one weight, renormalise the rest ---
        if ablation == "no_detection":
            cfg.alpha = 0.0
            total = cfg.beta + cfg.gamma
            if total > 0:
                cfg.beta /= total
                cfg.gamma /= total
        elif ablation == "no_stability":
            cfg.beta = 0.0
            total = cfg.alpha + cfg.gamma
            if total > 0:
                cfg.alpha /= total
                cfg.gamma /= total
        elif ablation == "no_boundary":
            cfg.gamma = 0.0
            total = cfg.alpha + cfg.beta
            if total > 0:
                cfg.alpha /= total
                cfg.beta /= total

        engine = AMPFEngine(self.seg_service, cfg)

        if ablation == "no_alignment":
            return await self._run_no_alignment(engine, image, gt_mask, prompts)
        elif ablation in ("fusion_max", "fusion_mean"):
            return await self._run_alt_fusion(
                engine, image, gt_mask, prompts, ablation
            )
        else:
            # Weight-only ablation — standard pipeline with modified config
            return await engine.run_pipeline(image, gt_mask, prompts)

    async def _run_no_alignment(
        self,
        engine: AMPFEngine,
        image: np.ndarray,
        gt_mask: np.ndarray,
        prompts: dict,
    ) -> np.ndarray:
        """AMPF without Stage 2 (instance alignment).

        All candidates are fused globally instead of per-instance.
        """
        h, w = gt_mask.shape[:2]

        # Stage 1
        text_cands = await engine.run_text_mode(
            image, prompts.get("text", "crack"), engine.config.sam3_confidence
        )
        box_cands: List[Candidate] = []
        if prompts.get("box") is not None:
            box_cands = await engine.run_box_mode(image, prompts["box"])
        point_cands: List[Candidate] = []
        for pt, lbl in zip(
            prompts.get("points", []), prompts.get("labels", [])
        ):
            c = await engine.run_point_mode(image, tuple(pt), lbl)
            if c is not None:
                point_cands.append(c)

        all_cands = text_cands + box_cands + point_cands
        if not all_cands:
            return np.zeros((h, w), dtype=np.uint8)

        # Skip Stage 2 — go directly to fusion on all candidates
        scored: List[Tuple[Candidate, float]] = []
        for c in all_cands:
            conf = await engine.compute_composite_confidence(image, c)
            scored.append((c, conf))

        return engine.fuse_instance(scored)

    async def _run_alt_fusion(
        self,
        engine: AMPFEngine,
        image: np.ndarray,
        gt_mask: np.ndarray,
        prompts: dict,
        strategy: str,
    ) -> np.ndarray:
        """AMPF with alternative fusion strategy (max or mean)."""
        h, w = gt_mask.shape[:2]

        # Stage 1
        text_cands = await engine.run_text_mode(
            image, prompts.get("text", "crack"), engine.config.sam3_confidence
        )
        box_cands: List[Candidate] = []
        if prompts.get("box") is not None:
            box_cands = await engine.run_box_mode(image, prompts["box"])
        point_cands: List[Candidate] = []
        for pt, lbl in zip(
            prompts.get("points", []), prompts.get("labels", [])
        ):
            c = await engine.run_point_mode(image, tuple(pt), lbl)
            if c is not None:
                point_cands.append(c)

        all_cands = text_cands + box_cands + point_cands
        if not all_cands:
            return np.zeros((h, w), dtype=np.uint8)

        # Stage 2 — normal alignment
        gt_instances = engine.extract_gt_instances(gt_mask)
        if not gt_instances:
            return np.zeros((h, w), dtype=np.uint8)

        aligned = engine.align_candidates(
            gt_instances, text_cands, box_cands, point_cands
        )

        # Stage 3 — alternative fusion per instance
        import cv2

        fused_masks: List[np.ndarray] = []
        for ai in aligned:
            inst_cands: List[Candidate] = []
            if ai.text_candidate is not None:
                inst_cands.append(ai.text_candidate)
            if ai.box_candidate is not None:
                inst_cands.append(ai.box_candidate)
            if ai.point_candidate is not None:
                inst_cands.append(ai.point_candidate)
            if not inst_cands:
                continue

            if strategy == "fusion_max":
                # Element-wise max across all candidate masks
                combined = np.zeros((h, w), dtype=np.uint8)
                for c in inst_cands:
                    combined = np.maximum(combined, c.mask)
                fused_masks.append(combined)
            else:
                # Unweighted mean → binarise at 0.5
                avg = np.zeros((h, w), dtype=np.float64)
                for c in inst_cands:
                    avg += c.mask.astype(np.float64) / 255.0
                avg /= len(inst_cands)
                binary = (avg >= 0.5).astype(np.uint8) * 255
                # Morphological post-processing
                ks = engine.config.morph_kernel_size
                kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (ks, ks)
                )
                binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
                binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
                fused_masks.append(binary)

        if not fused_masks:
            return np.zeros((h, w), dtype=np.uint8)

        combined = np.zeros((h, w), dtype=np.uint8)
        for fm in fused_masks:
            combined = np.maximum(combined, fm)
        return combined


    # ------------------------------------------------------------------
    # Mode combination experiments (Req 8.4) — Task 7.2
    # ------------------------------------------------------------------

    async def run_mode_combination(
        self, dataset: CrackDataset, modes: List[str]
    ) -> pd.DataFrame:
        """Run AMPF with a subset of prompt modes (e.g. text+box).

        Only the specified modes are used in Stage 1; Stages 2-3 proceed
        normally with whatever candidates are available.

        Args:
            dataset: CrackDataset instance.
            modes: Subset of ["text", "box", "point"], e.g. ["text", "box"].

        Returns:
            DataFrame with per-image metrics.
        """
        method_name = "+".join(sorted(modes))
        rows: List[Dict[str, Any]] = []
        engine = AMPFEngine(self.seg_service, self.config)

        for idx in range(len(dataset)):
            image_id = f"{dataset.dataset_name}_{idx:04d}"
            try:
                image, gt_mask = dataset[idx]
                prompts = PromptGenerator.generate_all_prompts(gt_mask)
                pred_mask = await self._run_combination_image(
                    engine, image, gt_mask, prompts, modes
                )
                metrics = evaluate_single(pred_mask, gt_mask)
                rows.append(
                    {
                        "image_id": image_id,
                        "method": method_name,
                        "dataset": dataset.dataset_name,
                        **metrics,
                    }
                )
            except Exception:
                logger.exception(
                    "Combination %s failed on image %s", method_name, image_id
                )

        return pd.DataFrame(rows)

    async def _run_combination_image(
        self,
        engine: AMPFEngine,
        image: np.ndarray,
        gt_mask: np.ndarray,
        prompts: dict,
        modes: List[str],
    ) -> np.ndarray:
        """Run AMPF pipeline using only the specified modes."""
        h, w = gt_mask.shape[:2]

        text_cands: List[Candidate] = []
        box_cands: List[Candidate] = []
        point_cands: List[Candidate] = []

        if "text" in modes:
            text_cands = await engine.run_text_mode(
                image, prompts.get("text", "crack"), engine.config.sam3_confidence
            )
        if "box" in modes and prompts.get("box") is not None:
            box_cands = await engine.run_box_mode(image, prompts["box"])
        if "point" in modes:
            for pt, lbl in zip(
                prompts.get("points", []), prompts.get("labels", [])
            ):
                c = await engine.run_point_mode(image, tuple(pt), lbl)
                if c is not None:
                    point_cands.append(c)

        all_cands = text_cands + box_cands + point_cands
        if not all_cands:
            return np.zeros((h, w), dtype=np.uint8)

        # Stage 2
        gt_instances = engine.extract_gt_instances(gt_mask)
        if not gt_instances:
            return np.zeros((h, w), dtype=np.uint8)

        aligned = engine.align_candidates(
            gt_instances, text_cands, box_cands, point_cands
        )

        # Stage 3
        fused_masks: List[np.ndarray] = []
        for ai in aligned:
            inst_cands: List[Candidate] = []
            if ai.text_candidate is not None:
                inst_cands.append(ai.text_candidate)
            if ai.box_candidate is not None:
                inst_cands.append(ai.box_candidate)
            if ai.point_candidate is not None:
                inst_cands.append(ai.point_candidate)
            if not inst_cands:
                continue

            scored: List[Tuple[Candidate, float]] = []
            for c in inst_cands:
                conf = await engine.compute_composite_confidence(image, c)
                scored.append((c, conf))
            fused_masks.append(engine.fuse_instance(scored))

        if not fused_masks:
            return np.zeros((h, w), dtype=np.uint8)

        combined = np.zeros((h, w), dtype=np.uint8)
        for fm in fused_masks:
            combined = np.maximum(combined, fm)
        return combined


    # ------------------------------------------------------------------
    # run_all — full 32-group orchestration (Task 7.3)
    # ------------------------------------------------------------------

    async def run_all(self, datasets: Dict[str, CrackDataset]) -> None:
        """Execute all 32 experiment groups, save CSVs, and print summaries.

        Groups:
          -  6 single-mode baselines  (text/box/point × 2 datasets)
          -  2 AMPF full pipeline     (× 2 datasets)
          - 12 ablation studies       (6 types × 2 datasets)
          -  6 mode combinations      (3 pairs × 2 datasets)
          -  6 supervised baselines   (U-Net / DeepLabV3+ / YOLOv8-seg × 2 datasets)

        Args:
            datasets: Mapping of dataset name to CrackDataset, e.g.
                      {"crackforest": cf_ds, "deepcrack": dc_ds}.
        """
        all_results: List[pd.DataFrame] = []

        for ds_name, ds in datasets.items():
            logger.info("=== Dataset: %s (%d images) ===", ds_name, len(ds))

            # --- Single-mode baselines (6 groups) ---
            for mode in ("text", "box", "point"):
                logger.info("Running single-mode: %s on %s", mode, ds_name)
                df = await self.run_single_mode(ds, mode)
                self.save_results(df, f"{ds_name}_{mode}.csv")
                self.print_summary(df)
                all_results.append(df)

            # --- AMPF full pipeline (2 groups) ---
            logger.info("Running AMPF on %s", ds_name)
            df = await self.run_ampf(ds)
            self.save_results(df, f"{ds_name}_ampf.csv")
            self.print_summary(df)
            all_results.append(df)

            # --- Ablation studies (12 groups) ---
            for abl in self.ABLATION_TYPES:
                logger.info("Running ablation '%s' on %s", abl, ds_name)
                df = await self.run_ablation(ds, abl)
                self.save_results(df, f"{ds_name}_ablation_{abl}.csv")
                self.print_summary(df)
                all_results.append(df)

            # --- Mode combinations (6 groups) ---
            for combo in (
                ["text", "box"],
                ["text", "point"],
                ["box", "point"],
            ):
                combo_name = "+".join(combo)
                logger.info(
                    "Running combination %s on %s", combo_name, ds_name
                )
                df = await self.run_mode_combination(ds, combo)
                self.save_results(df, f"{ds_name}_combo_{combo_name}.csv")
                self.print_summary(df)
                all_results.append(df)

        # --- Supervised baselines (6 groups: 3 models × 2 datasets) ---
        logger.info("Running supervised baselines (U-Net, DeepLabV3+, YOLOv8-seg)")
        try:
            baseline_df = run_all_baselines(datasets)
            if not baseline_df.empty:
                all_results.append(baseline_df)
                # Save per-model per-dataset CSVs
                for (method, ds_name), group_df in baseline_df.groupby(
                    ["method", "dataset"]
                ):
                    self.save_results(group_df, f"{ds_name}_{method}.csv")
                    self.print_summary(group_df)
        except Exception:
            logger.exception("Supervised baselines failed — continuing without them")

        # Consolidated summary
        if all_results:
            combined = pd.concat(all_results, ignore_index=True)
            self.save_results(combined, "all_results.csv")
            self.print_summary(combined)

            # Generate visualizations
            for ds_name, ds in datasets.items():
                self.generate_visualizations(combined, ds)


    # ------------------------------------------------------------------
    # Visualization (Req 10.1-10.4) — Task 7.3
    # ------------------------------------------------------------------

    def generate_visualizations(
        self, results_df: pd.DataFrame, dataset: CrackDataset
    ) -> None:
        """Generate comparison figures and grouped bar charts.

        1. Side-by-side comparison: Original | GT | Text | Box | Point | AMPF
           for representative images (where AMPF improves most over best
           single mode).
        2. Grouped bar chart of mean metrics across methods.

        All figures saved as 300 DPI PNG to ``experiments/visualizations/``.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        VIS_DIR.mkdir(parents=True, exist_ok=True)
        ds_name = dataset.dataset_name

        # ---- Bar chart: mean metrics per method ----
        ds_df = results_df[results_df["dataset"] == ds_name]
        if ds_df.empty:
            logger.warning("No results for %s — skipping visualizations", ds_name)
            return

        self._generate_bar_chart(ds_df, ds_name)
        self._generate_comparison_figure(ds_df, dataset)

    def _generate_bar_chart(
        self, ds_df: pd.DataFrame, ds_name: str
    ) -> None:
        """Grouped bar chart comparing metrics across methods."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        metrics = ["iou", "dice", "precision", "recall", "f1"]
        summary = ds_df.groupby("method")[metrics].mean()

        fig, ax = plt.subplots(figsize=(max(10, len(summary) * 1.2), 6))
        summary.plot(kind="bar", ax=ax)
        ax.set_title(f"Mean Metrics — {ds_name}")
        ax.set_ylabel("Score")
        ax.set_xlabel("Method")
        ax.set_ylim(0, 1)
        ax.legend(loc="lower right")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        path = VIS_DIR / f"{ds_name}_metrics_bar.png"
        fig.savefig(str(path), dpi=300)
        plt.close(fig)
        logger.info("Bar chart saved to %s", path)

    def _generate_comparison_figure(
        self, ds_df: pd.DataFrame, dataset: CrackDataset
    ) -> None:
        """Side-by-side comparison figure for representative images.

        Selects images where AMPF shows the largest improvement over the
        best single-mode result (by IoU).
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ds_name = dataset.dataset_name

        # Identify representative images
        single_modes = ["text", "box", "point"]
        single_df = ds_df[ds_df["method"].isin(single_modes)]
        ampf_df = ds_df[ds_df["method"] == "ampf"]

        if single_df.empty or ampf_df.empty:
            logger.warning(
                "Missing single-mode or AMPF results for %s — "
                "skipping comparison figure",
                ds_name,
            )
            return

        # Best single-mode IoU per image
        best_single = single_df.groupby("image_id")["iou"].max()
        ampf_iou = ampf_df.set_index("image_id")["iou"]

        # Compute improvement
        common_ids = best_single.index.intersection(ampf_iou.index)
        if common_ids.empty:
            return
        improvement = ampf_iou.loc[common_ids] - best_single.loc[common_ids]
        top_ids = improvement.nlargest(min(3, len(improvement))).index.tolist()

        # Build figure
        n_rows = len(top_ids)
        col_labels = [
            "Original", "GT", "Text", "Box", "Point", "AMPF",
            "U-Net", "DeepLabV3+", "YOLOv8-seg",
        ]
        _LABEL_TO_METHOD = {
            "Text": "text",
            "Box": "box",
            "Point": "point",
            "AMPF": "ampf",
            "U-Net": "unet",
            "DeepLabV3+": "deeplabv3plus",
            "YOLOv8-seg": "yolov8seg",
        }
        n_cols = len(col_labels)

        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows)
        )
        if n_rows == 1:
            axes = axes[np.newaxis, :]

        for row_idx, img_id in enumerate(top_ids):
            # Parse index from image_id (format: datasetname_XXXX)
            idx = int(img_id.split("_")[-1])
            if idx >= len(dataset):
                continue
            image, gt_mask = dataset[idx]

            # Collect per-method masks (run synchronously is not possible here,
            # so we just show GT and leave mask columns blank if not cached).
            # For the comparison figure we re-read from the per-image results.
            method_masks: Dict[str, Optional[np.ndarray]] = {}
            # We don't have cached masks in the DataFrame — show IoU overlay text
            for col_idx, label in enumerate(col_labels):
                ax = axes[row_idx, col_idx]
                ax.axis("off")

                if label == "Original":
                    ax.imshow(image)
                elif label == "GT":
                    ax.imshow(gt_mask, cmap="gray", vmin=0, vmax=255)
                else:
                    # Show method name + IoU as text overlay on blank
                    method_key = _LABEL_TO_METHOD.get(label, label.lower())
                    row_data = ds_df[
                        (ds_df["image_id"] == img_id)
                        & (ds_df["method"] == method_key)
                    ]
                    if not row_data.empty:
                        iou_val = row_data.iloc[0]["iou"]
                        ax.text(
                            0.5, 0.5,
                            f"{label}\nIoU={iou_val:.3f}",
                            ha="center", va="center",
                            transform=ax.transAxes,
                            fontsize=12,
                        )
                    else:
                        ax.text(
                            0.5, 0.5, f"{label}\nN/A",
                            ha="center", va="center",
                            transform=ax.transAxes,
                            fontsize=12,
                        )

                if row_idx == 0:
                    ax.set_title(label, fontsize=10)

        plt.suptitle(
            f"Representative Comparisons — {ds_name}", fontsize=14
        )
        plt.tight_layout()

        path = VIS_DIR / f"{ds_name}_comparison.png"
        fig.savefig(str(path), dpi=300)
        plt.close(fig)
        logger.info("Comparison figure saved to %s", path)
