"""
AMPF Engine — Adaptive Multi-Prompt Fusion for SAM3 crack segmentation.

Three-stage pipeline:
  Stage 1: Independent Prompting (text, box, point)
  Stage 2: Instance-Level Alignment (IoU-based greedy matching)
  Stage 3: Confidence-Aware Fusion (composite confidence + weighted pixel fusion)
"""

from __future__ import annotations

import base64
import logging
import random
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np

from app.models.schemas import SegmentationResult, MaskData
from app.services.segmentation_service import SegmentationService

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures (Task 3.1)
# ---------------------------------------------------------------------------

@dataclass
class Candidate:
    """A (mask, score) pair from one prompt mode."""
    mask: np.ndarray            # (H, W) uint8, values 0 or 255
    detection_score: float      # SAM3 output confidence [0, 1]
    mode: str                   # "text" | "box" | "point"
    prompt_info: Any = None     # original prompt used for stability perturbation


@dataclass
class ScoredCandidate:
    """A candidate with its composite confidence score."""
    candidate: Candidate
    stability_score: float
    boundary_score: float
    composite_confidence: float


@dataclass
class AlignedInstance:
    """Per-GT-instance aligned candidates from all modes."""
    gt_mask: np.ndarray
    text_candidate: Optional[Candidate] = None
    box_candidate: Optional[Candidate] = None
    point_candidate: Optional[Candidate] = None


@dataclass
class AMPFConfig:
    """Configuration for the AMPF pipeline."""
    # Confidence weights (sum to 1.0)
    alpha: float = 0.4
    beta: float = 0.35
    gamma: float = 0.25
    # Thresholds
    confidence_threshold: float = 0.5
    iou_match_threshold: float = 0.1
    sam3_confidence: float = 0.25
    # Stability perturbation
    stability_perturbations: int = 5
    text_synonyms: List[str] = field(
        default_factory=lambda: ["crack", "fracture", "fissure", "break"]
    )
    box_perturb_pct: float = 0.05
    point_perturb_px: int = 3
    # Post-processing
    morph_kernel_size: int = 3


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class AMPFEngine:
    """Adaptive Multi-Prompt Fusion engine."""

    def __init__(
        self,
        segmentation_service: SegmentationService,
        config: Optional[AMPFConfig] = None,
    ):
        self.seg = segmentation_service
        self.config = config or AMPFConfig()

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _decode_mask_base64(mask_base64: str) -> np.ndarray:
        """Decode a base64-encoded PNG mask to (H, W) uint8 numpy array."""
        raw = base64.b64decode(mask_base64)
        buf = np.frombuffer(raw, dtype=np.uint8)
        mask = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
        return mask

    @staticmethod
    def _compute_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
        """Pixel-level IoU between two binary masks."""
        a = mask_a > 0
        b = mask_b > 0
        intersection = np.count_nonzero(a & b)
        union = np.count_nonzero(a | b)
        if union == 0:
            return 0.0
        return float(intersection / union)

    # ------------------------------------------------------------------
    # Stage 1: Independent Prompting (Task 3.2)
    # ------------------------------------------------------------------

    async def run_text_mode(
        self,
        image: np.ndarray,
        text: str = "crack",
        confidence: float = 0.25,
    ) -> List[Candidate]:
        """Run text prompt and return decoded candidates."""
        try:
            result: SegmentationResult = await self.seg.segment_with_text(
                image, [text], confidence
            )
            candidates: List[Candidate] = []
            for md in result.masks:
                mask = self._decode_mask_base64(md.mask_base64)
                candidates.append(
                    Candidate(
                        mask=mask,
                        detection_score=md.score,
                        mode="text",
                        prompt_info=text,
                    )
                )
            return candidates
        except Exception:
            logger.exception("Text mode failed")
            return []

    async def run_box_mode(
        self, image: np.ndarray, bbox: List[float]
    ) -> List[Candidate]:
        """Run box prompt and return decoded candidates."""
        try:
            result: SegmentationResult = await self.seg.segment_with_boxes(
                image, [bbox]
            )
            candidates: List[Candidate] = []
            for md in result.masks:
                mask = self._decode_mask_base64(md.mask_base64)
                candidates.append(
                    Candidate(
                        mask=mask,
                        detection_score=md.score,
                        mode="box",
                        prompt_info=bbox,
                    )
                )
            return candidates
        except Exception:
            logger.exception("Box mode failed")
            return []

    async def run_point_mode(
        self,
        image: np.ndarray,
        point: Tuple[float, float],
        label: int = 1,
    ) -> Optional[Candidate]:
        """Run point prompt; select highest-score mask if multiple returned."""
        try:
            result: SegmentationResult = await self.seg.segment_with_points(
                image, [point], [label]
            )
            if not result.masks:
                return None
            best_md: MaskData = max(result.masks, key=lambda m: m.score)
            mask = self._decode_mask_base64(best_md.mask_base64)
            return Candidate(
                mask=mask,
                detection_score=best_md.score,
                mode="point",
                prompt_info=point,
            )
        except Exception:
            logger.exception("Point mode failed")
            return None

    # ------------------------------------------------------------------
    # Stage 2: Instance-Level Alignment (Task 3.3)
    # ------------------------------------------------------------------

    @staticmethod
    def extract_gt_instances(gt_mask: np.ndarray) -> List[np.ndarray]:
        """Extract connected components from GT mask as individual binary masks."""
        binary = (gt_mask > 0).astype(np.uint8)
        num_labels, labels = cv2.connectedComponents(binary)
        instances: List[np.ndarray] = []
        for lbl in range(1, num_labels):  # skip background (0)
            inst = np.where(labels == lbl, 255, 0).astype(np.uint8)
            instances.append(inst)
        return instances

    def align_candidates(
        self,
        gt_instances: List[np.ndarray],
        text_candidates: List[Candidate],
        box_candidates: List[Candidate],
        point_candidates: List[Candidate],
    ) -> List[AlignedInstance]:
        """Match candidates to GT instances via IoU-based greedy matching.

        Text/box: best IoU match above threshold (a single candidate CAN
        match multiple GT instances).
        Point: direct i-th correspondence.
        """
        threshold = self.config.iou_match_threshold
        aligned: List[AlignedInstance] = []

        for i, gt_inst in enumerate(gt_instances):
            ai = AlignedInstance(gt_mask=gt_inst)

            # --- text: greedy best IoU ---
            best_iou_text = 0.0
            best_text: Optional[Candidate] = None
            for c in text_candidates:
                iou = self._compute_iou(gt_inst, c.mask)
                if iou > best_iou_text:
                    best_iou_text = iou
                    best_text = c
            if best_iou_text >= threshold:
                ai.text_candidate = best_text

            # --- box: greedy best IoU ---
            best_iou_box = 0.0
            best_box: Optional[Candidate] = None
            for c in box_candidates:
                iou = self._compute_iou(gt_inst, c.mask)
                if iou > best_iou_box:
                    best_iou_box = iou
                    best_box = c
            if best_iou_box >= threshold:
                ai.box_candidate = best_box

            # --- point: direct i-th correspondence ---
            if i < len(point_candidates):
                ai.point_candidate = point_candidates[i]

            aligned.append(ai)

        return aligned

    # ------------------------------------------------------------------
    # Stage 3: Confidence-Aware Fusion (Task 3.5)
    # ------------------------------------------------------------------

    def compute_boundary_score(self, mask: np.ndarray) -> float:
        """Sobel gradient magnitude along mask boundary, normalised to [0, 1]."""
        if np.count_nonzero(mask) == 0:
            return 0.0

        m = mask.astype(np.float32) / 255.0
        gx = cv2.Sobel(m, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(m, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx ** 2 + gy ** 2)

        boundary_pixels = mag[mag > 0]
        if boundary_pixels.size == 0:
            return 0.0

        # Normalise: max possible Sobel magnitude for binary 0/1 image is ~4.0
        score = float(np.mean(boundary_pixels))
        # Clamp to [0, 1]
        return min(max(score / 4.0, 0.0), 1.0)

    async def compute_stability_score(
        self, image: np.ndarray, candidate: Candidate
    ) -> float:
        """Re-run inference with perturbed prompts, compute mean IoU with original.

        Text mode  → run each synonym, find best-matching mask per synonym, average.
        Box mode   → perturb bbox ±5% N times, find best match, average.
        Point mode → offset point ±3px N times, take best mask, average.
        """
        cfg = self.config
        original_mask = candidate.mask
        ious: List[float] = []

        try:
            if candidate.mode == "text":
                for synonym in cfg.text_synonyms:
                    try:
                        result = await self.seg.segment_with_text(
                            image, [synonym], cfg.sam3_confidence
                        )
                        best_iou = 0.0
                        for md in result.masks:
                            m = self._decode_mask_base64(md.mask_base64)
                            iou = self._compute_iou(original_mask, m)
                            if iou > best_iou:
                                best_iou = iou
                        ious.append(best_iou)
                    except Exception:
                        logger.debug("Stability: synonym '%s' failed", synonym)

            elif candidate.mode == "box":
                bbox = candidate.prompt_info  # [x1, y1, x2, y2]
                for _ in range(cfg.stability_perturbations):
                    try:
                        perturbed = self._perturb_bbox(bbox, cfg.box_perturb_pct)
                        result = await self.seg.segment_with_boxes(image, [perturbed])
                        best_iou = 0.0
                        for md in result.masks:
                            m = self._decode_mask_base64(md.mask_base64)
                            iou = self._compute_iou(original_mask, m)
                            if iou > best_iou:
                                best_iou = iou
                        ious.append(best_iou)
                    except Exception:
                        logger.debug("Stability: box perturbation failed")

            elif candidate.mode == "point":
                pt = candidate.prompt_info  # (x, y)
                for _ in range(cfg.stability_perturbations):
                    try:
                        px, py = self._perturb_point(pt, cfg.point_perturb_px)
                        result = await self.seg.segment_with_points(
                            image, [(px, py)], [1]
                        )
                        best_iou = 0.0
                        for md in result.masks:
                            m = self._decode_mask_base64(md.mask_base64)
                            iou = self._compute_iou(original_mask, m)
                            if iou > best_iou:
                                best_iou = iou
                        ious.append(best_iou)
                    except Exception:
                        logger.debug("Stability: point perturbation failed")

        except Exception:
            logger.exception("Stability score computation failed entirely")

        if not ious:
            return 0.0
        return float(np.mean(ious))

    # --- perturbation helpers ---

    @staticmethod
    def _perturb_bbox(bbox: List[float], pct: float) -> List[float]:
        """Perturb each bbox coordinate by a random amount in [-pct, +pct]."""
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        dx = w * random.uniform(-pct, pct)
        dy = h * random.uniform(-pct, pct)
        return [x1 + dx, y1 + dy, x2 + dx, y2 + dy]

    @staticmethod
    def _perturb_point(
        pt: Tuple[float, float], px: int
    ) -> Tuple[float, float]:
        """Offset point by a random amount in [-px, +px] for each axis."""
        x, y = pt
        return (
            x + random.uniform(-px, px),
            y + random.uniform(-px, px),
        )

    async def compute_composite_confidence(
        self, image: np.ndarray, candidate: Candidate
    ) -> float:
        """C = α*S_det + β*S_stab + γ*S_bound."""
        cfg = self.config
        s_det = candidate.detection_score
        s_stab = await self.compute_stability_score(image, candidate)
        s_bound = self.compute_boundary_score(candidate.mask)
        return cfg.alpha * s_det + cfg.beta * s_stab + cfg.gamma * s_bound

    def fuse_instance(
        self, candidates_with_confidence: List[Tuple[Candidate, float]]
    ) -> np.ndarray:
        """Weighted pixel-level fusion with morphological post-processing.

        1. Filter below threshold (fallback: keep highest confidence).
        2. Weighted pixel average → binarise at 0.5.
        3. Morphological closing + opening (elliptical kernel).
        Returns (H, W) uint8 mask, values 0 / 255.
        """
        if not candidates_with_confidence:
            raise ValueError("No candidates to fuse")

        threshold = self.config.confidence_threshold

        # Filter
        above = [(c, conf) for c, conf in candidates_with_confidence if conf >= threshold]
        if not above:
            # Fallback: keep highest confidence candidate
            best_c, best_conf = max(
                candidates_with_confidence, key=lambda x: x[1]
            )
            above = [(best_c, best_conf)]

        # Reference shape from first candidate
        h, w = above[0][0].mask.shape[:2]

        # Weighted pixel average
        weighted_sum = np.zeros((h, w), dtype=np.float64)
        weight_total = 0.0
        for c, conf in above:
            weighted_sum += (c.mask.astype(np.float64) / 255.0) * conf
            weight_total += conf

        if weight_total > 0:
            avg = weighted_sum / weight_total
        else:
            avg = weighted_sum

        # Binarise at 0.5
        binary = (avg >= 0.5).astype(np.uint8) * 255

        # Morphological closing then opening
        ks = self.config.morph_kernel_size
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        return binary

    # ------------------------------------------------------------------
    # Full Pipeline (Task 3.9)
    # ------------------------------------------------------------------

    async def run_pipeline(
        self,
        image: np.ndarray,
        gt_mask: np.ndarray,
        prompts: dict,
    ) -> np.ndarray:
        """Execute all 3 AMPF stages end-to-end.

        Args:
            image: (H, W, 3) uint8 RGB
            gt_mask: (H, W) uint8 binary, 0/255
            prompts: dict with keys 'text', 'box', 'points', 'labels'

        Returns:
            Final binary mask (H, W) uint8, values 0/255.
        """
        h, w = gt_mask.shape[:2]

        # ---- Stage 1: Independent Prompting ----
        text_candidates = await self.run_text_mode(
            image,
            text=prompts.get("text", "crack"),
            confidence=self.config.sam3_confidence,
        )

        box_candidates: List[Candidate] = []
        if prompts.get("box") is not None:
            box_candidates = await self.run_box_mode(image, prompts["box"])

        point_candidates: List[Candidate] = []
        points = prompts.get("points", [])
        labels = prompts.get("labels", [])
        for idx, pt in enumerate(points):
            lbl = labels[idx] if idx < len(labels) else 1
            cand = await self.run_point_mode(image, tuple(pt), lbl)
            if cand is not None:
                point_candidates.append(cand)

        # Check degraded mode — need at least one mode with results
        all_candidates = text_candidates + box_candidates + point_candidates
        if not all_candidates:
            logger.critical("All three prompt modes failed — returning empty mask")
            return np.zeros((h, w), dtype=np.uint8)

        # ---- Stage 2: Instance-Level Alignment ----
        gt_instances = self.extract_gt_instances(gt_mask)
        if not gt_instances:
            logger.warning("No GT instances found — returning empty mask")
            return np.zeros((h, w), dtype=np.uint8)

        aligned = self.align_candidates(
            gt_instances, text_candidates, box_candidates, point_candidates
        )

        # ---- Stage 3: Confidence-Aware Fusion per instance ----
        fused_masks: List[np.ndarray] = []

        for ai in aligned:
            instance_candidates: List[Candidate] = []
            if ai.text_candidate is not None:
                instance_candidates.append(ai.text_candidate)
            if ai.box_candidate is not None:
                instance_candidates.append(ai.box_candidate)
            if ai.point_candidate is not None:
                instance_candidates.append(ai.point_candidate)

            if not instance_candidates:
                continue

            # Compute composite confidence for each candidate
            scored: List[Tuple[Candidate, float]] = []
            for c in instance_candidates:
                conf = await self.compute_composite_confidence(image, c)
                scored.append((c, conf))

            fused = self.fuse_instance(scored)
            fused_masks.append(fused)

        # Combine per-instance fused masks via logical OR
        if not fused_masks:
            return np.zeros((h, w), dtype=np.uint8)

        combined = np.zeros((h, w), dtype=np.uint8)
        for fm in fused_masks:
            combined = np.maximum(combined, fm)

        return combined
