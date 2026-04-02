"""Automated prompt generation from ground-truth masks for SAM3 inference."""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


class PromptGenerator:
    """Derives text, box, and point prompts from a ground-truth binary mask."""

    @staticmethod
    def generate_text_prompt() -> str:
        """Returns fixed text prompt 'crack'."""
        return "crack"

    @staticmethod
    def generate_box_prompt(gt_mask: np.ndarray) -> Optional[List[float]]:
        """Compute minimum bounding rectangle of all foreground pixels.

        Args:
            gt_mask: (H, W) binary uint8 mask (0=background, 255=crack)

        Returns:
            [x1, y1, x2, y2] or None if mask has no foreground pixels.
        """
        # Find foreground pixels
        coords = cv2.findNonZero(gt_mask)
        if coords is None:
            return None

        x, y, w, h = cv2.boundingRect(coords)
        return [float(x), float(y), float(x + w), float(y + h)]

    @staticmethod
    def generate_point_prompts(
        gt_mask: np.ndarray,
    ) -> Tuple[List[Tuple[float, float]], List[int]]:
        """Compute centroid of each connected component.

        Args:
            gt_mask: (H, W) binary uint8 mask (0=background, 255=crack)

        Returns:
            (points, labels) where points are (x, y) centroids and labels are all 1.
            Empty mask returns ([], []).
        """
        if gt_mask.max() == 0:
            return ([], [])

        # Connected component analysis
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            gt_mask, connectivity=8
        )

        points: List[Tuple[float, float]] = []
        point_labels: List[int] = []

        # Skip label 0 (background)
        for i in range(1, num_labels):
            cx, cy = centroids[i]
            points.append((float(cx), float(cy)))
            point_labels.append(1)

        return (points, point_labels)

    @staticmethod
    def generate_all_prompts(gt_mask: np.ndarray) -> Dict:
        """Return all three prompt types for a given GT mask.

        Returns:
            {'text': str, 'box': Optional[list], 'points': list, 'labels': list}
        """
        points, labels = PromptGenerator.generate_point_prompts(gt_mask)
        return {
            "text": PromptGenerator.generate_text_prompt(),
            "box": PromptGenerator.generate_box_prompt(gt_mask),
            "points": points,
            "labels": labels,
        }
