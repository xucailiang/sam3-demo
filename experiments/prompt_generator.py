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
        """Compute one foreground-guaranteed point per connected component.

        For each component, the geometric centroid is rounded to the nearest integer
        pixel. If that pixel lies on the crack foreground it is used directly.
        Otherwise, the nearest foreground pixel in the same component is selected
        via Euclidean distance, ensuring every point prompt lands on a crack pixel.

        Args:
            gt_mask: (H, W) binary uint8 mask (0=background, 255=crack)

        Returns:
            (points, labels) where points are (x, y) and labels are all 1.
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
        h, w = gt_mask.shape[:2]

        # Skip label 0 (background)
        for i in range(1, num_labels):
            cx, cy = centroids[i]
            ix, iy = int(round(cx)), int(round(cy))
            ix = max(0, min(ix, w - 1))
            iy = max(0, min(iy, h - 1))

            # Check if rounded centroid lies on foreground
            if gt_mask[iy, ix] == 255 and labels[iy, ix] == i:
                points.append((float(ix), float(iy)))
            else:
                # Find nearest foreground pixel in this component
                fg_coords = cv2.findNonZero((labels == i).astype(np.uint8))
                if fg_coords is None or len(fg_coords) == 0:
                    # Fallback: use centroid as-is
                    points.append((float(cx), float(cy)))
                else:
                    fg_pts = fg_coords.squeeze(1)  # (N, 2) in (x, y) order
                    dists = (fg_pts[:, 0] - cx) ** 2 + (fg_pts[:, 1] - cy) ** 2
                    nearest_idx = int(np.argmin(dists))
                    px, py = fg_pts[nearest_idx]
                    points.append((float(px), float(py)))
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
