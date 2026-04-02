"""Pixel-level segmentation evaluation metrics.

Computes IoU, Dice, Precision, Recall, and F1 for binary segmentation masks.
All masks are expected as (H, W) uint8 arrays with values 0 (background) and 255 (foreground).
Non-binary inputs are binarized at threshold 127 before computation.
"""

from typing import Dict, List, Tuple

import numpy as np


def _binarize(mask: np.ndarray) -> np.ndarray:
    """Ensure mask is boolean. Binarize at 127 if not already binary."""
    return mask > 127


def compute_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """Intersection over Union: |P ∩ G| / |P ∪ G|.

    Returns 0.0 when the denominator is zero.
    """
    p, g = _binarize(pred), _binarize(gt)
    intersection = np.logical_and(p, g).sum()
    union = np.logical_or(p, g).sum()
    return float(intersection / union) if union > 0 else 0.0


def compute_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    """Dice coefficient: 2|P ∩ G| / (|P| + |G|).

    Returns 0.0 when the denominator is zero.
    """
    p, g = _binarize(pred), _binarize(gt)
    intersection = np.logical_and(p, g).sum()
    denom = p.sum() + g.sum()
    return float(2.0 * intersection / denom) if denom > 0 else 0.0


def compute_precision_recall_f1(
    pred: np.ndarray, gt: np.ndarray
) -> Tuple[float, float, float]:
    """Pixel-level Precision, Recall, and F1.

    Precision = TP / (TP + FP)
    Recall    = TP / (TP + FN)
    F1        = 2 * P * R / (P + R)

    Returns 0.0 for any metric whose denominator is zero.
    """
    p, g = _binarize(pred), _binarize(gt)
    tp = float(np.logical_and(p, g).sum())
    fp = float(np.logical_and(p, ~g).sum())
    fn = float(np.logical_and(~p, g).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def evaluate_single(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    """Compute all metrics for a single prediction/ground-truth pair.

    Returns:
        Dict with keys: iou, dice, precision, recall, f1
    """
    precision, recall, f1 = compute_precision_recall_f1(pred, gt)
    return {
        "iou": compute_iou(pred, gt),
        "dice": compute_dice(pred, gt),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def evaluate_batch(
    predictions: List[np.ndarray], ground_truths: List[np.ndarray]
) -> Dict[str, Dict[str, float]]:
    """Compute dataset-level mean and standard deviation for each metric.

    Args:
        predictions: List of predicted masks.
        ground_truths: List of ground-truth masks (same length).

    Returns:
        Dict mapping metric name to {"mean": float, "std": float}.
    """
    if len(predictions) != len(ground_truths):
        raise ValueError(
            f"Length mismatch: {len(predictions)} predictions vs "
            f"{len(ground_truths)} ground truths"
        )

    all_metrics: Dict[str, List[float]] = {
        "iou": [], "dice": [], "precision": [], "recall": [], "f1": []
    }

    for pred, gt in zip(predictions, ground_truths):
        m = evaluate_single(pred, gt)
        for k in all_metrics:
            all_metrics[k].append(m[k])

    return {
        k: {"mean": float(np.mean(v)), "std": float(np.std(v))}
        for k, v in all_metrics.items()
    }
