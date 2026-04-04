# Feature: ampf-crack-segmentation, Property 7: IoU metric correctness and self-consistency
"""Property-based tests for evaluation metrics."""

import sys
from pathlib import Path

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluation import compute_iou, compute_dice, compute_precision_recall_f1, evaluate_single, evaluate_batch


def mask_pair_strategy():
    """Generate pairs of binary masks."""
    return (
        st.tuples(
            st.integers(min_value=5, max_value=50),
            st.integers(min_value=5, max_value=50),
        )
        .flatmap(
            lambda hw: st.tuples(
                st.just(hw),
                st.binary(min_size=hw[0] * hw[1], max_size=hw[0] * hw[1]),
                st.binary(min_size=hw[0] * hw[1], max_size=hw[0] * hw[1]),
            )
        )
        .map(lambda t: _build_pair(t[0], t[1], t[2]))
    )


def _build_pair(hw, b1, b2):
    h, w = hw
    p = np.frombuffer(b1, dtype=np.uint8).reshape(h, w)
    g = np.frombuffer(b2, dtype=np.uint8).reshape(h, w)
    # Binarize: >127 → 255, else 0
    p = ((p > 127).astype(np.uint8)) * 255
    g = ((g > 127).astype(np.uint8)) * 255
    return p, g


# --- Property 7: IoU metric correctness and self-consistency ---

@given(data=mask_pair_strategy())
@settings(max_examples=100)
def test_iou_range(data):
    pred, gt = data
    iou = compute_iou(pred, gt)
    assert 0.0 <= iou <= 1.0


@given(data=mask_pair_strategy())
@settings(max_examples=100)
def test_iou_formula(data):
    pred, gt = data
    p = pred > 127
    g = gt > 127
    intersection = np.logical_and(p, g).sum()
    union = np.logical_or(p, g).sum()
    expected = float(intersection / union) if union > 0 else 0.0
    assert abs(compute_iou(pred, gt) - expected) < 1e-9


def test_iou_self_consistency():
    """IoU(M, M) == 1.0 for non-empty mask."""
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[5:15, 5:15] = 255
    assert compute_iou(mask, mask) == 1.0


def test_iou_both_empty():
    mask = np.zeros((20, 20), dtype=np.uint8)
    assert compute_iou(mask, mask) == 0.0


@given(data=mask_pair_strategy())
@settings(max_examples=100)
def test_dice_range(data):
    pred, gt = data
    dice = compute_dice(pred, gt)
    assert 0.0 <= dice <= 1.0


@given(data=mask_pair_strategy())
@settings(max_examples=100)
def test_precision_recall_f1_range(data):
    pred, gt = data
    p, r, f1 = compute_precision_recall_f1(pred, gt)
    assert 0.0 <= p <= 1.0
    assert 0.0 <= r <= 1.0
    assert 0.0 <= f1 <= 1.0


def test_evaluate_single_keys():
    pred = np.zeros((20, 20), dtype=np.uint8)
    gt = np.zeros((20, 20), dtype=np.uint8)
    pred[5:15, 5:15] = 255
    gt[5:15, 5:15] = 255
    result = evaluate_single(pred, gt)
    assert set(result.keys()) == {"iou", "dice", "precision", "recall", "f1"}
    assert result["iou"] == 1.0


def test_evaluate_batch():
    preds = [np.ones((10, 10), dtype=np.uint8) * 255] * 3
    gts = [np.ones((10, 10), dtype=np.uint8) * 255] * 3
    result = evaluate_batch(preds, gts)
    assert "iou" in result
    assert result["iou"]["mean"] == 1.0
    assert result["iou"]["std"] == 0.0
