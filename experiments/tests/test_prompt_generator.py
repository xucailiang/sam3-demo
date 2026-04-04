# Feature: ampf-crack-segmentation, Property 1: Bounding box contains all foreground pixels
# Feature: ampf-crack-segmentation, Property 2: Point prompts match connected components
"""Property-based tests for PromptGenerator."""

import sys
from pathlib import Path

import cv2
import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from prompt_generator import PromptGenerator


def binary_mask_strategy(min_fg=1):
    """Generate random binary masks with foreground pixels."""
    return (
        st.tuples(
            st.integers(min_value=10, max_value=100),
            st.integers(min_value=10, max_value=100),
        )
        .flatmap(
            lambda hw: st.tuples(
                st.just(hw[0]),
                st.just(hw[1]),
                st.lists(
                    st.tuples(
                        st.integers(min_value=0, max_value=hw[0] - 1),
                        st.integers(min_value=0, max_value=hw[1] - 1),
                    ),
                    min_size=min_fg,
                    max_size=200,
                ),
            )
        )
        .map(lambda t: _build_mask(t[0], t[1], t[2]))
    )


def _build_mask(h, w, coords):
    mask = np.zeros((h, w), dtype=np.uint8)
    for r, c in coords:
        mask[r, c] = 255
    return mask


def connected_component_mask_strategy(min_components=1, max_components=10):
    """Generate masks with distinct connected components (blobs)."""
    return (
        st.integers(min_value=1, max_value=max_components)
        .flatmap(
            lambda n: st.tuples(
                st.just(n),
                st.lists(
                    st.tuples(
                        st.integers(min_value=5, max_value=95),
                        st.integers(min_value=5, max_value=95),
                    ),
                    min_size=n,
                    max_size=n,
                ),
            )
        )
        .map(lambda t: _build_blob_mask(100, 100, t[1]))
    )


def _build_blob_mask(h, w, centers):
    mask = np.zeros((h, w), dtype=np.uint8)
    for cy, cx in centers:
        cv2.circle(mask, (cx, cy), 2, 255, -1)
    return mask


# --- Property 1: Bounding box contains all foreground pixels ---

@given(mask=binary_mask_strategy(min_fg=1))
@settings(max_examples=100)
def test_box_contains_all_foreground(mask):
    box = PromptGenerator.generate_box_prompt(mask)
    assert box is not None
    x1, y1, x2, y2 = box
    rows, cols = np.where(mask > 0)
    # All foreground pixels within box
    assert np.all(cols >= x1), f"col {cols.min()} < x1 {x1}"
    assert np.all(cols <= x2), f"col {cols.max()} > x2 {x2}"
    assert np.all(rows >= y1), f"row {rows.min()} < y1 {y1}"
    assert np.all(rows <= y2), f"row {rows.max()} > y2 {y2}"


def test_box_empty_mask():
    mask = np.zeros((50, 50), dtype=np.uint8)
    assert PromptGenerator.generate_box_prompt(mask) is None


# --- Property 2: Point prompts match connected components ---

@given(mask=connected_component_mask_strategy(min_components=1, max_components=5))
@settings(max_examples=100)
def test_points_match_components(mask):
    num_labels, labels = cv2.connectedComponents(mask, connectivity=8)[:2]
    n_components = num_labels - 1  # exclude background
    if n_components == 0:
        return
    points, point_labels = PromptGenerator.generate_point_prompts(mask)
    assert len(points) == n_components
    assert len(point_labels) == n_components
    assert all(lbl == 1 for lbl in point_labels)
    # Each point falls within its component's foreground
    for x, y in points:
        ix, iy = int(round(x)), int(round(y))
        iy = max(0, min(iy, mask.shape[0] - 1))
        ix = max(0, min(ix, mask.shape[1] - 1))
        assert mask[iy, ix] == 255, f"Point ({x},{y}) not on foreground"


def test_points_empty_mask():
    mask = np.zeros((50, 50), dtype=np.uint8)
    points, labels = PromptGenerator.generate_point_prompts(mask)
    assert points == []
    assert labels == []


def test_text_prompt():
    assert PromptGenerator.generate_text_prompt() == "crack"


def test_generate_all_prompts():
    mask = np.zeros((50, 50), dtype=np.uint8)
    cv2.circle(mask, (25, 25), 5, 255, -1)
    result = PromptGenerator.generate_all_prompts(mask)
    assert "text" in result
    assert "box" in result
    assert "points" in result
    assert "labels" in result
    assert result["text"] == "crack"
    assert result["box"] is not None
    assert len(result["points"]) == 1
