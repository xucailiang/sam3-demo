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


# --- Deterministic tests: concave crack structures ---


def _old_centroid_method(gt_mask):
    """Replicate original centroid-only behavior for regression testing."""
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        gt_mask, connectivity=8
    )
    points = []
    for i in range(1, num_labels):
        cx, cy = centroids[i]
        points.append((float(cx), float(cy)))
    return points


def test_foreground_point_on_u_shape():
    """U-shaped crack: geometric centroid falls in the hollow center (bg).

    Old centroid method would return a point on background. The nearest-fg
    fallback must select a crack pixel.
    """
    mask = np.zeros((100, 100), dtype=np.uint8)
    # Draw a U shape: two vertical bars + bottom bar, 3px thickness
    cv2.line(mask, (20, 20), (20, 70), 255, 3)  # left vertical
    cv2.line(mask, (20, 70), (60, 70), 255, 3)  # bottom horizontal
    cv2.line(mask, (60, 70), (60, 20), 255, 3)  # right vertical

    # 1. Old centroid method returns a background point
    old_points = _old_centroid_method(mask)
    assert len(old_points) == 1
    old_x, old_y = int(round(old_points[0][0])), int(round(old_points[0][1]))
    assert mask[old_y, old_x] == 0, (
        f"U-shape centroid should be on background, but ({old_x},{old_y}) is fg"
    )

    # 2. New method guarantees a foreground point
    new_points, _ = PromptGenerator.generate_point_prompts(mask)
    assert len(new_points) == 1
    new_x, new_y = int(round(new_points[0][0])), int(round(new_points[0][1]))
    assert mask[new_y, new_x] == 255, (
        f"Nearest-fg point ({new_x},{new_y}) should be on foreground"
    )

    # 3. New point belongs to the same connected component
    num_labels, labels = cv2.connectedComponents(mask, connectivity=8)[:2]
    old_label = labels[old_y, old_x] if (0 <= old_y < 100 and 0 <= old_x < 100) else 0
    new_label = labels[new_y, new_x]
    assert new_label != 0, "Point should be on a foreground component"
    assert old_label == 0, "Centroid should be background"
    assert new_label == 1, "Point should be on the only crack component"


def test_foreground_point_on_c_ring():
    """C-shaped thin ring (nearly closed ellipse): centroid falls in interior bg.

    A thin crack forming a C shape is common in surface spalling imagery.
    The geometric centroid is in the hollow interior.
    """
    mask = np.zeros((100, 100), dtype=np.uint8)
    # Thin ring: ellipse arc from 30° to 330°, 2px thickness
    cv2.ellipse(mask, (50, 50), (25, 25), 0, 30, 330, 255, 2)

    old_points = _old_centroid_method(mask)
    assert len(old_points) >= 1, "Should have at least one crack component"

    new_points, _ = PromptGenerator.generate_point_prompts(mask)
    assert len(new_points) == len(old_points)

    # Each new point must be on foreground
    num_labels, labels = cv2.connectedComponents(mask, connectivity=8)[:2]
    for i, (px, py) in enumerate(new_points):
        ix, iy = int(round(px)), int(round(py))
        assert mask[iy, ix] == 255, (
            f"C-ring point {i} at ({ix},{iy}) not on foreground"
        )
        assert labels[iy, ix] == i + 1, (
            f"C-ring point {i} component mismatch: expected {i+1}, got {labels[iy, ix]}"
        )


def test_foreground_point_on_multi_component_mix():
    """One blob (centroid on fg) + one U-shape (centroid on bg).

    The nearest-fg fallback should only activate for the component where the
    centroid is off-foreground.
    """
    mask = np.zeros((100, 100), dtype=np.uint8)
    # Blob: centroid on fg
    cv2.circle(mask, (75, 25), 6, 255, -1)
    # U-shape: centroid on bg
    cv2.line(mask, (10, 50), (10, 90), 255, 3)
    cv2.line(mask, (10, 90), (40, 90), 255, 3)
    cv2.line(mask, (40, 90), (40, 50), 255, 3)

    old_points = _old_centroid_method(mask)
    assert len(old_points) == 2

    new_points, _ = PromptGenerator.generate_point_prompts(mask)
    assert len(new_points) == 2

    num_labels, labels = cv2.connectedComponents(mask, connectivity=8)[:2]
    # Both new points must be on foreground
    for i, (px, py) in enumerate(new_points):
        ix, iy = int(round(px)), int(round(py))
        assert mask[iy, ix] == 255, (
            f"Mixed point {i} at ({ix},{iy}) not on foreground"
        )
        assert labels[iy, ix] == i + 1, (
            f"Mixed point {i} component mismatch"
        )

    # At least one old centroid should be on background (the U-shape one)
    old_has_bg = False
    for ox, oy in old_points:
        oix, oiy = int(round(ox)), int(round(oy))
        if mask[oiy, oix] == 0:
            old_has_bg = True
            break
    assert old_has_bg, "At least one old centroid should fall on background"


def test_nearest_fg_is_closest_pixel():
    """On a U-shape, the nearest-fg point should be closer to the centroid
    than any other foreground pixel in the same component.
    """
    mask = np.zeros((100, 100), dtype=np.uint8)
    cv2.line(mask, (20, 20), (20, 70), 255, 3)
    cv2.line(mask, (20, 70), (60, 70), 255, 3)
    cv2.line(mask, (60, 70), (60, 20), 255, 3)

    _, labels, _, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cx, cy = centroids[1]  # geometric centroid of the U

    new_points, _ = PromptGenerator.generate_point_prompts(mask)
    nx, ny = new_points[0]
    nearest_dist = (nx - cx) ** 2 + (ny - cy) ** 2

    # Enumerate all foreground pixels in this component, verify none is closer
    component_mask = (labels == 1).astype(np.uint8)
    fg_coords = cv2.findNonZero(component_mask)
    for pt in fg_coords.squeeze(1):
        px, py = pt[0], pt[1]
        dist = (px - cx) ** 2 + (py - cy) ** 2
        assert dist + 1e-6 >= nearest_dist, (
            f"Found pixel ({px},{py}) closer to centroid ({cx:.1f},{cy:.1f})"
            f" than nearest-fg point ({nx:.0f},{ny:.0f})"
            f"  (dist={dist:.1f} vs nearest={nearest_dist:.1f})"
        )
