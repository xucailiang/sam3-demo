"""Unit tests for stitch_images and filter_and_offset_results."""
import numpy as np
import pytest

from .segmentation_service import stitch_images, filter_and_offset_results


class TestStitchImages:
    """Tests for stitch_images utility function."""

    def test_same_height(self):
        sample = np.ones((100, 80, 3), dtype=np.uint8) * 128
        target = np.ones((100, 120, 3), dtype=np.uint8) * 64
        stitched, sw, tw = stitch_images(sample, target)

        assert stitched.shape == (100, 200, 3)
        assert sw == 80
        assert tw == 120
        np.testing.assert_array_equal(stitched[:, :80], sample)
        np.testing.assert_array_equal(stitched[:, 80:], target)

    def test_sample_taller(self):
        sample = np.ones((150, 80, 3), dtype=np.uint8) * 128
        target = np.ones((100, 120, 3), dtype=np.uint8) * 64
        stitched, sw, tw = stitch_images(sample, target)

        assert stitched.shape == (150, 200, 3)
        # Target area below H2 should be black (zero-padded)
        np.testing.assert_array_equal(stitched[100:, 80:], 0)

    def test_target_taller(self):
        sample = np.ones((80, 60, 3), dtype=np.uint8) * 200
        target = np.ones((120, 90, 3), dtype=np.uint8) * 50
        stitched, sw, tw = stitch_images(sample, target)

        assert stitched.shape == (120, 150, 3)
        assert sw == 60
        assert tw == 90
        # Sample area below H1 should be black
        np.testing.assert_array_equal(stitched[80:, :60], 0)


class TestFilterAndOffsetResults:
    """Tests for filter_and_offset_results utility function."""

    def test_filters_sample_region_boxes(self):
        """Boxes fully in sample region (center_x <= sample_width) are removed."""
        masks = np.ones((3, 100, 200), dtype=bool)
        boxes = np.array([
            [10, 10, 50, 50],    # center_x=30 <= 80 -> filtered
            [60, 10, 150, 50],   # center_x=105 > 80 -> kept
            [90, 20, 180, 60],   # center_x=135 > 80 -> kept
        ], dtype=np.float32)
        scores = np.array([0.9, 0.8, 0.7])

        fm, fb, fs = filter_and_offset_results(masks, boxes, scores, 80, 120, 100, 100, 200)

        assert len(fm) == 2
        assert len(fb) == 2
        assert len(fs) == 2

    def test_offsets_x_coordinates(self):
        """Kept boxes have x coords shifted by -sample_width."""
        masks = np.ones((1, 100, 200), dtype=bool)
        boxes = np.array([[90, 20, 180, 60]], dtype=np.float32)
        scores = np.array([0.95])

        fm, fb, fs = filter_and_offset_results(masks, boxes, scores, 80, 120, 100, 100, 200)

        assert fb[0, 0] == pytest.approx(10)   # 90 - 80
        assert fb[0, 1] == pytest.approx(20)   # y unchanged
        assert fb[0, 2] == pytest.approx(100)  # 180 - 80
        assert fb[0, 3] == pytest.approx(60)   # y unchanged

    def test_crops_masks(self):
        """Masks are cropped to target region dimensions."""
        masks = np.ones((2, 100, 200), dtype=bool)
        boxes = np.array([
            [60, 10, 150, 50],
            [90, 20, 180, 60],
        ], dtype=np.float32)
        scores = np.array([0.8, 0.7])

        fm, fb, fs = filter_and_offset_results(masks, boxes, scores, 80, 120, 100, 100, 200)

        assert fm.shape == (2, 100, 120)

    def test_empty_input(self):
        """Empty inputs return correctly shaped empty arrays."""
        masks = np.empty((0, 100, 200), dtype=bool)
        boxes = np.empty((0, 4), dtype=np.float32)
        scores = np.empty((0,), dtype=np.float32)

        fm, fb, fs = filter_and_offset_results(masks, boxes, scores, 80, 120, 100, 100, 200)

        assert fm.shape == (0, 100, 120)
        assert fb.shape == (0, 4)
        assert fs.shape == (0,)

    def test_all_filtered(self):
        """When all boxes are in sample region (center_x <= sample_width), result is empty."""
        masks = np.ones((2, 100, 200), dtype=bool)
        boxes = np.array([
            [10, 10, 50, 50],   # center_x=30 <= 80 -> filtered
            [20, 20, 70, 70],   # center_x=45 <= 80 -> filtered
        ], dtype=np.float32)
        scores = np.array([0.9, 0.8])

        fm, fb, fs = filter_and_offset_results(masks, boxes, scores, 80, 120, 100, 100, 200)

        assert len(fm) == 0
