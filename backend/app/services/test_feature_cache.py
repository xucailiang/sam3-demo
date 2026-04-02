"""Unit tests for sample cache functionality in SegmentationService.

Tests cover:
- create_sample returns valid sample_id
- delete_sample correctly clears cache
- infer_with_sample raises SampleNotFoundError for non-existent sample_id

Requirements: 1.1, 1.3, 4.6
"""
import asyncio
import uuid
from unittest.mock import MagicMock
import numpy as np
import pytest
import torch

from .segmentation_service import SegmentationService
from .exceptions import SampleNotFoundError
from ..models.schemas import DefectBox, CachedSample


class MockPredictor:
    """Mock SAM3SemanticPredictor for testing."""
    
    def __init__(self):
        self.features = torch.randn(1, 256, 64, 64)
        self.args = MagicMock()
        self.args.conf = 0.25
    
    def set_image(self, image):
        """Mock set_image - stores features."""
        pass
    
    def __call__(self, bboxes=None, text=None):
        """Mock inference - returns empty results."""
        return []


class MockModelManager:
    """Mock SAM3ModelManager for testing."""
    
    def __init__(self):
        self._predictor = MockPredictor()
    
    def get_semantic_predictor(self):
        return self._predictor
    
    def clear_cache(self):
        pass


def run_async(coro):
    """Helper to run async functions in sync tests."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class TestCreateSample:
    """Tests for create_sample method."""
    
    @pytest.fixture
    def service(self):
        """Create SegmentationService with mock model manager."""
        return SegmentationService(model_manager=MockModelManager())
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample test image."""
        return np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    @pytest.fixture
    def defect_boxes(self):
        """Create sample defect boxes."""
        return [
            DefectBox(x1=10, y1=10, x2=50, y2=50, category="scratch"),
            DefectBox(x1=60, y1=60, x2=90, y2=90, category="bubble"),
        ]
    
    def test_create_sample_returns_valid_uuid(self, service, sample_image, defect_boxes):
        """Test that create_sample returns a valid UUID string."""
        sample_id, cache_time_ms = run_async(service.create_sample(sample_image, defect_boxes))
        
        # Verify sample_id is a valid UUID
        assert sample_id is not None
        assert isinstance(sample_id, str)
        assert len(sample_id) > 0
        
        # Verify it's a valid UUID format
        parsed_uuid = uuid.UUID(sample_id)
        assert str(parsed_uuid) == sample_id
    
    def test_create_sample_returns_positive_cache_time(self, service, sample_image, defect_boxes):
        """Test that create_sample returns positive cache time."""
        sample_id, cache_time_ms = run_async(service.create_sample(sample_image, defect_boxes))
        
        assert cache_time_ms >= 0
    
    def test_create_sample_caches_image(self, service, sample_image, defect_boxes):
        """Test that create_sample stores sample image in cache."""
        sample_id, _ = run_async(service.create_sample(sample_image, defect_boxes))
        
        # Verify sample is in cache
        assert sample_id in service._sample_cache
        
        cached = service._sample_cache[sample_id]
        assert isinstance(cached, CachedSample)
        assert cached.sample_image is not None
        assert cached.src_shape == sample_image.shape[:2]
        assert len(cached.boxes) == len(defect_boxes)
    
    def test_create_sample_preserves_box_categories(self, service, sample_image, defect_boxes):
        """Test that create_sample preserves defect box categories."""
        sample_id, _ = run_async(service.create_sample(sample_image, defect_boxes))
        
        cached = service._sample_cache[sample_id]
        
        for i, box in enumerate(cached.boxes):
            assert box.category == defect_boxes[i].category
    
    def test_create_multiple_samples_unique_ids(self, service, sample_image, defect_boxes):
        """Test that multiple create_sample calls return unique IDs."""
        sample_id1, _ = run_async(service.create_sample(sample_image, defect_boxes))
        sample_id2, _ = run_async(service.create_sample(sample_image, defect_boxes))
        
        assert sample_id1 != sample_id2
        assert sample_id1 in service._sample_cache
        assert sample_id2 in service._sample_cache


class TestDeleteSample:
    """Tests for delete_sample method."""
    
    @pytest.fixture
    def service(self):
        """Create SegmentationService with mock model manager."""
        return SegmentationService(model_manager=MockModelManager())
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample test image."""
        return np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    @pytest.fixture
    def defect_boxes(self):
        """Create sample defect boxes."""
        return [DefectBox(x1=10, y1=10, x2=50, y2=50, category="scratch")]
    
    def test_delete_sample_removes_from_cache(self, service, sample_image, defect_boxes):
        """Test that delete_sample removes the sample from cache."""
        sample_id, _ = run_async(service.create_sample(sample_image, defect_boxes))
        
        # Verify sample exists
        assert sample_id in service._sample_cache
        
        # Delete sample
        result = service.delete_sample(sample_id)
        
        # Verify deletion
        assert result is True
        assert sample_id not in service._sample_cache
    
    def test_delete_sample_returns_false_for_nonexistent(self, service):
        """Test that delete_sample returns False for non-existent sample."""
        fake_id = str(uuid.uuid4())
        
        result = service.delete_sample(fake_id)
        
        assert result is False
    
    def test_delete_sample_does_not_affect_other_samples(self, service, sample_image, defect_boxes):
        """Test that deleting one sample doesn't affect others."""
        sample_id1, _ = run_async(service.create_sample(sample_image, defect_boxes))
        sample_id2, _ = run_async(service.create_sample(sample_image, defect_boxes))
        
        # Delete first sample
        service.delete_sample(sample_id1)
        
        # Verify second sample still exists
        assert sample_id1 not in service._sample_cache
        assert sample_id2 in service._sample_cache


class TestInferWithSampleNotFound:
    """Tests for infer_with_sample with non-existent sample_id."""
    
    @pytest.fixture
    def service(self):
        """Create SegmentationService with mock model manager."""
        return SegmentationService(model_manager=MockModelManager())
    
    @pytest.fixture
    def target_image(self):
        """Create a target test image."""
        return np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    def test_infer_with_nonexistent_sample_raises_error(self, service, target_image):
        """Test that infer_with_sample raises SampleNotFoundError for non-existent sample."""
        fake_id = str(uuid.uuid4())
        
        with pytest.raises(SampleNotFoundError) as exc_info:
            run_async(service.infer_with_sample(fake_id, target_image))
        
        assert exc_info.value.sample_id == fake_id
    
    def test_infer_after_delete_raises_error(self, service, target_image):
        """Test that infer_with_sample raises error after sample is deleted."""
        # Create and then delete a sample
        sample_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        defect_boxes = [DefectBox(x1=10, y1=10, x2=50, y2=50, category="scratch")]
        
        sample_id, _ = run_async(service.create_sample(sample_image, defect_boxes))
        service.delete_sample(sample_id)
        
        # Try to infer with deleted sample
        with pytest.raises(SampleNotFoundError) as exc_info:
            run_async(service.infer_with_sample(sample_id, target_image))
        
        assert exc_info.value.sample_id == sample_id


class TestListSamples:
    """Tests for list_samples method."""
    
    @pytest.fixture
    def service(self):
        """Create SegmentationService with mock model manager."""
        return SegmentationService(model_manager=MockModelManager())
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample test image."""
        return np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    def test_list_samples_empty_initially(self, service):
        """Test that list_samples returns empty list initially."""
        samples = service.list_samples()
        
        assert samples == []
    
    def test_list_samples_returns_created_samples(self, service, sample_image):
        """Test that list_samples returns all created samples."""
        boxes1 = [DefectBox(x1=10, y1=10, x2=50, y2=50, category="scratch")]
        boxes2 = [
            DefectBox(x1=10, y1=10, x2=50, y2=50, category="scratch"),
            DefectBox(x1=60, y1=60, x2=90, y2=90, category="bubble"),
        ]
        
        sample_id1, _ = run_async(service.create_sample(sample_image, boxes1))
        sample_id2, _ = run_async(service.create_sample(sample_image, boxes2))
        
        samples = service.list_samples()
        
        assert len(samples) == 2
        
        sample_ids = [s.sample_id for s in samples]
        assert sample_id1 in sample_ids
        assert sample_id2 in sample_ids
        
        # Verify boxes_count
        for sample in samples:
            if sample.sample_id == sample_id1:
                assert sample.boxes_count == 1
            elif sample.sample_id == sample_id2:
                assert sample.boxes_count == 2
