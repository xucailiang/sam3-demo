"""API endpoint tests for sample routes.

Tests cover:
- POST /api/sample/create returns correct response format
- POST /api/sample/infer returns 404 for invalid sample_id
- Batch inference fault tolerance (continues processing on single image failure)

Requirements: 4.3, 4.6, 5.4
"""
import io
import json
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from ..main import app
from ..models.schemas import (
    BatchSampleInferResult,
    SampleInferResult,
    MaskDataWithCategory,
)
from ..services.exceptions import SampleNotFoundError

# Module path for patching (relative to where tests run from)
SAMPLE_MODULE = "app.api.sample"


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def sample_image_bytes():
    """Create a sample image as bytes for upload."""
    img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer.getvalue()


@pytest.fixture
def valid_boxes_json():
    """Create valid boxes JSON string."""
    return json.dumps([
        {"x1": 10, "y1": 10, "x2": 50, "y2": 50, "category": "scratch"},
        {"x1": 60, "y1": 60, "x2": 90, "y2": 90, "category": "bubble"},
    ])


class TestCreateEndpoint:
    """Tests for POST /api/sample/create endpoint."""
    
    def test_create_returns_correct_format(self, client, sample_image_bytes, valid_boxes_json):
        """Test that create endpoint returns correct response format.
        
        Requirements: 4.3
        """
        mock_sample_id = str(uuid.uuid4())
        mock_feature_time = 120.5
        
        with patch(f"{SAMPLE_MODULE}.get_segmentation_service") as mock_get_service:
            mock_service = MagicMock()
            mock_service.create_sample = AsyncMock(
                return_value=(mock_sample_id, mock_feature_time)
            )
            mock_get_service.return_value = mock_service
            
            response = client.post(
                "/api/sample/create",
                files={"image": ("test.png", sample_image_bytes, "image/png")},
                data={"boxes": valid_boxes_json},
            )
        
        assert response.status_code == 200
        
        data = response.json()
        
        # Verify response structure
        assert "success" in data
        assert "data" in data
        assert "error" in data
        
        assert data["success"] is True
        assert data["error"] is None
        
        # Verify data fields
        result = data["data"]
        assert "sample_id" in result
        assert "boxes_count" in result
        assert "feature_time_ms" in result
        
        assert result["sample_id"] == mock_sample_id
        assert result["boxes_count"] == 2
        assert result["feature_time_ms"] == mock_feature_time
    
    def test_create_with_invalid_boxes_json_returns_400(self, client, sample_image_bytes):
        """Test that create endpoint returns 400 for invalid boxes JSON.
        
        Requirements: 4.3
        """
        response = client.post(
            "/api/sample/create",
            files={"image": ("test.png", sample_image_bytes, "image/png")},
            data={"boxes": "invalid json"},
        )
        
        assert response.status_code == 400
        
        data = response.json()
        assert data["success"] is False
        assert data["error"] is not None
        assert "JSON" in data["error"] or "json" in data["error"].lower()
    
    def test_create_with_empty_boxes_returns_400(self, client, sample_image_bytes):
        """Test that create endpoint returns 400 for empty boxes array.
        
        Requirements: 4.3
        """
        response = client.post(
            "/api/sample/create",
            files={"image": ("test.png", sample_image_bytes, "image/png")},
            data={"boxes": "[]"},
        )
        
        assert response.status_code == 400
        
        data = response.json()
        assert data["success"] is False
        assert "box" in data["error"].lower() or "required" in data["error"].lower()


class TestInferEndpoint:
    """Tests for POST /api/sample/infer endpoint."""
    
    def test_infer_with_invalid_sample_id_returns_404(self, client, sample_image_bytes):
        """Test that infer endpoint returns 404 for non-existent sample_id.
        
        Requirements: 4.6
        """
        fake_sample_id = str(uuid.uuid4())
        
        with patch(f"{SAMPLE_MODULE}.get_segmentation_service") as mock_get_service:
            mock_service = MagicMock()
            mock_service.batch_infer_with_sample = AsyncMock(
                side_effect=SampleNotFoundError(fake_sample_id)
            )
            mock_get_service.return_value = mock_service
            
            response = client.post(
                "/api/sample/infer",
                files=[("files", ("test.png", sample_image_bytes, "image/png"))],
                data={"sample_id": fake_sample_id, "confidence": "0.25"},
            )
        
        assert response.status_code == 404
        
        data = response.json()
        assert data["success"] is False
        assert fake_sample_id in data["error"]
    
    def test_infer_with_no_files_returns_400(self, client):
        """Test that infer endpoint returns 400 when no files provided.
        
        Requirements: 4.3
        """
        response = client.post(
            "/api/sample/infer",
            data={"sample_id": str(uuid.uuid4()), "confidence": "0.25"},
        )
        
        # FastAPI returns 422 for missing required fields
        assert response.status_code in [400, 422]


class TestBatchInferenceFaultTolerance:
    """Tests for batch inference fault tolerance.
    
    Requirements: 5.4
    """
    
    def test_batch_infer_continues_on_partial_failure(self, client, sample_image_bytes):
        """Test that batch inference continues processing when some images fail.
        
        Requirements: 5.4
        """
        sample_id = str(uuid.uuid4())
        
        # Create mock result with partial success
        mock_result = BatchSampleInferResult(
            results=[
                SampleInferResult(
                    masks=[],
                    count=0,
                    processing_time_ms=30.0,
                    image_size=(100, 100),
                ),
            ],
            total=3,
            success_count=1,
            failed_count=2,
            total_time_ms=100.0,
            feature_reuse_saved_ms=50.0,
        )
        
        with patch(f"{SAMPLE_MODULE}.get_segmentation_service") as mock_get_service:
            mock_service = MagicMock()
            mock_service.batch_infer_with_sample = AsyncMock(return_value=mock_result)
            mock_get_service.return_value = mock_service
            
            # Send 3 images
            files = [
                ("files", ("test1.png", sample_image_bytes, "image/png")),
                ("files", ("test2.png", sample_image_bytes, "image/png")),
                ("files", ("test3.png", sample_image_bytes, "image/png")),
            ]
            
            response = client.post(
                "/api/sample/infer",
                files=files,
                data={"sample_id": sample_id, "confidence": "0.25"},
            )
        
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        
        result = data["data"]
        assert result["total"] == 3
        assert result["success_count"] == 1
        assert result["failed_count"] == 2
        # Verify that results are still returned despite failures
        assert len(result["results"]) == 1
    
    def test_batch_infer_returns_performance_stats(self, client, sample_image_bytes):
        """Test that batch inference returns performance statistics.
        
        Requirements: 5.4
        """
        sample_id = str(uuid.uuid4())
        
        mock_result = BatchSampleInferResult(
            results=[
                SampleInferResult(
                    masks=[
                        MaskDataWithCategory(
                            mask_base64="base64data",
                            bbox=[10, 10, 50, 50],
                            score=0.95,
                            area=1600,
                            category="scratch",
                        )
                    ],
                    count=1,
                    processing_time_ms=30.0,
                    image_size=(100, 100),
                ),
            ],
            total=1,
            success_count=1,
            failed_count=0,
            total_time_ms=30.0,
            feature_reuse_saved_ms=120.0,
        )
        
        with patch(f"{SAMPLE_MODULE}.get_segmentation_service") as mock_get_service:
            mock_service = MagicMock()
            mock_service.batch_infer_with_sample = AsyncMock(return_value=mock_result)
            mock_get_service.return_value = mock_service
            
            response = client.post(
                "/api/sample/infer",
                files=[("files", ("test.png", sample_image_bytes, "image/png"))],
                data={"sample_id": sample_id, "confidence": "0.25"},
            )
        
        assert response.status_code == 200
        
        data = response.json()
        result = data["data"]
        
        # Verify performance stats are present
        assert "total_time_ms" in result
        assert "feature_reuse_saved_ms" in result
        assert result["total_time_ms"] >= 0
        assert result["feature_reuse_saved_ms"] >= 0


class TestDeleteEndpoint:
    """Tests for DELETE /api/sample/{sample_id} endpoint."""
    
    def test_delete_nonexistent_sample_returns_404(self, client):
        """Test that delete endpoint returns 404 for non-existent sample.
        
        Requirements: 4.6
        """
        fake_sample_id = str(uuid.uuid4())
        
        with patch(f"{SAMPLE_MODULE}.get_segmentation_service") as mock_get_service:
            mock_service = MagicMock()
            mock_service.delete_sample = MagicMock(return_value=False)
            mock_get_service.return_value = mock_service
            
            response = client.delete(f"/api/sample/{fake_sample_id}")
        
        assert response.status_code == 404
        
        data = response.json()
        assert data["success"] is False
        assert "not found" in data["error"].lower()
    
    def test_delete_existing_sample_returns_success(self, client):
        """Test that delete endpoint returns success for existing sample.
        
        Requirements: 4.5
        """
        sample_id = str(uuid.uuid4())
        
        with patch(f"{SAMPLE_MODULE}.get_segmentation_service") as mock_get_service:
            mock_service = MagicMock()
            mock_service.delete_sample = MagicMock(return_value=True)
            mock_get_service.return_value = mock_service
            
            response = client.delete(f"/api/sample/{sample_id}")
        
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert data["data"] is True
