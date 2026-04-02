"""
SAM3 Demo Backend - Sample API Routes

This module provides the API endpoints for defect sample management:
- POST /api/sample/create - Create a defect sample with cached features
- DELETE /api/sample/{sample_id} - Delete a sample and release cache
- GET /api/sample/list - List all cached samples
- POST /api/sample/infer - Batch inference using cached features
"""
import io
import json
import logging
from typing import List

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from PIL import Image

from ..models.schemas import (
    CreateSampleApiResponse,
    CreateSampleResponse,
    DefectBox,
    DefectBoxSchema,
    DeleteSampleResponse,
    SampleInferApiResponse,
    SampleListResponse,
)
from ..services.segmentation_service import get_segmentation_service
from ..services.exceptions import SampleNotFoundError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/sample", tags=["sample"])


def _handle_inference_error(e: Exception, context: str) -> None:
    """Classify inference errors and re-raise appropriately."""
    err_msg = str(e).lower()
    if "out of memory" in err_msg or "cuda" in err_msg and "oom" in err_msg:
        from ..main import GPUOutOfMemoryError
        raise GPUOutOfMemoryError(str(e))
    logger.error(f"{context}: {e}")
    raise HTTPException(
        status_code=500,
        detail=f"Sample operation failed: {str(e)}",
    )


async def _load_image_from_upload(file: UploadFile) -> np.ndarray:
    """Load and validate an uploaded image file."""
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        if image.mode != "RGB":
            image = image.convert("RGB")
        return np.array(image)
    except Exception as e:
        logger.error(f"Failed to load image {file.filename}: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid image format: {str(e)}"
        )


@router.post("/create", response_model=CreateSampleApiResponse)
async def create_sample(
    image: UploadFile = File(..., description="Sample image file"),
    boxes: str = Form(..., description="JSON array of defect boxes"),
) -> CreateSampleApiResponse:
    """Create a defect sample and cache features.
    
    Extracts features from the sample image and caches them for reuse
    in subsequent inference operations.
    
    Args:
        image: Sample image file (jpg, jpeg, png, webp)
        boxes: JSON array of defect boxes, each as 
               {"x1": float, "y1": float, "x2": float, "y2": float, "category": str}
    
    Returns:
        CreateSampleApiResponse with sample_id and feature extraction time
    
    Raises:
        HTTPException 400: Invalid image or boxes format
        HTTPException 500: Feature extraction error
    """
    # Parse boxes
    try:
        boxes_data = json.loads(boxes)
        if not isinstance(boxes_data, list):
            raise ValueError("boxes must be a JSON array")
        
        defect_boxes: List[DefectBox] = []
        for i, box in enumerate(boxes_data):
            try:
                schema = DefectBoxSchema(**box)
                defect_boxes.append(DefectBox.from_schema(schema))
            except Exception as e:
                raise ValueError(f"Invalid box at index {i}: {str(e)}")
        
        if len(defect_boxes) == 0:
            raise ValueError("At least one defect box is required")
            
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid boxes JSON format: {str(e)}"
        )
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    
    # Load image
    image_np = await _load_image_from_upload(image)
    
    try:
        service = get_segmentation_service()
        sample_id, feature_time_ms = await service.create_sample(
            image=image_np,
            boxes=defect_boxes,
        )
        
        return CreateSampleApiResponse(
            success=True,
            data=CreateSampleResponse(
                sample_id=sample_id,
                boxes_count=len(defect_boxes),
                feature_time_ms=feature_time_ms,
            ),
            error=None,
        )
    except Exception as e:
        _handle_inference_error(e, "Create sample failed")


@router.delete("/{sample_id}", response_model=DeleteSampleResponse)
async def delete_sample(sample_id: str) -> DeleteSampleResponse:
    """Delete a sample and release cached features.
    
    Args:
        sample_id: The unique identifier of the sample to delete
    
    Returns:
        DeleteSampleResponse indicating success or failure
    
    Raises:
        HTTPException 404: Sample not found
    """
    service = get_segmentation_service()
    deleted = service.delete_sample(sample_id)
    
    if not deleted:
        raise HTTPException(
            status_code=404,
            detail=f"Sample not found: {sample_id}"
        )
    
    return DeleteSampleResponse(
        success=True,
        data=True,
        error=None,
    )


@router.get("/list", response_model=SampleListResponse)
async def list_samples() -> SampleListResponse:
    """List all cached samples.
    
    Returns:
        SampleListResponse with list of sample metadata
    """
    service = get_segmentation_service()
    samples = service.list_samples()
    
    return SampleListResponse(
        success=True,
        data=samples,
        error=None,
    )



@router.post("/infer", response_model=SampleInferApiResponse)
async def infer_with_sample(
    sample_id: str = Form(..., description="Sample ID to use for inference"),
    files: List[UploadFile] = File(..., description="Target image files"),
    confidence: float = Form(0.25, description="Confidence threshold (0-1)"),
) -> SampleInferApiResponse:
    """Batch inference using cached sample features.
    
    Uses the cached features from a previously created sample to perform
    segmentation on multiple target images. This is more efficient than
    re-extracting features for each image.
    
    Args:
        sample_id: The unique identifier of the sample to use
        files: Target image files (jpg, jpeg, png, webp)
        confidence: Confidence threshold for inference (0-1)
    
    Returns:
        SampleInferApiResponse with segmentation results and performance stats
    
    Raises:
        HTTPException 400: Invalid images or parameters
        HTTPException 404: Sample not found
        HTTPException 500: Inference error
    """
    if not files:
        raise HTTPException(
            status_code=400,
            detail="At least one target image is required"
        )
    
    # Load all target images
    target_images: List[np.ndarray] = []
    for f in files:
        target_images.append(await _load_image_from_upload(f))
    
    try:
        service = get_segmentation_service()
        result = await service.batch_infer_with_sample(
            sample_id=sample_id,
            target_images=target_images,
            confidence=confidence,
        )
        
        return SampleInferApiResponse(
            success=True,
            data=result,
            error=None,
        )
    except SampleNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=str(e)
        )
    except Exception as e:
        _handle_inference_error(e, "Sample inference failed")
