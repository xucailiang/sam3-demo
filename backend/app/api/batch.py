"""
SAM3 Demo Backend - Batch Segmentation API Routes

This module provides the API endpoints for batch image segmentation:
- POST /api/batch/text - Batch text prompt segmentation
- POST /api/batch/stitch - Batch stitch segmentation
"""
import io
import json
import logging
import time
from typing import List

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from PIL import Image

from ..models.schemas import (
    BatchSegmentationResponse,
    BatchSegmentationResult,
)
from ..services.segmentation_service import (
    get_segmentation_service,
    parse_text_prompts,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/batch", tags=["batch"])


def _handle_batch_error(e: Exception, context: str) -> None:
    """Classify batch inference errors and re-raise appropriately."""
    err_msg = str(e).lower()
    if "out of memory" in err_msg or "cuda" in err_msg and "oom" in err_msg:
        from ..main import GPUOutOfMemoryError
        raise GPUOutOfMemoryError(str(e))
    logger.error(f"{context}: {e}")
    raise HTTPException(
        status_code=500,
        detail=f"Batch segmentation failed: {str(e)}",
    )


async def _load_image(file: UploadFile) -> np.ndarray:
    """Load and validate an uploaded image file."""
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        if image.mode != "RGB":
            image = image.convert("RGB")
        return np.array(image)
    except Exception as e:
        logger.error(f"Failed to load image {file.filename}: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid image: {file.filename}")


@router.post("/text", response_model=BatchSegmentationResponse)
async def batch_segment_with_text(
    files: List[UploadFile] = File(..., description="Image files to segment"),
    prompts: str = Form(..., description="Comma-separated text prompts"),
) -> BatchSegmentationResponse:
    """Batch text prompt segmentation endpoint.

    Segments multiple images using the same text prompts.

    Requirements: 8.2
    """
    prompt_list = parse_text_prompts(prompts)
    if not prompt_list:
        raise HTTPException(status_code=400, detail="At least one text prompt is required")

    if not files:
        raise HTTPException(status_code=400, detail="At least one image file is required")

    # Load all images
    images: List[np.ndarray] = []
    for f in files:
        images.append(await _load_image(f))

    total_start = time.time()

    try:
        service = get_segmentation_service()
        results = await service.batch_segment_with_text(images, prompt_list)

        success_count = sum(1 for r in results if r.count > 0)
        total_time = (time.time() - total_start) * 1000

        return BatchSegmentationResponse(
            success=True,
            data=BatchSegmentationResult(
                results=results,
                total=len(results),
                success_count=success_count,
                failed_count=len(results) - success_count,
                total_processing_time_ms=total_time,
            ),
        )
    except Exception as e:
        _handle_batch_error(e, "Batch text segmentation failed")


@router.post("/stitch", response_model=BatchSegmentationResponse)
async def batch_segment_with_stitch(
    files: List[UploadFile] = File(..., description="Target image files"),
    sample_file: UploadFile = File(..., description="Sample image file"),
    sample_box: str = Form(..., description="JSON [x1, y1, x2, y2] for sample region"),
) -> BatchSegmentationResponse:
    """Batch stitch segmentation endpoint.

    Segments multiple target images using a sample image region.
    Each target image is stitched with the sample image and SAM3 finds
    similar objects in the target region.

    Args:
        files: Target image files (jpg, jpeg, png, webp)
        sample_file: Sample image file containing the reference object
        sample_box: JSON array [x1, y1, x2, y2] defining the sample region

    Returns:
        BatchSegmentationResponse with results for all target images

    Raises:
        HTTPException 400: Invalid images, missing parameters, or sample_box format error
        HTTPException 500: Model inference error
    """
    # Validate sample_box
    try:
        box_data = json.loads(sample_box)
        if not isinstance(box_data, list) or len(box_data) != 4:
            raise ValueError("sample_box must be [x1, y1, x2, y2]")
        x1, y1, x2, y2 = [float(v) for v in box_data]
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1
        validated_box = [x1, y1, x2, y2]
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid sample_box JSON: {str(e)}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if not files:
        raise HTTPException(status_code=400, detail="At least one target image is required")

    # Load all target images
    target_images: List[np.ndarray] = []
    for f in files:
        target_images.append(await _load_image(f))

    # Load sample image
    sample_image = await _load_image(sample_file)

    total_start = time.time()

    try:
        service = get_segmentation_service()
        results = await service.batch_segment_with_stitch(
            target_images, sample_image, validated_box,
        )

        success_count = sum(1 for r in results if r.count > 0)
        total_time = (time.time() - total_start) * 1000

        return BatchSegmentationResponse(
            success=True,
            data=BatchSegmentationResult(
                results=results,
                total=len(results),
                success_count=success_count,
                failed_count=len(results) - success_count,
                total_processing_time_ms=total_time,
            ),
        )
    except Exception as e:
        _handle_batch_error(e, "Batch stitch segmentation failed")