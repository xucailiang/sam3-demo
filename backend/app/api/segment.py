"""
SAM3 Demo Backend - Segmentation API Routes

This module provides the API endpoints for image segmentation:
- POST /api/segment/text - Text prompt segmentation
- POST /api/segment/points - Point prompt segmentation
- POST /api/segment/boxes - Bounding box prompt segmentation
- POST /api/segment/stitch - Stitch segmentation (sample + target image)
"""
import io
import json
import logging
from typing import List

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from PIL import Image

from ..models.schemas import (
    BoxPrompt,
    PointPrompt,
    SegmentationResponse,
)
from ..services.segmentation_service import (
    get_segmentation_service,
    parse_text_prompts,
)

from typing import Optional

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/segment", tags=["segmentation"])


def _handle_inference_error(e: Exception, context: str) -> None:
    """Classify inference errors and re-raise appropriately.

    Detects GPU OOM conditions and raises GPUOutOfMemoryError (→ 503).
    All other errors become HTTPException 500.
    """
    err_msg = str(e).lower()
    if "out of memory" in err_msg or "cuda" in err_msg and "oom" in err_msg:
        from ..main import GPUOutOfMemoryError
        raise GPUOutOfMemoryError(str(e))
    logger.error(f"{context}: {e}")
    raise HTTPException(
        status_code=500,
        detail=f"Segmentation failed: {str(e)}",
    )


async def _load_image_from_upload(file: UploadFile) -> np.ndarray:
    """Load and validate an uploaded image file.
    
    Args:
        file: Uploaded file from the request
        
    Returns:
        Image as numpy array in RGB format (H, W, C)
        
    Raises:
        HTTPException: If the image cannot be loaded or is invalid
    """
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Convert to RGB if necessary
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        return np.array(image)
    except Exception as e:
        logger.error(f"Failed to load image: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid image format: {str(e)}"
        )


@router.post("/text", response_model=SegmentationResponse)
async def segment_with_text(
    file: UploadFile = File(..., description="Image file to segment"),
    prompts: str = Form(..., description="Comma-separated text prompts"),
) -> SegmentationResponse:
    """Text prompt segmentation endpoint.
    
    Segments the image based on text descriptions. Multiple prompts can be
    provided as comma-separated values.
    
    Args:
        file: Image file (jpg, jpeg, png, webp)
        prompts: Comma-separated text prompts (e.g., "person, car, dog")
        
    Returns:
        SegmentationResponse with masks for all matching objects
        
    Raises:
        HTTPException 400: Invalid image or empty prompts
        HTTPException 500: Model inference error
    """
    # Validate prompts
    prompt_list = parse_text_prompts(prompts)
    if not prompt_list:
        raise HTTPException(
            status_code=400,
            detail="At least one text prompt is required"
        )
    
    # Load image
    image = await _load_image_from_upload(file)
    
    try:
        # Perform segmentation
        service = get_segmentation_service()
        result = await service.segment_with_text(image, prompt_list)
        
        return SegmentationResponse(
            success=True,
            data=result,
            error=None,
        )
    except Exception as e:
        _handle_inference_error(e, "Text segmentation failed")


@router.post("/points", response_model=SegmentationResponse)
async def segment_with_points(
    file: UploadFile = File(..., description="Image file to segment"),
    points: str = Form(..., description="JSON array of point coordinates"),
    labels: str = Form(..., description="JSON array of point labels (1=foreground, 0=background)"),
) -> SegmentationResponse:
    """Point prompt segmentation endpoint.
    
    Segments the image based on clicked points. Each point has a label
    indicating whether it's a foreground (1) or background (0) point.
    
    Args:
        file: Image file (jpg, jpeg, png, webp)
        points: JSON array of [x, y] coordinates, e.g., "[[100, 200], [150, 250]]"
        labels: JSON array of labels (0 or 1), e.g., "[1, 0]"
        
    Returns:
        SegmentationResponse with mask containing foreground points
        and excluding background points
        
    Raises:
        HTTPException 400: Invalid image, points, or labels
        HTTPException 500: Model inference error
    """
    # Parse points
    try:
        points_data = json.loads(points)
        if not isinstance(points_data, list) or len(points_data) == 0:
            raise ValueError("Points must be a non-empty array")
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid points JSON format: {str(e)}"
        )
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    
    # Parse labels
    try:
        labels_data = json.loads(labels)
        if not isinstance(labels_data, list) or len(labels_data) == 0:
            raise ValueError("Labels must be a non-empty array")
        if len(labels_data) != len(points_data):
            raise ValueError(
                f"Number of labels ({len(labels_data)}) must match "
                f"number of points ({len(points_data)})"
            )
        # Validate label values
        for label in labels_data:
            if label not in [0, 1]:
                raise ValueError(f"Invalid label value: {label}. Must be 0 or 1")
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid labels JSON format: {str(e)}"
        )
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    
    # Convert to tuples
    point_tuples = [(float(p[0]), float(p[1])) for p in points_data]
    
    # Load image
    image = await _load_image_from_upload(file)
    
    try:
        # Perform segmentation
        service = get_segmentation_service()
        result = await service.segment_with_points(image, point_tuples, labels_data)
        
        return SegmentationResponse(
            success=True,
            data=result,
            error=None,
        )
    except Exception as e:
        _handle_inference_error(e, "Point segmentation failed")


@router.post("/boxes", response_model=SegmentationResponse)
async def segment_with_boxes(
    file: UploadFile = File(..., description="Image file to segment"),
    boxes: str = Form(..., description="JSON array of bounding boxes"),
) -> SegmentationResponse:
    """Bounding box prompt segmentation endpoint.
    
    Segments the image based on bounding boxes. Each box defines a region
    of interest for segmentation.
    
    Args:
        file: Image file (jpg, jpeg, png, webp)
        boxes: JSON array of boxes, each as [x1, y1, x2, y2],
               e.g., "[[10, 20, 100, 200], [150, 50, 300, 250]]"
        
    Returns:
        SegmentationResponse with masks for objects in the bounding boxes
        
    Raises:
        HTTPException 400: Invalid image or boxes
        HTTPException 500: Model inference error
    """
    # Parse boxes
    try:
        boxes_data = json.loads(boxes)
        if not isinstance(boxes_data, list) or len(boxes_data) == 0:
            raise ValueError("Boxes must be a non-empty array")
        
        # Validate each box
        validated_boxes = []
        for i, box in enumerate(boxes_data):
            if not isinstance(box, list) or len(box) != 4:
                raise ValueError(
                    f"Box {i} must be an array of 4 numbers [x1, y1, x2, y2]"
                )
            x1, y1, x2, y2 = [float(v) for v in box]
            
            # Normalize coordinates (ensure x1 <= x2 and y1 <= y2)
            if x1 > x2:
                x1, x2 = x2, x1
            if y1 > y2:
                y1, y2 = y2, y1
            
            validated_boxes.append([x1, y1, x2, y2])
            
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
    image = await _load_image_from_upload(file)
    
    try:
        # Perform segmentation
        service = get_segmentation_service()
        result = await service.segment_with_boxes(image, validated_boxes)
        
        return SegmentationResponse(
            success=True,
            data=result,
            error=None,
        )
    except Exception as e:
        _handle_inference_error(e, "Box segmentation failed")


@router.post("/stitch", response_model=SegmentationResponse)
async def segment_with_stitch(
    file: UploadFile = File(..., description="Target image file to segment"),
    sample_file: UploadFile = File(..., description="Sample image file"),
    sample_box: str = Form(..., description="JSON array [x1, y1, x2, y2] for the sample region"),
    confidence: float = Form(0.25, description="Confidence threshold for inference (0-1)"),
) -> SegmentationResponse:
    """Stitch segmentation endpoint.

    Segments the target image by stitching it with the sample image and using
    the sample_box as a visual prompt. The sample image is placed on the left,
    target image on the right, and SAM3 finds similar objects in the target region.

    Args:
        file: Target image file (jpg, jpeg, png, webp)
        sample_file: Sample image file containing the reference object
        sample_box: JSON array [x1, y1, x2, y2] defining the sample region

    Returns:
        SegmentationResponse with masks for all similar objects in target image

    Raises:
        HTTPException 400: Invalid images, missing parameters, or sample_box format error
        HTTPException 500: Model inference error
    """
    # Validate sample_box
    try:
        box_data = json.loads(sample_box)
        if not isinstance(box_data, list) or len(box_data) != 4:
            raise ValueError("sample_box must be an array of 4 numbers [x1, y1, x2, y2]")
        box_floats = [float(v) for v in box_data]
        x1, y1, x2, y2 = box_floats
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1
        validated_box = [x1, y1, x2, y2]
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid sample_box JSON format: {str(e)}"
        )
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

    # Load both images
    target_image = await _load_image_from_upload(file)
    sample_image = await _load_image_from_upload(sample_file)

    try:
        service = get_segmentation_service()
        result = await service.segment_with_stitch(
            target_image, sample_image, validated_box, confidence=confidence,
        )

        return SegmentationResponse(
            success=True,
            data=result,
            error=None,
        )
    except ValueError as e:
        # Stitched image size exceeded limit
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        _handle_inference_error(e, "Stitch segmentation failed")