"""
SAM3 Demo Backend - FastAPI Application Entry Point

This module provides the main FastAPI application with:
- CORS middleware configuration
- Unified exception handling middleware
- Model preloading on startup
- Health check endpoint
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from .api import segment_router, batch_router
from .models import HealthResponse
from .services import get_model_manager, initialize_model_manager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager
    
    Handles startup and shutdown events:
    - Startup: Preload SAM3 models to GPU
    - Shutdown: Cleanup resources
    """
    # Startup
    logger.info("Starting SAM3 Demo API...")
    
    try:
        # Initialize and preload models
        # Set preload=False for faster startup during development
        # In production, set preload=True to ensure models are ready
        initialize_model_manager(preload=False)
        logger.info("Model manager initialized")
    except Exception as e:
        logger.warning(f"Failed to preload models: {e}")
        logger.info("Models will be loaded on first request")
    
    yield
    
    # Shutdown
    logger.info("Shutting down SAM3 Demo API...")


app = FastAPI(
    title="SAM3 Demo API",
    description="SAM3 图像分割演示项目 API 服务",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS 中间件配置 - 允许前端跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 开发环境允许所有来源，生产环境应限制
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(segment_router)
app.include_router(batch_router)


# --- Unified Exception Handlers (Requirements: 6.3, 6.4) ---


class GPUOutOfMemoryError(Exception):
    """Raised when GPU runs out of memory during inference."""
    pass


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Handle request validation errors with 400 status.

    Returns a consistent JSON error envelope for malformed requests
    (missing fields, wrong types, etc.).
    """
    errors = exc.errors()
    detail = "; ".join(
        f"{'.'.join(str(loc) for loc in e.get('loc', []))}: {e.get('msg', 'invalid')}"
        for e in errors
    )
    logger.warning(f"Validation error: {detail}")
    return JSONResponse(
        status_code=400,
        content={"success": False, "data": None, "error": f"Validation error: {detail}"},
    )


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(
    request: Request, exc: StarletteHTTPException
) -> JSONResponse:
    """Handle HTTPException with consistent JSON envelope.

    Ensures all HTTP errors (400, 404, 500, etc.) return the same
    {success, data, error} format.
    """
    logger.warning(f"HTTP {exc.status_code} on {request.url.path}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"success": False, "data": None, "error": str(exc.detail)},
    )


@app.exception_handler(GPUOutOfMemoryError)
async def gpu_oom_handler(
    request: Request, exc: GPUOutOfMemoryError
) -> JSONResponse:
    """Handle GPU out-of-memory errors with 503 status."""
    logger.error(f"GPU OOM: {exc}")
    return JSONResponse(
        status_code=503,
        content={
            "success": False,
            "data": None,
            "error": "GPU out of memory, please try a smaller image",
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    """Catch-all handler for unhandled exceptions — returns 500."""
    logger.error(f"Unhandled error on {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "data": None,
            "error": f"Internal server error: {str(exc)}",
        },
    )


@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "SAM3 Demo API", "status": "running"}


@app.get("/api/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint
    
    Returns the service status and model loading state.
    """
    model_manager = get_model_manager()
    
    return HealthResponse(
        status="healthy",
        model_loaded=model_manager.is_ready(),
    )
