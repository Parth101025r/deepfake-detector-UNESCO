import logging
from typing import Any, Dict
from fastapi import APIRouter, File, HTTPException, Request, UploadFile, status
from pydantic import BaseModel, ConfigDict, Field
from app.config import settings

logger = logging.getLogger("truthlens.api")
router = APIRouter()


class ProbabilitiesResponse(BaseModel):
    real: float
    ai_generated: float


class PredictionResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    class_name: str = Field(..., alias="class")
    confidence: float
    probabilities: ProbabilitiesResponse
    inference_time_ms: float
    model_version: str


class ExplanationResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    class_name: str = Field(..., alias="class")
    confidence: float
    probabilities: ProbabilitiesResponse
    gradcam_image: str
    inference_time_ms: float


class HealthResponse(BaseModel):
    status: str
    app_name: str
    version: str
    device: str
    checkpoint_loaded: bool


@router.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check(request: Request):
    """Health check endpoint providing model and environment diagnostic status."""
    engine = request.app.state.inference_engine
    return HealthResponse(
        status="healthy",
        app_name=settings.APP_NAME,
        version=settings.APP_VERSION,
        device=settings.DEVICE,
        checkpoint_loaded=engine.is_custom_checkpoint_loaded,
    )


@router.post(
    "/predict",
    summary="Predict AI vs Real Image using Test-Time Augmentation",
    description="Upload an image to classify whether it is a Real Image (Class 0) or AI Generated Image (Class 1).",
    response_model=PredictionResponse,
    tags=["Inference"],
)
async def predict_image(request: Request, file: UploadFile = File(...)):
    """Classifies uploaded image with Test-Time Augmentation (TTA)."""
    if not file:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="No file provided."
        )

    try:
        contents = await file.read()
        engine = request.app.state.inference_engine
        result = engine.predict_with_tta(contents, filename=file.filename or "image.jpg")
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred during image classification: {str(e)}",
        )


@router.post(
    "/explain",
    summary="Generate Grad-CAM Heatmap Explanation for Image Prediction",
    description="Upload an image to retrieve prediction along with Base64 Grad-CAM activation overlay.",
    response_model=ExplanationResponse,
    tags=["Explainability"],
)
async def explain_image(request: Request, file: UploadFile = File(...)):
    """Generates prediction and Grad-CAM visual explanation overlay."""
    if not file:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="No file provided."
        )

    try:
        contents = await file.read()
        engine = request.app.state.inference_engine
        result = engine.explain_with_gradcam(
            contents, filename=file.filename or "image.jpg"
        )
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Explanation error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred during Grad-CAM generation: {str(e)}",
        )
