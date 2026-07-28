from contextlib import asynccontextmanager
import logging
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.api.routes import router
from app.config import settings
from app.services.inference import engine

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("truthlens.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown event handling."""
    logger.info("Initializing TruthLens AI application resources...")
    # Register inference engine singleton in app state
    app.state.inference_engine = engine
    logger.info(
        f"Model loaded successfully on device '{settings.DEVICE}' | Checkpoint status: {engine.is_custom_checkpoint_loaded}"
    )
    yield
    logger.info("Shutting down TruthLens AI application resources...")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="TruthLens AI - Production Grade API for AI-Generated vs Real Image Detection using EfficientNet-B4, Transfer Learning, TTA & Grad-CAM.",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# Enable CORS for cross-origin request compatibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Exception handlers
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled server error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server processing error. Please check logs for details."},
    )


# Include API routes (both root level and /api/v1 prefix)
app.include_router(router)
app.include_router(router, prefix="/api/v1")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )
