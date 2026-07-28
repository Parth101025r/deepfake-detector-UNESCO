from pathlib import Path
from typing import List, Tuple
import torch
try:
    from pydantic_settings import BaseSettings
except ImportError:
    try:
        from pydantic.v1 import BaseSettings
    except ImportError:
        from pydantic import BaseSettings


class Settings(BaseSettings):
    # App Information
    APP_NAME: str = "TruthLens AI - AI Generated Image Detection"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # Server Configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Model & Preprocessing Configuration
    IMAGE_SIZE: int = 380
    NUM_CLASSES: int = 2
    CLASS_NAMES: List[str] = ["Real Image", "AI Generated"]
    
    # ImageNet Mean & Std for Albumentations
    IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)

    # Hardware & Performance
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    CHECKPOINT_DIR: Path = BASE_DIR / "checkpoints"
    LOG_DIR: Path = BASE_DIR / "logs"
    DATASET_DIR: Path = Path("c:/Unesco/Dataset/diffusion_coco_5k")
    BEST_MODEL_PATH: Path = CHECKPOINT_DIR / "best_model.pth"
    LAST_MODEL_PATH: Path = CHECKPOINT_DIR / "last_model.pth"
    ONNX_MODEL_PATH: Path = CHECKPOINT_DIR / "truthlens.onnx"

    # File Upload Constraints
    MAX_FILE_SIZE_MB: int = 10
    ALLOWED_EXTENSIONS: List[str] = [".jpg", ".jpeg", ".png", ".webp", ".bmp"]

    # Training Hyperparameters
    BATCH_SIZE: int = 16
    EPOCHS: int = 15
    INITIAL_LR: float = 1e-3
    FINE_TUNE_LR: float = 1e-4
    WEIGHT_DECAY: float = 1e-4
    FREEZE_EPOCHS: int = 5
    EARLY_STOPPING_PATIENCE: int = 5

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
