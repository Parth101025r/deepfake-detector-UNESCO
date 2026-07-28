import base64
import io
from pathlib import Path
from typing import Tuple
import cv2
from PIL import Image
import numpy as np
from fastapi import HTTPException, status
from app.config import settings


def validate_image_file(file_bytes: bytes, filename: str) -> None:
    """Validates image payload size, extension, and integrity."""
    if not file_bytes or len(file_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    # Check file size limit
    max_size_bytes = settings.MAX_FILE_SIZE_MB * 1024 * 1024
    if len(file_bytes) > max_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size exceeds maximum allowed limit of {settings.MAX_FILE_SIZE_MB}MB.",
        )

    # Check extension
    ext = Path(filename).suffix.lower()
    if ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file format '{ext}'. Allowed extensions: {', '.join(settings.ALLOWED_EXTENSIONS)}",
        )

    # Verify corrupt or non-decodable image format
    try:
        image = Image.open(io.BytesIO(file_bytes))
        image.verify()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Corrupted or unreadable image file: {str(e)}",
        )


def bytes_to_cv2(file_bytes: bytes) -> np.ndarray:
    """Decodes image bytes into an RGB NumPy array."""
    nparr = np.frombuffer(file_bytes, np.uint8)
    image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to decode image bytes into valid OpenCV matrix.",
        )
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def bytes_to_pil(file_bytes: bytes) -> Image.Image:
    """Converts image bytes to RGB PIL Image."""
    return Image.open(io.BytesIO(file_bytes)).convert("RGB")


def ndarray_to_base64(img_rgb: np.ndarray) -> str:
    """Encodes an RGB NumPy array to a Base64 PNG data URL string."""
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    success, buffer = cv2.imencode(".png", img_bgr)
    if not success:
        raise ValueError("Failed to encode image to PNG format.")
    b64_str = base64.b64encode(buffer.tobytes()).decode("utf-8")
    return f"data:image/png;base64,{b64_str}"


def overlay_gradcam(
    img_rgb: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Overlays normalized Grad-CAM heatmap (0..1) on RGB input image.
    Returns (overlayed_rgb, heatmap_colored_rgb).
    """
    # Resize heatmap to match image dimensions
    h, w, _ = img_rgb.shape
    heatmap_resized = cv2.resize(heatmap, (w, h))

    # Convert normalized heatmap float to 0..255 uint8
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_colored_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_colored_rgb = cv2.cvtColor(heatmap_colored_bgr, cv2.COLOR_BGR2RGB)

    # Blend original image and heatmap
    overlay_rgb = cv2.addWeighted(img_rgb, 1 - alpha, heatmap_colored_rgb, alpha, 0)
    return overlay_rgb, heatmap_colored_rgb
