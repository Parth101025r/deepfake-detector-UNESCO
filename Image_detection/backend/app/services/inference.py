import logging
import time
from typing import Any, Dict
import numpy as np
import torch
import torch.nn.functional as F
from app.config import settings
from app.models.efficientnet import GradCAM, TruthLensModel
from app.services.preprocessing import apply_transform_to_image, apply_tta_transforms, get_val_transforms
from app.utils.image_utils import (
    bytes_to_cv2,
    ndarray_to_base64,
    overlay_gradcam,
    validate_image_file,
)

logger = logging.getLogger("truthlens.inference")


class InferenceEngine:
    """Production Inference Engine supporting TTA, Grad-CAM, and logging."""

    def __init__(self):
        self.device = torch.device(settings.DEVICE)
        self.model = TruthLensModel(
            num_classes=settings.NUM_CLASSES, pretrained=True
        ).to(self.device)
        self.gradcam = None
        self.is_custom_checkpoint_loaded = False
        self._load_checkpoint()

    def _load_checkpoint(self) -> None:
        """Loads weights from checkpoint file if available."""
        checkpoint_path = settings.BEST_MODEL_PATH
        if checkpoint_path.exists():
            try:
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                    self.model.load_state_dict(checkpoint["state_dict"])
                else:
                    self.model.load_state_dict(checkpoint)
                self.is_custom_checkpoint_loaded = True
                logger.info(
                    f"Successfully loaded trained model checkpoint from: {checkpoint_path}"
                )
            except Exception as e:
                logger.error(
                    f"Failed to load model checkpoint at {checkpoint_path}: {e}. Running with ImageNet pretrained baseline."
                )
        else:
            logger.warning(
                f"Checkpoint {checkpoint_path} not found. Running with default pretrained backbone."
            )

        self.model.eval()
        self.gradcam = GradCAM(self.model)

    def predict_with_tta(
        self, file_bytes: bytes, filename: str = "upload.jpg"
    ) -> Dict[str, Any]:
        """
        Executes Test-Time Augmentation (TTA) inference across:
        1. Original image
        2. Horizontally flipped image
        3. Brightness-adjusted image
        Averages softmax probabilities for enhanced stability.
        """
        start_time = time.time()
        validate_image_file(file_bytes, filename)

        img_rgb = bytes_to_cv2(file_bytes)
        tta_tensors = apply_tta_transforms(img_rgb, image_size=settings.IMAGE_SIZE)

        all_probs = []
        with torch.no_grad():
            for tensor in tta_tensors:
                tensor = tensor.to(self.device)
                logits = self.model(tensor)
                probs = F.softmax(logits, dim=1).cpu().numpy()[0]
                all_probs.append(probs)

        # Average probabilities across 3 TTA passes
        avg_probs = np.mean(all_probs, axis=0)
        real_prob = float(avg_probs[0] * 100)
        ai_prob = float(avg_probs[1] * 100)

        predicted_class_idx = int(np.argmax(avg_probs))
        confidence = float(np.max(avg_probs) * 100)

        predicted_label = (
            "AI Generated" if predicted_class_idx == 1 else "Real Image"
        )
        elapsed_ms = (time.time() - start_time) * 1000

        logger.info(
            f"Inference completed in {elapsed_ms:.2f}ms | Pred: {predicted_label} ({confidence:.2f}%)"
        )

        return {
            "class": predicted_label,
            "confidence": round(confidence, 2),
            "probabilities": {
                "real": round(real_prob, 2),
                "ai_generated": round(ai_prob, 2),
            },
            "inference_time_ms": round(elapsed_ms, 2),
            "model_version": "EfficientNet-B4-TTA",
        }

    def explain_with_gradcam(
        self, file_bytes: bytes, filename: str = "upload.jpg"
    ) -> Dict[str, Any]:
        """
        Executes prediction and generates a Grad-CAM heatmap visualization.
        """
        start_time = time.time()
        validate_image_file(file_bytes, filename)

        img_rgb = bytes_to_cv2(file_bytes)
        val_transform = get_val_transforms(image_size=settings.IMAGE_SIZE)
        input_tensor = apply_transform_to_image(val_transform, img_rgb).unsqueeze(0).to(self.device)

        # Generate Grad-CAM heatmap
        heatmap = self.gradcam.generate_heatmap(input_tensor)

        # Forward pass for final prediction
        with torch.no_grad():
            logits = self.model(input_tensor)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]

        real_prob = float(probs[0] * 100)
        ai_prob = float(probs[1] * 100)

        predicted_class_idx = int(np.argmax(probs))
        confidence = float(np.max(probs) * 100)
        predicted_label = (
            "AI Generated" if predicted_class_idx == 1 else "Real Image"
        )

        # Overlay heatmap on original resized RGB image
        overlay_rgb, _ = overlay_gradcam(img_rgb, heatmap)
        gradcam_b64 = ndarray_to_base64(overlay_rgb)

        elapsed_ms = (time.time() - start_time) * 1000
        logger.info(
            f"Grad-CAM explanation generated in {elapsed_ms:.2f}ms for {predicted_label}"
        )

        return {
            "class": predicted_label,
            "confidence": round(confidence, 2),
            "probabilities": {
                "real": round(real_prob, 2),
                "ai_generated": round(ai_prob, 2),
            },
            "gradcam_image": gradcam_b64,
            "inference_time_ms": round(elapsed_ms, 2),
        }


# Singleton engine instance
engine = InferenceEngine()
