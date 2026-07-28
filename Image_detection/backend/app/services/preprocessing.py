from typing import Any, Callable, List
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
from app.config import settings

# Optional Albumentations integration with automatic torchvision fallback
HAS_ALBUMENTATIONS = False
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False


def get_train_transforms(image_size: int = settings.IMAGE_SIZE) -> Any:
    """Returns training data transformation pipeline."""
    if HAS_ALBUMENTATIONS:
        return A.Compose(
            [
                A.Resize(image_size, image_size),
                A.HorizontalFlip(p=0.5),
                A.RandomRotate90(p=0.3),
                A.ShiftScaleRotate(
                    shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, contrast_limit=0.2, p=0.5
                ),
                A.Normalize(mean=settings.IMAGENET_MEAN, std=settings.IMAGENET_STD),
                ToTensorV2(),
            ]
        )
    else:
        # Fallback to Torchvision transforms
        return T.Compose(
            [
                T.ToPILImage(),
                T.Resize((image_size, image_size)),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomRotation(degrees=15),
                T.ColorJitter(brightness=0.2, contrast=0.2),
                T.ToTensor(),
                T.Normalize(mean=settings.IMAGENET_MEAN, std=settings.IMAGENET_STD),
            ]
        )


def get_val_transforms(image_size: int = settings.IMAGE_SIZE) -> Any:
    """Returns validation & inference transformation pipeline."""
    if HAS_ALBUMENTATIONS:
        return A.Compose(
            [
                A.Resize(image_size, image_size),
                A.Normalize(mean=settings.IMAGENET_MEAN, std=settings.IMAGENET_STD),
                ToTensorV2(),
            ]
        )
    else:
        # Torchvision fallback
        return T.Compose(
            [
                T.ToPILImage(),
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                T.Normalize(mean=settings.IMAGENET_MEAN, std=settings.IMAGENET_STD),
            ]
        )


def apply_transform_to_image(transform: Any, image_rgb: np.ndarray) -> torch.Tensor:
    """Applies either Albumentations or Torchvision transform to an RGB array."""
    if HAS_ALBUMENTATIONS and hasattr(transform, "processors"):
        return transform(image=image_rgb)["image"]
    elif HAS_ALBUMENTATIONS and isinstance(transform, A.Compose):
        return transform(image=image_rgb)["image"]
    else:
        # Torchvision callable pipeline
        if isinstance(image_rgb, np.ndarray):
            return transform(image_rgb)
        return transform(np.array(image_rgb))


def apply_tta_transforms(
    image_rgb: np.ndarray, image_size: int = settings.IMAGE_SIZE
) -> List[torch.Tensor]:
    """
    Applies Test-Time Augmentations (TTA):
    1. Standard (Original)
    2. Horizontally Flipped
    3. Brightness Adjusted
    Returns list of PyTorch tensors (1, 3, H, W).
    """
    if HAS_ALBUMENTATIONS:
        t1 = A.Compose(
            [
                A.Resize(image_size, image_size),
                A.Normalize(mean=settings.IMAGENET_MEAN, std=settings.IMAGENET_STD),
                ToTensorV2(),
            ]
        )
        t2 = A.Compose(
            [
                A.Resize(image_size, image_size),
                A.HorizontalFlip(p=1.0),
                A.Normalize(mean=settings.IMAGENET_MEAN, std=settings.IMAGENET_STD),
                ToTensorV2(),
            ]
        )
        t3 = A.Compose(
            [
                A.Resize(image_size, image_size),
                A.RandomBrightnessContrast(
                    brightness_limit=(0.15, 0.15), contrast_limit=0.0, p=1.0
                ),
                A.Normalize(mean=settings.IMAGENET_MEAN, std=settings.IMAGENET_STD),
                ToTensorV2(),
            ]
        )
        tensor1 = t1(image=image_rgb)["image"].unsqueeze(0)
        tensor2 = t2(image=image_rgb)["image"].unsqueeze(0)
        tensor3 = t3(image=image_rgb)["image"].unsqueeze(0)
    else:
        # Torchvision TTA
        norm = T.Normalize(mean=settings.IMAGENET_MEAN, std=settings.IMAGENET_STD)

        # 1. Original
        pil_img = Image.fromarray(image_rgb)
        t1_pil = pil_img.resize((image_size, image_size))
        tensor1 = norm(T.ToTensor()(t1_pil)).unsqueeze(0)

        # 2. Horizontal Flip
        t2_pil = t1_pil.transpose(Image.FLIP_LEFT_RIGHT)
        tensor2 = norm(T.ToTensor()(t2_pil)).unsqueeze(0)

        # 3. Brightness adjustment
        t3_transform = T.Compose(
            [
                T.ColorJitter(brightness=(1.15, 1.15)),
                T.ToTensor(),
                norm,
            ]
        )
        tensor3 = t3_transform(t1_pil).unsqueeze(0)

    return [tensor1, tensor2, tensor3]
