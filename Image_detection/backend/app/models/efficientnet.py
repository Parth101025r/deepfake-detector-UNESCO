from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import EfficientNet_B4_Weights
from app.config import settings


class TruthLensModel(nn.Module):
    """
    EfficientNet-B4 wrapper with Transfer Learning head and Grad-CAM support.
    Class 0: Real Image
    Class 1: AI Generated Image
    """

    def __init__(
        self, num_classes: int = settings.NUM_CLASSES, pretrained: bool = True
    ):
        super(TruthLensModel, self).__init__()
        weights = EfficientNet_B4_Weights.DEFAULT if pretrained else None
        self.backbone = models.efficientnet_b4(weights=weights)

        # Retrieve output feature count of backbone (1792 for EfficientNet-B4)
        in_features = self.backbone.classifier[1].in_features

        # Replace classification head
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def freeze_backbone(self) -> None:
        """Freezes all feature extractor layers for initial transfer learning."""
        for param in self.backbone.features.parameters():
            param.requires_grad = False
        for param in self.backbone.classifier.parameters():
            param.requires_grad = True

    def unfreeze_fine_tune_layers(self, unfreeze_blocks_from: int = 6) -> None:
        """
        Unfreezes backbone feature blocks starting from `unfreeze_blocks_from` (0 to 8).
        Block 6 onwards corresponds to high-level domain representations.
        """
        for i, block in enumerate(self.backbone.features):
            requires_grad = i >= unfreeze_blocks_from
            for param in block.parameters():
                param.requires_grad = requires_grad
        for param in self.backbone.classifier.parameters():
            param.requires_grad = True

    def get_target_layer_for_gradcam(self) -> nn.Module:
        """Returns target convolutional layer for Grad-CAM activation mapping."""
        return self.backbone.features[-1]


class GradCAM:
    """
    Grad-CAM implementation for EfficientNet-B4.
    Computes gradient-weighted class activation map overlay.
    """

    def __init__(self, model: TruthLensModel, target_layer: nn.Module = None):
        self.model = model
        self.model.eval()
        self.target_layer = (
            target_layer if target_layer is not None else model.get_target_layer_for_gradcam()
        )

        self.gradients = None
        self.activations = None

        # Register forward and backward hooks
        self.target_layer.register_forward_hook(self._save_activations)
        self.target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_heatmap(
        self, input_tensor: torch.Tensor, target_class: int = None
    ) -> torch.Tensor:
        """
        Generates normalized Grad-CAM heatmap (H, W) array between 0 and 1.
        """
        self.model.zero_grad()
        output = self.model(input_tensor)

        if target_class is None:
            target_class = torch.argmax(output, dim=1).item()

        score = output[0, target_class]
        score.backward(retain_graph=True)

        # Pooled gradients across spatial dimensions (channels)
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)

        # ReLU to keep only positive contributions
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()

        # Normalize to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam
