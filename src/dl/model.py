"""
model.py — ResNet-18 transfer learning setup for HAM10000 skin lesion classification.

Strategy:
  - Load ResNet-18 pretrained on ImageNet.
  - Freeze all layers except the last 2 residual blocks (layer3, layer4) and the FC head.
  - Replace FC head with: Dropout → Linear(512 → 256) → ReLU → Dropout → Linear(256 → 7).
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn
from torchvision import models

from src.utils.config import NUM_CLASSES, FREEZE_LAYERS, MODEL_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DermatologyClassifier(nn.Module):
    """
    ResNet-18 fine-tuned for 7-class skin lesion classification.

    Architecture:
        ResNet-18 backbone (pretrained ImageNet)
        └── Freeze: conv1, bn1, layer1, layer2
        └── Trainable: layer3, layer4, avgpool
        └── Custom FC head:
              Dropout(0.4)
              Linear(512 → 256)
              ReLU
              Dropout(0.3)
              Linear(256 → 7)
    """

    def __init__(self, num_classes: int = NUM_CLASSES, pretrained: bool = True, freeze: bool = FREEZE_LAYERS):
        super().__init__()

        # Load backbone
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet18(weights=weights)
        logger.info(f"Loaded ResNet-18 ({'pretrained' if pretrained else 'random init'})")

        # ── Freeze layers ──────────────────────────────────────────────────────
        if freeze:
            frozen = ["conv1", "bn1", "relu", "maxpool", "layer1", "layer2"]
            for name, module in backbone.named_children():
                if name in frozen:
                    for param in module.parameters():
                        param.requires_grad = False
            logger.info(f"Frozen layers: {frozen}")

        # ── Extract all layers except the original FC ──────────────────────────
        self.features = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
            backbone.avgpool,
        )

        in_features = backbone.fc.in_features  # 512 for ResNet-18

        # ── Custom classification head ─────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes),
        )

        # Register the final conv layer name for Grad-CAM
        self.target_layer = backbone.layer4[-1]
        self._register_hooks()

        self._log_trainable_params()

    def _register_hooks(self):
        """Register forward/backward hooks on layer4's last block for Grad-CAM."""
        self._gradients = None
        self._activations = None

        def forward_hook(module, input, output):
            self._activations = output

        def backward_hook(module, grad_input, grad_output):
            self._gradients = grad_output[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def _log_trainable_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            f"Parameters — Total: {total:,} | Trainable: {trainable:,} "
            f"({100 * trainable / total:.1f}%)"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def get_activations_gradients(self):
        return self._gradients

    def get_activations(self):
        return self._activations


def build_model(num_classes: int = NUM_CLASSES, pretrained: bool = True) -> DermatologyClassifier:
    """Convenience function to build the model."""
    model = DermatologyClassifier(num_classes=num_classes, pretrained=pretrained)
    return model


def save_checkpoint(model: nn.Module, optimizer, epoch: int, metrics: dict, path: str = None):
    """Save model checkpoint."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    if path is None:
        path = os.path.join(MODEL_DIR, f"derma_resnet18_epoch{epoch}.pt")
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }, path)
    logger.info(f"Checkpoint saved → {path}")
    return path


def load_checkpoint(path: str, model: nn.Module = None, optimizer=None):
    """Load model checkpoint."""
    checkpoint = torch.load(path, map_location="cpu")
    if model is not None:
        model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    logger.info(f"Loaded checkpoint from {path} (epoch {checkpoint['epoch']})")
    return checkpoint


if __name__ == "__main__":
    model = build_model()
    dummy = torch.randn(2, 3, 224, 224)
    out = model(dummy)
    print(f"Output shape: {out.shape}")  # (2, 7)
