"""
gradcam.py — Grad-CAM visualization for the Dermatology Classifier.

Implements Class Activation Mapping (Selvaraju et al., 2017) to highlight
which skin regions activate each diagnosis class in the CNN.

Usage:
    python src/dl/gradcam.py --image path/to/image.jpg --class_idx 4
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import argparse
import numpy as np
from PIL import Image
import cv2

import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from src.dl.dataset import get_transforms
from src.dl.model import build_model, load_checkpoint
from src.utils.config import CLASS_LABELS, CLASS_DESCRIPTIONS, MODEL_DIR, OUTPUT_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)


class GradCAM:
    """
    Grad-CAM for DermatologyClassifier.

    The model registers forward/backward hooks on layer4's last block
    during construction, so we just need to do a forward + backward pass.
    """

    def __init__(self, model: torch.nn.Module, device: torch.device):
        self.model = model
        self.device = device

    def generate(self, image_tensor: torch.Tensor, class_idx: int = None) -> tuple:
        """
        Generate Grad-CAM heatmap for a single image.

        Args:
            image_tensor: Preprocessed image tensor of shape (1, 3, H, W).
            class_idx:    Target class index. If None, uses predicted class.

        Returns:
            (heatmap, predicted_class_idx, confidence)
            heatmap: numpy array (H, W) in [0, 1]
        """
        self.model.eval()
        image_tensor = image_tensor.to(self.device)
        image_tensor.requires_grad_(True)

        # Forward pass
        output = self.model(image_tensor)
        probs  = torch.softmax(output, dim=1)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        confidence = probs[0, class_idx].item()

        # Backward pass for target class
        self.model.zero_grad()
        score = output[0, class_idx]
        score.backward()

        # Retrieve gradients and activations from hooks
        gradients  = self.model.get_activations_gradients()   # (1, C, H, W)
        activations = self.model.get_activations()             # (1, C, H, W)

        if gradients is None or activations is None:
            raise RuntimeError("Hooks did not capture gradients/activations. Check model architecture.")

        # Global average pooling of gradients → weights
        weights = gradients.mean(dim=[2, 3], keepdim=True)    # (1, C, 1, 1)

        # Weighted sum of activation maps
        cam = (weights * activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        cam = F.relu(cam)

        # Upsample to input size
        cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()

        # Normalize to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam, class_idx, confidence


def overlay_heatmap(original_image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Overlay Grad-CAM heatmap onto original image.

    Args:
        original_image: RGB numpy array (H, W, 3) in [0, 255].
        heatmap:        Float array (H, W) in [0, 1].
        alpha:          Blend weight for heatmap.

    Returns:
        Blended RGB image as uint8 numpy array.
    """
    colormap = cm.get_cmap("jet")
    heatmap_colored = colormap(heatmap)[:, :, :3]  # Drop alpha channel
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)

    original_resized = cv2.resize(original_image, (224, 224))
    blended = (alpha * heatmap_colored + (1 - alpha) * original_resized).astype(np.uint8)
    return blended


def visualize_gradcam(
    image_path: str,
    model: torch.nn.Module,
    device: torch.device,
    class_idx: int = None,
    save_path: str = None,
) -> dict:
    """
    Full Grad-CAM visualization pipeline for a single image.

    Args:
        image_path: Path to input dermoscopic image.
        model:      Loaded DermatologyClassifier.
        device:     Torch device.
        class_idx:  Target class (None = predicted class).
        save_path:  Path to save visualization PNG.

    Returns:
        dict with predicted_class, confidence, and heatmap.
    """
    # Load and preprocess
    original = np.array(Image.open(image_path).convert("RGB"))
    transform = get_transforms("test")
    image_tensor = transform(Image.fromarray(original)).unsqueeze(0)

    # Generate CAM
    gradcam = GradCAM(model, device)
    heatmap, pred_idx, confidence = gradcam.generate(image_tensor, class_idx)
    overlay = overlay_heatmap(original, heatmap)

    pred_label = CLASS_LABELS[pred_idx]
    pred_desc  = CLASS_DESCRIPTIONS.get(pred_label, pred_label)

    logger.info(
        f"Grad-CAM | Predicted: {pred_label} ({pred_desc}) | Confidence: {confidence:.3f}"
    )

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].imshow(cv2.resize(original, (224, 224)))
    axes[0].set_title("Original Image", fontsize=12)
    axes[0].axis("off")

    im = axes[1].imshow(heatmap, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("Grad-CAM Heatmap", fontsize=12)
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(overlay)
    axes[2].set_title(f"Overlay\n{pred_label}: {confidence:.1%}", fontsize=12)
    axes[2].axis("off")

    plt.suptitle(
        f"Grad-CAM — {pred_desc}\n"
        f"Confidence: {confidence:.1%}",
        fontsize=13,
    )
    plt.tight_layout()

    if save_path is None:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        basename = os.path.splitext(os.path.basename(image_path))[0]
        save_path = os.path.join(OUTPUT_DIR, f"gradcam_{basename}_{pred_label}.png")

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    logger.info(f"Grad-CAM visualization saved → {save_path}")
    plt.close()

    return {
        "predicted_class":  pred_label,
        "predicted_desc":   pred_desc,
        "confidence":       confidence,
        "heatmap":          heatmap,
        "overlay_path":     save_path,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Grad-CAM for skin lesion image")
    parser.add_argument("--image",     required=True, help="Path to input dermoscopic image")
    parser.add_argument("--class_idx", type=int, default=None, help="Target class index (default: predicted)")
    parser.add_argument("--checkpoint", default=None, help="Model checkpoint path")
    parser.add_argument("--save_path",  default=None, help="Output PNG path")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = args.checkpoint or os.path.join(MODEL_DIR, "best_model.pt")
    model = build_model().to(device)
    load_checkpoint(ckpt, model)

    result = visualize_gradcam(
        args.image, model, device,
        class_idx=args.class_idx,
        save_path=args.save_path,
    )
    print(f"\nResult: {result['predicted_class']} ({result['confidence']:.1%})")
    print(f"Saved: {result['overlay_path']}")
