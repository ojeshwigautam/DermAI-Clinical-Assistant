"""
evaluate.py — Evaluation metrics and visualization for the CNN classifier.

Computes:
  - Overall accuracy
  - Macro-F1 score
  - Per-class recall
  - Confusion matrix (saved as PNG)
  - Classification report
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score,
    classification_report, confusion_matrix,
)

from src.dl.dataset import get_dataloaders
from src.dl.model import build_model, load_checkpoint
from src.utils.config import CLASS_LABELS, MODEL_DIR, OUTPUT_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)


@torch.no_grad()
def get_predictions(model, loader, device):
    """Collect all predictions and ground truths from a DataLoader."""
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        probs   = torch.softmax(outputs, dim=1)
        preds   = outputs.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def compute_metrics(y_true, y_pred) -> dict:
    """Compute and log all evaluation metrics."""
    acc       = accuracy_score(y_true, y_pred)
    macro_f1  = f1_score(y_true, y_pred, average="macro", zero_division=0)
    per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)

    metrics = {
        "accuracy":        round(acc, 4),
        "macro_f1":        round(macro_f1, 4),
        "per_class_recall": {
            CLASS_LABELS[i]: round(float(per_class_recall[i]), 4)
            for i in range(len(CLASS_LABELS))
        },
    }

    logger.info(f"\n{'='*50}")
    logger.info(f"Accuracy:   {acc:.4f}")
    logger.info(f"Macro-F1:   {macro_f1:.4f}")
    logger.info("Per-class Recall:")
    for cls, rec in metrics["per_class_recall"].items():
        logger.info(f"  {cls:>6}: {rec:.4f}")
    logger.info(f"\nClassification Report:\n{classification_report(y_true, y_pred, target_names=CLASS_LABELS, zero_division=0)}")

    return metrics


def plot_confusion_matrix(y_true, y_pred, save_path: str = None):
    """Plot and save a normalized confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS,
        ax=ax, linewidths=0.5,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title("HAM10000 Confusion Matrix (Normalized)", fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Confusion matrix saved → {save_path}")
    plt.close()


def plot_training_curves(history_path: str, save_path: str = None):
    """Plot train/val loss and F1 curves from saved history JSON."""
    with open(history_path) as f:
        history = json.load(f)

    epochs = range(1, len(history["train"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    axes[0].plot(epochs, [e["loss"] for e in history["train"]], label="Train")
    axes[0].plot(epochs, [e["loss"] for e in history["val"]],   label="Val")
    axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch"); axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Macro-F1
    axes[1].plot(epochs, [e["macro_f1"] for e in history["train"]], label="Train")
    axes[1].plot(epochs, [e["macro_f1"] for e in history["val"]],   label="Val")
    axes[1].set_title("Macro-F1"); axes[1].set_xlabel("Epoch"); axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("Training Curves — Dermatology Classifier", fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Training curves saved → {save_path}")
    plt.close()


def run_evaluation(checkpoint_path: str = None):
    """Full evaluation pipeline on the test set."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if checkpoint_path is None:
        checkpoint_path = os.path.join(MODEL_DIR, "best_model.pt")

    logger.info(f"Evaluating model: {checkpoint_path}")
    model = build_model().to(device)
    load_checkpoint(checkpoint_path, model)

    loaders = get_dataloaders()
    y_true, y_pred, y_probs = get_predictions(model, loaders["test"], device)

    metrics = compute_metrics(y_true, y_pred)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plot_confusion_matrix(y_true, y_pred, save_path=os.path.join(OUTPUT_DIR, "confusion_matrix.png"))

    history_path = os.path.join(OUTPUT_DIR, "training_history.json")
    if os.path.exists(history_path):
        plot_training_curves(history_path, save_path=os.path.join(OUTPUT_DIR, "training_curves.png"))

    metrics_path = os.path.join(OUTPUT_DIR, "test_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Test metrics saved → {metrics_path}")

    return metrics


if __name__ == "__main__":
    run_evaluation()
