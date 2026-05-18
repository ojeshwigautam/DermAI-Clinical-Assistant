"""
train.py — Training loop for skin lesion CNN classifier.

Features:
  - Class-weighted CrossEntropyLoss for imbalanced HAM10000
  - SGD + momentum optimizer
  - ReduceLROnPlateau learning rate scheduler
  - Best-model checkpoint saving (macro-F1)
  - Training curve logging
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import argparse
import json
import time

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score

from src.dl.dataset import get_dataloaders
from src.dl.model import build_model, save_checkpoint
from src.utils.config import (
    NUM_EPOCHS, LEARNING_RATE, MOMENTUM, WEIGHT_DECAY,
    BATCH_SIZE, MODEL_DIR, OUTPUT_DIR,
)
from src.utils.logger import get_logger

logger = get_logger(__name__, log_dir=OUTPUT_DIR)


def train_one_epoch(model, loader, criterion, optimizer, device) -> dict:
    """Run one training epoch. Returns loss and accuracy."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds, all_labels = [], []

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        if (batch_idx + 1) % 20 == 0:
            logger.info(f"  Batch [{batch_idx+1}/{len(loader)}] Loss: {loss.item():.4f}")

    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return {
        "loss":     total_loss / total,
        "accuracy": correct / total,
        "macro_f1": macro_f1,
    }


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> dict:
    """Evaluate model on a dataloader. Returns loss, accuracy, macro-F1."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds, all_labels = [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return {
        "loss":     total_loss / total,
        "accuracy": correct / total,
        "macro_f1": macro_f1,
    }


def train(args):
    """Full training pipeline."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    logger.info("Loading data...")
    loaders = get_dataloaders(batch_size=args.batch_size)
    class_weights = loaders["class_weights"].to(device)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model().to(device)

    # ── Loss, Optimizer, Scheduler ────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=3, factor=0.5,
    )

    # ── Training Loop ─────────────────────────────────────────────────────────
    history = {"train": [], "val": []}
    best_f1 = 0.0
    best_ckpt = None

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        logger.info(f"\n{'='*60}\nEpoch {epoch}/{args.epochs}")

        train_metrics = train_one_epoch(model, loaders["train"], criterion, optimizer, device)
        val_metrics   = evaluate(model, loaders["val"], criterion, device)

        scheduler.step(val_metrics["macro_f1"])

        elapsed = time.time() - start
        logger.info(
            f"Epoch {epoch} | Time: {elapsed:.1f}s\n"
            f"  Train → Loss: {train_metrics['loss']:.4f} | "
            f"Acc: {train_metrics['accuracy']:.4f} | F1: {train_metrics['macro_f1']:.4f}\n"
            f"  Val   → Loss: {val_metrics['loss']:.4f} | "
            f"Acc: {val_metrics['accuracy']:.4f} | F1: {val_metrics['macro_f1']:.4f}"
        )

        history["train"].append(train_metrics)
        history["val"].append(val_metrics)

        # Save best model
        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            best_ckpt = save_checkpoint(
                model, optimizer, epoch,
                {"val_f1": best_f1, **val_metrics},
                path=os.path.join(MODEL_DIR, "best_model.pt"),
            )
            logger.info(f"  ★ New best model! Val macro-F1: {best_f1:.4f}")

    # ── Save history ──────────────────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    history_path = os.path.join(OUTPUT_DIR, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info(f"Training history saved → {history_path}")
    logger.info(f"Best Val macro-F1: {best_f1:.4f} | Checkpoint: {best_ckpt}")

    return model, history


def parse_args():
    parser = argparse.ArgumentParser(description="Train Dermatology CNN Classifier")
    parser.add_argument("--epochs",     type=int,   default=NUM_EPOCHS)
    parser.add_argument("--batch_size", type=int,   default=BATCH_SIZE)
    parser.add_argument("--lr",         type=float, default=LEARNING_RATE)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
