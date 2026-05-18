"""
severity_model.py — Fine-tune DistilBERT for 3-class symptom severity classification.

Classes:
  0 → mild
  1 → moderate
  2 → severe

Pipeline:
  1. Load synthetic dataset.
  2. Tokenize with DistilBERT tokenizer.
  3. Fine-tune DistilBertForSequenceClassification for 5 epochs.
  4. Evaluate: accuracy, per-class F1.
  5. Save model to models/severity_classifier/
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score

from src.nlp.data_gen import generate_synthetic_dataset
from src.utils.config import (
    NLP_MODEL_NAME, NLP_MAX_LENGTH, NLP_BATCH_SIZE, NLP_EPOCHS, NLP_LR,
    NUM_SEVERITY_CLASSES, SEVERITY_LABELS, SYNTHETIC_REPORTS_CSV,
    MODEL_DIR, OUTPUT_DIR, RANDOM_SEED,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

SEVERITY_MODEL_DIR = os.path.join(MODEL_DIR, "severity_classifier")
SEVERITY_LABEL2ID  = {label: i for i, label in enumerate(SEVERITY_LABELS)}
SEVERITY_ID2LABEL  = {i: label for i, label in enumerate(SEVERITY_LABELS)}


# ─── Dataset ──────────────────────────────────────────────────────────────────

class SeverityDataset(Dataset):
    def __init__(self, texts: list, labels: list, tokenizer, max_length: int = NLP_MAX_LENGTH):
        self.encodings = tokenizer(
            texts, truncation=True, padding="max_length",
            max_length=max_length, return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx],
        }


# ─── Training ─────────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    if os.path.exists(SYNTHETIC_REPORTS_CSV):
        df = pd.read_csv(SYNTHETIC_REPORTS_CSV)
    else:
        df = generate_synthetic_dataset()
        os.makedirs(os.path.dirname(SYNTHETIC_REPORTS_CSV), exist_ok=True)
        df.to_csv(SYNTHETIC_REPORTS_CSV, index=False)
    logger.info(f"Loaded {len(df)} reports.")
    return df


def train_severity_classifier():
    """Full fine-tuning pipeline for symptom severity classification."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    df = load_data()
    df["label_id"] = df["severity"].map(SEVERITY_LABEL2ID)

    texts  = df["raw_report"].tolist()
    labels = df["label_id"].tolist()

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, stratify=labels, random_state=RANDOM_SEED
    )

    # ── Tokenizer & Model ─────────────────────────────────────────────────────
    tokenizer = DistilBertTokenizerFast.from_pretrained(NLP_MODEL_NAME)
    model = DistilBertForSequenceClassification.from_pretrained(
        NLP_MODEL_NAME,
        num_labels=NUM_SEVERITY_CLASSES,
        id2label=SEVERITY_ID2LABEL,
        label2id=SEVERITY_LABEL2ID,
    ).to(device)

    train_dataset = SeverityDataset(train_texts, train_labels, tokenizer)
    val_dataset   = SeverityDataset(val_texts,   val_labels,   tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=NLP_BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=NLP_BATCH_SIZE)

    # ── Optimizer & Scheduler ─────────────────────────────────────────────────
    optimizer = AdamW(model.parameters(), lr=NLP_LR, weight_decay=0.01)
    total_steps = len(train_loader) * NLP_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps
    )

    # ── Training Loop ─────────────────────────────────────────────────────────
    best_val_f1   = 0.0
    history = []

    for epoch in range(1, NLP_EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad()
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_batch   = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels_batch)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        # Validation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids      = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = outputs.logits.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(batch["labels"].numpy())

        acc      = accuracy_score(all_labels, all_preds)
        macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        avg_loss = total_loss / len(train_loader)

        logger.info(f"Epoch {epoch}/{NLP_EPOCHS} | Loss: {avg_loss:.4f} | Val Acc: {acc:.4f} | Val F1: {macro_f1:.4f}")
        history.append({"epoch": epoch, "loss": avg_loss, "val_acc": acc, "val_f1": macro_f1})

        if macro_f1 > best_val_f1:
            best_val_f1 = macro_f1
            os.makedirs(SEVERITY_MODEL_DIR, exist_ok=True)
            model.save_pretrained(SEVERITY_MODEL_DIR)
            tokenizer.save_pretrained(SEVERITY_MODEL_DIR)
            logger.info(f"  ★ Best model saved (Val F1: {best_val_f1:.4f})")

    logger.info(f"\nFull classification report:\n{classification_report(all_labels, all_preds, target_names=SEVERITY_LABELS, zero_division=0)}")

    # Save history
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "severity_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    return model, tokenizer


# ─── Inference ────────────────────────────────────────────────────────────────

class SeverityPredictor:
    """Wrapper for inference with the fine-tuned severity classifier."""

    def __init__(self, model_dir: str = SEVERITY_MODEL_DIR):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)
        self.model = DistilBertForSequenceClassification.from_pretrained(model_dir).to(self.device)
        self.model.eval()
        logger.info(f"SeverityPredictor loaded from {model_dir}")

    def predict(self, text: str) -> dict:
        """
        Predict severity class for a patient report.

        Returns:
            dict with predicted_label, confidence, all_probs
        """
        encoding = self.tokenizer(
            text, truncation=True, padding="max_length",
            max_length=NLP_MAX_LENGTH, return_tensors="pt",
        )
        with torch.no_grad():
            outputs = self.model(
                input_ids=encoding["input_ids"].to(self.device),
                attention_mask=encoding["attention_mask"].to(self.device),
            )
        probs = torch.softmax(outputs.logits, dim=1)[0].cpu().numpy()
        pred_idx   = int(probs.argmax())
        confidence = float(probs[pred_idx])

        return {
            "predicted_label": SEVERITY_ID2LABEL[pred_idx],
            "predicted_idx":   pred_idx,
            "confidence":      confidence,
            "probabilities":   {SEVERITY_ID2LABEL[i]: float(p) for i, p in enumerate(probs)},
        }


if __name__ == "__main__":
    train_severity_classifier()
