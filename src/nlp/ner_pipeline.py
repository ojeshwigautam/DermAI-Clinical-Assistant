"""
ner_pipeline.py — spaCy-based NER for patient symptom reports.

Extracts entities:
  - BODY_PART       → e.g., "left forearm", "upper back"
  - SYMPTOM_DURATION → e.g., "2 weeks", "3 months", "since childhood"
  - LESION_TYPE     → e.g., "melanoma", "basal cell carcinoma"

Strategy:
  1. Load spaCy's en_core_web_sm model.
  2. Add a custom EntityRuler with pattern rules for each entity type.
  3. Evaluate using token-level F1 per entity type.

Usage:
    python src/nlp/ner_pipeline.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
import pandas as pd
from collections import defaultdict

import spacy
from spacy.pipeline import EntityRuler

from src.nlp.data_gen import generate_synthetic_dataset
from src.utils.config import SYNTHETIC_REPORTS_CSV, DATA_DIR, OUTPUT_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ─── Pattern Dictionaries ─────────────────────────────────────────────────────

BODY_PART_TERMS = [
    "left forearm", "right forearm", "upper back", "lower back",
    "left shoulder", "right shoulder", "left cheek", "right cheek",
    "neck", "scalp", "chest", "abdomen", "left calf", "right calf",
    "left thigh", "right thigh", "left hand", "right hand",
    "forearm", "shoulder", "cheek", "calf", "thigh", "hand", "back",
    "face", "arm", "leg", "foot", "nose", "ear", "forehead", "temple",
]

LESION_TERMS = [
    "mole", "nevus", "melanoma", "basal cell carcinoma",
    "actinic keratosis", "dermatofibroma", "vascular lesion",
    "benign keratosis", "seborrheic keratosis", "lesion", "growth",
    "spot", "patch", "nodule", "papule", "plaque",
]

DURATION_PATTERNS = [
    {"label": "SYMPTOM_DURATION", "pattern": [{"LIKE_NUM": True}, {"LOWER": {"IN": ["week", "weeks", "month", "months", "year", "years", "day", "days"]}}]},
    {"label": "SYMPTOM_DURATION", "pattern": [{"LOWER": "since"}, {"LOWER": {"IN": ["childhood", "birth", "infancy", "adolescence"]}}]},
    {"label": "SYMPTOM_DURATION", "pattern": [{"LOWER": "several"}, {"LOWER": {"IN": ["weeks", "months", "years"]}}]},
]


def build_patterns() -> list:
    """Build spaCy EntityRuler patterns for all three entity types."""
    patterns = []

    # Body part patterns
    for term in BODY_PART_TERMS:
        words = term.split()
        pattern = [{"LOWER": w} for w in words]
        patterns.append({"label": "BODY_PART", "pattern": pattern})

    # Lesion type patterns
    for term in LESION_TERMS:
        words = term.split()
        pattern = [{"LOWER": w} for w in words]
        patterns.append({"label": "LESION_TYPE", "pattern": pattern})

    # Duration patterns (regex-style token patterns)
    patterns.extend(DURATION_PATTERNS)

    return patterns


def build_ner_pipeline() -> spacy.language.Language:
    """
    Load spaCy model and attach custom EntityRuler for dermatology NER.

    Returns:
        Configured spaCy nlp pipeline.
    """
    try:
        nlp = spacy.load("en_core_web_sm", disable=["ner"])
        logger.info("Loaded spaCy en_core_web_sm (NER disabled, using EntityRuler)")
    except OSError:
        logger.warning("en_core_web_sm not found. Run: python -m spacy download en_core_web_sm")
        nlp = spacy.blank("en")
        logger.info("Using blank spaCy model")

    # Add EntityRuler BEFORE default NER
    ruler = nlp.add_pipe("entity_ruler", before="ner" if "ner" in nlp.pipe_names else None)
    patterns = build_patterns()
    ruler.add_patterns(patterns)
    logger.info(f"EntityRuler loaded with {len(patterns)} patterns.")

    return nlp


def extract_entities(text: str, nlp) -> dict:
    """
    Extract NER entities from a patient report text.

    Args:
        text: Raw patient report string.
        nlp:  spaCy pipeline with EntityRuler.

    Returns:
        dict with keys body_part, lesion_type, symptom_duration (lists of strings).
    """
    doc = nlp(text)
    entities = defaultdict(list)

    for ent in doc.ents:
        key = ent.label_.lower()
        entities[key].append(ent.text)

    return {
        "body_part":        entities.get("body_part", []),
        "lesion_type":      entities.get("lesion_type", []),
        "symptom_duration": entities.get("symptom_duration", []),
    }


# ─── Evaluation ───────────────────────────────────────────────────────────────

def _token_level_match(predicted: list, ground_truth: str) -> tuple:
    """Simple substring matching for evaluation."""
    gt_lower = ground_truth.lower()
    for pred in predicted:
        if gt_lower in pred.lower() or pred.lower() in gt_lower:
            return True, True
    return False, False  # (true_positive, false_negative)


def evaluate_ner(df: pd.DataFrame, nlp) -> dict:
    """
    Evaluate NER on synthetic dataset.

    Computes precision, recall, F1 per entity type using substring matching.

    Returns:
        dict with per-entity and macro metrics.
    """
    entity_cols = {
        "body_part":        "body_part",
        "lesion_type":      "lesion_type",
        "symptom_duration": "symptom_duration",
    }

    results = {}
    for ent_type, col in entity_cols.items():
        tp = fp = fn = 0
        for _, row in df.iterrows():
            extracted = extract_entities(row["raw_report"], nlp)
            preds = extracted.get(ent_type, [])
            gt    = str(row[col]).lower()

            if preds:
                matched = any(gt in p.lower() or p.lower() in gt for p in preds)
                if matched:
                    tp += 1
                else:
                    fp += len(preds)
                    fn += 1
            else:
                fn += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        results[ent_type] = {
            "precision": round(precision, 4),
            "recall":    round(recall, 4),
            "f1":        round(f1, 4),
            "tp": tp, "fp": fp, "fn": fn,
        }
        logger.info(f"NER [{ent_type}] → P: {precision:.4f} | R: {recall:.4f} | F1: {f1:.4f}")

    macro_f1 = sum(v["f1"] for v in results.values()) / len(results)
    results["macro_f1"] = round(macro_f1, 4)
    logger.info(f"NER Macro-F1: {macro_f1:.4f}")
    return results


def run_ner_pipeline():
    """Run full NER pipeline: load data → build NLP → extract → evaluate."""
    # Load or generate dataset
    if os.path.exists(SYNTHETIC_REPORTS_CSV):
        df = pd.read_csv(SYNTHETIC_REPORTS_CSV)
        logger.info(f"Loaded {len(df)} reports from {SYNTHETIC_REPORTS_CSV}")
    else:
        df = generate_synthetic_dataset()
        df.to_csv(SYNTHETIC_REPORTS_CSV, index=False)

    nlp = build_ner_pipeline()

    # Demo extraction on first 5 reports
    logger.info("\n── Sample Extractions ──")
    for _, row in df.head(5).iterrows():
        entities = extract_entities(row["raw_report"], nlp)
        logger.info(f"\n  Report: {row['raw_report'][:80]}...")
        logger.info(f"  Extracted: {entities}")

    # Evaluate
    metrics = evaluate_ner(df, nlp)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    metrics_path = os.path.join(OUTPUT_DIR, "ner_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"NER metrics saved → {metrics_path}")

    return nlp, metrics


if __name__ == "__main__":
    run_ner_pipeline()
