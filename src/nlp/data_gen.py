"""
data_gen.py — Generator for 100 synthetic patient symptom reports.

Each report includes:
  - Patient demographics
  - Lesion description (body_part, lesion_type, symptom_duration)
  - Symptom severity label (mild / moderate / severe)
  - Reference report text for NLG evaluation (BLEU)

Output: data/synthetic_reports.csv
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import random
import pandas as pd
import numpy as np

from src.utils.config import SYNTHETIC_REPORTS_CSV, SYNTHETIC_REPORT_COUNT, RANDOM_SEED, DATA_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ─── Vocabulary ───────────────────────────────────────────────────────────────

BODY_PARTS = [
    "left forearm", "right forearm", "upper back", "lower back",
    "left shoulder", "right shoulder", "left cheek", "right cheek",
    "neck", "scalp", "chest", "abdomen", "left calf", "right calf",
    "left thigh", "right thigh", "left hand", "right hand",
]

LESION_TYPES = [
    "mole", "nevus", "melanoma", "basal cell carcinoma",
    "actinic keratosis", "dermatofibroma", "vascular lesion",
    "benign keratosis", "seborrheic keratosis",
]

DURATIONS = [
    "2 weeks", "1 month", "3 months", "6 months",
    "1 year", "2 years", "several years", "since childhood",
    "4 weeks", "8 months",
]

COLORS = ["brown", "dark brown", "black", "red", "pink", "flesh-colored", "blue-grey"]
TEXTURES = ["smooth", "rough", "scaly", "crusty", "pearly", "waxy", "ulcerated"]
SIZES = ["2mm", "4mm", "6mm", "8mm", "10mm", "12mm", "15mm", "20mm"]

SEVERITIES = ["mild", "moderate", "severe"]

# Severity weights: mild lesions most common
SEVERITY_WEIGHTS = [0.40, 0.38, 0.22]

SEVERITY_INDICATORS = {
    "mild": [
        "The lesion appears stable with no recent changes.",
        "No pain or bleeding reported.",
        "Patient reports occasional mild itching.",
        "The borders appear well-defined and regular.",
    ],
    "moderate": [
        "Patient reports intermittent itching and occasional bleeding.",
        "The lesion has shown gradual enlargement over the past few months.",
        "Some asymmetry noted upon examination.",
        "Border irregularity present; requires monitoring.",
    ],
    "severe": [
        "Rapid growth observed over the past 4 weeks.",
        "Patient reports persistent pain, bleeding, and ulceration.",
        "Marked asymmetry and irregular borders with satellite lesions.",
        "Urgent biopsy recommended due to concerning morphology.",
    ],
}

REFERENCE_TEMPLATES = {
    "mild": (
        "Patient presents with a {size} {color} {lesion_type} on the {body_part} "
        "present for {duration}. Lesion is {texture} with regular borders. "
        "No alarming features noted. Follow-up in 6 months recommended."
    ),
    "moderate": (
        "Patient reports a {size} {color} {lesion_type} on the {body_part} "
        "for approximately {duration}. The lesion exhibits mild asymmetry and "
        "{texture} texture. Monitoring with dermoscopy advised; biopsy to be considered."
    ),
    "severe": (
        "Urgent evaluation of a {size} {color} {lesion_type} located on the {body_part}. "
        "Lesion has been present for {duration} with recent rapid change. "
        "Irregular borders, {texture} surface, and satellite lesions noted. "
        "Immediate excision biopsy strongly recommended."
    ),
}


# ─── Report Generation ────────────────────────────────────────────────────────

def _make_raw_report(body_part: str, lesion_type: str, duration: str,
                     color: str, texture: str, size: str, severity: str) -> str:
    """Generate a natural-language symptom report (patient voice)."""
    intro = random.choice([
        f"I've had a {color} {texture} {lesion_type} on my {body_part} for {duration}.",
        f"There's a {color} {lesion_type} on my {body_part} that's been there for {duration}.",
        f"I noticed a {size} {color} spot on my {body_part} about {duration} ago.",
    ])
    severity_note = random.choice(SEVERITY_INDICATORS[severity])
    context = random.choice([
        "My dermatologist referred me for further evaluation.",
        "I am concerned about its recent change in appearance.",
        "A family member noticed a change in the lesion's appearance.",
        "I have a family history of skin cancer.",
        "I spend significant time outdoors and rarely use sunscreen.",
    ])
    return f"{intro} {severity_note} {context}"


def _make_reference_report(body_part, lesion_type, duration, color, texture, size, severity) -> str:
    """Generate a clinical reference report for BLEU evaluation."""
    template = REFERENCE_TEMPLATES[severity]
    return template.format(
        size=size, color=color, lesion_type=lesion_type,
        body_part=body_part, duration=duration, texture=texture,
    )


def generate_synthetic_dataset(n: int = SYNTHETIC_REPORT_COUNT) -> pd.DataFrame:
    """
    Generate n synthetic patient reports.

    Returns:
        DataFrame with columns:
            report_id, raw_report, body_part, lesion_type,
            symptom_duration, severity, reference_report
    """
    records = []
    for i in range(n):
        body_part  = random.choice(BODY_PARTS)
        lesion_type = random.choice(LESION_TYPES)
        duration   = random.choice(DURATIONS)
        color      = random.choice(COLORS)
        texture    = random.choice(TEXTURES)
        size       = random.choice(SIZES)
        severity   = random.choices(SEVERITIES, weights=SEVERITY_WEIGHTS)[0]

        raw_report = _make_raw_report(body_part, lesion_type, duration, color, texture, size, severity)
        ref_report = _make_reference_report(body_part, lesion_type, duration, color, texture, size, severity)

        records.append({
            "report_id":        f"RPT{i+1:04d}",
            "raw_report":       raw_report,
            "body_part":        body_part,
            "lesion_type":      lesion_type,
            "symptom_duration": duration,
            "color":            color,
            "texture":          texture,
            "size":             size,
            "severity":         severity,
            "reference_report": ref_report,
        })

    df = pd.DataFrame(records)
    logger.info(f"Generated {len(df)} synthetic reports.")
    logger.info(f"Severity distribution:\n{df['severity'].value_counts().to_string()}")
    return df


def save_dataset(df: pd.DataFrame, path: str = SYNTHETIC_REPORTS_CSV):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    logger.info(f"Saved synthetic reports → {path}")


if __name__ == "__main__":
    df = generate_synthetic_dataset(SYNTHETIC_REPORT_COUNT)
    save_dataset(df)
    print(df[["report_id", "raw_report", "severity"]].head(3).to_string())
