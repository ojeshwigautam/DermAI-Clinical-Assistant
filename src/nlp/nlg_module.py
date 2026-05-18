"""
nlg_module.py — Template-based NLG module for patient report generation.

Pipeline:
  Input:  CNN predicted class + NER extracted entities + severity prediction
  Output: Structured clinical draft report

Evaluation:
  BLEU score (corpus-level) comparing generated vs reference reports.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
import random
import pandas as pd
import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

from src.utils.config import (
    CLASS_DESCRIPTIONS, SEVERITY_LABELS, REPORT_DIR, OUTPUT_DIR,
    SYNTHETIC_REPORTS_CSV, RANDOM_SEED,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)
random.seed(RANDOM_SEED)

# Ensure NLTK tokenizer is available
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)


# ─── Report Templates ─────────────────────────────────────────────────────────

REPORT_HEADER = "DERMATOLOGY CONSULTATION REPORT\n" + "=" * 40

REPORT_TEMPLATES = {
    "mild": """
CLINICAL IMPRESSION:
The patient presents with a {lesion_type} located on the {body_part}.
The lesion has been present for approximately {duration}.
Severity assessment: MILD — No immediate intervention required.

DIAGNOSIS:
Preliminary classification: {cnn_diagnosis} ({cnn_description}).
Confidence: {confidence:.1%}

CLINICAL FINDINGS:
- Lesion site: {body_part}
- Lesion type: {lesion_type}
- Symptom duration: {duration}
- Severity: Mild

RECOMMENDATIONS:
• Routine follow-up in 6 months.
• Patient advised on sun protection and self-monitoring.
• No urgent biopsy indicated at this time.
• Return immediately if rapid change, bleeding, or pain develops.

NEXT STEPS:
☐ Dermoscopic monitoring at 6-month review.
☐ Patient education on ABCDE criteria.
""",

    "moderate": """
CLINICAL IMPRESSION:
The patient presents with a {lesion_type} on the {body_part},
reported for approximately {duration}.
Severity assessment: MODERATE — Requires monitoring and possible biopsy.

DIAGNOSIS:
Preliminary classification: {cnn_diagnosis} ({cnn_description}).
Confidence: {confidence:.1%}

CLINICAL FINDINGS:
- Lesion site: {body_part}
- Lesion type: {lesion_type}
- Symptom duration: {duration}
- Severity: Moderate — mild asymmetry or border irregularity noted.

RECOMMENDATIONS:
• Schedule follow-up within 4–6 weeks.
• Consider punch biopsy if further change observed.
• Dermoscopy recommended at next visit.
• Patient advised on self-examination techniques.

NEXT STEPS:
☐ Repeat dermoscopic evaluation in 4 weeks.
☐ Histopathological analysis if clinically indicated.
☐ Document lesion measurements for comparison.
""",

    "severe": """
CLINICAL IMPRESSION:
URGENT: The patient presents with a {lesion_type} on the {body_part}
with a history of approximately {duration} and recent concerning changes.
Severity assessment: SEVERE — Immediate evaluation required.

DIAGNOSIS:
Preliminary classification: {cnn_diagnosis} ({cnn_description}).
Confidence: {confidence:.1%}

CLINICAL FINDINGS:
- Lesion site: {body_part}
- Lesion type: {lesion_type}
- Symptom duration: {duration}
- Severity: SEVERE — Rapid growth, asymmetry, irregular borders, possible ulceration.

RECOMMENDATIONS:
• URGENT excision biopsy within 2 weeks.
• Sentinel lymph node biopsy to be considered pending histology.
• Oncology referral if malignancy confirmed.
• Patient counselled on urgency and prognosis.

NEXT STEPS:
☐ Immediate referral to dermatologic surgery.
☐ Pre-operative blood panel and imaging as indicated.
☐ Histopathology with immunohistochemistry.
☐ Multidisciplinary team review if melanoma confirmed.
""",
}


# ─── NLG Module ───────────────────────────────────────────────────────────────

class PatientReportGenerator:
    """
    Template-based NLG module.

    Given CNN prediction + NER entities + severity → generate clinical report.
    """

    def generate(
        self,
        cnn_class: str,
        cnn_confidence: float,
        entities: dict,
        severity: str,
        report_id: str = "UNKNOWN",
    ) -> str:
        """
        Generate a structured patient report.

        Args:
            cnn_class:      Predicted class label (e.g., 'mel').
            cnn_confidence: Model confidence (0–1).
            entities:       Dict from NER: {'body_part': [...], 'lesion_type': [...], 'symptom_duration': [...]}.
            severity:       Predicted severity ('mild' | 'moderate' | 'severe').
            report_id:      Patient/report identifier.

        Returns:
            Formatted report string.
        """
        if severity not in REPORT_TEMPLATES:
            logger.warning(f"Unknown severity '{severity}', defaulting to 'mild'")
            severity = "mild"

        # Best entity extraction (first found, or fallback)
        body_part  = entities.get("body_part", ["unspecified location"])[0] if entities.get("body_part") else "unspecified location"
        lesion_type = entities.get("lesion_type", ["skin lesion"])[0] if entities.get("lesion_type") else "skin lesion"
        duration   = entities.get("symptom_duration", ["unknown duration"])[0] if entities.get("symptom_duration") else "unknown duration"

        cnn_description = CLASS_DESCRIPTIONS.get(cnn_class, cnn_class)

        body = REPORT_TEMPLATES[severity].format(
            body_part=body_part,
            lesion_type=lesion_type,
            duration=duration,
            cnn_diagnosis=cnn_class.upper(),
            cnn_description=cnn_description,
            confidence=cnn_confidence,
        )

        report = (
            f"{REPORT_HEADER}\n"
            f"Report ID: {report_id}\n"
            f"{'─'*40}\n"
            f"{body}\n"
            f"{'='*40}\n"
            f"⚠ This report is AI-generated for clinical decision support only.\n"
            f"   Final diagnosis must be confirmed by a qualified dermatologist.\n"
        )
        return report

    def batch_generate(self, df: pd.DataFrame, ner_results: list, severity_results: list) -> list:
        """
        Generate reports for a batch of patients.

        Args:
            df:               DataFrame with ground truth entities.
            ner_results:      List of dicts from NER pipeline.
            severity_results: List of dicts from severity classifier.

        Returns:
            List of generated report strings.
        """
        reports = []
        for i, row in df.iterrows():
            entities = ner_results[i] if i < len(ner_results) else {}
            severity_info = severity_results[i] if i < len(severity_results) else {"predicted_label": "mild", "confidence": 0.5}
            report = self.generate(
                cnn_class="nv",         # placeholder; in real pipeline from CNN
                cnn_confidence=0.80,    # placeholder
                entities=entities,
                severity=severity_info.get("predicted_label", "mild"),
                report_id=row.get("report_id", f"RPT{i:04d}"),
            )
            reports.append(report)
        return reports


# ─── BLEU Evaluation ──────────────────────────────────────────────────────────

def compute_bleu(generated_reports: list, reference_reports: list) -> float:
    """
    Compute corpus BLEU score between generated and reference reports.

    Args:
        generated_reports:  List of generated report strings.
        reference_reports:  List of reference report strings.

    Returns:
        BLEU score (float, 0–1).
    """
    smoother = SmoothingFunction().method1

    references = []
    hypotheses = []

    for gen, ref in zip(generated_reports, reference_reports):
        hyp_tokens = nltk.word_tokenize(gen.lower())
        ref_tokens = nltk.word_tokenize(ref.lower())
        hypotheses.append(hyp_tokens)
        references.append([ref_tokens])

    bleu = corpus_bleu(references, hypotheses, smoothing_function=smoother)
    logger.info(f"Corpus BLEU Score: {bleu:.4f}")
    return bleu


def run_nlg_evaluation():
    """Run NLG pipeline and compute BLEU score."""
    from src.nlp.ner_pipeline import build_ner_pipeline, extract_entities

    # Load dataset
    if os.path.exists(SYNTHETIC_REPORTS_CSV):
        df = pd.read_csv(SYNTHETIC_REPORTS_CSV)
    else:
        from src.nlp.data_gen import generate_synthetic_dataset
        df = generate_synthetic_dataset()

    nlp = build_ner_pipeline()
    generator = PatientReportGenerator()

    logger.info("Extracting NER entities for all reports...")
    ner_results = [extract_entities(text, nlp) for text in df["raw_report"].tolist()]

    # Use ground truth severity for BLEU evaluation (simulating classifier output)
    severity_results = [{"predicted_label": sev, "confidence": 0.85} for sev in df["severity"].tolist()]

    logger.info("Generating reports...")
    generated = generator.batch_generate(df, ner_results, severity_results)
    references = df["reference_report"].tolist()

    bleu_score = compute_bleu(generated, references)

    # Save sample reports
    os.makedirs(REPORT_DIR, exist_ok=True)
    for i, (report, row) in enumerate(zip(generated[:5], df.itertuples())):
        path = os.path.join(REPORT_DIR, f"{row.report_id}.txt")
        with open(path, "w") as f:
            f.write(report)

    # Save BLEU score
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "nlg_bleu_score.json"), "w") as f:
        json.dump({"corpus_bleu": round(bleu_score, 4)}, f, indent=2)

    logger.info(f"Sample reports saved to {REPORT_DIR}")
    logger.info(f"BLEU Score: {bleu_score:.4f}")
    return bleu_score


if __name__ == "__main__":
    run_nlg_evaluation()
