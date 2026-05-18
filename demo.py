"""
demo.py — End-to-end Pipeline Demo: Image → CNN Diagnosis → NLP Report

Usage:
    python demo.py --image path/to/lesion.jpg
    python demo.py --image path/to/lesion.jpg --report path/to/report.txt
    python demo.py --demo   # Runs with synthetic example (no real image needed)

Pipeline:
    1. Load dermoscopic image
    2. CNN inference → predicted class + confidence
    3. Grad-CAM visualization
    4. NER entity extraction from patient report text
    5. Severity classification
    6. NLG report generation
    7. Print end-to-end summary
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import textwrap

import torch

from src.dl.model import build_model, load_checkpoint
from src.dl.dataset import get_transforms
from src.dl.gradcam import visualize_gradcam
from src.nlp.ner_pipeline import build_ner_pipeline, extract_entities
from src.nlp.nlg_module import PatientReportGenerator
from src.utils.config import CLASS_LABELS, CLASS_DESCRIPTIONS, MODEL_DIR, OUTPUT_DIR, REPORT_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ─── Colours for terminal output ──────────────────────────────────────────────
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"


def print_banner():
    banner = f"""
{BOLD}{CYAN}
╔══════════════════════════════════════════════════════════╗
║   P11 — Dermatology Image Classifier + Patient NLP       ║
║   End-to-End Pipeline Demo                               ║
╚══════════════════════════════════════════════════════════╝
{RESET}"""
    print(banner)


def run_cnn_inference(image_path: str, model, device) -> dict:
    """Run CNN inference on a single image."""
    from PIL import Image
    import torch

    transform = get_transforms("test")
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]

    pred_idx    = probs.argmax().item()
    pred_label  = CLASS_LABELS[pred_idx]
    confidence  = probs[pred_idx].item()
    all_probs   = {CLASS_LABELS[i]: round(probs[i].item(), 4) for i in range(len(CLASS_LABELS))}

    return {
        "predicted_class": pred_label,
        "description":     CLASS_DESCRIPTIONS.get(pred_label, pred_label),
        "confidence":      confidence,
        "all_probs":       all_probs,
        "pred_idx":        pred_idx,
    }


def run_demo_mode():
    """
    Demo mode: runs the full NLP pipeline on a synthetic example.
    No real image or trained model required.
    """
    print(f"\n{BOLD}[DEMO MODE] Running with synthetic example{RESET}\n")

    # ── Simulated CNN output ───────────────────────────────────────────────────
    cnn_result = {
        "predicted_class": "mel",
        "description":     CLASS_DESCRIPTIONS["mel"],
        "confidence":      0.812,
        "all_probs":       {c: round(1/7, 4) for c in CLASS_LABELS},
    }
    cnn_result["all_probs"]["mel"] = 0.812

    print(f"{BOLD}━━━ STEP 1: CNN Prediction ━━━{RESET}")
    print(f"  Predicted Class: {GREEN}{cnn_result['predicted_class'].upper()}{RESET} — {cnn_result['description']}")
    print(f"  Confidence:      {cnn_result['confidence']:.1%}")

    # ── Simulated patient report ───────────────────────────────────────────────
    sample_report = (
        "I've had a dark brown mole on my left forearm for 3 months. "
        "It has been growing rapidly and started bleeding last week. "
        "I'm very concerned as I have a family history of melanoma."
    )

    print(f"\n{BOLD}━━━ STEP 2: Patient Report (Input) ━━━{RESET}")
    for line in textwrap.wrap(sample_report, width=70):
        print(f"  {line}")

    # ── NER ────────────────────────────────────────────────────────────────────
    print(f"\n{BOLD}━━━ STEP 3: NER Entity Extraction ━━━{RESET}")
    nlp = build_ner_pipeline()
    entities = extract_entities(sample_report, nlp)
    print(f"  body_part:        {GREEN}{entities.get('body_part', ['—'])}{RESET}")
    print(f"  lesion_type:      {GREEN}{entities.get('lesion_type', ['—'])}{RESET}")
    print(f"  symptom_duration: {GREEN}{entities.get('symptom_duration', ['—'])}{RESET}")

    # ── Severity ───────────────────────────────────────────────────────────────
    print(f"\n{BOLD}━━━ STEP 4: Severity Classification ━━━{RESET}")
    # Simulated (no trained model needed in demo)
    severity_result = {"predicted_label": "severe", "confidence": 0.91}
    sev = severity_result["predicted_label"]
    color = RED if sev == "severe" else (YELLOW if sev == "moderate" else GREEN)
    print(f"  Severity: {color}{BOLD}{sev.upper()}{RESET}  (confidence: {severity_result['confidence']:.1%})")

    # ── NLG Report ────────────────────────────────────────────────────────────
    print(f"\n{BOLD}━━━ STEP 5: Generated Patient Report ━━━{RESET}")
    generator = PatientReportGenerator()
    report = generator.generate(
        cnn_class=cnn_result["predicted_class"],
        cnn_confidence=cnn_result["confidence"],
        entities=entities,
        severity=severity_result["predicted_label"],
        report_id="DEMO-001",
    )

    print(report)

    # Save report
    os.makedirs(REPORT_DIR, exist_ok=True)
    report_path = os.path.join(REPORT_DIR, "DEMO-001.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"{BOLD}[✓] Report saved → {report_path}{RESET}")


def run_full_pipeline(args):
    """Full pipeline with real image and optional report text."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ── Load Model ────────────────────────────────────────────────────────────
    ckpt = args.checkpoint or os.path.join(MODEL_DIR, "best_model.pt")
    if not os.path.exists(ckpt):
        print(f"{RED}[ERROR] No trained model found at {ckpt}{RESET}")
        print("  Train the model first: python src/dl/train.py")
        sys.exit(1)

    model = build_model().to(device)
    load_checkpoint(ckpt, model)

    # ── STEP 1: CNN Inference ─────────────────────────────────────────────────
    print(f"\n{BOLD}━━━ STEP 1: CNN Inference ━━━{RESET}")
    cnn_result = run_cnn_inference(args.image, model, device)
    pred = cnn_result["predicted_class"]
    conf = cnn_result["confidence"]
    print(f"  Predicted: {GREEN}{pred.upper()}{RESET} — {cnn_result['description']}")
    print(f"  Confidence: {conf:.1%}")
    print(f"  Class Probabilities:")
    for cls, prob in sorted(cnn_result["all_probs"].items(), key=lambda x: -x[1]):
        bar = "█" * int(prob * 30)
        print(f"    {cls:>6}: {bar:<30} {prob:.3f}")

    # ── STEP 2: Grad-CAM ──────────────────────────────────────────────────────
    print(f"\n{BOLD}━━━ STEP 2: Grad-CAM Visualization ━━━{RESET}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    gradcam_result = visualize_gradcam(args.image, model, device)
    print(f"  Grad-CAM saved → {gradcam_result['overlay_path']}")

    # ── STEP 3: NER ───────────────────────────────────────────────────────────
    print(f"\n{BOLD}━━━ STEP 3: NER Entity Extraction ━━━{RESET}")
    report_text = ""
    if args.report and os.path.exists(args.report):
        with open(args.report) as f:
            report_text = f.read()
    elif args.report_text:
        report_text = args.report_text
    else:
        report_text = (
            "Patient reports a dark brown lesion on the upper back for 2 months. "
            "The lesion has shown signs of recent growth."
        )
        print(f"  (No report provided — using default example)")

    nlp = build_ner_pipeline()
    entities = extract_entities(report_text, nlp)
    print(f"  body_part:        {entities.get('body_part')}")
    print(f"  lesion_type:      {entities.get('lesion_type')}")
    print(f"  symptom_duration: {entities.get('symptom_duration')}")

    # ── STEP 4: Severity ──────────────────────────────────────────────────────
    print(f"\n{BOLD}━━━ STEP 4: Severity Prediction ━━━{RESET}")
    severity_dir = os.path.join(MODEL_DIR, "severity_classifier")
    if os.path.exists(severity_dir):
        from src.nlp.severity_model import SeverityPredictor
        predictor = SeverityPredictor(severity_dir)
        severity_result = predictor.predict(report_text)
    else:
        logger.warning("Severity model not found. Using heuristic fallback.")
        severity_result = {"predicted_label": "moderate", "confidence": 0.7}

    sev = severity_result["predicted_label"]
    print(f"  Severity: {BOLD}{sev.upper()}{RESET} (confidence: {severity_result['confidence']:.1%})")

    # ── STEP 5: NLG Report ────────────────────────────────────────────────────
    print(f"\n{BOLD}━━━ STEP 5: Generated Patient Report ━━━{RESET}")
    generator = PatientReportGenerator()
    report = generator.generate(
        cnn_class=pred,
        cnn_confidence=conf,
        entities=entities,
        severity=sev,
        report_id=os.path.splitext(os.path.basename(args.image))[0],
    )
    print(report)

    # Save
    os.makedirs(REPORT_DIR, exist_ok=True)
    report_path = os.path.join(REPORT_DIR, f"{os.path.splitext(os.path.basename(args.image))[0]}_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"{BOLD}[✓] Report saved → {report_path}{RESET}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="P11 End-to-End Dermatology AI Pipeline Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--image",       default=None, help="Path to input dermoscopic image (.jpg/.png)")
    parser.add_argument("--report",      default=None, help="Path to patient report text file")
    parser.add_argument("--report_text", default=None, help="Patient report text (inline string)")
    parser.add_argument("--checkpoint",  default=None, help="CNN model checkpoint path")
    parser.add_argument("--demo",        action="store_true", help="Run demo mode (no image/model needed)")
    return parser.parse_args()


if __name__ == "__main__":
    print_banner()
    args = parse_args()

    if args.demo or args.image is None:
        run_demo_mode()
    else:
        if not os.path.exists(args.image):
            print(f"{RED}[ERROR] Image not found: {args.image}{RESET}")
            sys.exit(1)
        run_full_pipeline(args)
