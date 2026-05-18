"""
config.py — Global configuration and hyperparameters for P11 project.
"""

import os

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR       = os.path.join(BASE_DIR, "data")
MODEL_DIR      = os.path.join(BASE_DIR, "models")
OUTPUT_DIR     = os.path.join(BASE_DIR, "outputs")
REPORT_DIR     = os.path.join(BASE_DIR, "reports")

# HAM10000 paths (adjust after download)
HAM_IMAGE_DIR  = os.path.join(DATA_DIR, "ham10000_images_part_1")
HAM_IMAGE_DIR2 = os.path.join(DATA_DIR, "ham10000_images_part_2")
HAM_CSV        = os.path.join(DATA_DIR, "HAM10000_metadata.csv")

# ─── DL Hyperparameters ───────────────────────────────────────────────────────
IMAGE_SIZE     = 224          # ResNet-18 expected input
BATCH_SIZE     = 32
NUM_EPOCHS     = 15
LEARNING_RATE  = 0.01
MOMENTUM       = 0.9
WEIGHT_DECAY   = 1e-4
FREEZE_LAYERS  = True         # Freeze all except last 2 blocks + FC head
NUM_CLASSES    = 7
RANDOM_SEED    = 42
TRAIN_SPLIT    = 0.8
VAL_SPLIT      = 0.1
TEST_SPLIT     = 0.1

# Class labels
CLASS_NAMES = {
    0: "akiec",  # Actinic Keratoses
    1: "bcc",    # Basal Cell Carcinoma
    2: "bkl",    # Benign Keratosis
    3: "df",     # Dermatofibroma
    4: "mel",    # Melanoma
    5: "nv",     # Melanocytic Nevi
    6: "vasc",   # Vascular Lesions
}
CLASS_LABELS = list(CLASS_NAMES.values())

CLASS_DESCRIPTIONS = {
    "akiec": "Actinic Keratoses / Intraepithelial Carcinoma",
    "bcc":   "Basal Cell Carcinoma",
    "bkl":   "Benign Keratosis-like Lesions",
    "df":    "Dermatofibroma",
    "mel":   "Melanoma",
    "nv":    "Melanocytic Nevi",
    "vasc":  "Vascular Lesions",
}

# ─── NLP Hyperparameters ──────────────────────────────────────────────────────
NLP_MODEL_NAME      = "distilbert-base-uncased"
NLP_MAX_LENGTH      = 128
NLP_BATCH_SIZE      = 16
NLP_EPOCHS          = 5
NLP_LR              = 2e-5
NUM_SEVERITY_CLASSES = 3
SEVERITY_LABELS      = ["mild", "moderate", "severe"]

SYNTHETIC_REPORT_COUNT = 100
SYNTHETIC_REPORTS_CSV  = os.path.join(DATA_DIR, "synthetic_reports.csv")

# NER entity labels
NER_ENTITIES = ["body_part", "symptom_duration", "lesion_type"]

# ─── Augmentation ─────────────────────────────────────────────────────────────
AUGMENTATION = {
    "horizontal_flip": 0.5,
    "vertical_flip":   0.3,
    "rotation":        30,
    "color_jitter":    {"brightness": 0.2, "contrast": 0.2, "saturation": 0.2, "hue": 0.1},
}
