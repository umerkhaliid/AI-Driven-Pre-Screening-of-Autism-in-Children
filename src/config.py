from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

# Place the trained EfficientNet-B0 checkpoint here (.pth), or set FACE_CLASSIFIER_MODEL_PATH.
FACE_CLASSIFIER_DEFAULT_PATH = MODELS_DIR / "efficientnet_b0_autism.pth"

RAW_DATASET_PATH = RAW_DATA_DIR / "synthetic_autism_screening_5000.csv"

PROCESSED_TRAIN_PATH = PROCESSED_DATA_DIR / "train_ready.csv"

RISK_LABELS = {0: "No Risk", 1: "Mild Risk", 2: "Moderate Risk", 3: "Severe Risk"}
NUM_CLASSES = 4
