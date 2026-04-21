import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any

from src.config import MODELS_DIR, RISK_LABELS, NUM_CLASSES
from src.qchat_mapper import map_all_answers_to_features, compute_total_score
from src.scoring import screening_risk_level, screening_referral_interpretation


CALIBRATED_MODEL_PATH = MODELS_DIR / "calibrated_model.joblib"
THRESHOLD_PATH = MODELS_DIR / "threshold_config.joblib"


def load_calibrated_model():
    if not CALIBRATED_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Calibrated model not found at {CALIBRATED_MODEL_PATH}. "
            "Run calibrate_and_tune_threshold.py first."
        )
    return joblib.load(CALIBRATED_MODEL_PATH)


def load_threshold_config():
    if not THRESHOLD_PATH.exists():
        raise FileNotFoundError(
            f"Threshold config not found at {THRESHOLD_PATH}. "
            "Run calibrate_and_tune_threshold.py first."
        )
    return joblib.load(THRESHOLD_PATH)


def predict_with_thresholds(class_probs, thresholds):
    """
    Multi-class prediction using per-class thresholds.

    For screening: among classes exceeding their threshold,
    pick the highest-severity class (conservative approach).
    Falls back to argmax if none pass.
    """
    passing = []
    for cls in range(NUM_CLASSES):
        if class_probs[cls] >= thresholds[cls]:
            passing.append(cls)

    if len(passing) > 0:
        return max(passing)
    return int(class_probs.argmax())


def validate_payload(payload: Dict[str, Any]):
    required = ["age_mons", "jaundice", "family_mem_with_asd",
                "qchat_answers", "mchat_answers"]
    missing = [x for x in required if x not in payload]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")

    age = payload["age_mons"]
    if not isinstance(age, (int, float)):
        raise ValueError("age_mons must be numeric")
    if age <= 0 or age > 144:
        raise ValueError("age_mons must be between 1 and 144 months")

    gender = get_gender_value(payload)
    if gender not in ["male", "female", "m", "f", "other", "not defined", "undefined"]:
        raise ValueError("gender must be 'male', 'female', or 'other'")

    for f in ["jaundice", "family_mem_with_asd"]:
        v = str(payload[f]).strip().lower()
        if v not in ["yes", "no", "y", "n", "true", "false", "0", "1"]:
            raise ValueError(f"{f} must be yes/no")

    qa = payload["qchat_answers"]
    for q in range(1, 11):
        if q not in qa:
            raise ValueError(f"Missing Q-CHAT answer for question {q}")

    ma = payload["mchat_answers"]
    for q in range(11, 25):
        if q not in ma:
            raise ValueError(f"Missing M-CHAT-R answer for question {q}")


def normalize_yes_no(value: str) -> int:
    value = str(value).strip().lower()
    if value in ["yes", "y", "true", "1"]:
        return 1
    if value in ["no", "n", "false", "0"]:
        return 0
    raise ValueError("Invalid yes/no value")


def get_gender_value(payload: Dict[str, Any]) -> str:
    gender = payload.get("gender")
    if gender is None or str(gender).strip() == "":
        gender = payload.get("sex")
    if gender is None or str(gender).strip() == "":
        raise ValueError("Missing required field: gender")
    return str(gender).strip().lower()


def normalize_gender(value: str) -> int:
    value = str(value).strip().lower()
    if value in ["male", "m"]:
        return 1
    if value in ["female", "f", "other", "not defined", "undefined"]:
        return 0
    raise ValueError("Invalid gender value")


def build_feature_row(mapped_features: Dict[str, int], payload: Dict[str, Any]) -> pd.DataFrame:
    row = {}
    for i in range(1, 25):
        row[f"a{i}"] = mapped_features[f"a{i}"]
    row["age_mons"] = int(payload["age_mons"])
    # Keep the trained-model feature name for compatibility with existing artifacts.
    row["sex"] = normalize_gender(get_gender_value(payload))
    row["jaundice"] = normalize_yes_no(payload["jaundice"])
    row["family_mem_with_asd"] = normalize_yes_no(payload["family_mem_with_asd"])
    return pd.DataFrame([row])


def predict_autism_risk(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Full screening prediction pipeline.

    Returns:
    - Default prediction (argmax of probabilities)
    - Screening prediction (per-class tuned thresholds, conservative for recall)
    - All class probabilities
    """
    validate_payload(payload)

    mapped = map_all_answers_to_features(
        payload["qchat_answers"],
        payload["mchat_answers"]
    )

    total_score = compute_total_score(mapped)
    score_risk_level = screening_risk_level(total_score)
    referral = screening_referral_interpretation(total_score)

    # ML inference
    model = load_calibrated_model()
    threshold_cfg = load_threshold_config()
    per_class_thresholds = threshold_cfg["per_class_thresholds"]

    X = build_feature_row(mapped, payload)
    class_probs = model.predict_proba(X)[0]

    # Default prediction (argmax)
    default_class = int(class_probs.argmax())

    # Screening prediction (per-class thresholds, conservative)
    screening_class = predict_with_thresholds(class_probs, per_class_thresholds)

    # Build probability dict
    class_probabilities = {}
    for cls_id, label in RISK_LABELS.items():
        class_probabilities[label] = round(float(class_probs[cls_id]), 4)

    return {
        "inputs_used": {
            "age_mons": int(payload["age_mons"]),
            "gender": get_gender_value(payload),
            "jaundice": str(payload["jaundice"]).strip().lower(),
            "family_mem_with_asd": str(payload["family_mem_with_asd"]).strip().lower(),
            "qchat_answers": payload["qchat_answers"],
            "mchat_answers": payload["mchat_answers"]
        },
        "screening_score": total_score,
        "screening_score_max": 24,
        "score_risk_level": score_risk_level,
        "referral_interpretation": referral,

        "prediction_default": {
            "predicted_class": default_class,
            "predicted_label": RISK_LABELS[default_class],
        },
        "prediction_screening": {
            "predicted_class": screening_class,
            "predicted_label": RISK_LABELS[screening_class],
            "thresholds_used": {RISK_LABELS[k]: v for k, v in per_class_thresholds.items()},
        },

        "class_probabilities": class_probabilities,

        "disclaimer": (
            "This tool is a screening aid and not a medical diagnosis. "
            "If you are concerned about your child's development, "
            "consult a qualified healthcare professional."
        )
    }
