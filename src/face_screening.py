"""
Binary face-image screening (Autistic vs Non_Autistic).

Uses the calibrated + threshold-tuned CNN pipeline produced by
Kaggle-Autism/Kaggle-Autism/{train_cnn_candidates,select_best_cnn,calibrate_and_tune_cnn}.py.

Inference flow:
  1. Load best CNN checkpoint (models/cnn/best_cnn.pth) — architecture known from
     best_cnn_info.joblib (e.g., ResNet-50 / EfficientNet-B0 / MobileNetV3-Large).
  2. Forward pass to get logits -> softmax probabilities.
  3. Apply Platt-scaling calibrator (models/cnn/cnn_calibrator.joblib) to get
     a calibrated P(Autistic).
  4. Classify using tuned threshold from models/cnn/cnn_threshold_config.joblib
     (threshold < 0.5 -> high recall for screening).

Legacy fallback: if the new pipeline artifacts are missing, fall back to the
original uncalibrated EfficientNet-B0 at models/efficientnet_b0_autism.pth.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from PIL import Image

from src.config import FACE_CLASSIFIER_DEFAULT_PATH, MODELS_DIR

FACE_CLASS_LABELS: Tuple[str, ...] = ("Autistic", "Non_Autistic")
AUTISTIC_CLASS_INDEX = 0

CNN_DIR = MODELS_DIR / "cnn"
BEST_CNN_PATH = CNN_DIR / "best_cnn.pth"
BEST_CNN_INFO_PATH = CNN_DIR / "best_cnn_info.joblib"
CNN_CALIBRATOR_PATH = CNN_DIR / "cnn_calibrator.joblib"
CNN_THRESHOLD_PATH = CNN_DIR / "cnn_threshold_config.joblib"


def dev_bypass_face_screening() -> bool:
    return os.getenv("DEV_BYPASS_FACE_SCREENING", "").strip().lower() in ("1", "true", "yes")


def _using_calibrated_pipeline() -> bool:
    return all(p.is_file() for p in (BEST_CNN_PATH, BEST_CNN_INFO_PATH, CNN_CALIBRATOR_PATH, CNN_THRESHOLD_PATH))


# ---------- architecture builders (must mirror Kaggle-Autism/cnn_pipeline.py) ----------

def _build_efficientnet_b0(num_classes: int = 2):
    import torch
    from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    in_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(in_features, num_classes)
    return model, EfficientNet_B0_Weights.DEFAULT.transforms()


def _build_resnet50(num_classes: int = 2):
    import torch
    from torchvision.models import resnet50, ResNet50_Weights
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    in_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features, num_classes)
    return model, ResNet50_Weights.DEFAULT.transforms()


def _build_mobilenet_v3_large(num_classes: int = 2):
    import torch
    from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
    model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)
    in_features = model.classifier[3].in_features
    model.classifier[3] = torch.nn.Linear(in_features, num_classes)
    return model, MobileNet_V3_Large_Weights.DEFAULT.transforms()


_ARCH_BUILDERS = {
    "EfficientNet-B0": _build_efficientnet_b0,
    "ResNet-50": _build_resnet50,
    "MobileNetV3-Large": _build_mobilenet_v3_large,
}


# ---------- loading (cached) ----------

@lru_cache(maxsize=1)
def _load_calibrated_pipeline():
    import torch
    import joblib

    info = joblib.load(BEST_CNN_INFO_PATH)
    arch = info["model_name"]
    builder = _ARCH_BUILDERS[arch]
    model, tfms = builder(num_classes=2)

    ckpt = torch.load(BEST_CNN_PATH, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state") or ckpt.get("state_dict") or ckpt
    model.load_state_dict(state, strict=False)
    model.eval()

    calibrator = joblib.load(CNN_CALIBRATOR_PATH)
    threshold_config = joblib.load(CNN_THRESHOLD_PATH)
    return model, tfms, calibrator, threshold_config, arch


@lru_cache(maxsize=1)
def _load_legacy_efficientnet():
    import torch
    path = FACE_CLASSIFIER_DEFAULT_PATH
    if not path.is_file():
        env = os.getenv("FACE_CLASSIFIER_MODEL_PATH", "").strip()
        if env and Path(env).is_file():
            path = Path(env)
        else:
            raise FileNotFoundError(
                f"No face classifier found at {FACE_CLASSIFIER_DEFAULT_PATH} and no "
                "calibrated pipeline at models/cnn/. Train one first."
            )
    model, tfms = _build_efficientnet_b0(num_classes=2)
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    state = ckpt.get("model_state") or ckpt.get("state_dict") or ckpt
    model.load_state_dict(state, strict=False)
    model.eval()
    return model, tfms, str(path)


# ---------- prediction ----------

def predict_face_binary(pil_img: Image.Image) -> Dict[str, Any]:
    """
    Returns dict with probabilities (raw + calibrated if available),
    predicted label, and is_autistic (True => continue questionnaire).

    When calibrated pipeline is present:
      is_autistic = (P_calibrated(Autistic) >= tuned_threshold)
    Otherwise falls back to argmax on raw probs.
    """
    import torch

    if _using_calibrated_pipeline():
        model, tfms, calibrator, threshold_config, arch = _load_calibrated_pipeline()
        x = tfms(pil_img.convert("RGB")).unsqueeze(0)
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

        # Platt-scaling uses (pos_logit - neg_logit) as feature
        logits_np = logits.cpu().numpy()[0]
        margin = np.array([[logits_np[AUTISTIC_CLASS_INDEX] - logits_np[1 - AUTISTIC_CLASS_INDEX]]])
        p_autistic_calibrated = float(calibrator.predict_proba(margin)[0, 1])

        threshold = float(threshold_config["decision_threshold"])
        is_autistic = p_autistic_calibrated >= threshold
        predicted_label = FACE_CLASS_LABELS[AUTISTIC_CLASS_INDEX] if is_autistic else FACE_CLASS_LABELS[1 - AUTISTIC_CLASS_INDEX]

        return {
            "predicted_class_index": AUTISTIC_CLASS_INDEX if is_autistic else 1 - AUTISTIC_CLASS_INDEX,
            "predicted_label": predicted_label,
            "is_autistic": bool(is_autistic),
            "probabilities": {FACE_CLASS_LABELS[i]: float(probs[i]) for i in range(len(FACE_CLASS_LABELS))},
            "calibrated_probability_autistic": p_autistic_calibrated,
            "decision_threshold": threshold,
            "model_arch": arch,
            "model_path": str(BEST_CNN_PATH),
            "pipeline": "calibrated",
        }

    # Legacy fallback
    model, tfms, path_str = _load_legacy_efficientnet()
    x = tfms(pil_img.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    class_idx = int(probs.argmax())
    return {
        "predicted_class_index": class_idx,
        "predicted_label": FACE_CLASS_LABELS[class_idx],
        "is_autistic": class_idx == AUTISTIC_CLASS_INDEX,
        "probabilities": {FACE_CLASS_LABELS[i]: float(probs[i]) for i in range(len(FACE_CLASS_LABELS))},
        "model_path": path_str,
        "pipeline": "legacy_uncalibrated",
    }


def predict_face_binary_or_bypass(pil_img: Optional[Image.Image]) -> Dict[str, Any]:
    if dev_bypass_face_screening():
        return {
            "predicted_class_index": AUTISTIC_CLASS_INDEX,
            "predicted_label": FACE_CLASS_LABELS[AUTISTIC_CLASS_INDEX],
            "is_autistic": True,
            "probabilities": {FACE_CLASS_LABELS[0]: 1.0, FACE_CLASS_LABELS[1]: 0.0},
            "model_path": None,
            "dev_bypass": True,
        }
    if pil_img is None:
        raise ValueError("Image is required when DEV_BYPASS_FACE_SCREENING is not enabled.")
    return predict_face_binary(pil_img)


# Backwards-compat helpers (kept in case other modules import them)
def resolve_face_classifier_path() -> Optional[Path]:
    if BEST_CNN_PATH.is_file():
        return BEST_CNN_PATH
    if FACE_CLASSIFIER_DEFAULT_PATH.is_file():
        return FACE_CLASSIFIER_DEFAULT_PATH
    return None
