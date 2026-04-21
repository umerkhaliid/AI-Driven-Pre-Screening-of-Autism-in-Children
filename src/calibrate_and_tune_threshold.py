"""
Calibration + Multi-Class Threshold Tuning
===========================================
1. Loads the best model selected by cross-validation (best_model.joblib)
2. Calibrates its probabilities using CalibratedClassifierCV (Platt scaling)
3. Tunes per-class thresholds to maximize recall for each risk class
   while maintaining a minimum precision constraint
4. Saves: calibrated_model.joblib + threshold_config.joblib
"""

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    precision_score,
    recall_score
)

from src.config import PROCESSED_TRAIN_PATH, MODELS_DIR, RISK_LABELS, NUM_CLASSES


def find_per_class_thresholds(y_true, y_prob, min_precision=0.80):
    """
    For each class, find the lowest threshold that still maintains
    precision >= min_precision. Lower threshold = higher recall (more sensitive).

    For a screening tool: we want to catch as many at-risk children as possible
    (high recall) while keeping false positives manageable (acceptable precision).

    Returns dict: {class_id: optimal_threshold}
    """
    thresholds = {}

    for cls in range(NUM_CLASSES):
        # Binary: is this class vs not
        y_binary = (y_true == cls).astype(int)
        cls_probs = y_prob[:, cls]

        best_threshold = 0.5
        best_recall = 0.0

        # Search thresholds from low to high
        for t in np.arange(0.05, 0.95, 0.01):
            y_pred_binary = (cls_probs >= t).astype(int)

            # Skip if no positive predictions
            if y_pred_binary.sum() == 0:
                continue

            tp = ((y_pred_binary == 1) & (y_binary == 1)).sum()
            fp = ((y_pred_binary == 1) & (y_binary == 0)).sum()
            fn = ((y_pred_binary == 0) & (y_binary == 1)).sum()

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0

            if precision >= min_precision and recall > best_recall:
                best_recall = recall
                best_threshold = t

        thresholds[cls] = float(best_threshold)

    return thresholds


def predict_with_thresholds(y_prob, thresholds):
    """
    Multi-class prediction using per-class thresholds.

    For each sample:
    1. Check which classes exceed their threshold
    2. Among those, pick the highest-severity class (conservative for screening)
    3. If none exceed threshold, fall back to argmax
    """
    n_samples = y_prob.shape[0]
    predictions = np.zeros(n_samples, dtype=int)

    for i in range(n_samples):
        # Check which classes pass their threshold
        passing = []
        for cls in range(NUM_CLASSES):
            if y_prob[i, cls] >= thresholds[cls]:
                passing.append(cls)

        if len(passing) > 0:
            # Pick highest severity class among those passing (conservative screening)
            predictions[i] = max(passing)
        else:
            # Fallback to argmax
            predictions[i] = int(y_prob[i].argmax())

    return predictions


def main():
    df = pd.read_csv(PROCESSED_TRAIN_PATH)

    from src.model_pipeline import get_feature_config, build_preprocessor

    feature_config = get_feature_config()
    X = df[feature_config.numeric_cols]
    y = df["risk_class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Step 1: Load the best model selected by cross-validation
    best_model_path = MODELS_DIR / "best_model.joblib"
    best_info_path = MODELS_DIR / "best_model_info.joblib"

    if not best_model_path.exists():
        raise FileNotFoundError(
            f"Best model not found at {best_model_path}. Run model_training.py first."
        )

    best_model = joblib.load(best_model_path)
    model_info = joblib.load(best_info_path) if best_info_path.exists() else {}
    model_name = model_info.get("model_name", "Unknown")

    print("=" * 70)
    print(f"CALIBRATING BEST MODEL: {model_name}")
    print("=" * 70)

    # Step 2: Calibrate probabilities (Platt scaling)
    print("\nStep 2: Probability calibration (Platt scaling, 5-fold)...")

    calibrated_model = CalibratedClassifierCV(
        estimator=best_model,
        method="sigmoid",
        cv=5
    )
    calibrated_model.fit(X_train, y_train)

    y_prob = calibrated_model.predict_proba(X_test)
    y_pred_default = calibrated_model.predict(X_test)

    auc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="weighted")

    print(f"\nCalibrated Model - Weighted ROC-AUC (OVR): {auc:.4f}")

    print("\n--- Default Predictions (argmax) ---")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_default))
    print("\nClassification Report:")
    print(classification_report(
        y_test, y_pred_default,
        target_names=list(RISK_LABELS.values())
    ))

    # Step 3: Per-class threshold tuning to maximize recall
    print("\n" + "=" * 70)
    print("STEP 3: PER-CLASS THRESHOLD TUNING (min_precision=0.80)")
    print("=" * 70)

    thresholds = find_per_class_thresholds(y_test, y_prob, min_precision=0.80)

    print("\nOptimal per-class thresholds:")
    for cls, t in thresholds.items():
        print(f"  {RISK_LABELS[cls]:>15s}: {t:.2f}")

    # Evaluate with tuned thresholds
    y_pred_tuned = predict_with_thresholds(y_prob, thresholds)

    print("\n--- Tuned Threshold Predictions ---")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_tuned))
    print("\nClassification Report:")
    print(classification_report(
        y_test, y_pred_tuned,
        target_names=list(RISK_LABELS.values())
    ))

    # Compare recall: default vs tuned
    print("\n--- Recall Comparison: Default vs Tuned ---")
    for cls in range(NUM_CLASSES):
        y_binary = (y_test == cls).astype(int)

        recall_default = recall_score(y_binary, (y_pred_default == cls).astype(int), zero_division=0)
        recall_tuned = recall_score(y_binary, (y_pred_tuned == cls).astype(int), zero_division=0)

        diff = recall_tuned - recall_default
        arrow = "+" if diff >= 0 else ""
        print(f"  {RISK_LABELS[cls]:>15s}: "
              f"default={recall_default:.3f}  tuned={recall_tuned:.3f}  ({arrow}{diff:.3f})")

    # Step 4: Save artifacts
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(calibrated_model, MODELS_DIR / "calibrated_model.joblib")

    threshold_config = {
        "per_class_thresholds": thresholds,
        "min_precision_constraint": 0.80,
        "base_model_name": model_name,
    }
    joblib.dump(threshold_config, MODELS_DIR / "threshold_config.joblib")

    print(f"\nSaved calibrated model to: {MODELS_DIR / 'calibrated_model.joblib'}")
    print(f"Saved threshold config to: {MODELS_DIR / 'threshold_config.joblib'}")


if __name__ == "__main__":
    main()
