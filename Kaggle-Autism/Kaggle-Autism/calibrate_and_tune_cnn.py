"""
Step 3 of CNN pipeline — mirrors src/calibrate_and_tune_threshold.py.

Uses OUT-OF-FOLD (OOF) predictions saved by cross_validate_cnn.py. This is
the equivalent of sklearn's CalibratedClassifierCV(cv=5): calibration is fit
on predictions the training folds never saw, so the calibrator is unbiased.
Test set stays untouched until the final evaluation.

1. Loads OOF logits for the winning model from models/cnn/cv_oof_<arch>.npz
2. Fits Platt scaling (sigmoid on logit margin) on those OOF logits
3. Tunes decision threshold by MAXIMIZING F_BETA with beta=2 (F2-score).
   F2 weights recall 2x more than precision — standard objective for medical
   screening where missed positive cases (FN) are costlier than false alarms.
4. Evaluates the winner-retrained-on-full-pool model on the held-out TEST set
   with default argmax vs calibrated + tuned threshold.

Saves:
  - models/cnn/cnn_calibrator.joblib
  - models/cnn/cnn_threshold_config.joblib
  - reports/cnn_calibration_report.{md,json}
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import joblib
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, log_loss,
)

from cnn_pipeline import (
    CLASSES, POSITIVE_CLASS_IDX, DATA_ROOT, MODELS_DIR, REPORTS_DIR,
    get_candidates, load_model_from_ckpt,
)


F_BETA = 2.0               # F2-score: medical screening standard (weights recall 2x)
MIN_PRECISION_FLOOR = 0.50  # hard floor to prevent absurdly low thresholds


def platt_fit(logits: np.ndarray, labels: np.ndarray):
    """Fit Platt scaling on the positive-class logit margin."""
    margin = (logits[:, POSITIVE_CLASS_IDX] - logits[:, 1 - POSITIVE_CLASS_IDX]).reshape(-1, 1)
    y_pos = (labels == POSITIVE_CLASS_IDX).astype(int)
    calibrator = LogisticRegression(C=1e6, solver="lbfgs")
    calibrator.fit(margin, y_pos)
    return calibrator


def platt_apply(calibrator: LogisticRegression, logits: np.ndarray) -> np.ndarray:
    margin = (logits[:, POSITIVE_CLASS_IDX] - logits[:, 1 - POSITIVE_CLASS_IDX]).reshape(-1, 1)
    return calibrator.predict_proba(margin)[:, 1]


def _fbeta(prec: float, rec: float, beta: float) -> float:
    denom = (beta ** 2) * prec + rec
    return ((1 + beta ** 2) * prec * rec / denom) if denom > 0 else 0.0


def find_best_threshold(y_pos: np.ndarray, p_pos: np.ndarray,
                        beta: float = F_BETA,
                        min_precision_floor: float = MIN_PRECISION_FLOOR):
    """
    Pick the threshold that maximizes F_beta (default F2). This is the
    standard medical-screening objective — false negatives cost more than
    false positives.

    A soft floor min_precision_floor is applied so the threshold can't
    collapse to absurd values (e.g., predicting 'Autistic' for everyone).
    """
    best_t, best_fb, best_prec, best_rec = 0.5, -1.0, 0.0, 0.0
    sweep = []
    for t in np.arange(0.05, 0.96, 0.01):
        y_hat = (p_pos >= t).astype(int)
        if y_hat.sum() == 0:
            continue
        prec = precision_score(y_pos, y_hat, zero_division=0)
        rec = recall_score(y_pos, y_hat, zero_division=0)
        f1 = f1_score(y_pos, y_hat, zero_division=0)
        fb = _fbeta(prec, rec, beta)
        sweep.append({
            "threshold": round(float(t), 2),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1": round(float(f1), 4),
            f"f{int(beta)}": round(float(fb), 4),
        })
        if prec >= min_precision_floor and fb > best_fb:
            best_t, best_fb, best_prec, best_rec = float(t), float(fb), float(prec), float(rec)
    return best_t, best_prec, best_rec, best_fb, sweep


@torch.no_grad()
def collect_test_logits(arch: str, ckpt_path: Path, device: str, batch_size: int = 32):
    cand = get_candidates()[arch]
    tfms = cand.transforms_fn()
    ds = datasets.ImageFolder(DATA_ROOT / "test", transform=tfms)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    model = load_model_from_ckpt(arch, ckpt_path, device=device)
    all_logits, all_labels = [], []
    for x, y in loader:
        x = x.to(device)
        all_logits.append(model(x).cpu().numpy())
        all_labels.extend(y.cpu().tolist())
    return np.concatenate(all_logits, axis=0), np.array(all_labels)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cv_info_path = MODELS_DIR / "cv_winner_info.joblib"
    best_info_path = MODELS_DIR / "best_cnn_info.joblib"
    if not cv_info_path.exists():
        raise FileNotFoundError(f"{cv_info_path} missing. Run cross_validate_cnn.py.")
    if not best_info_path.exists():
        raise FileNotFoundError(f"{best_info_path} missing. Run train_final_cnn.py.")
    cv_info = joblib.load(cv_info_path)
    best_info = joblib.load(best_info_path)
    arch = cv_info["model_name"]
    oof_path = Path(cv_info["oof_npz_path"])
    ckpt_path = Path(best_info["ckpt_path"])

    print("=" * 70)
    print(f"CALIBRATING (OOF) + THRESHOLD TUNING: {arch}")
    print("=" * 70)
    print(f"OOF source: {oof_path}")
    print(f"Final model: {ckpt_path}")

    # Load OOF logits from CV
    data = np.load(oof_path, allow_pickle=True)
    oof_logits = data["oof_logits"]
    oof_labels = data["oof_labels"]
    y_pos_oof = (oof_labels == POSITIVE_CLASS_IDX).astype(int)

    # Raw probs on OOF
    oof_probs_raw = torch.softmax(torch.from_numpy(oof_logits), dim=1).numpy()
    oof_p_pos_raw = oof_probs_raw[:, POSITIVE_CLASS_IDX]
    raw_logloss = log_loss(y_pos_oof, np.clip(oof_p_pos_raw, 1e-7, 1 - 1e-7))
    raw_auc = roc_auc_score(y_pos_oof, oof_p_pos_raw)

    # Fit Platt on OOF
    calibrator = platt_fit(oof_logits, oof_labels)
    oof_p_pos_cal = platt_apply(calibrator, oof_logits)
    cal_logloss = log_loss(y_pos_oof, np.clip(oof_p_pos_cal, 1e-7, 1 - 1e-7))
    cal_auc = roc_auc_score(y_pos_oof, oof_p_pos_cal)

    print(f"\nOOF calibration:")
    print(f"  log-loss raw={raw_logloss:.4f}  calibrated={cal_logloss:.4f}")
    print(f"  ROC-AUC  raw={raw_auc:.4f}   calibrated={cal_auc:.4f}")

    # Threshold on OOF (maximize F2)
    best_t, best_prec, best_rec, best_fb, sweep = find_best_threshold(
        y_pos_oof, oof_p_pos_cal, beta=F_BETA, min_precision_floor=MIN_PRECISION_FLOOR
    )
    print(f"\nOOF threshold tuning (maximize F{int(F_BETA)}, precision floor={MIN_PRECISION_FLOOR}):")
    print(f"  threshold={best_t:.2f}  precision={best_prec:.4f}  recall={best_rec:.4f}  F{int(F_BETA)}={best_fb:.4f}")

    # Evaluate on TEST set with final model
    test_logits, test_labels = collect_test_logits(arch, ckpt_path, device)
    test_probs_raw = torch.softmax(torch.from_numpy(test_logits), dim=1).numpy()
    test_p_pos_raw = test_probs_raw[:, POSITIVE_CLASS_IDX]
    test_p_pos_cal = platt_apply(calibrator, test_logits)
    y_pos_test = (test_labels == POSITIVE_CLASS_IDX).astype(int)

    test_default = (test_p_pos_raw >= 0.5).astype(int)
    test_tuned = (test_p_pos_cal >= best_t).astype(int)

    def _metrics(y, yhat):
        return {
            "precision": round(float(precision_score(y, yhat, zero_division=0)), 4),
            "recall": round(float(recall_score(y, yhat, zero_division=0)), 4),
            "f1": round(float(f1_score(y, yhat, zero_division=0)), 4),
            "accuracy": round(float((y == yhat).mean()), 4),
            "confusion_matrix": confusion_matrix(y, yhat, labels=[1, 0]).tolist(),
        }

    default_m = _metrics(y_pos_test, test_default)
    tuned_m = _metrics(y_pos_test, test_tuned)
    test_auc_raw = roc_auc_score(y_pos_test, test_p_pos_raw)
    test_auc_cal = roc_auc_score(y_pos_test, test_p_pos_cal)

    print("\nTEST (default argmax on raw probs):")
    print(classification_report(y_pos_test, test_default,
          target_names=["Non_Autistic", "Autistic"], digits=4, zero_division=0))
    print(f"TEST (calibrated + threshold={best_t:.2f}):")
    print(classification_report(y_pos_test, test_tuned,
          target_names=["Non_Autistic", "Autistic"], digits=4, zero_division=0))

    # Save artifacts
    joblib.dump(calibrator, MODELS_DIR / "cnn_calibrator.joblib")
    threshold_config = {
        "model_name": arch,
        "ckpt_path": str(ckpt_path),
        "positive_class_idx": POSITIVE_CLASS_IDX,
        "positive_class_label": CLASSES[POSITIVE_CLASS_IDX],
        "decision_threshold": best_t,
        "tuning_objective": f"F{int(F_BETA)}-score (medical screening standard)",
        "min_precision_floor": MIN_PRECISION_FLOOR,
        "calibration_source": "5-fold OOF predictions",
        "oof_precision_at_threshold": round(best_prec, 4),
        "oof_recall_at_threshold": round(best_rec, 4),
        "oof_fbeta_at_threshold": round(best_fb, 4),
        "oof_log_loss_raw": round(float(raw_logloss), 4),
        "oof_log_loss_calibrated": round(float(cal_logloss), 4),
        "oof_roc_auc": round(float(raw_auc), 4),
    }
    joblib.dump(threshold_config, MODELS_DIR / "cnn_threshold_config.joblib")

    # Markdown
    md = [f"# CNN Calibration & Threshold Tuning (OOF-based)\n"]
    md.append(f"- **Model:** `{arch}`")
    md.append(f"- **Final checkpoint:** `{ckpt_path}` (trained on full train+valid pool)")
    md.append(f"- **Calibration source:** 5-fold out-of-fold predictions (`{oof_path.name}`)")
    md.append(f"- **Calibration method:** Platt scaling (sigmoid on logit margin)")
    md.append(f"- **Tuning objective:** Maximize F{int(F_BETA)}-score (medical screening standard; weights recall {int(F_BETA)}x over precision)")
    md.append(f"- **Precision floor:** {MIN_PRECISION_FLOOR} (prevents absurdly low thresholds)\n")

    md.append("## OOF Calibration\n")
    md.append("| Metric | Raw | Calibrated |")
    md.append("|---|---|---|")
    md.append(f"| Log-loss | {raw_logloss:.4f} | {cal_logloss:.4f} |")
    md.append(f"| ROC-AUC | {raw_auc:.4f} | {cal_auc:.4f} |\n")

    md.append("## Decision Threshold\n")
    md.append(f"- **Threshold on P(Autistic):** {best_t:.2f}")
    md.append(f"- **OOF precision:** {best_prec:.4f}")
    md.append(f"- **OOF recall:** {best_rec:.4f}")
    md.append(f"- **OOF F{int(F_BETA)}:** {best_fb:.4f}\n")

    md.append("## Test-Set Comparison\n")
    md.append("| Strategy | Accuracy | Precision(A) | Recall(A) | F1(A) | ROC-AUC |")
    md.append("|---|---|---|---|---|---|")
    md.append(f"| Default (argmax) | {default_m['accuracy']:.4f} | "
              f"{default_m['precision']:.4f} | {default_m['recall']:.4f} | "
              f"{default_m['f1']:.4f} | {test_auc_raw:.4f} |")
    md.append(f"| Calibrated + threshold={best_t:.2f} | {tuned_m['accuracy']:.4f} | "
              f"{tuned_m['precision']:.4f} | {tuned_m['recall']:.4f} | "
              f"{tuned_m['f1']:.4f} | {test_auc_cal:.4f} |\n")

    md.append("## Threshold Sweep (on OOF)\n")
    md.append("| Threshold | Precision | Recall | F1 |")
    md.append("|---|---|---|---|")
    for s in sweep:
        if int(round(s["threshold"] * 100)) % 5 == 0:
            md.append(f"| {s['threshold']:.2f} | {s['precision']:.3f} | {s['recall']:.3f} | {s['f1']:.3f} |")

    (REPORTS_DIR / "cnn_calibration_report.md").write_text("\n".join(md))
    (REPORTS_DIR / "cnn_calibration_report.json").write_text(json.dumps({
        "threshold_config": threshold_config,
        "oof_calibration": {
            "raw_log_loss": round(float(raw_logloss), 4),
            "calibrated_log_loss": round(float(cal_logloss), 4),
            "raw_auc": round(float(raw_auc), 4),
            "calibrated_auc": round(float(cal_auc), 4),
        },
        "test_default": {**default_m, "roc_auc": round(float(test_auc_raw), 4)},
        "test_tuned": {**tuned_m, "roc_auc": round(float(test_auc_cal), 4)},
        "threshold_sweep": sweep,
    }, indent=2))

    print(f"\n[saved] {MODELS_DIR / 'cnn_calibrator.joblib'}")
    print(f"[saved] {MODELS_DIR / 'cnn_threshold_config.joblib'}")
    print(f"[saved] {REPORTS_DIR / 'cnn_calibration_report.md'}")


if __name__ == "__main__":
    main()
