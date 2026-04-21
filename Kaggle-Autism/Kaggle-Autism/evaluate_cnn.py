"""
Full evaluation of the trained EfficientNet-B0 autism face classifier.

Computes accuracy, loss, per-class precision/recall/F1, confusion matrix,
sensitivity, specificity, ROC-AUC, and saves:
  - reports/cnn_eval_metrics.json
  - reports/cnn_eval_report.md
  - reports/cnn_confusion_matrix.png
  - reports/cnn_roc_curve.png
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(__file__).resolve().parent / "data"
CKPT_PATH = PROJECT_ROOT / "models" / "efficientnet_b0_autism.pth"
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def build_model(num_classes: int = 2) -> nn.Module:
    weights = EfficientNet_B0_Weights.DEFAULT
    model = efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def evaluate(split: str = "test", batch_size: int = 32, num_workers: int = 0):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[info] device={device}")
    print(f"[info] checkpoint={CKPT_PATH}")
    print(f"[info] data={DATA_ROOT / split}")

    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    classes = ckpt.get("classes", ["Autistic", "Non_Autistic"])
    print(f"[info] classes={classes}")

    model = build_model(num_classes=len(classes))
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.to(device).eval()

    weights = EfficientNet_B0_Weights.DEFAULT
    tfms = weights.transforms()

    ds = datasets.ImageFolder(DATA_ROOT / split, transform=tfms)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    criterion = nn.CrossEntropyLoss()

    y_true, y_pred, y_prob = [], [], []
    total_loss, n = 0.0, 0
    start = time.time()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            total_loss += float(loss.item()) * x.size(0)
            n += int(x.size(0))
            y_true.extend(y.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())
            y_prob.extend(probs.cpu().numpy().tolist())

    elapsed = time.time() - start
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    # Overall metrics
    acc = accuracy_score(y_true, y_pred)
    avg_loss = total_loss / max(n, 1)

    # Per-class
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(classes))), zero_division=0
    )

    # Weighted + macro
    prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    # Confusion matrix — treat "Autistic" (index 0) as POSITIVE class for clinical framing
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    TP = int(cm[0, 0])
    FN = int(cm[0, 1])
    FP = int(cm[1, 0])
    TN = int(cm[1, 1])

    sensitivity = TP / max(TP + FN, 1)   # Recall for Autistic
    specificity = TN / max(TN + FP, 1)   # Recall for Non_Autistic
    ppv = TP / max(TP + FP, 1)           # Precision
    npv = TN / max(TN + FN, 1)
    fpr_val = FP / max(FP + TN, 1)
    fnr_val = FN / max(FN + TP, 1)

    # ROC-AUC (positive class prob = Autistic index 0)
    try:
        auc = roc_auc_score(y_true == 0, y_prob[:, 0])
        fpr_curve, tpr_curve, _ = roc_curve(y_true == 0, y_prob[:, 0])
    except Exception as e:
        auc = float("nan")
        fpr_curve = tpr_curve = np.array([])
        print(f"[warn] roc_auc failed: {e}")

    # Pretty classification report (string)
    report_str = classification_report(
        y_true, y_pred, target_names=classes, digits=4, zero_division=0
    )

    # ---- Save JSON ----
    metrics = {
        "split": split,
        "num_samples": int(n),
        "classes": classes,
        "elapsed_seconds": round(elapsed, 2),
        "loss": round(avg_loss, 6),
        "accuracy": round(acc, 6),
        "per_class": {
            classes[i]: {
                "precision": round(float(prec[i]), 6),
                "recall": round(float(rec[i]), 6),
                "f1": round(float(f1[i]), 6),
                "support": int(support[i]),
            }
            for i in range(len(classes))
        },
        "macro_avg": {
            "precision": round(float(prec_m), 6),
            "recall": round(float(rec_m), 6),
            "f1": round(float(f1_m), 6),
        },
        "weighted_avg": {
            "precision": round(float(prec_w), 6),
            "recall": round(float(rec_w), 6),
            "f1": round(float(f1_w), 6),
        },
        "confusion_matrix": {
            "labels": classes,
            "matrix": cm.tolist(),
            "TP_autistic": TP,
            "FN_autistic": FN,
            "FP_autistic": FP,
            "TN_autistic": TN,
        },
        "clinical_metrics_positive_class_autistic": {
            "sensitivity_recall": round(sensitivity, 6),
            "specificity": round(specificity, 6),
            "precision_ppv": round(ppv, 6),
            "npv": round(npv, 6),
            "fpr": round(fpr_val, 6),
            "fnr": round(fnr_val, 6),
            "roc_auc": round(float(auc), 6) if not np.isnan(auc) else None,
        },
    }

    json_path = REPORTS_DIR / "cnn_eval_metrics.json"
    json_path.write_text(json.dumps(metrics, indent=2))
    print(f"[ok] saved {json_path}")

    # ---- Save Markdown ----
    md = []
    md.append(f"# EfficientNet-B0 Autism Face Classifier — Evaluation Report\n")
    md.append(f"- **Checkpoint:** `{CKPT_PATH}`")
    md.append(f"- **Split:** `{split}` ({n} samples)")
    md.append(f"- **Classes:** {classes}")
    md.append(f"- **Device:** {device}")
    md.append(f"- **Eval time:** {elapsed:.2f}s\n")

    md.append("## Overall Metrics\n")
    md.append("| Metric | Value |")
    md.append("|---|---|")
    md.append(f"| Loss (CrossEntropy) | {avg_loss:.4f} |")
    md.append(f"| Accuracy | {acc:.4f} |")
    md.append(f"| Macro Precision | {prec_m:.4f} |")
    md.append(f"| Macro Recall | {rec_m:.4f} |")
    md.append(f"| Macro F1 | {f1_m:.4f} |")
    md.append(f"| Weighted Precision | {prec_w:.4f} |")
    md.append(f"| Weighted Recall | {rec_w:.4f} |")
    md.append(f"| Weighted F1 | {f1_w:.4f} |")
    md.append(f"| ROC-AUC (Autistic = positive) | {auc:.4f} |\n")

    md.append("## Per-Class Metrics\n")
    md.append("| Class | Precision | Recall | F1 | Support |")
    md.append("|---|---|---|---|---|")
    for i, c in enumerate(classes):
        md.append(
            f"| {c} | {prec[i]:.4f} | {rec[i]:.4f} | {f1[i]:.4f} | {int(support[i])} |"
        )
    md.append("")

    md.append("## Confusion Matrix\n")
    md.append(f"Rows = true label, Columns = predicted label. Labels: {classes}\n")
    md.append("|               | Pred Autistic | Pred Non_Autistic |")
    md.append("|---|---|---|")
    md.append(f"| **True Autistic**     | {TP} | {FN} |")
    md.append(f"| **True Non_Autistic** | {FP} | {TN} |\n")

    md.append("## Clinical Metrics (positive class = Autistic)\n")
    md.append("| Metric | Value |")
    md.append("|---|---|")
    md.append(f"| Sensitivity (TPR / Recall) | {sensitivity:.4f} |")
    md.append(f"| Specificity (TNR) | {specificity:.4f} |")
    md.append(f"| Precision (PPV) | {ppv:.4f} |")
    md.append(f"| NPV | {npv:.4f} |")
    md.append(f"| FPR | {fpr_val:.4f} |")
    md.append(f"| FNR | {fnr_val:.4f} |")
    md.append(f"| ROC-AUC | {auc:.4f} |\n")

    md.append("## Sklearn classification_report\n")
    md.append("```\n" + report_str + "\n```\n")

    md_path = REPORTS_DIR / "cnn_eval_report.md"
    md_path.write_text("\n".join(md))
    print(f"[ok] saved {md_path}")

    # ---- Save confusion matrix plot ----
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix — EfficientNet-B0")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    cm_path = REPORTS_DIR / "cnn_confusion_matrix.png"
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    print(f"[ok] saved {cm_path}")

    # ---- Save ROC curve ----
    if fpr_curve.size > 0:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(fpr_curve, tpr_curve, label=f"ROC (AUC = {auc:.4f})")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve — EfficientNet-B0 (Autistic = positive)")
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        roc_path = REPORTS_DIR / "cnn_roc_curve.png"
        fig.savefig(roc_path, dpi=150)
        plt.close(fig)
        print(f"[ok] saved {roc_path}")

    # ---- Console summary ----
    print("\n=== EVALUATION SUMMARY ===")
    print(f"Accuracy: {acc:.4f}   Loss: {avg_loss:.4f}   ROC-AUC: {auc:.4f}")
    print(f"Macro P/R/F1: {prec_m:.4f} / {rec_m:.4f} / {f1_m:.4f}")
    print(f"Sensitivity: {sensitivity:.4f}   Specificity: {specificity:.4f}")
    print(f"Confusion matrix:\n{cm}")
    print("\nClassification report:")
    print(report_str)

    return metrics


if __name__ == "__main__":
    evaluate(split="test")
