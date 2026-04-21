"""
Step 2 of CNN pipeline — mirrors src/model_training.py's select_best_model_by_cv().

Loads the three trained candidate checkpoints, evaluates each on the validation
split, and picks the best by ROC-AUC (same criterion as the ML pipeline).

Saves:
  - models/cnn/best_cnn.pth         (copy of winning architecture's checkpoint)
  - models/cnn/best_cnn_info.joblib (name, metrics, ckpt path)
  - reports/cnn_model_comparison.json
  - reports/cnn_model_comparison.md
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import torch
import joblib
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from sklearn.metrics import (
    roc_auc_score, accuracy_score,
    precision_recall_fscore_support, confusion_matrix,
)

from cnn_pipeline import (
    CLASSES, POSITIVE_CLASS_IDX, DATA_ROOT, MODELS_DIR, REPORTS_DIR,
    get_candidates, load_model_from_ckpt,
)


@torch.no_grad()
def evaluate_on_valid(name: str, ckpt_path: Path, device: str, batch_size: int = 32):
    cand = get_candidates()[name]
    tfms = cand.transforms_fn()
    ds = datasets.ImageFolder(DATA_ROOT / "valid", transform=tfms)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = load_model_from_ckpt(name, ckpt_path, device=device)
    criterion = nn.CrossEntropyLoss()

    y_true, y_pred, y_prob_pos = [], [], []
    total_loss, n = 0.0, 0
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
        y_prob_pos.extend(probs[:, POSITIVE_CLASS_IDX].cpu().tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob_pos = np.array(y_prob_pos)

    y_binary_pos = (y_true == POSITIVE_CLASS_IDX).astype(int)
    auc = roc_auc_score(y_binary_pos, y_prob_pos)
    acc = accuracy_score(y_true, y_pred)
    avg_loss = total_loss / max(n, 1)

    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    return {
        "model": name,
        "ckpt": str(ckpt_path),
        "valid_samples": int(n),
        "loss": round(avg_loss, 4),
        "accuracy": round(float(acc), 4),
        "roc_auc": round(float(auc), 4),
        "per_class": {
            CLASSES[i]: {
                "precision": round(float(prec[i]), 4),
                "recall": round(float(rec[i]), 4),
                "f1": round(float(f1[i]), 4),
            } for i in range(len(CLASSES))
        },
        "confusion_matrix": cm.tolist(),
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    results = []
    for name, cand in get_candidates().items():
        ckpt_path = MODELS_DIR / cand.ckpt_name
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Checkpoint missing: {ckpt_path}. Run train_cnn_candidates.py first."
            )
        r = evaluate_on_valid(name, ckpt_path, device=device)
        results.append(r)
        print(
            f"  {name:<25s}: ROC-AUC={r['roc_auc']:.4f}  "
            f"Acc={r['accuracy']:.4f}  Loss={r['loss']:.4f}"
        )

    # Select best by ROC-AUC (matches ML pipeline criterion)
    best = max(results, key=lambda r: r["roc_auc"])
    print(f"\n  >> Best CNN by validation ROC-AUC: {best['model']}  ({best['roc_auc']:.4f})")

    # Copy winning checkpoint to best_cnn.pth + save info
    best_ckpt_src = Path(best["ckpt"])
    best_ckpt_dst = MODELS_DIR / "best_cnn.pth"
    shutil.copy2(best_ckpt_src, best_ckpt_dst)

    best_info = {
        "model_name": best["model"],
        "ckpt_path": str(best_ckpt_dst),
        "valid_metrics": best,
        "selection_criterion": "roc_auc (binary, Autistic = positive class)",
    }
    joblib.dump(best_info, MODELS_DIR / "best_cnn_info.joblib")

    # Save comparison reports
    comparison_path = REPORTS_DIR / "cnn_model_comparison.json"
    comparison_path.write_text(json.dumps({
        "criterion": "ROC-AUC on validation split (Autistic = positive class)",
        "candidates": results,
        "best": best["model"],
    }, indent=2))

    md = ["# CNN Model Comparison — Validation Split\n"]
    md.append(f"- **Split:** valid ({results[0]['valid_samples']} samples)")
    md.append(f"- **Selection criterion:** ROC-AUC (binary, Autistic = positive class)")
    md.append(f"- **Winner:** `{best['model']}` (ROC-AUC = {best['roc_auc']:.4f})\n")
    md.append("| Model | ROC-AUC | Accuracy | Loss | Autistic P/R/F1 | Non_Autistic P/R/F1 |")
    md.append("|---|---|---|---|---|---|")
    for r in results:
        a = r["per_class"]["Autistic"]
        na = r["per_class"]["Non_Autistic"]
        md.append(
            f"| {r['model']} | {r['roc_auc']:.4f} | {r['accuracy']:.4f} | {r['loss']:.4f} | "
            f"{a['precision']:.3f}/{a['recall']:.3f}/{a['f1']:.3f} | "
            f"{na['precision']:.3f}/{na['recall']:.3f}/{na['f1']:.3f} |"
        )
    (REPORTS_DIR / "cnn_model_comparison.md").write_text("\n".join(md))

    print(f"\n[ok] saved best checkpoint to {best_ckpt_dst}")
    print(f"[ok] saved best info to {MODELS_DIR / 'best_cnn_info.joblib'}")
    print(f"[ok] saved comparison to {comparison_path}")


if __name__ == "__main__":
    main()
