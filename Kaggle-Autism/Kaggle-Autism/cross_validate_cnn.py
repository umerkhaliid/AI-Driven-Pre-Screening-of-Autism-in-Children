"""
5-Fold Stratified Cross-Validation for CNN candidates.

Mirrors src/cross_validate_models.py + the CV selection logic in
src/model_training.py::select_best_model_by_cv().

Pools `train/` + `valid/` into one dataset (test/ stays held out), then for
each candidate:
  - StratifiedKFold(n_splits=5) on labels
  - Train each fold from scratch with the shared recipe
  - Collect held-out (OOF) logits + metrics per fold
  - Report mean +/- std across folds

Picks the winning architecture by mean CV ROC-AUC (same criterion as ML pipeline).

Also saves out-of-fold (OOF) logits for every candidate — used by
calibrate_and_tune_cnn.py to fit Platt scaling + tune threshold WITHOUT
touching the test set.

Outputs:
  - reports/cnn_cv_results.json, cnn_cv_report.md
  - models/cnn/cv_oof_<arch>.npz            (oof_logits, oof_labels, order_paths)
  - models/cnn/cv_winner_info.joblib        (winning arch + mean/std AUC)
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import joblib
from torch import nn
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torchvision import datasets
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, accuracy_score,
    precision_score, recall_score, f1_score, log_loss,
)

from cnn_pipeline import (
    CLASSES, POSITIVE_CLASS_IDX, DATA_ROOT, MODELS_DIR, REPORTS_DIR,
    get_candidates, freeze_backbone, unfreeze_backbone,
    get_param_groups, set_bn_eval, get_train_transforms,
)


N_SPLITS = 5
WARMUP_EPOCHS = 2          # stage 1: frozen backbone, train head
FINETUNE_EPOCHS = 6        # stage 2: unfrozen backbone, fine-tune whole model
BATCH_SIZE = 16
HEAD_LR = 3e-4
BACKBONE_LR = 1e-5
RANDOM_STATE = 42


def build_label_index(eval_tfms):
    """
    Labels/paths are identical across candidates (ImageFolder is deterministic),
    so we build them once using any transform.
    """
    train_ds = datasets.ImageFolder(DATA_ROOT / "train", transform=eval_tfms)
    valid_ds = datasets.ImageFolder(DATA_ROOT / "valid", transform=eval_tfms)
    samples = list(train_ds.samples) + list(valid_ds.samples)
    paths = [s[0] for s in samples]
    labels = np.array([s[1] for s in samples], dtype=int)
    classes = train_ds.classes
    return labels, paths, classes


def _train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    set_bn_eval(model)  # keep BN stable during fine-tuning
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()


def train_fold(candidate, train_loader, val_loader, device,
               warmup_epochs=WARMUP_EPOCHS, finetune_epochs=FINETUNE_EPOCHS,
               head_lr=HEAD_LR, backbone_lr=BACKBONE_LR):
    """
    Two-stage transfer learning:
      Stage 1: freeze backbone, train head (warmup) with head_lr
      Stage 2: unfreeze all, fine-tune with discriminative LRs (backbone_lr << head_lr)
    """
    model = candidate.builder(num_classes=len(CLASSES)).to(device)
    criterion = nn.CrossEntropyLoss()

    # Stage 1 — head warm-up
    freeze_backbone(model, candidate.name)
    opt1 = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=head_lr
    )
    for _ in range(warmup_epochs):
        _train_epoch(model, train_loader, criterion, opt1, device)

    # Stage 2 — full fine-tune with discriminative LRs
    unfreeze_backbone(model, candidate.name)
    opt2 = torch.optim.AdamW(
        get_param_groups(model, candidate.name, head_lr=head_lr, backbone_lr=backbone_lr)
    )
    for _ in range(finetune_epochs):
        _train_epoch(model, train_loader, criterion, opt2, device)

    # Collect held-out logits
    model.eval()
    fold_logits, fold_labels = [], []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            logits = model(x)
            fold_logits.append(logits.cpu().numpy())
            fold_labels.extend(y.cpu().tolist())
    return np.concatenate(fold_logits, axis=0), np.array(fold_labels)


def fold_metrics(y_true, logits):
    probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    p_pos = probs[:, POSITIVE_CLASS_IDX]
    y_pos = (y_true == POSITIVE_CLASS_IDX).astype(int)
    y_pred = probs.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_pos, p_pos)),
        "log_loss": float(log_loss(y_pos, np.clip(p_pos, 1e-7, 1 - 1e-7))),
        "precision_autistic": float(precision_score(y_pos, y_pred == POSITIVE_CLASS_IDX, zero_division=0)),
        "recall_autistic": float(recall_score(y_pos, y_pred == POSITIVE_CLASS_IDX, zero_division=0)),
        "f1_autistic": float(f1_score(y_pos, y_pred == POSITIVE_CLASS_IDX, zero_division=0)),
    }


def cross_validate_one(name, candidate, labels, paths, device):
    print("\n" + "=" * 70)
    print(f"5-FOLD CV (fine-tuning + augmentation): {name}")
    print("=" * 70)

    train_tfms = get_train_transforms()                 # augmentation for training folds
    eval_tfms = candidate.transforms_fn()               # standard resize+centercrop for held-out eval

    # Two pools with identical ordering but different transforms
    train_pool = ConcatDataset([
        datasets.ImageFolder(DATA_ROOT / "train", transform=train_tfms),
        datasets.ImageFolder(DATA_ROOT / "valid", transform=train_tfms),
    ])
    eval_pool = ConcatDataset([
        datasets.ImageFolder(DATA_ROOT / "train", transform=eval_tfms),
        datasets.ImageFolder(DATA_ROOT / "valid", transform=eval_tfms),
    ])
    assert len(train_pool) == len(eval_pool) == len(labels)

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    oof_logits = np.zeros((len(train_pool), len(CLASSES)), dtype=np.float32)
    oof_filled = np.zeros(len(train_pool), dtype=bool)
    per_fold = []

    t_total = time.time()
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels), start=1):
        t0 = time.time()
        train_loader = DataLoader(Subset(train_pool, train_idx), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        val_loader = DataLoader(Subset(eval_pool, val_idx), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        logits, y_val = train_fold(candidate, train_loader, val_loader, device)
        oof_logits[val_idx] = logits
        oof_filled[val_idx] = True

        m = fold_metrics(y_val, logits)
        m["fold"] = fold_idx
        m["elapsed_seconds"] = round(time.time() - t0, 1)
        per_fold.append(m)
        print(
            f"  fold {fold_idx}/{N_SPLITS}: "
            f"auc={m['roc_auc']:.4f}  acc={m['accuracy']:.4f}  "
            f"recall(autistic)={m['recall_autistic']:.3f}  ({m['elapsed_seconds']}s)"
        )

    assert oof_filled.all(), "OOF coverage incomplete"
    t_total = round(time.time() - t_total, 1)

    # Aggregate
    auc_mean = float(np.mean([f["roc_auc"] for f in per_fold]))
    auc_std = float(np.std([f["roc_auc"] for f in per_fold]))
    acc_mean = float(np.mean([f["accuracy"] for f in per_fold]))
    acc_std = float(np.std([f["accuracy"] for f in per_fold]))
    logloss_mean = float(np.mean([f["log_loss"] for f in per_fold]))
    recall_mean = float(np.mean([f["recall_autistic"] for f in per_fold]))
    f1_mean = float(np.mean([f["f1_autistic"] for f in per_fold]))

    # Save OOF
    oof_path = MODELS_DIR / f"cv_oof_{candidate.ckpt_name.replace('.pth', '')}.npz"
    np.savez(oof_path, oof_logits=oof_logits, oof_labels=labels, paths=np.array(paths))
    print(f"  [saved] {oof_path}")

    summary = {
        "model": name,
        "oof_npz": str(oof_path),
        "cv_total_elapsed_seconds": t_total,
        "mean_roc_auc": round(auc_mean, 4),
        "std_roc_auc": round(auc_std, 4),
        "mean_accuracy": round(acc_mean, 4),
        "std_accuracy": round(acc_std, 4),
        "mean_log_loss": round(logloss_mean, 4),
        "mean_recall_autistic": round(recall_mean, 4),
        "mean_f1_autistic": round(f1_mean, 4),
        "per_fold": per_fold,
    }
    print(f"  ==> {name}: ROC-AUC {auc_mean:.4f} +/- {auc_std:.4f}  "
          f"(acc {acc_mean:.4f} +/- {acc_std:.4f}, total {t_total}s)")
    return summary


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Data: {DATA_ROOT}")
    print(f"Pool = train + valid (test kept held out)")

    # Build flat labels list once (ImageFolder is deterministic, identical order across candidates)
    any_cand = next(iter(get_candidates().values()))
    labels, paths, classes = build_label_index(any_cand.transforms_fn())
    print(f"Pool size: {len(labels)}   classes: {classes}")
    print(f"Class counts: {np.bincount(labels).tolist()}\n")

    all_summaries = []
    for name, cand in get_candidates().items():
        summary = cross_validate_one(name, cand, labels, paths, device)
        all_summaries.append(summary)

    # Pick winner by mean CV ROC-AUC
    winner = max(all_summaries, key=lambda s: s["mean_roc_auc"])

    # Save winner info
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    cv_winner_info = {
        "model_name": winner["model"],
        "mean_roc_auc": winner["mean_roc_auc"],
        "std_roc_auc": winner["std_roc_auc"],
        "mean_accuracy": winner["mean_accuracy"],
        "oof_npz_path": winner["oof_npz"],
        "n_splits": N_SPLITS,
        "random_state": RANDOM_STATE,
        "warmup_epochs": WARMUP_EPOCHS,
        "finetune_epochs": FINETUNE_EPOCHS,
        "batch_size": BATCH_SIZE,
        "head_lr": HEAD_LR,
        "backbone_lr": BACKBONE_LR,
        "augmentation": "RandomResizedCrop+HFlip+ColorJitter",
    }
    joblib.dump(cv_winner_info, MODELS_DIR / "cv_winner_info.joblib")
    print(f"\n[saved] {MODELS_DIR / 'cv_winner_info.joblib'}")

    # Reports
    (REPORTS_DIR / "cnn_cv_results.json").write_text(json.dumps({
        "criterion": "Mean ROC-AUC over 5 stratified folds (Autistic = positive)",
        "winner": winner["model"],
        "candidates": all_summaries,
        "config": {
            "n_splits": N_SPLITS,
            "warmup_epochs": WARMUP_EPOCHS,
            "finetune_epochs": FINETUNE_EPOCHS,
            "batch_size": BATCH_SIZE,
            "head_lr": HEAD_LR,
            "backbone_lr": BACKBONE_LR,
            "random_state": RANDOM_STATE,
            "augmentation": "RandomResizedCrop+HFlip+ColorJitter",
        },
    }, indent=2))

    md = ["# CNN 5-Fold Stratified Cross-Validation Results\n"]
    md.append(f"- **Pool:** train + valid ({len(labels)} images)  |  Test held out ({len([1 for _ in (DATA_ROOT / 'test').rglob('*.jpg')])} images)")
    md.append(f"- **Splits:** {N_SPLITS} (StratifiedKFold, random_state={RANDOM_STATE})")
    md.append(f"- **Recipe:** 2-stage fine-tuning ({WARMUP_EPOCHS} head-warmup + {FINETUNE_EPOCHS} fine-tune), AdamW, batch {BATCH_SIZE}, head_lr={HEAD_LR}, backbone_lr={BACKBONE_LR}")
    md.append(f"- **Augmentation:** RandomResizedCrop(0.85-1.0) + HFlip + ColorJitter")
    md.append(f"- **Selection criterion:** mean ROC-AUC (binary, Autistic = positive)")
    md.append(f"- **Winner:** `{winner['model']}` (ROC-AUC = {winner['mean_roc_auc']:.4f} +/- {winner['std_roc_auc']:.4f})\n")

    md.append("## Summary (mean +/- std across folds)\n")
    md.append("| Model | ROC-AUC | Accuracy | Log-Loss | Recall(Autistic) | F1(Autistic) | Time (s) |")
    md.append("|---|---|---|---|---|---|---|")
    for s in all_summaries:
        md.append(
            f"| {s['model']} | {s['mean_roc_auc']:.4f} ± {s['std_roc_auc']:.4f} | "
            f"{s['mean_accuracy']:.4f} ± {s['std_accuracy']:.4f} | "
            f"{s['mean_log_loss']:.4f} | {s['mean_recall_autistic']:.4f} | "
            f"{s['mean_f1_autistic']:.4f} | {s['cv_total_elapsed_seconds']} |"
        )
    md.append("")

    md.append("## Per-Fold Detail\n")
    for s in all_summaries:
        md.append(f"### {s['model']}\n")
        md.append("| Fold | ROC-AUC | Accuracy | Log-Loss | Recall(A) | F1(A) | Time (s) |")
        md.append("|---|---|---|---|---|---|---|")
        for f in s["per_fold"]:
            md.append(
                f"| {f['fold']} | {f['roc_auc']:.4f} | {f['accuracy']:.4f} | "
                f"{f['log_loss']:.4f} | {f['recall_autistic']:.4f} | "
                f"{f['f1_autistic']:.4f} | {f['elapsed_seconds']} |"
            )
        md.append("")

    (REPORTS_DIR / "cnn_cv_report.md").write_text("\n".join(md))
    print(f"[saved] {REPORTS_DIR / 'cnn_cv_report.md'}")

    print("\n" + "=" * 70)
    print("CROSS-VALIDATION COMPLETE")
    print("=" * 70)
    for s in all_summaries:
        marker = "  <-- WINNER" if s["model"] == winner["model"] else ""
        print(f"  {s['model']:<25s}: ROC-AUC {s['mean_roc_auc']:.4f} +/- {s['std_roc_auc']:.4f}{marker}")
    print(f"\nNext: python train_final_cnn.py  (retrains {winner['model']} on full pool)")


if __name__ == "__main__":
    main()
