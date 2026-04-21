"""
Retrain the CV-winning CNN on the FULL train+valid pool (test kept held out).

Uses the same two-stage fine-tuning + augmentation recipe as cross_validate_cnn.py
so the final model is trained identically to the CV folds.

Saves:
  - models/cnn/best_cnn.pth
  - models/cnn/best_cnn_info.joblib
"""

from __future__ import annotations

import time
from pathlib import Path

import torch
import joblib
from torch import nn
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets

from cnn_pipeline import (
    CLASSES, DATA_ROOT, MODELS_DIR,
    get_candidates, freeze_backbone, unfreeze_backbone,
    get_param_groups, set_bn_eval, get_train_transforms,
)

WARMUP_EPOCHS = 2
FINETUNE_EPOCHS = 6
BATCH_SIZE = 16
HEAD_LR = 3e-4
BACKBONE_LR = 1e-5


def _train_epoch(model, loader, criterion, optimizer, device, label):
    model.train()
    set_bn_eval(model)
    total_loss, total_correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * x.size(0)
        total_correct += int((logits.argmax(dim=1) == y).sum().item())
        n += int(x.size(0))
    print(f"  {label}  loss={total_loss/n:.4f}  acc={total_correct/n:.4f}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    info_path = MODELS_DIR / "cv_winner_info.joblib"
    if not info_path.exists():
        raise FileNotFoundError(f"{info_path} missing. Run cross_validate_cnn.py first.")
    cv_info = joblib.load(info_path)
    name = cv_info["model_name"]
    print(f"Retraining CV winner: {name} on full (train+valid) pool")
    print(f"CV score: ROC-AUC {cv_info['mean_roc_auc']:.4f} +/- {cv_info['std_roc_auc']:.4f}")

    cand = get_candidates()[name]
    train_tfms = get_train_transforms()
    train_ds = datasets.ImageFolder(DATA_ROOT / "train", transform=train_tfms)
    valid_ds = datasets.ImageFolder(DATA_ROOT / "valid", transform=train_tfms)
    pool = ConcatDataset([train_ds, valid_ds])
    loader = DataLoader(pool, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    print(f"Pool size: {len(pool)}")

    model = cand.builder(num_classes=len(CLASSES)).to(device)
    criterion = nn.CrossEntropyLoss()

    t0 = time.time()
    # Stage 1: frozen backbone, head warm-up
    freeze_backbone(model, name)
    opt1 = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=HEAD_LR
    )
    for epoch in range(1, WARMUP_EPOCHS + 1):
        _train_epoch(model, loader, criterion, opt1, device, f"warmup {epoch:02d}")

    # Stage 2: unfreeze, fine-tune
    unfreeze_backbone(model, name)
    opt2 = torch.optim.AdamW(
        get_param_groups(model, name, head_lr=HEAD_LR, backbone_lr=BACKBONE_LR)
    )
    for epoch in range(1, FINETUNE_EPOCHS + 1):
        _train_epoch(model, loader, criterion, opt2, device, f"finetune {epoch:02d}")

    elapsed = round(time.time() - t0, 1)
    out_path = MODELS_DIR / "best_cnn.pth"
    ckpt = {
        "arch": name,
        "classes": CLASSES,
        "class_to_idx": train_ds.class_to_idx,
        "model_state": model.state_dict(),
    }
    torch.save(ckpt, out_path)

    best_info = {
        "model_name": name,
        "ckpt_path": str(out_path),
        "selection_criterion": "5-fold CV mean ROC-AUC (fine-tuned with augmentation)",
        "cv_mean_roc_auc": cv_info["mean_roc_auc"],
        "cv_std_roc_auc": cv_info["std_roc_auc"],
        "cv_mean_accuracy": cv_info["mean_accuracy"],
        "trained_on": "train + valid (pool)",
        "pool_size": len(pool),
        "warmup_epochs": WARMUP_EPOCHS,
        "finetune_epochs": FINETUNE_EPOCHS,
        "head_lr": HEAD_LR,
        "backbone_lr": BACKBONE_LR,
        "augmentation": "RandomResizedCrop+HFlip+ColorJitter",
        "elapsed_seconds": elapsed,
    }
    joblib.dump(best_info, MODELS_DIR / "best_cnn_info.joblib")
    print(f"\n[saved] {out_path}  ({elapsed}s)")
    print(f"[saved] {MODELS_DIR / 'best_cnn_info.joblib'}")


if __name__ == "__main__":
    main()
