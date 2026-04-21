"""
Step 1 of CNN pipeline — mirrors src/model_training.py.

Trains 3 candidate CNN architectures (EfficientNet-B0, ResNet-50,
MobileNetV3-Large) with the SAME recipe (frozen backbone + classifier head,
AdamW, CrossEntropy), then saves each best-validation checkpoint.

Selection by ROC-AUC on validation happens in select_best_cnn.py.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets

from cnn_pipeline import (
    CLASSES, DATA_ROOT, MODELS_DIR, REPORTS_DIR,
    get_candidates, freeze_backbone,
)


def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return float((preds == y).float().mean().item())


def run_epoch(model, loader, criterion, optimizer=None, device="cpu"):
    if optimizer is None:
        model.eval()
        torch.set_grad_enabled(False)
    else:
        model.train()
        torch.set_grad_enabled(True)

    total_loss = 0.0
    total_acc = 0.0
    n = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = criterion(logits, y)

        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        bs = x.size(0)
        total_loss += float(loss.item()) * bs
        total_acc += accuracy(logits, y) * bs
        n += bs

    return total_loss / max(n, 1), total_acc / max(n, 1)


def train_one(cand, epochs: int, batch_size: int, lr: float, num_workers: int, device: str):
    print("\n" + "=" * 70)
    print(f"TRAINING: {cand.name}")
    print("=" * 70)

    tfms = cand.transforms_fn()
    train_ds = datasets.ImageFolder(DATA_ROOT / "train", transform=tfms)
    valid_ds = datasets.ImageFolder(DATA_ROOT / "valid", transform=tfms)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = cand.builder(num_classes=len(CLASSES)).to(device)
    freeze_backbone(model, cand.name)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )

    best_val_acc = -1.0
    best_val_loss = float("inf")
    out_path = MODELS_DIR / cand.ckpt_name
    history = []
    start = time.time()

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc = run_epoch(model, valid_loader, criterion, None, device)
        history.append({
            "epoch": epoch,
            "train_loss": round(tr_loss, 4),
            "train_acc": round(tr_acc, 4),
            "val_loss": round(va_loss, 4),
            "val_acc": round(va_acc, 4),
        })
        print(f"  Epoch {epoch:02d} | train loss {tr_loss:.4f} acc {tr_acc:.3f} | val loss {va_loss:.4f} acc {va_acc:.3f}")

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_val_loss = va_loss
            ckpt = {
                "arch": cand.name,
                "classes": CLASSES,
                "class_to_idx": train_ds.class_to_idx,
                "model_state": model.state_dict(),
            }
            torch.save(ckpt, out_path)

    elapsed = round(time.time() - start, 1)
    print(f"  [saved] {out_path}  (best val acc {best_val_acc:.3f} in {elapsed}s)")
    return {
        "model": cand.name,
        "ckpt": str(out_path),
        "best_val_acc": round(best_val_acc, 4),
        "best_val_loss": round(best_val_loss, 4),
        "epochs": epochs,
        "elapsed_seconds": elapsed,
        "history": history,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Data: {DATA_ROOT}")
    print(f"Models dir: {MODELS_DIR}")

    candidates = get_candidates()
    all_results = []
    for cand in candidates.values():
        result = train_one(
            cand,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            num_workers=args.num_workers,
            device=device,
        )
        all_results.append(result)

    # Save training summary
    summary_path = REPORTS_DIR / "cnn_training_summary.json"
    summary_path.write_text(json.dumps({
        "hyperparams": {"epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr},
        "device": device,
        "results": all_results,
    }, indent=2))
    print(f"\n[ok] saved training summary to {summary_path}")

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE — all candidates saved to", MODELS_DIR)
    print("=" * 70)
    for r in all_results:
        print(f"  {r['model']:<25s}: best_val_acc={r['best_val_acc']:.4f}  ({r['elapsed_seconds']}s)")


if __name__ == "__main__":
    main()
