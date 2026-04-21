"""
PyTorch EfficientNet-B0 training script for Kaggle Autism dataset.

Folder structure expected (ImageFolder):
  <root>/train/Autistic/*.jpg
  <root>/train/Non_Autistic/*.jpg
  <root>/valid/Autistic/*.jpg
  <root>/valid/Non_Autistic/*.jpg
  <root>/test/Autistic/*.jpg
  <root>/test/Non_Autistic/*.jpg

Produces a checkpoint (.pth) that the main Streamlit app can load for Step 1 gating.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


def build_model(num_classes: int = 2) -> nn.Module:
    weights = EfficientNet_B0_Weights.DEFAULT
    model = efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, required=True, help="D:\FYP\autism-prescreening-tool (3)\autism-prescreening-tool\Kaggle-Autism\Kaggle-Autism\data")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--out", type=str, default="models/efficientnet_b0_autism.pth")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    weights = EfficientNet_B0_Weights.DEFAULT
    train_tfms = weights.transforms()
    eval_tfms = weights.transforms()

    root = Path(args.data_root)
    train_ds = datasets.ImageFolder(root / "train", transform=train_tfms)
    valid_ds = datasets.ImageFolder(root / "valid", transform=eval_tfms)
    test_ds = datasets.ImageFolder(root / "test", transform=eval_tfms)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = build_model(num_classes=len(train_ds.classes)).to(device)

    # Fine-tuning: freeze backbone, train classifier head first
    for p in model.features.parameters():
        p.requires_grad = False
    for p in model.classifier.parameters():
        p.requires_grad = True

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    best_val_acc = -1.0
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    start = time.time()
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer=optimizer, device=device)
        va_loss, va_acc = run_epoch(model, valid_loader, criterion, optimizer=None, device=device)
        print(f"Epoch {epoch:02d} | train loss {tr_loss:.4f} acc {tr_acc:.3f} | val loss {va_loss:.4f} acc {va_acc:.3f}")

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            ckpt = {
                "arch": "efficientnet_b0",
                "weights": "EfficientNet_B0_Weights.DEFAULT",
                "classes": train_ds.classes,
                "class_to_idx": train_ds.class_to_idx,
                "model_state": model.state_dict(),
            }
            torch.save(ckpt, out_path)
            print("Saved checkpoint:", out_path)

    te_loss, te_acc = run_epoch(model, test_loader, criterion, optimizer=None, device=device)
    print(f"Test loss {te_loss:.4f} acc {te_acc:.3f}")
    print("Done in", round(time.time() - start, 1), "s")


if __name__ == "__main__":
    main()
