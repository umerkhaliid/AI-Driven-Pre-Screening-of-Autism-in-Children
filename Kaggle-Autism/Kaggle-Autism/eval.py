"""
Evaluate a trained EfficientNet-B0 checkpoint on a dataset split (ImageFolder).
"""

from __future__ import annotations

import argparse
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["train", "valid", "test"])
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location="cpu")

    classes = ckpt.get("classes", ["Autistic", "Non_Autistic"])
    model = build_model(num_classes=len(classes))
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.to(device).eval()

    weights = EfficientNet_B0_Weights.DEFAULT
    tfms = weights.transforms()

    ds = datasets.ImageFolder(Path(args.data_root) / args.split, transform=tfms)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += float(loss.item()) * x.size(0)
            correct += int((logits.argmax(dim=1) == y).sum().item())
            total += int(x.size(0))

    print({"split": args.split, "loss": total_loss / max(total, 1), "accuracy": correct / max(total, 1)})


if __name__ == "__main__":
    main()
