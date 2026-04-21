"""
CNN candidate architectures for face-based autism classification.

Mirrors the ML pipeline pattern in src/model_training.py (3 models, same recipe,
compared by ROC-AUC, then best is calibrated + threshold-tuned).

Models:
  - EfficientNet-B0     (compound-scaled, ~5.3M params)
  - ResNet-50           (deep residual, ~25M params)
  - MobileNetV3-Large   (inverted residual + SE, ~5.5M params)
"""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn
from torchvision import transforms
from torchvision.models import (
    efficientnet_b0, EfficientNet_B0_Weights,
    resnet50, ResNet50_Weights,
    mobilenet_v3_large, MobileNet_V3_Large_Weights,
)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(img_size: int = 224):
    """
    Face-image training augmentation used during CV folds + final retrain.
    Evaluation still uses the default weights.transforms() (resize + center crop).
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(__file__).resolve().parent / "data"
MODELS_DIR = PROJECT_ROOT / "models" / "cnn"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = ["Autistic", "Non_Autistic"]
POSITIVE_CLASS_IDX = 0  # "Autistic" is the positive class for clinical framing


@dataclass
class CNNCandidate:
    name: str
    builder: Callable[[int], nn.Module]
    transforms_fn: Callable
    ckpt_name: str


def _build_efficientnet_b0(num_classes: int = 2) -> nn.Module:
    weights = EfficientNet_B0_Weights.DEFAULT
    model = efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def _build_resnet50(num_classes: int = 2) -> nn.Module:
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def _build_mobilenet_v3_large(num_classes: int = 2) -> nn.Module:
    weights = MobileNet_V3_Large_Weights.DEFAULT
    model = mobilenet_v3_large(weights=weights)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    return model


def get_candidates() -> dict[str, CNNCandidate]:
    return {
        "EfficientNet-B0": CNNCandidate(
            name="EfficientNet-B0",
            builder=_build_efficientnet_b0,
            transforms_fn=lambda: EfficientNet_B0_Weights.DEFAULT.transforms(),
            ckpt_name="efficientnet_b0.pth",
        ),
        "ResNet-50": CNNCandidate(
            name="ResNet-50",
            builder=_build_resnet50,
            transforms_fn=lambda: ResNet50_Weights.DEFAULT.transforms(),
            ckpt_name="resnet50.pth",
        ),
        "MobileNetV3-Large": CNNCandidate(
            name="MobileNetV3-Large",
            builder=_build_mobilenet_v3_large,
            transforms_fn=lambda: MobileNet_V3_Large_Weights.DEFAULT.transforms(),
            ckpt_name="mobilenet_v3_large.pth",
        ),
    }


def freeze_backbone(model: nn.Module, name: str) -> None:
    """Freeze backbone, keep classifier head trainable (transfer learning recipe)."""
    if name == "EfficientNet-B0":
        for p in model.features.parameters():
            p.requires_grad = False
        for p in model.classifier.parameters():
            p.requires_grad = True
    elif name == "ResNet-50":
        for p in model.parameters():
            p.requires_grad = False
        for p in model.fc.parameters():
            p.requires_grad = True
    elif name == "MobileNetV3-Large":
        for p in model.features.parameters():
            p.requires_grad = False
        for p in model.classifier.parameters():
            p.requires_grad = True
    else:
        raise ValueError(f"Unknown model: {name}")


def unfreeze_backbone(model: nn.Module, name: str) -> None:
    """Unfreeze all params for stage-2 fine-tuning."""
    for p in model.parameters():
        p.requires_grad = True


def get_param_groups(model: nn.Module, name: str, head_lr: float, backbone_lr: float):
    """Discriminative learning rates: small LR for backbone, larger for head."""
    if name == "EfficientNet-B0":
        backbone_params = list(model.features.parameters())
        head_params = list(model.classifier.parameters())
    elif name == "ResNet-50":
        head_params = list(model.fc.parameters())
        head_ids = {id(p) for p in head_params}
        backbone_params = [p for p in model.parameters() if id(p) not in head_ids]
    elif name == "MobileNetV3-Large":
        backbone_params = list(model.features.parameters())
        head_params = list(model.classifier.parameters())
    else:
        raise ValueError(f"Unknown model: {name}")
    return [
        {"params": backbone_params, "lr": backbone_lr},
        {"params": head_params, "lr": head_lr},
    ]


def set_bn_eval(model: nn.Module) -> None:
    """
    Keep BatchNorm layers in eval mode during fine-tuning so their running stats
    don't drift on the small autism dataset. Standard trick for transfer learning.
    """
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.eval()


def load_model_from_ckpt(name: str, ckpt_path: Path, device: str = "cpu") -> nn.Module:
    cand = get_candidates()[name]
    model = cand.builder(num_classes=len(CLASSES))
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.to(device).eval()
    return model
