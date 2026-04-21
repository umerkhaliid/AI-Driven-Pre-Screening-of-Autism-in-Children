"""
Run inference on a folder of images using a trained EfficientNet-B0 checkpoint.

Folder should be in ImageFolder format (one subfolder per class) OR a single folder
that contains images; for the latter, the script will treat it as one pseudo-class.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

# # example of loading an image with the Keras API
# from keras.preprocessing.image import load_img
# # load the image
# img = load_img('C:/Users/Mikian/Desktop/processimgs/jacobtest.PNG')
# # report details about the image
# print(type(img))
# print(img.format)
# print(img.mode)
# print(img.size)
# # show the image
# img.show()

def build_model(num_classes: int = 2) -> nn.Module:
    weights = EfficientNet_B0_Weights.DEFAULT
    model = efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-root", type=str, required=True, help="Folder with images or ImageFolder structure")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to .pth checkpoint")
    parser.add_argument("--batch-size", type=int, default=8)
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

    root = Path(args.img_root)
    if any((root / c).is_dir() for c in classes):
        ds = datasets.ImageFolder(root, transform=tfms)
    else:
        # Wrap a single folder into ImageFolder shape by using one subfolder
        ds = datasets.ImageFolder(root.parent, transform=tfms)
        raise ValueError("Provide an ImageFolder-style directory with class subfolders.")

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    with torch.no_grad():
        for i, (x, _) in enumerate(loader):
            x = x.to(device)
            probs = torch.softmax(model(x), dim=1).cpu()
            for j in range(probs.size(0)):
                p = probs[j].tolist()
                pred = int(torch.tensor(p).argmax().item())
                print({"index": i * args.batch_size + j, "pred": classes[pred], "probs": dict(zip(classes, p))})


if __name__ == "__main__":
    main()
