# CNN 5-Fold Stratified Cross-Validation Results

- **Pool:** train + valid (2734 images)  |  Test held out (280 images)
- **Splits:** 5 (StratifiedKFold, random_state=42)
- **Recipe:** 2-stage fine-tuning (2 head-warmup + 6 fine-tune), AdamW, batch 16, head_lr=0.0003, backbone_lr=1e-05
- **Augmentation:** RandomResizedCrop(0.85-1.0) + HFlip + ColorJitter
- **Selection criterion:** mean ROC-AUC (binary, Autistic = positive)
- **Winner:** `ResNet-50` (ROC-AUC = 0.8967 +/- 0.0046)

## Summary (mean +/- std across folds)

| Model | ROC-AUC | Accuracy | Log-Loss | Recall(Autistic) | F1(Autistic) | Time (s) |
|---|---|---|---|---|---|---|
| EfficientNet-B0 | 0.8879 ± 0.0030 | 0.7996 ± 0.0194 | 0.4523 | 0.7528 | 0.7865 | 718.5 |
| ResNet-50 | 0.8967 ± 0.0046 | 0.8050 ± 0.0128 | 0.4756 | 0.8142 | 0.8064 | 1166.8 |
| MobileNetV3-Large | 0.8811 ± 0.0139 | 0.7776 ± 0.0167 | 0.5492 | 0.7389 | 0.7660 | 644.5 |

## Per-Fold Detail

### EfficientNet-B0

| Fold | ROC-AUC | Accuracy | Log-Loss | Recall(A) | F1(A) | Time (s) |
|---|---|---|---|---|---|---|
| 1 | 0.8894 | 0.8117 | 0.4481 | 0.8759 | 0.8233 | 144.3 |
| 2 | 0.8904 | 0.7623 | 0.5062 | 0.5766 | 0.7085 | 143.5 |
| 3 | 0.8826 | 0.8062 | 0.4407 | 0.7692 | 0.7985 | 143.9 |
| 4 | 0.8906 | 0.8007 | 0.4227 | 0.7729 | 0.7947 | 143.5 |
| 5 | 0.8866 | 0.8168 | 0.4436 | 0.7692 | 0.8077 | 143.4 |

### ResNet-50

| Fold | ROC-AUC | Accuracy | Log-Loss | Recall(A) | F1(A) | Time (s) |
|---|---|---|---|---|---|---|
| 1 | 0.8903 | 0.7934 | 0.5003 | 0.8139 | 0.7979 | 232.4 |
| 2 | 0.8963 | 0.7934 | 0.4487 | 0.7847 | 0.7919 | 232.3 |
| 3 | 0.9047 | 0.8282 | 0.4718 | 0.8864 | 0.8374 | 233.1 |
| 4 | 0.8973 | 0.8080 | 0.4955 | 0.7582 | 0.7977 | 234.5 |
| 5 | 0.8950 | 0.8022 | 0.4615 | 0.8278 | 0.8071 | 234.4 |

### MobileNetV3-Large

| Fold | ROC-AUC | Accuracy | Log-Loss | Recall(A) | F1(A) | Time (s) |
|---|---|---|---|---|---|---|
| 1 | 0.8582 | 0.7660 | 0.5488 | 0.7226 | 0.7557 | 128.7 |
| 2 | 0.8758 | 0.7569 | 0.5772 | 0.6241 | 0.7200 | 121.9 |
| 3 | 0.8968 | 0.8062 | 0.4609 | 0.7473 | 0.7938 | 135.2 |
| 4 | 0.8938 | 0.7788 | 0.6106 | 0.9304 | 0.8076 | 135.5 |
| 5 | 0.8806 | 0.7802 | 0.5484 | 0.6703 | 0.7531 | 123.3 |
