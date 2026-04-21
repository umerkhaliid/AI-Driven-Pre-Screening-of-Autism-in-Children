# CNN Calibration & Threshold Tuning (OOF-based)

- **Model:** `ResNet-50`
- **Final checkpoint:** `C:\Users\Hassaan\Desktop\aid_updated_v1\aid_updated_v1\models\cnn\best_cnn.pth` (trained on full train+valid pool)
- **Calibration source:** 5-fold out-of-fold predictions (`cv_oof_resnet50.npz`)
- **Calibration method:** Platt scaling (sigmoid on logit margin)
- **Tuning objective:** Maximize F2-score (medical screening standard; weights recall 2x over precision)
- **Precision floor:** 0.5 (prevents absurdly low thresholds)

## OOF Calibration

| Metric | Raw | Calibrated |
|---|---|---|
| Log-loss | 0.4756 | 0.4099 |
| ROC-AUC | 0.8947 | 0.8947 |

## Decision Threshold

- **Threshold on P(Autistic):** 0.16
- **OOF precision:** 0.6563
- **OOF recall:** 0.9737
- **OOF F2:** 0.8878

## Test-Set Comparison

| Strategy | Accuracy | Precision(A) | Recall(A) | F1(A) | ROC-AUC |
|---|---|---|---|---|---|
| Default (argmax) | 0.9107 | 0.9528 | 0.8643 | 0.9064 | 0.9812 |
| Calibrated + threshold=0.16 | 0.8643 | 0.7865 | 1.0000 | 0.8805 | 0.9812 |

## Threshold Sweep (on OOF)

| Threshold | Precision | Recall | F1 |
|---|---|---|---|
| 0.05 | 0.547 | 0.997 | 0.706 |
| 0.10 | 0.602 | 0.988 | 0.749 |
| 0.15 | 0.648 | 0.977 | 0.779 |
| 0.20 | 0.686 | 0.954 | 0.798 |
| 0.25 | 0.709 | 0.927 | 0.803 |
| 0.30 | 0.735 | 0.904 | 0.811 |
| 0.35 | 0.755 | 0.884 | 0.815 |
| 0.40 | 0.773 | 0.861 | 0.815 |
| 0.45 | 0.791 | 0.829 | 0.809 |
| 0.50 | 0.808 | 0.804 | 0.806 |
| 0.55 | 0.825 | 0.770 | 0.796 |
| 0.60 | 0.846 | 0.737 | 0.787 |
| 0.65 | 0.866 | 0.696 | 0.772 |
| 0.70 | 0.881 | 0.652 | 0.750 |
| 0.75 | 0.903 | 0.599 | 0.720 |
| 0.80 | 0.920 | 0.538 | 0.679 |
| 0.85 | 0.940 | 0.466 | 0.623 |
| 0.90 | 0.954 | 0.379 | 0.542 |
| 0.95 | 0.968 | 0.244 | 0.389 |