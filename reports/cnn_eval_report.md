# EfficientNet-B0 Autism Face Classifier — Evaluation Report

- **Checkpoint:** `C:\Users\Hassaan\Desktop\aid_updated_v1\aid_updated_v1\models\efficientnet_b0_autism.pth`
- **Split:** `test` (280 samples)
- **Classes:** ['Autistic', 'Non_Autistic']
- **Device:** cuda
- **Eval time:** 6.33s

## Overall Metrics

| Metric | Value |
|---|---|
| Loss (CrossEntropy) | 0.5085 |
| Accuracy | 0.7893 |
| Macro Precision | 0.8005 |
| Macro Recall | 0.7893 |
| Macro F1 | 0.7873 |
| Weighted Precision | 0.8005 |
| Weighted Recall | 0.7893 |
| Weighted F1 | 0.7873 |
| ROC-AUC (Autistic = positive) | 0.8829 |

## Per-Class Metrics

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Autistic | 0.8584 | 0.6929 | 0.7668 | 140 |
| Non_Autistic | 0.7425 | 0.8857 | 0.8078 | 140 |

## Confusion Matrix

Rows = true label, Columns = predicted label. Labels: ['Autistic', 'Non_Autistic']

|               | Pred Autistic | Pred Non_Autistic |
|---|---|---|
| **True Autistic**     | 97 | 43 |
| **True Non_Autistic** | 16 | 124 |

## Clinical Metrics (positive class = Autistic)

| Metric | Value |
|---|---|
| Sensitivity (TPR / Recall) | 0.6929 |
| Specificity (TNR) | 0.8857 |
| Precision (PPV) | 0.8584 |
| NPV | 0.7425 |
| FPR | 0.1143 |
| FNR | 0.3071 |
| ROC-AUC | 0.8829 |

## Sklearn classification_report

```
              precision    recall  f1-score   support

    Autistic     0.8584    0.6929    0.7668       140
Non_Autistic     0.7425    0.8857    0.8078       140

    accuracy                         0.7893       280
   macro avg     0.8005    0.7893    0.7873       280
weighted avg     0.8005    0.7893    0.7873       280

```
