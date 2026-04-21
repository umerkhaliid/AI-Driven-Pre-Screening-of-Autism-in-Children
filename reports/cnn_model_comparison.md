# CNN Model Comparison — Validation Split

- **Split:** valid (80 samples)
- **Selection criterion:** ROC-AUC (binary, Autistic = positive class)
- **Winner:** `ResNet-50` (ROC-AUC = 0.9800)

| Model | ROC-AUC | Accuracy | Loss | Autistic P/R/F1 | Non_Autistic P/R/F1 |
|---|---|---|---|---|---|
| EfficientNet-B0 | 0.9700 | 0.9375 | 0.3674 | 0.973/0.900/0.935 | 0.907/0.975/0.940 |
| ResNet-50 | 0.9800 | 0.9500 | 0.3475 | 0.974/0.925/0.949 | 0.929/0.975/0.951 |
| MobileNetV3-Large | 0.9719 | 0.9250 | 0.2659 | 0.947/0.900/0.923 | 0.905/0.950/0.927 |