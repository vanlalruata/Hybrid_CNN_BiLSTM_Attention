# Model Architecture
**Model:** Parallel Fusion CNN + LSTM  
**Author:** Dr. Vanlalruata Hnamte  
**Framework:** PyTorch 2.9 (GPU Accelerated)


## 1. Motivation

Intrusion detection in IoT/IIoT systems involves analyzing large-scale network traffic with:
- High dimensionality,
- Temporal dependencies,
- Complex nonlinear patterns.

To effectively capture both **spatial correlations** (feature-level dependencies) and **temporal dynamics** (sequence behavior), a **hybrid parallel CNN–LSTM fusion architecture** is designed.

---

## 2. Architecture Overview

The model consists of **two parallel encoding paths**:
```aiignore
            ┌────────────────────┐
            │ Input Features (X) │
            └─────────┬──────────┘
                      │
     ┌────────────────┴────────────────┐
     │                                 │
┌────▼────┐                      ┌─────▼─────┐
│ CNN │                          │    LSTM   │
│ Branch │                       │    Branch │
└────┬────┘                      └────┬──────┘
     │                                │
     └───────────┬────────────────────┘
                 ▼
        Fusion Layer (Concat)
                 │
               Dense → Dropout → Output (Softmax)

```


---

## 3. Branch Details

### **3.1 CNN Branch (Feature Abstraction)**

- Input: `[batch, features]`
- Layers:
  - Linear → ReLU
  - Reshape → Conv1D (32 filters, kernel=3, padding=1)
  - BatchNorm + AdaptiveMaxPool1D(16)
  - Flatten → Dense (output: hidden_dim)

The CNN learns **localized correlations** among feature subsets (e.g., packet rate, flags, entropy).

---

### **3.2 LSTM Branch (Sequential Context)**

- Input: `[batch, 1, features]`
- Layers:
  - Linear projection
  - Bi-directional LSTM (hidden=64)
  - Linear output projection

The LSTM captures **temporal and contextual dependencies**, modeling latent traffic patterns over time.

---

## 4. Fusion and Classification

- CNN and LSTM outputs are **concatenated**
- Passed through:
  - Dense(128) → ReLU → Dropout(0.25)
  - Final layer: Dense(num_classes)

Output:
\[
\hat{y} = \text{Softmax}(W_f \cdot [h_{cnn} \oplus h_{lstm}] + b_f)
\]

---

## 5. Training Objective

Loss:
\[
\mathcal{L} = -\frac{1}{N} \sum_{i=1}^N y_i \log(\hat{y}_i)
\]
using **CrossEntropyLoss**, optimized with **AdamW**.

---

## 6. Key Design Advantages

| Component | Contribution |
|------------|---------------|
| CNN | Extracts local and nonlinear feature interactions |
| LSTM | Learns sequential correlations between network states |
| Fusion | Combines short-term (CNN) and long-term (LSTM) dependencies |
| Dropout | Regularizes the fusion representation |
| Adaptive Pooling | Reduces spatial variance and computational load |

---

## 7. Model Complexity

- **Parameters:** ~1.2–2.5M (depending on input size)
- **Estimated MACs per sample:** ~4×(I×H + H² + H)
- **Training Epochs:** 50–100 (early stopping applied)
- **Energy Proxy:** Time × Power (P = 120 W default)

Complexity metrics are automatically exported as:
```outputs/<dataset>_<type>/complexity.json```


---

## 8. Comparative Baselines

For evaluation, a **Simple MLP** is included as a lightweight baseline:
- 2 Hidden Layers (128 → 64 → num_classes)
- ReLU activation
- Dropout regularization

This helps validate the **incremental gain** from hybridization.

---

## 9. Interpretability Integration

The model supports **SHAP**, **LIME**, and **ANOVA** explainers:
- CNN filters reveal *feature group importance*.
- LSTM activations expose *temporal influence*.
- SHAP and LIME quantify per-feature contributions to decisions.

---

## 10. Deployment Readiness

Saved as PyTorch checkpoints (`.pt`), containing:
- Model architecture
- Input dimension
- Class metadata
- State dictionary

Reloadable for real-time inference:
```python
from src.inference import load_model, predict_live
model, state = load_model("EDGE_IIoT", "binary")
pred = predict_live(model, live_traffic_features)
```

## 11. Conclusion

The Parallel CNN–LSTM Fusion model achieves:

* High detection accuracy across multiple IoT/IIoT datasets
* Robustness to feature variation
* Interpretability through SHAP/LIME/ANOVA
* Efficiency via GPU optimization and adaptive pooling

It forms a strong foundation for **edge-based intelligent intrusion detection systems** capable of real-time operation and post-hoc explainability.
