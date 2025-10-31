# Implementation Details
**Project:** Hybrid Deep Learning Model for Intrusion Detection in IoT/IIoT Networks  
**Author:** Dr. Vanlalruata Hnamte  
**Framework:** PyTorch 2.9 (CUDA 12.9 Enabled)

---

## 1. System Overview

The implementation integrates five major research datasets:
- **EDGE_IIoT Dataset**
- **CIC-IoT-2023 Dataset**
- **Aposemat-IoT-23 Dataset**
- **CIC-IoMT-2024 Dataset**
- **CIC-IIoT-2025 Dataset**

Each dataset contains flow-based or packet-level network traffic labeled as *normal* or *attack*.  
All datasets are preprocessed, standardized, and stored under a common layout:

```aiignore
data/
├── raw/
│ ├── EDGE_IIoT/.csv
│ ├── CICIoT2023/.csv
│ ├── AposeIoT23/.csv
│ ├── CICIoMT2024/.csv
│ ├── CICIoT2025/.csv
│ └── BoT-IoT/.csv
└── processed/
└── <dataset>_<type>/
├── X_train.npy
├── y_train.npy
├── X_test.npy
├── y_test.npy
├── scaler.joblib
└── meta.json
```


---

## 2. Experimental Framework

The project is modularized under `src/` and orchestrated by `main.py`.

### Directory Structure
```aiignore
src/
├── utils.py # General-purpose utilities, seeding, timers
├── datasets.py # Data loading, encoding, normalization, splitting
├── viz.py # Visualization (histograms, correlation heatmaps)
├── models.py # CNN+LSTM Fusion and MLP architectures
├── train.py # Training loops with early stopping
├── evaluate.py # Metrics, confusion matrix, ROC-AUC, PR curves
├── inference.py # Model loading and live prediction
├── xai.py # SHAP, LIME, ANOVA interpretability
└── ablation.py # User-selected feature subset retraining
```


All modules are GPU-compatible and leverage efficient tensor operations.  
Models are saved as `.pt` files containing both weights and metadata.

---

## 3. Model Implementation

- **Architecture:** Parallel CNN–LSTM Fusion  
- **Activation Function:** ReLU  
- **Optimization:** AdamW (weight decay = 1e-4)  
- **Loss Function:** Cross-Entropy  
- **Batch Size:** 256 (default)  
- **Learning Rate:** 0.001 (default)  
- **Early Stopping:** Patience = 8 epochs  
- **Number of Epochs:** 50–100 depending on convergence  

Training outputs include:
- Model checkpoints (`.pt`)
- Epoch-wise metrics (`history.csv`)
- Learning curves (`learning_curve.eps`)
- Complexity and resource report (`complexity.json`)

---

## 4. Classification & Regression Metrics

### **Classification**
- Accuracy  
- Precision  
- Recall  
- F1-Score  
- ROC-AUC  
- Confusion Matrix  
- Precision–Recall Curve  

### **Regression (if used)**
- Mean Squared Error (MSE)  
- Root Mean Squared Error (RMSE)  
- Mean Absolute Error (MAE)

Metrics are logged as both JSON and CSV in each model’s `outputs/<dataset>_binary|multiclass/eval/` folder.

---

## 5. Explainable AI Integration

The framework implements:
- **SHAP (SHapley Additive Explanations):** Global and local feature importance.
- **LIME (Local Interpretable Model-Agnostic Explanations):** Instance-level explanations.
- **ANOVA F-score Analysis:** Feature variance-based discriminative ranking.

Output files:
```aiignore
outputs/<dataset>/xai/
├── shap_summary.eps
├── lime_example.eps
├── anova_top20.eps
├── shap_feature_importance.csv
├── lime_feature_importance.csv
├── anova_feature_importance.csv
└── xai_combined_features.csv
```


---

## 6. Ablation Study

Ablation can be performed via **user-selected feature subsets**.  
The CLI prompts for a list of features to retain, retrains the model, and compares performance degradation or improvement.

This allows analyzing:
- Impact of top-k ranked features.
- Model robustness against feature removal.
- Energy/time trade-offs after feature reduction.

---

## 7. Resource and Energy Efficiency

During training and inference:
- GPU time per epoch and inference latency per batch are measured.
- Energy is estimated via proxy:
  \[
  E_{proxy} = T_{batch} \times P_{GPU}
  \]
  where \( P_{GPU} = 120 \text{ W} \) by default.

The script automatically exports:
```
energy_j_per_batch_proxy
inference_time_ms_per_batch
```


---

## 8. Learning Curves

Learning curves are automatically plotted:
- X-axis: Epochs  
- Y-axis: Training and Validation Loss  

These help detect:
- **Overfitting:** Training loss ↓ while Validation loss ↑  
- **Underfitting:** Both losses high and flat  

Saved as:
```aiignore
outputs/<dataset>/learning_curve.eps
```


---

## 9. Reproducibility and Randomization

All random states are fixed using:
```
set_all_seeds(seed)
```
ensuring reproducibility across runs.

## 10. Deployment and Inference

Models can be reloaded for online deployment:
```aiignore
from src.inference import load_model, predict_live
model, state = load_model("EDGE_IIoT", "binary")
result = predict_live(model, live_data, return_proba=True)
```

Inference results:
* y_pred.npy
* y_proba.npy
* inference_meta.json (timing + energy)

are stored under: ```outputs/<dataset>_<type>/inference/```

## 11. Hardware and Software Requirements
| Component | Specification                                                 |
| --- |---------------------------------------------------------------|
| CPU | Intel Core i7-12600K (8 cores, 16 threads)                    |
| GPU | NVIDIA GeForce RTX 3060                                       |
| RAM | 32 GB                                                         |
| Storage | 1 TB SSD                                                      |
|Framework | 	PyTorch 2.9                                                  |
|CUDA Version | 	12.9                                                         |
|Python | 	≥ 3.10                                                       |
|GPU Memory | 	≥ 8 GB (recommended)                                         |
|Libraries | 	numpy, pandas, matplotlib, seaborn, shap, lime, scikit-learn |

## 12. Summary

This implementation offers a scalable, interpretable, and energy-efficient deep learning pipeline for intrusion detection in IoT and IIoT networks.

It is modular, reproducible, and extensible for:

* New datasets
* Model architecture experiments
* Explainability benchmarking
* Live deployment in real-world environments

