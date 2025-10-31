"""
xai.py — Explainable AI for IDS models (SHAP, LIME, ANOVA)
Author: Dr. Vanlalruata Hnamte
Version: 1.0
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.feature_selection import f_classif
from sklearn.preprocessing import StandardScaler
from lime.lime_tabular import LimeTabularExplainer

from .models import CNN_LSTM_Fusion, SimpleMLP


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------------------------------------------------------
# Helper: rebuild model from checkpoint
# -------------------------------------------------------------------------

def _load_model(ckpt_path: str):
    """
    Loads model for explanation.
    Returns (model, meta dict)
    """
    state = torch.load(ckpt_path, map_location=DEVICE)
    name = (state.get("model_name") or "").lower()
    input_dim = int(state.get("input_dim"))
    num_classes = int(state.get("num_classes", 2))
    hidden_dim = int(state.get("hidden_dim", 64))

    if name in ["cnn_lstm_fusion", "cnn+lstm", "fusion"]:
        model = CNN_LSTM_Fusion(input_dim, num_classes, hidden_dim=hidden_dim)
    elif name in ["mlp", "simplemlp"]:
        model = SimpleMLP(input_dim, num_classes)
    else:
        raise ValueError(f"Unknown model type in checkpoint: {name}")

    model.load_state_dict(state["state_dict"])
    model.to(DEVICE).eval()
    meta = {
        "model_name": state.get("model_name"),
        "input_dim": input_dim,
        "num_classes": num_classes,
        "class_names": state.get("class_names"),
        "task_type": state.get("task_type")
    }
    return model, meta


# -------------------------------------------------------------------------
# ANOVA feature importance
# -------------------------------------------------------------------------

def compute_anova_importance(X: np.ndarray, y: np.ndarray, feature_names: list, outdir: str):
    """
    Performs ANOVA F-test for feature significance.
    Saves ranked table & bar plot.
    """
    os.makedirs(outdir, exist_ok=True)
    fvals, pvals = f_classif(X, y)
    df = pd.DataFrame({
        "feature": feature_names,
        "f_value": fvals,
        "p_value": pvals
    }).sort_values(by="f_value", ascending=False)

    df.to_csv(os.path.join(outdir, "anova_feature_importance.csv"), index=False)

    plt.figure(figsize=(10, 6))
    top_n = min(20, len(df))
    plt.barh(df["feature"].iloc[:top_n][::-1], df["f_value"].iloc[:top_n][::-1])
    plt.xlabel("F-value (ANOVA)")
    plt.ylabel("Feature")
    plt.title("Top Feature Importance (ANOVA F-test)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "anova_importance_top20.eps"), format="eps", dpi=200)
    plt.close()

    return df


# -------------------------------------------------------------------------
# SHAP explanations
# -------------------------------------------------------------------------

def compute_shap_explanations(
    X: np.ndarray,
    feature_names: list,
    ckpt_path: str,
    outdir: str,
    sample_size: int = 500
):
    """
    Computes SHAP values (KernelExplainer for tabular data).
    Produces global summary and per-class beeswarm/heatmaps.
    """
    os.makedirs(outdir, exist_ok=True)

    model, meta = _load_model(ckpt_path)
    num_classes = meta["num_classes"]
    class_names = meta.get("class_names") or [f"class_{i}" for i in range(num_classes)]

    # Reduce sample for tractable SHAP computation
    idx = np.random.choice(np.arange(len(X)), size=min(sample_size, len(X)), replace=False)
    X_sample = X[idx]

    # Define prediction function for SHAP
    def predict_fn(x):
        with torch.no_grad():
            x_t = torch.tensor(x, dtype=torch.float32).to(DEVICE)
            logits = model(x_t)
            probs = torch.softmax(logits, dim=1)
            return probs.cpu().numpy()

    # Initialize SHAP kernel explainer (model-agnostic)
    explainer = shap.KernelExplainer(predict_fn, shap.sample(X_sample, 100))
    shap_values = explainer.shap_values(X_sample, nsamples=200)

    # Save SHAP arrays
    np.save(os.path.join(outdir, "shap_values.npy"), shap_values)
    np.save(os.path.join(outdir, "shap_sample.npy"), X_sample)

    # Global summary plot
    plt.figure()
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "shap_summary_all.eps"), format="eps", dpi=200)
    plt.close()

    # Per-class beeswarm
    for c in range(num_classes):
        plt.figure()
        shap.summary_plot(shap_values[c], X_sample, feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"shap_class_{class_names[c]}.eps"), format="eps", dpi=200)
        plt.close()

    # Mean absolute SHAP value ranking
    imp = np.abs(shap_values).mean(axis=1).mean(axis=0) if isinstance(shap_values, list) else np.abs(shap_values).mean(axis=0)
    shap_df = pd.DataFrame({"feature": feature_names, "mean_abs_shap": imp}).sort_values("mean_abs_shap", ascending=False)
    shap_df.to_csv(os.path.join(outdir, "shap_feature_importance.csv"), index=False)

    return shap_df


# -------------------------------------------------------------------------
# LIME explanations
# -------------------------------------------------------------------------

def compute_lime_explanations(
    X: np.ndarray,
    feature_names: list,
    ckpt_path: str,
    outdir: str,
    sample_size: int = 10
):
    """
    Generate LIME local explanations for a few representative samples.
    Saves per-sample EPS visualizations.
    """
    os.makedirs(outdir, exist_ok=True)

    model, meta = _load_model(ckpt_path)
    num_classes = meta["num_classes"]
    class_names = meta.get("class_names") or [f"class_{i}" for i in range(num_classes)]

    # Subset
    idx = np.random.choice(np.arange(len(X)), size=min(sample_size, len(X)), replace=False)
    X_sample = X[idx]

    # Scaler for interpretability normalization
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    # Define predict function
    def predict_fn(x):
        with torch.no_grad():
            x_t = torch.tensor(x, dtype=torch.float32).to(DEVICE)
            logits = model(x_t)
            probs = torch.softmax(logits, dim=1)
            return probs.cpu().numpy()

    # Initialize LIME explainer
    explainer = LimeTabularExplainer(
        X_scaled,
        mode="classification",
        feature_names=feature_names,
        class_names=class_names,
        discretize_continuous=True
    )

    explanations = []
    for i, row in enumerate(X_sample):
        exp = explainer.explain_instance(scaler.transform([row])[0], predict_fn, num_features=10)
        fig = exp.as_pyplot_figure()
        plt.title(f"LIME explanation — Sample {i}")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"lime_sample_{i}.eps"), format="eps", dpi=200)
        plt.close()
        explanations.append({
            "sample_index": int(i),
            "top_features": exp.as_list()
        })

    with open(os.path.join(outdir, "lime_explanations.json"), "w") as f:
        json.dump(explanations, f, indent=2)

    return explanations


# -------------------------------------------------------------------------
# Unified entrypoint
# -------------------------------------------------------------------------

def explain_model(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list,
    ckpt_path: str,
    outdir: str,
    do_shap: bool = True,
    do_lime: bool = True,
    do_anova: bool = True
) -> dict:
    """
    Runs SHAP, LIME, and ANOVA analyses together.
    Each produces artifacts saved under outdir/xai/.
    """
    outdir = os.path.join(outdir, "xai")
    os.makedirs(outdir, exist_ok=True)

    results = {}
    if do_anova:
        print("[XAI] Computing ANOVA feature importance…")
        results["anova"] = compute_anova_importance(X, y, feature_names, os.path.join(outdir, "anova"))
    if do_shap:
        print("[XAI] Computing SHAP explanations (global + per-class)…")
        results["shap"] = compute_shap_explanations(X, feature_names, ckpt_path, os.path.join(outdir, "shap"))
    if do_lime:
        print("[XAI] Computing LIME local explanations…")
        results["lime"] = compute_lime_explanations(X, feature_names, ckpt_path, os.path.join(outdir, "lime"))

    print(f"[XAI] Explanations stored in: {outdir}")
    return results
