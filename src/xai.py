"""
xai.py — Explainable AI (SHAP, LIME, ANOVA) for IDS Models
Author: Dr. Vanlalruata Hnamte
Version: 1.0
"""

import os
import json
import numpy as np
import pandas as pd
import shap
import torch
import lime.lime_tabular
from sklearn.feature_selection import f_classif
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .utils import ensure_dir


def _safe_save_plot(path):
    ensure_dir(os.path.dirname(path))
    plt.tight_layout()
    plt.savefig(path, format="eps", dpi=200)
    plt.close()


def run_xai(
    X_train, X_test, y_train, y_test,
    checkpoint: str, feature_names, class_names,
    methods=["shap", "lime", "anova"], outdir="outputs/xai"
):
    """Run SHAP, LIME, and/or ANOVA feature importance on a trained model checkpoint."""
    ensure_dir(outdir)
    state = torch.load(checkpoint, map_location="cpu")
    input_dim = int(state["input_dim"])
    num_classes = int(state["num_classes"])

    # Reload the real trained model
    from .models import CNN_LSTM_Fusion, SimpleMLP
    model_name = (state.get("model_name") or "").lower()
    if model_name in ["cnn_lstm_fusion", "fusion"]:
        model = CNN_LSTM_Fusion(input_dim, num_classes)
    else:
        model = SimpleMLP(input_dim, num_classes)
    model.load_state_dict(state["state_dict"])
    model.eval()

    # Prediction wrapper returning probabilities for LIME
    def _predict_proba_np(m):
        def fn(x_np):
            with torch.no_grad():
                xb = torch.tensor(np.array(x_np, dtype=np.float32))
                logits = m(xb)
                return torch.softmax(logits, dim=1).cpu().numpy()
        return fn

    # -------------------------
    # 1️⃣ SHAP
    # -------------------------
    shap_scores = None
    if "shap" in methods:
        print("[XAI] Running SHAP...")
        explainer = shap.Explainer(model, np.array(X_train[:500], dtype=np.float32))
        shap_values = explainer(np.array(X_test[:500], dtype=np.float32))
        shap.summary_plot(shap_values, features=X_test[:500], feature_names=feature_names, show=False)
        _safe_save_plot(os.path.join(outdir, "shap_summary.eps"))

        shap_mean = np.abs(shap_values.values).mean(axis=0)
        shap_scores = pd.DataFrame({"feature": feature_names, "importance": shap_mean}).sort_values("importance", ascending=False)
        shap_scores.to_csv(os.path.join(outdir, "shap_feature_importance.csv"), index=False)

    # -------------------------
    # 2️⃣ LIME
    # -------------------------
    lime_scores = None
    if "lime" in methods:
        print("[XAI] Running LIME...")
        lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            X_train, feature_names=feature_names, class_names=class_names, mode="classification"
        )
        predict_fn = _predict_proba_np(model)
        lime_exp = lime_explainer.explain_instance(X_test[0], predict_fn, num_features=min(10, len(feature_names)))
        fig = lime_exp.as_pyplot_figure()
        _safe_save_plot(os.path.join(outdir, "lime_example.eps"))
        lime_scores = pd.DataFrame(lime_exp.as_list(), columns=["feature", "weight"])
        lime_scores.to_csv(os.path.join(outdir, "lime_feature_importance.csv"), index=False)

    # -------------------------
    # 3️⃣ ANOVA (F-score)
    # -------------------------
    anova_scores = None
    if "anova" in methods:
        print("[XAI] Running ANOVA F-score...")
        f_vals, _ = f_classif(X_train, y_train)
        anova_scores = pd.DataFrame({"feature": feature_names, "f_score": f_vals}).sort_values("f_score", ascending=False)
        anova_scores.to_csv(os.path.join(outdir, "anova_feature_importance.csv"), index=False)

        plt.figure(figsize=(8, 5))
        plt.barh(anova_scores["feature"].head(20), anova_scores["f_score"].head(20))
        plt.title("Top 20 Features by ANOVA F-score")
        plt.gca().invert_yaxis()
        _safe_save_plot(os.path.join(outdir, "anova_top20.eps"))

    # -------------------------
    # Consolidate all rankings
    # -------------------------
    combined = []
    if shap_scores is not None:
        shap_scores["method"] = "shap"
        combined.append(shap_scores.rename(columns={"importance": "score"}))
    if lime_scores is not None:
        lime_scores["method"] = "lime"
        lime_scores.rename(columns={"weight": "score"}, inplace=True)
        combined.append(lime_scores)
    if anova_scores is not None:
        anova_scores["method"] = "anova"
        anova_scores.rename(columns={"f_score": "score"}, inplace=True)
        combined.append(anova_scores)

    if combined:
        top = pd.concat(combined, ignore_index=True)
        top.to_csv(os.path.join(outdir, "xai_combined_features.csv"), index=False)
        print(f"[XAI] Results saved under {outdir}")
