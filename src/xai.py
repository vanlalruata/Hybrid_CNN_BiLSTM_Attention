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
import warnings
import lime.lime_tabular
from sklearn.feature_selection import f_classif
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .utils import ensure_dir

# Silence NumPy FutureWarning about global RNG seeding from downstream libs (e.g., SHAP)
warnings.filterwarnings("ignore", message="The NumPy global RNG was seeded by calling `np.random.seed`", category=FutureWarning)
# Silence EPS transparency warning (artists with alpha will be rendered opaque in PS/EPS)
warnings.filterwarnings("ignore", message="The PostScript backend does not support transparency", category=UserWarning)


def _safe_save_plot(path):
    ensure_dir(os.path.dirname(path))
    plt.tight_layout()
    plt.savefig(path, format="eps", dpi=200, transparent=False)
    plt.close()

def _coerce_feature_dim_np(X, expected: int):
    """
    Coerce numpy array X to have expected feature dimension.
    - If 1D: pad/truncate along axis 0
    - If 2D+: pad/truncate along last dimension (axis=1 for [N, D])
    """
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 1:
        feat = X.shape[0]
        if feat == expected:
            return X
        if feat < expected:
            pad = np.zeros((expected - feat,), dtype=X.dtype)
            return np.concatenate([X, pad], axis=0)
        return X[:expected]
    elif X.ndim >= 2:
        feat = X.shape[1]
        if feat == expected:
            return X
        if feat < expected:
            pad = np.zeros((X.shape[0], expected - feat), dtype=X.dtype)
            print(f"[WARN] Coercing features from {feat} to expected {expected} by zero-padding.")
            return np.hstack([X, pad])
        print(f"[WARN] Coercing features from {feat} to expected {expected} by truncation.")
        return X[:, :expected]
    return X

def _align_matrix_and_names(X, names, expected: int):
    """
    Align both the data matrix and feature_names to the expected dim.
    Returns (X_aligned, names_aligned).
    """
    X = np.asarray(X, dtype=np.float32)
    names = list(names) if names is not None else [f"f{i}" for i in range(X.shape[1])]
    if X.shape[1] < expected:
        add = expected - X.shape[1]
        X2 = np.hstack([X, np.zeros((X.shape[0], add), dtype=X.dtype)])
        new_names = names + [f"pad_{i}" for i in range(add)]
        print(f"[WARN] Feature mismatch: data has {X.shape[1]}, expected {expected}. Padded {add} zero cols.")
        return X2, new_names
    if X.shape[1] > expected:
        print(f"[WARN] Feature mismatch: data has {X.shape[1]}, expected {expected}. Truncated to {expected}.")
        return X[:, :expected], names[:expected]
    return X, names


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
    model_name_raw = (state.get("model_name") or "").lower()
    model_suffix = "fusion" if "fusion" in model_name_raw else "dnn"

    if model_name_raw in ["cnn_lstm_fusion", "fusion"]:
        model = CNN_LSTM_Fusion(input_dim, num_classes)
    elif model_name_raw in ["mlp", "simplemlp", "dnn"]:
        model = SimpleMLP(input_dim, num_classes)
    else:
        # Fallback
        model = SimpleMLP(input_dim, num_classes)
    model.load_state_dict(state["state_dict"])
    model.eval()

    # Align features to checkpoint's expected input dimension
    expected_dim = int(input_dim)
    X_train, feature_names = _align_matrix_and_names(np.asarray(X_train, dtype=np.float32), list(feature_names), expected_dim)
    X_test, feature_names = _align_matrix_and_names(np.asarray(X_test, dtype=np.float32), feature_names, expected_dim)

    # Prediction wrapper returning probabilities for LIME/SHAP (robust to pandas/numpy inputs)
    def _predict_proba_np(m, expected_dim: int):
        def _to_numpy2d(x):
            if isinstance(x, pd.DataFrame):
                a = x.to_numpy(dtype=np.float32, copy=False)
            elif isinstance(x, pd.Series):
                a = x.to_numpy(dtype=np.float32, copy=False)
            else:
                a = np.array(x, dtype=np.float32)
            if a.ndim == 1:
                a = a.reshape(1, -1)
            return a
        def fn(x_np):
            with torch.no_grad():
                a = _to_numpy2d(x_np)
                a = _coerce_feature_dim_np(a, expected_dim)
                xb = torch.tensor(a, dtype=torch.float32)
                logits = m(xb)
                prob = torch.softmax(logits, dim=1).cpu().numpy()
                return prob
        return fn

    # -------------------------
    # SHAP
    # -------------------------
    shap_scores = None
    if "shap" in methods:
        print("[XAI] Running SHAP...")

        # Wrap model to accept numpy input and return numpy output
        def _predict_logits_np(m, expected_dim: int):
            def fn(x_np):
                with torch.no_grad():
                    x_adj = _coerce_feature_dim_np(np.array(x_np, dtype=np.float32), expected_dim)
                    xb = torch.tensor(x_adj, dtype=torch.float32)
                    logits = m(xb)
                    return logits.cpu().numpy()
            return fn

        # Use probabilities for explainability stability (optional: switch to logits by using _predict_logits_np)
        shap_predict_fn = _predict_proba_np(model, expected_dim)

        # Limit background and eval set for speed
        bg = np.array(X_train[:min(500, len(X_train))], dtype=np.float32)
        te = np.array(X_test[:min(500, len(X_test))], dtype=np.float32)

        # SHAP permutation explainer on black-box predictor
        explainer = shap.Explainer(shap_predict_fn, bg)
        shap_values = explainer(te)

        # Handle multi-output (multiclass) shapes from SHAP
        vals = getattr(shap_values, "values", shap_values)
        if vals.ndim == 3:
            # Select positive class for binary, class 0 otherwise
            cls_ix = 1 if num_classes > 1 else 0
            vals_2d = vals[:, :, cls_ix]
        else:
            vals_2d = vals

        # Align plot inputs and names to SHAP output feature dimension
        n_feat_vals = vals_2d.shape[1]
        # Align feature names
        if len(feature_names) < n_feat_vals:
            missing = n_feat_vals - len(feature_names)
            feature_names_plot = list(feature_names) + [f"pad_{i}" for i in range(missing)]
            print(f"[WARN] SHAP names shorter ({len(feature_names)}) than SHAP features ({n_feat_vals}); padding names.")
        elif len(feature_names) > n_feat_vals:
            feature_names_plot = list(feature_names)[:n_feat_vals]
            print(f"[WARN] SHAP names longer ({len(feature_names)}) than SHAP features ({n_feat_vals}); truncating names.")
        else:
            feature_names_plot = feature_names

        # Align features matrix for summary plot
        if te.shape[1] != n_feat_vals:
            te_plot = te[:, :n_feat_vals] if te.shape[1] > n_feat_vals else np.hstack(
                [te, np.zeros((te.shape[0], n_feat_vals - te.shape[1]), dtype=te.dtype)]
            )
            if te.shape[1] < n_feat_vals:
                print(f"[WARN] SHAP features shorter ({te.shape[1]}) than SHAP values ({n_feat_vals}); padding zeros for plot.")
            else:
                print(f"[WARN] SHAP features longer ({te.shape[1]}) than SHAP values ({n_feat_vals}); truncating for plot.")
        else:
            te_plot = te

        shap.summary_plot(vals_2d, features=te_plot, feature_names=feature_names_plot, show=False)
        _safe_save_plot(os.path.join(outdir, f"shap_summary_{model_suffix}.eps"))

        shap_mean = np.abs(vals_2d).mean(axis=0)
        shap_scores = pd.DataFrame({"feature": feature_names_plot, "importance": shap_mean}).sort_values("importance", ascending=False)
        shap_scores.to_csv(os.path.join(outdir, f"shap_feature_importance_{model_suffix}.csv"), index=False)

    # -------------------------
    # LIME
    # -------------------------
    lime_scores = None
    if "lime" in methods:
        print("[XAI] Running LIME...")
        lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            np.asarray(X_train, dtype=np.float32),
            feature_names=feature_names, class_names=class_names, mode="classification"
        )
        predict_fn = _predict_proba_np(model, expected_dim)
        instance = _coerce_feature_dim_np(np.asarray(X_test[0], dtype=np.float32), expected_dim)
        lime_exp = lime_explainer.explain_instance(instance, predict_fn, num_features=min(10, len(feature_names)))
        fig = lime_exp.as_pyplot_figure()
        _safe_save_plot(os.path.join(outdir, f"lime_example_{model_suffix}.eps"))
        lime_scores = pd.DataFrame(lime_exp.as_list(), columns=["feature", "weight"])
        lime_scores.to_csv(os.path.join(outdir, f"lime_feature_importance_{model_suffix}.csv"), index=False)

    # -------------------------
    # ANOVA (F-score)
    # -------------------------
    anova_scores = None
    if "anova" in methods:
        print("[XAI] Running ANOVA F-score...")
        f_vals, _ = f_classif(X_train, y_train)
        # Guard against NaNs/Infs produced by constant features
        f_vals = np.nan_to_num(f_vals, nan=0.0, posinf=0.0, neginf=0.0)

        # Align feature names to length of f_vals to avoid length mismatch
        n_feat = int(len(f_vals))
        if len(feature_names) < n_feat:
            missing = n_feat - len(feature_names)
            feature_names_anova = list(feature_names) + [f"pad_{i}" for i in range(missing)]
            print(f"[WARN] ANOVA names shorter ({len(feature_names)}) than features ({n_feat}); padding names.")
        elif len(feature_names) > n_feat:
            feature_names_anova = list(feature_names)[:n_feat]
            print(f"[WARN] ANOVA names longer ({len(feature_names)}) than features ({n_feat}); truncating names.")
        else:
            feature_names_anova = feature_names

        anova_scores = pd.DataFrame({"feature": feature_names_anova, "f_score": f_vals}).sort_values("f_score", ascending=False)
        anova_scores.to_csv(os.path.join(outdir, f"anova_feature_importance_{model_suffix}.csv"), index=False)

        plt.figure(figsize=(8, 5))
        plt.barh(anova_scores["feature"].head(20), anova_scores["f_score"].head(20))
        plt.title("Top 20 Features by ANOVA F-score")
        plt.gca().invert_yaxis()
        _safe_save_plot(os.path.join(outdir, f"anova_top20_{model_suffix}.eps"))

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
        top.to_csv(os.path.join(outdir, f"xai_combined_features_{model_suffix}.csv"), index=False)
        print(f"[XAI] Results saved under {outdir}")
