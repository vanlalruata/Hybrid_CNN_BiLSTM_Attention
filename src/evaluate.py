"""
evaluate.py — Unified evaluation utilities for IoT/IIoT IDS
Author: Dr. Vanlalruata Hnamte
Version: 1.0
"""

import os
import time
import json
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    RocCurveDisplay,
    PrecisionRecallDisplay,
    mean_squared_error,
    mean_absolute_error,
    brier_score_loss,
)
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import label_binarize

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import warnings

# Silence EPS transparency warning (artists with alpha will be rendered opaque in PS/EPS)
warnings.filterwarnings("ignore", message="The PostScript backend does not support transparency", category=UserWarning)

# Local models (must match what you trained with)
from .models import CNN_LSTM_Fusion, SimpleMLP


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
POWER_WATTS = float(os.environ.get("POWER_WATTS", "120"))  # proxy for Joules = seconds * watts


# -----------------------------------------------------------------------------
# Model loading
# -----------------------------------------------------------------------------

def _build_from_state(state: Dict) -> nn.Module:
    """
    Recreate the model architecture from a saved checkpoint dict.
    The training code stores:
      - 'model_name', 'input_dim', 'num_classes', 'hidden_dim', 'task_type'
    """
    name = (state.get("model_name") or "").lower()
    input_dim = int(state.get("input_dim"))
    num_classes = int(state.get("num_classes", 1))
    hidden_dim = int(state.get("hidden_dim", 64))

    if name in ["cnn_lstm_fusion", "cnn+lstm", "fusion"]:
        model = CNN_LSTM_Fusion(input_dim, num_classes, hidden_dim=hidden_dim)
    elif name in ["mlp", "simplemlp", "dnn"]:
        model = SimpleMLP(input_dim, num_classes)
    else:
        raise ValueError(f"Unknown model in checkpoint: {name}")

    model.load_state_dict(state["state_dict"])
    model.to(DEVICE).eval()
    return model


def load_checkpoint(ckpt_path: str) -> Tuple[nn.Module, Dict]:
    """
    Loads a training checkpoint and rebuilds the model ready for inference.
    Returns (model, metadata_dict).
    """
    state = torch.load(ckpt_path, map_location=DEVICE)
    model = _build_from_state(state)
    meta = {
        "model_name": state.get("model_name"),
        "input_dim": state.get("input_dim"),
        "num_classes": state.get("num_classes"),
        "class_names": state.get("class_names"),
        "task_type": state.get("task_type"),
        "hidden_dim": state.get("hidden_dim"),
        "epoch": state.get("epoch"),
        "checkpoint_path": ckpt_path
    }
    return model, meta


# -----------------------------------------------------------------------------
# Data helpers
# -----------------------------------------------------------------------------

def make_loader(X: np.ndarray, y: Optional[np.ndarray], batch_size: int = 512, shuffle: bool = False) -> DataLoader:
    if y is None:
        ds = TensorDataset(torch.tensor(X, dtype=torch.float32))
    else:
        ds = TensorDataset(torch.tensor(X, dtype=torch.float32),
                           torch.tensor(y, dtype=torch.long))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, pin_memory=torch.cuda.is_available())


def _coerce_feature_dim(X: np.ndarray, expected: int) -> np.ndarray:
    """
    Ensure X has the feature dimension expected by the checkpoint.
    If fewer columns: right-pad zeros. If more: truncate extra columns.
    """
    X = np.asarray(X)
    if X.shape[1] == expected:
        return X
    if X.shape[1] < expected:
        pad = np.zeros((X.shape[0], expected - X.shape[1]), dtype=X.dtype)
        print(f"[WARN] Feature mismatch: X has {X.shape[1]} cols, expected {expected}. Padding {pad.shape[1]} zeros.")
        return np.hstack([X, pad])
    else:
        print(f"[WARN] Feature mismatch: X has {X.shape[1]} cols, expected {expected}. Truncating to {expected}.")
        return X[:, :expected]


# -----------------------------------------------------------------------------
# Plot helpers
# -----------------------------------------------------------------------------

def _tight_eps(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, format="eps", dpi=200, transparent=False)
    plt.close()


def plot_confusion(cm: np.ndarray, class_names: List[str], outdir: str, title: str = "Confusion Matrix", model_suffix: str = "model"):
    plt.figure()
    im = plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=45, ha="right")
    plt.yticks(ticks, class_names)
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], "d"),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    _tight_eps(os.path.join(outdir, "plots", f"confusion_matrix_{model_suffix}.eps"))


def plot_roc_pr(y_true: np.ndarray, y_proba: np.ndarray, class_names: List[str], task_type: str, outdir: str, model_suffix: str = "model"):
    # ROC
    try:
        if task_type == "binary":
            RocCurveDisplay.from_predictions(y_true, y_proba[:, 1])
        else:
            y_bin = label_binarize(y_true, classes=list(range(len(class_names))))
            RocCurveDisplay.from_predictions(y_bin.ravel(), y_proba.ravel())
        plt.title("ROC Curve")
        _tight_eps(os.path.join(outdir, "plots", f"roc_curve_{model_suffix}.eps"))
    except Exception:
        plt.figure(); plt.text(0.5, 0.5, "ROC not available", ha="center")
        _tight_eps(os.path.join(outdir, "plots", f"roc_curve_{model_suffix}.eps"))

    # PR
    try:
        if task_type == "binary":
            PrecisionRecallDisplay.from_predictions(y_true, y_proba[:, 1])
        else:
            y_bin = label_binarize(y_true, classes=list(range(len(class_names))))
            PrecisionRecallDisplay.from_predictions(y_bin.ravel(), y_proba.ravel())
        plt.title("Precision-Recall Curve")
        _tight_eps(os.path.join(outdir, "plots", f"pr_curve_{model_suffix}.eps"))
    except Exception:
        plt.figure(); plt.text(0.5, 0.5, "PR not available", ha="center")
        _tight_eps(os.path.join(outdir, "plots", f"pr_curve_{model_suffix}.eps"))


def plot_calibration(y_true: np.ndarray, y_proba: np.ndarray, task_type: str, outdir: str, model_suffix: str = "model"):
    """
    Reliability diagram. For multiclass we use the max-class confidence proxy.
    """
    plt.figure()
    if task_type == "binary":
        prob_pos = y_proba[:, 1]
        frac_pos, mean_pred = calibration_curve(y_true, prob_pos, n_bins=10, strategy="uniform")
        plt.plot(mean_pred, frac_pos, marker="o", label="Model")
        plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect")
        bs = brier_score_loss(y_true, prob_pos)
        plt.title(f"Calibration (Brier={bs:.4f})")
    else:
        conf = y_proba.max(axis=1)
        # Treat correct/incorrect as surrogate for reliability (ECE-style)
        correct = (np.argmax(y_proba, axis=1) == y_true).astype(int)
        frac_pos, mean_pred = calibration_curve(correct, conf, n_bins=10, strategy="uniform")
        plt.plot(mean_pred, frac_pos, marker="o", label="Model (Top-1)")
        plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect")
        plt.title("Calibration (Top-1 confidence)")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.legend()
    _tight_eps(os.path.join(outdir, "plots", f"calibration_curve_{model_suffix}.eps"))


def plot_regression(y_true: np.ndarray, y_pred: np.ndarray, outdir: str, model_suffix: str = "model"):
    # Scatter y_true vs y_pred
    plt.figure()
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.scatter(y_true, y_pred, s=8, alpha=0.6)
    plt.plot(lims, lims, "--")
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title("Regression: Prediction Scatter")
    _tight_eps(os.path.join(outdir, "plots", f"reg_scatter_{model_suffix}.eps"))

    # Residuals
    plt.figure()
    resid = y_pred - y_true
    plt.hist(resid, bins=30, edgecolor="black")
    plt.title("Regression: Residuals")
    plt.xlabel("Residual")
    plt.ylabel("Count")
    _tight_eps(os.path.join(outdir, "plots", f"reg_residuals_{model_suffix}.eps"))


# -----------------------------------------------------------------------------
# Core evaluation
# -----------------------------------------------------------------------------

@torch.no_grad()
def _infer_batched(model: nn.Module, loader: DataLoader, task_type: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Returns (y_pred, y_proba_or_none).
    For regression task_type, y_proba is None and y_pred is float values.
    """
    y_pred_all, y_proba_all = [], []
    for batch in loader:
        if len(batch) == 2:
            xb, = batch[0:1]
        else:
            xb = batch[0]
        xb = xb.to(DEVICE)
        logits = model(xb)

        if task_type in ["binary", "multiclass"]:
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            y_proba_all.append(probs)
            y_pred_all.append(np.argmax(probs, axis=1))
        else:
            # regression: raw output
            y_hat = logits.squeeze(-1).cpu().numpy()
            y_pred_all.append(y_hat)

    y_pred = np.concatenate(y_pred_all)
    y_proba = np.concatenate(y_proba_all) if len(y_proba_all) else None
    return y_pred, y_proba


def evaluate_classification(
    X_test: np.ndarray,
    y_test: np.ndarray,
    ckpt_path: str,
    outdir: str,
    batch_size: int = 512
) -> Dict:
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(os.path.join(outdir, "plots"), exist_ok=True)

    model, meta = load_checkpoint(ckpt_path)
    model_name = (meta.get("model_name") or "").lower()
    model_suffix = "fusion" if "fusion" in model_name else "dnn"

    class_names = meta.get("class_names") or [str(i) for i in range(int(meta.get("num_classes", 2)))]
    # Align feature dimension to what the checkpoint expects (guards against stale checkpoints vs new splits)
    exp_in = int(meta.get("input_dim") or X_test.shape[1])
    X_test = _coerce_feature_dim(X_test, exp_in)
    loader = make_loader(X_test, y_test, batch_size=batch_size, shuffle=False)

    # Inference timing
    if torch.cuda.is_available(): torch.cuda.synchronize()
    t0 = time.perf_counter()
    y_pred, y_proba = _infer_batched(model, loader, task_type="multiclass" if len(class_names) > 2 else "binary")
    if torch.cuda.is_available(): torch.cuda.synchronize()
    t1 = time.perf_counter()

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    rep = classification_report(y_test, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    # ROC-AUC
    try:
        if len(class_names) == 2:
            roc_auc = roc_auc_score(y_test, y_proba[:, 1])
        else:
            roc_auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")
    except Exception:
        roc_auc = float("nan")

    # Plots
    plot_confusion(cm, class_names, outdir, model_suffix=model_suffix)
    if y_proba is not None:
        plot_roc_pr(y_test, y_proba, class_names, "binary" if len(class_names) == 2 else "multiclass", outdir, model_suffix=model_suffix)
        plot_calibration(y_test, y_proba, "binary" if len(class_names) == 2 else "multiclass", outdir, model_suffix=model_suffix)

    # Save predictions
    pred_df = pd.DataFrame({
        "y_true": y_test,
        "y_pred": y_pred
    })
    if y_proba is not None:
        for i, cname in enumerate(class_names):
            pred_df[f"proba_{cname}"] = y_proba[:, i]
    pred_csv = os.path.join(outdir, f"predictions_{model_suffix}.csv")
    pred_df.to_csv(pred_csv, index=False)

    # Save report CSV
    rep_df = pd.DataFrame(rep).T
    rep_csv = os.path.join(outdir, f"classification_report_{model_suffix}.csv")
    rep_df.to_csv(rep_csv)

    # Timing / energy
    total_time = t1 - t0
    n_samples = len(X_test)
    batches = max(1, len(loader))
    ms_per_batch = (total_time / batches) * 1000.0
    samples_per_sec = n_samples / total_time if total_time > 0 else float("inf")
    energy_j = total_time * POWER_WATTS

    # Echo params/flops if available (from train metrics.json sitting next to checkpoint)
    maybe_train_metrics = None
    train_metrics_guess = os.path.join(os.path.dirname(os.path.dirname(ckpt_path)),  # outdir/
                                       f"metrics_{meta.get('model_name')}_{os.path.basename(os.path.dirname(ckpt_path))}.json")
    # We won't hard-depend on this file; keep robust.
    if os.path.exists(train_metrics_guess):
        try:
            with open(train_metrics_guess) as f:
                maybe_train_metrics = json.load(f)
        except Exception:
            pass

    metrics = {
        "checkpoint": ckpt_path,
        "model_name": meta.get("model_name"),
        "num_classes": len(class_names),
        "class_names": class_names,
        "accuracy": float(acc),
        "roc_auc_macro": float(roc_auc),
        "samples": int(n_samples),
        "batches": int(batches),
        "total_infer_sec": float(total_time),
        "ms_per_batch": float(ms_per_batch),
        "samples_per_sec": float(samples_per_sec),
        "energy_j_proxy": float(energy_j),
    }
    if isinstance(maybe_train_metrics, dict):
        for k in ["param_count", "flops_est_per_batch", "power_watts_assumed", "device"]:
            if k in maybe_train_metrics:
                metrics[k] = maybe_train_metrics[k]

    # Build a complexity snapshot for evaluation time (detection on validation)
    try:
        param_count = sum(p.numel() for p in model.parameters())
    except Exception:
        param_count = None
    complexity_eval = {
        "checkpoint": ckpt_path,
        "model_name": meta.get("model_name"),
        "param_count": int(param_count) if param_count is not None else None,
        "device": str(DEVICE),
        "power_watts_assumed": float(POWER_WATTS),
        "eval_infer_total_sec": float(total_time),
        "eval_ms_per_batch": float(ms_per_batch),
        "eval_ms_per_sample": float((total_time / max(1, n_samples)) * 1000.0),
        "eval_samples_per_sec": float(samples_per_sec),
        "eval_energy_j_proxy": float(energy_j),
        "batches": int(batches),
        "samples": int(n_samples),
    }

    # Write files
    metrics_path = os.path.join(outdir, f"metrics_{model_suffix}.json")
    complexity_eval_path = os.path.join(outdir, f"complexity_eval_{model_suffix}.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    with open(complexity_eval_path, "w") as f:
        json.dump(complexity_eval, f, indent=2)

    return {
        "metrics_json": metrics_path,
        "complexity_eval_json": complexity_eval_path,
        "predictions_csv": pred_csv,
        "report_csv": rep_csv,
        "confusion_matrix": cm.tolist()
    }


def evaluate_regression(
    X_test: np.ndarray,
    y_test: np.ndarray,
    ckpt_path: str,
    outdir: str,
    batch_size: int = 512
) -> Dict:
    """
    For completeness if you configure a regression head (e.g., models returning 1 value).
    """
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(os.path.join(outdir, "plots"), exist_ok=True)

    model, meta = load_checkpoint(ckpt_path)
    model_name = (meta.get("model_name") or "").lower()
    model_suffix = "fusion" if "fusion" in model_name else "dnn"

    exp_in = int(meta.get("input_dim") or X_test.shape[1])
    X_test = _coerce_feature_dim(X_test, exp_in)
    loader = make_loader(X_test, y_test, batch_size=batch_size, shuffle=False)

    if torch.cuda.is_available(): torch.cuda.synchronize()
    t0 = time.perf_counter()
    # Run inference (regression path)
    y_pred, _ = _infer_batched(model, loader, task_type="regression")
    if torch.cuda.is_available(): torch.cuda.synchronize()
    t1 = time.perf_counter()

    # Metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(y_test, y_pred)

    # Plots
    plot_regression(y_test, y_pred, outdir, model_suffix=model_suffix)

    # Save predictions
    pred_df = pd.DataFrame({"y_true": y_test, "y_pred": y_pred})
    pred_csv = os.path.join(outdir, f"predictions_regression_{model_suffix}.csv")
    pred_df.to_csv(pred_csv, index=False)

    # Timing / energy
    total_time = t1 - t0
    n_samples = len(X_test)
    batches = max(1, len(loader))
    ms_per_batch = (total_time / batches) * 1000.0
    samples_per_sec = n_samples / total_time if total_time > 0 else float("inf")
    energy_j = total_time * POWER_WATTS

    metrics = {
        "checkpoint": ckpt_path,
        "model_name": meta.get("model_name"),
        "task": "regression",
        "samples": int(n_samples),
        "batches": int(batches),
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae),
        "total_infer_sec": float(total_time),
        "ms_per_batch": float(ms_per_batch),
        "samples_per_sec": float(samples_per_sec),
        "energy_j_proxy": float(energy_j),
    }
    with open(os.path.join(outdir, f"metrics_regression_{model_suffix}.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    return {
        "metrics_json": os.path.join(outdir, f"metrics_regression_{model_suffix}.json"),
        "predictions_csv": pred_csv
    }


# -----------------------------------------------------------------------------
# Public convenience APIs
# -----------------------------------------------------------------------------

def evaluate_on_test(
    X_test: np.ndarray,
    y_test: np.ndarray,
    checkpoint_path: str,
    outdir: str,
    task_type: str,
    class_names: Optional[List[str]] = None,
    batch_size: int = 512
) -> Dict:
    """
    Single entrypoint used by your CLI/main.
    - task_type: "binary" | "multiclass" | "regression"
    """
    # Allow overriding class_names (else use those in checkpoint)
    model, meta = load_checkpoint(checkpoint_path)
    if class_names is None:
        class_names = meta.get("class_names")

    # Re-dispatch to specialized evaluators
    if task_type in ["binary", "multiclass"]:
        # repackage model meta path so the specialized function can re-load (keeps logic simple)
        del model  # noqa
        return evaluate_classification(X_test, y_test, checkpoint_path, outdir, batch_size=batch_size)
    elif task_type == "regression":
        del model  # noqa
        return evaluate_regression(X_test, y_test, checkpoint_path, outdir, batch_size=batch_size)
    else:
        raise ValueError(f"Unknown task_type: {task_type}")


@torch.no_grad()
def load_model_and_predict(
    checkpoint_path: str,
    X: np.ndarray,
    batch_size: int = 1024,
    return_proba: bool = True
) -> Dict[str, np.ndarray]:
    """
    Lightweight utility for **deployment/online use**:
      y_pred, (optional) y_proba, and model metadata.
    """
    model, meta = load_checkpoint(checkpoint_path)
    exp_in = int(meta.get("input_dim") or X.shape[1])
    X = _coerce_feature_dim(X, exp_in)
    loader = make_loader(X, y=None, batch_size=batch_size, shuffle=False)

    if torch.cuda.is_available(): torch.cuda.synchronize()
    t0 = time.perf_counter()

    # Inference
    y_pred_all, y_proba_all = [], []
    for (xb,) in loader:
        xb = xb.to(DEVICE)
        logits = model(xb)
        if meta.get("task_type") in ["binary", "multiclass"]:
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            y_proba_all.append(probs)
            y_pred_all.append(np.argmax(probs, axis=1))
        else:
            y_pred_all.append(logits.squeeze(-1).cpu().numpy())

    if torch.cuda.is_available(): torch.cuda.synchronize()
    t1 = time.perf_counter()

    y_pred = np.concatenate(y_pred_all)
    out = {
        "y_pred": y_pred,
        "meta": meta,
        "infer_total_sec": float(t1 - t0),
        "energy_j_proxy": float((t1 - t0) * POWER_WATTS)
    }
    if return_proba and len(y_proba_all):
        out["y_proba"] = np.concatenate(y_proba_all)
    return out


def aggregate_seed_stats(
    results_list: List[Dict],
    outdir: str,
    model_suffix: str
) -> Dict:
    """
    Given a list of result dictionaries containing file paths of evaluated runs,
    compute the mean and standard deviation of key metrics and save to seed_stats.json.
    """
    keys = ["accuracy", "macro_f1", "macro_precision", "macro_recall", "eval_total_sec", "energy_j"]
    stats = {}
    
    extracted = {k: [] for k in keys}
    for res in results_list:
        # Load metrics json
        m_path = res["metrics_json"]
        with open(m_path, "r") as f:
            m_data = json.load(f)
        
        # Load classification report csv
        rep_csv = res["report_csv"]
        rep_df = pd.read_csv(rep_csv, index_col=0)
        
        extracted["accuracy"].append(m_data.get("accuracy", 0.0))
        extracted["macro_f1"].append(rep_df.loc["macro avg", "f1-score"])
        extracted["macro_precision"].append(rep_df.loc["macro avg", "precision"])
        extracted["macro_recall"].append(rep_df.loc["macro avg", "recall"])
        extracted["eval_total_sec"].append(m_data.get("total_infer_sec", 0.0))
        extracted["energy_j"].append(m_data.get("energy_j_proxy", 0.0))
        
    for k, vals in extracted.items():
        v_arr = np.array(vals)
        stats[f"{k}_mean"] = float(np.mean(v_arr))
        stats[f"{k}_std"] = float(np.std(v_arr))
        
    # Write to seed_stats.json
    out_path = os.path.join(outdir, f"seed_stats_{model_suffix}.json")
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)
        
    print(f"[STATS] Seed statistics saved to {out_path}")
    return stats
