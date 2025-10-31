"""
train.py — Unified training utilities for IoT/IIoT IDS
Author: Dr. Vanlalruata Hnamte
Version: 1.0
"""

import os
import time
import json
import math
import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)
from sklearn.preprocessing import label_binarize

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

# Local models
from HLC.models import CNN_LSTM_Fusion, SimpleMLP


# ============================================================
# Device / Energy proxy
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
POWER_WATTS = float(os.environ.get("POWER_WATTS", "120"))  # proxy for energy calc


# ============================================================
# Helpers: time complexity (params & approximate FLOPs)
# ============================================================

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def estimate_flops_linear(in_f, out_f):
    # MACs ~ in_f * out_f; FLOPs ~ 2 * MACs
    return 2.0 * in_f * out_f


def estimate_flops_conv1d(in_ch, out_ch, k, out_len):
    # per location MACs: in_ch * k; total MACs: out_ch * out_len * in_ch * k
    # FLOPs ~ 2 * MACs
    macs = out_ch * out_len * in_ch * k
    return 2.0 * macs


def estimate_flops_lstm(seq_len, input_size, hidden_size, bidirectional=True, layers=1):
    """
    Rough estimate for LSTM FLOPs per sequence:
    For one layer unidirectional:
      gates = 4
      FLOPs per time-step ~ 2 * (input_size*hidden + hidden*hidden + hidden)
    Multiply by seq_len and number of directions/layers.
    """
    dirs = 2 if bidirectional else 1
    per_timestep = 2.0 * (input_size * hidden_size + hidden_size * hidden_size + hidden_size) * 4
    return per_timestep * seq_len * dirs * layers


def estimate_model_flops(model_name: str, input_dim: int, batch_size: int, hidden_dim: int = 64) -> float:
    """
    Very coarse FLOPs estimator for our two built-in models.
    Returns FLOPs per forward pass (per batch).
    """
    if model_name.lower() in ["cnn_lstm_fusion", "cnn+lstm", "fusion"]:
        # CNN path
        L = input_dim
        # Conv1: 1->16, k=5, pool/2
        flops = estimate_flops_conv1d(1, 16, 5, out_len=L)
        L = L // 2
        # Conv2: 16->32, k=3, pool/2
        flops += estimate_flops_conv1d(16, 32, 3, out_len=L)
        L = L // 2
        # Conv3: 32->64, k=3, GAP -> out_len ~ L then pooled to 1
        flops += estimate_flops_conv1d(32, 64, 3, out_len=L)

        # LSTM path (BiLSTM, 1 layer, input_size=1, hidden=64)
        flops += estimate_flops_lstm(seq_len=input_dim, input_size=1, hidden_size=hidden_dim, bidirectional=True, layers=1)

        # FC: (64 + 128) -> 128 -> 64 -> C (ignore last C for now, caller can add with num_classes)
        flops_fc = estimate_flops_linear(64 + 2 * hidden_dim, 128)
        flops_fc += estimate_flops_linear(128, 64)
        return batch_size * (flops + flops_fc)

    elif model_name.lower() in ["mlp", "simplemlp"]:
        #  input_dim -> 256 -> 128 -> C
        flops = estimate_flops_linear(input_dim, 256)
        flops += estimate_flops_linear(256, 128)
        return batch_size * flops

    # Default: unknown
    return float("nan")


# ============================================================
# Data utilities
# ============================================================

def as_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool = True) -> DataLoader:
    ds = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long)
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, pin_memory=torch.cuda.is_available())


# ============================================================
# Training configuration
# ============================================================

@dataclass
class TrainConfig:
    model_name: str = "cnn_lstm_fusion"
    input_dim: int = 0
    num_classes: int = 2
    outdir: str = "outputs/run"
    epochs: int = 100
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    early_stop_patience: int = 10
    use_amp: bool = True
    seed: int = 42
    class_names: Optional[List[str]] = None
    task_type: str = "binary"   # "binary" | "multiclass"
    hidden_dim: int = 64        # for LSTM path
    save_tag: Optional[str] = None  # extra name tag (dataset, mode, etc.)


# ============================================================
# Model factory
# ============================================================

def build_model(cfg: TrainConfig) -> nn.Module:
    name = cfg.model_name.lower()
    if name in ["cnn_lstm_fusion", "cnn+lstm", "fusion"]:
        model = CNN_LSTM_Fusion(cfg.input_dim, cfg.num_classes, hidden_dim=cfg.hidden_dim)
    elif name in ["mlp", "simplemlp"]:
        model = SimpleMLP(cfg.input_dim, cfg.num_classes)
    else:
        raise ValueError(f"Unknown model name: {cfg.model_name}")
    return model.to(DEVICE)


# ============================================================
# Plot helpers (EPS)
# ============================================================

def _safe_eps(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, format="eps", dpi=200)
    plt.close()


def plot_learning_curves(history_df: pd.DataFrame, outdir: str):
    fig = plt.figure()
    plt.plot(history_df["epoch"], history_df["train_loss"], label="train_loss")
    if "val_loss" in history_df:
        plt.plot(history_df["epoch"], history_df["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Learning Curves (Loss vs. Epoch)")
    plt.legend()
    _safe_eps(os.path.join(outdir, "plots", "learning_curves.eps"))


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], outdir: str, title: str = "Confusion Matrix"):
    fig = plt.figure()
    im = plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], "d"),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    _safe_eps(os.path.join(outdir, "plots", "confusion_matrix.eps"))


def plot_roc_pr_curves(y_true: np.ndarray,
                       y_proba: np.ndarray,
                       class_names: List[str],
                       task_type: str,
                       outdir: str):
    os.makedirs(os.path.join(outdir, "plots"), exist_ok=True)

    # ROC
    try:
        if task_type == "binary":
            RocCurveDisplay.from_predictions(y_true, y_proba[:, 1])
        else:
            y_bin = label_binarize(y_true, classes=list(range(len(class_names))))
            RocCurveDisplay.from_predictions(y_bin.ravel(), y_proba.ravel())
        plt.title("ROC Curve")
        _safe_eps(os.path.join(outdir, "plots", "roc_curve.eps"))
    except Exception:
        plt.figure()
        plt.text(0.5, 0.5, "ROC failed (insufficient scores/classes)", ha="center")
        _safe_eps(os.path.join(outdir, "plots", "roc_curve.eps"))

    # PR
    try:
        if task_type == "binary":
            PrecisionRecallDisplay.from_predictions(y_true, y_proba[:, 1])
        else:
            y_bin = label_binarize(y_true, classes=list(range(len(class_names))))
            PrecisionRecallDisplay.from_predictions(y_bin.ravel(), y_proba.ravel())
        plt.title("Precision-Recall Curve")
        _safe_eps(os.path.join(outdir, "plots", "pr_curve.eps"))
    except Exception:
        plt.figure()
        plt.text(0.5, 0.5, "PR failed (insufficient scores/classes)", ha="center")
        _safe_eps(os.path.join(outdir, "plots", "pr_curve.eps"))


# ============================================================
# Training core
# ============================================================

def train_and_evaluate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cfg: TrainConfig
) -> Dict:
    """
    Trains a model, logs metrics/plots, saves checkpoint & returns a summary dict.
    """
    os.makedirs(cfg.outdir, exist_ok=True)
    os.makedirs(os.path.join(cfg.outdir, "plots"), exist_ok=True)
    os.makedirs(os.path.join(cfg.outdir, "checks"), exist_ok=True)

    # Build model
    model = build_model(cfg)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2, verbose=False)
    scaler = GradScaler(enabled=cfg.use_amp and DEVICE.type == "cuda")

    # Dataloaders
    train_loader = as_loader(X_train, y_train, cfg.batch_size, shuffle=True)
    val_loader   = as_loader(X_val,   y_val,   cfg.batch_size, shuffle=False)

    # Complexity proxies
    param_count = count_params(model)
    flops_est   = estimate_model_flops(cfg.model_name, cfg.input_dim, cfg.batch_size, cfg.hidden_dim)
    if not math.isnan(flops_est):
        flops_note = "FLOPs are rough estimates per forward pass (per batch)."
    else:
        flops_note = "FLOPs unavailable for this model."

    # Training loop
    best_loss = float("inf")
    patience  = cfg.early_stop_patience
    history_rows = []
    start_train_wall = time.perf_counter()

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        epoch_loss, correct, total = 0.0, 0, 0

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=cfg.use_amp and DEVICE.type == "cuda"):
                logits = model(xb)
                loss = criterion(logits, yb)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item() * yb.size(0)
            pred = logits.argmax(1)
            correct += (pred == yb).sum().item()
            total += yb.size(0)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        train_time_epoch = t1 - t0
        train_energy_epoch = train_time_epoch * POWER_WATTS

        train_loss = epoch_loss / max(1, total)
        train_acc  = correct / max(1, total)

        # Validation
        model.eval()
        val_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss += loss.item() * yb.size(0)
                v_correct += (logits.argmax(1) == yb).sum().item()
                v_total += yb.size(0)

        val_loss /= max(1, v_total)
        val_acc = v_correct / max(1, v_total)

        scheduler.step(val_loss)

        history_rows.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "train_time_sec": train_time_epoch,
            "train_energy_j_proxy": train_energy_epoch,
            "lr": optimizer.param_groups[0]["lr"]
        })

        # Early stopping on val loss
        if val_loss < best_loss - 1e-6:
            best_loss = val_loss
            patience = cfg.early_stop_patience
            # Save best checkpoint (unique)
            tag = cfg.save_tag or "run"
            ck_name = f"best_{cfg.model_name}_{tag}_e{epoch}_uid{uuid.uuid4().hex[:8]}.pt"
            best_ck_path = os.path.join(cfg.outdir, "checks", ck_name)
            torch.save({
                "state_dict": model.state_dict(),
                "model_name": cfg.model_name,
                "input_dim": cfg.input_dim,
                "num_classes": cfg.num_classes,
                "class_names": cfg.class_names,
                "task_type": cfg.task_type,
                "hidden_dim": cfg.hidden_dim,
                "epoch": epoch,
            }, best_ck_path)
            last_best_path = best_ck_path
        else:
            patience -= 1

        if patience == 0:
            break

    total_train_wall = time.perf_counter() - start_train_wall

    # Persist history
    hist_df = pd.DataFrame(history_rows)
    hist_csv = os.path.join(cfg.outdir, f"history_{cfg.model_name}_{cfg.save_tag or 'run'}.csv")
    hist_df.to_csv(hist_csv, index=False)
    plot_learning_curves(hist_df, cfg.outdir)

    # Load best checkpoint for evaluation
    state = torch.load(last_best_path, map_location=DEVICE)
    model.load_state_dict(state["state_dict"])
    model.eval()

    # Validation predictions & metrics
    y_true, y_pred, y_proba = [], [], []
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(DEVICE)
            logits = model(xb)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            y_proba.append(probs)
            y_pred.append(np.argmax(probs, axis=1))
            y_true.append(yb.numpy())

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    y_proba = np.concatenate(y_proba)

    # Classification metrics
    cls_names = cfg.class_names or [str(i) for i in range(cfg.num_classes)]
    report = classification_report(y_true, y_pred, target_names=cls_names, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    # ROC-AUC (macro)
    try:
        if cfg.task_type == "binary":
            roc_auc = roc_auc_score(y_true, y_proba[:, 1])
        else:
            roc_auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
    except Exception:
        roc_auc = float("nan")

    # Plots
    plot_confusion_matrix(cm, cls_names, cfg.outdir, title="Confusion Matrix (Validation)")
    plot_roc_pr_curves(y_true, y_proba, cls_names, cfg.task_type, cfg.outdir)

    # Save metrics JSON
    infer_time_per_batch = (t1 - t0) / max(1, len(val_loader))
    metrics = {
        "model_name": cfg.model_name,
        "save_tag": cfg.save_tag,
        "epochs_trained": int(hist_df["epoch"].max()),
        "best_val_loss": float(best_loss),
        "final_val_acc": float(hist_df["val_acc"].iloc[-1]),
        "final_val_loss": float(hist_df["val_loss"].iloc[-1]),
        "roc_auc_macro": float(roc_auc),
        "param_count": int(param_count),
        "flops_est_per_batch": None if math.isnan(flops_est) else float(flops_est),
        "flops_note": flops_note,
        "train_wall_time_sec": float(total_train_wall),
        "infer_time_ms_per_batch": float(infer_time_per_batch * 1000.0),
        "energy_j_proxy_per_train_sec": POWER_WATTS,
        "power_watts_assumed": POWER_WATTS,
        "device": str(DEVICE),
        "class_names": cls_names,
        "task_type": cfg.task_type,
        "history_csv": os.path.relpath(hist_csv, cfg.outdir),
        "best_checkpoint": os.path.relpath(last_best_path, cfg.outdir),
    }
    with open(os.path.join(cfg.outdir, f"metrics_{cfg.model_name}_{cfg.save_tag or 'run'}.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Also save classification report table
    rep_df = pd.DataFrame(report).T
    rep_csv = os.path.join(cfg.outdir, f"classification_report_{cfg.model_name}_{cfg.save_tag or 'run'}.csv")
    rep_df.to_csv(rep_csv)

    return {
        "metrics": metrics,
        "report_df_path": rep_csv,
        "history_csv_path": hist_csv,
        "confusion_matrix": cm.tolist()
    }


# ============================================================
# Public API (simple wrapper)
# ============================================================

def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    model_name: str,
    outdir: str,
    class_names: List[str],
    task_type: str,
    epochs: int = 100,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    early_stop_patience: int = 10,
    seed: int = 42,
    hidden_dim: int = 64,
    save_tag: Optional[str] = None
) -> Dict:
    """
    Convenience entrypoint that assembles TrainConfig and calls train_and_evaluate.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    cfg = TrainConfig(
        model_name=model_name,
        input_dim=X_train.shape[1],
        num_classes=len(class_names),
        outdir=outdir,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        early_stop_patience=early_stop_patience,
        use_amp=True,
        seed=seed,
        class_names=class_names,
        task_type=task_type,
        hidden_dim=hidden_dim,
        save_tag=save_tag
    )

    return train_and_evaluate(X_train, y_train, X_val, y_val, cfg)
