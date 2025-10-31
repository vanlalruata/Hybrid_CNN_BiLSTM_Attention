"""
utils.py — Utility functions for IoT/IIoT IDS framework
Author:  Dr. Vanlalruata Hnamte
Version: 1.0
"""

import os
import re
import time
import math
import random
import json
from datetime import datetime
from typing import Tuple, List, Dict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless mode for servers
import matplotlib.pyplot as plt
import seaborn as sns

import torch

from HLC.src.config import DIR_OUTPUTS, FIG_DPI


# ============================================================
# REPRODUCIBILITY & FILE MANAGEMENT
# ============================================================

def set_all_seeds(seed: int):
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    """Create a directory if it doesn’t exist."""
    os.makedirs(path, exist_ok=True)


def timestamp() -> str:
    """Return current timestamp string."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def unique_path(path: str) -> str:
    """Avoid overwriting by appending __n to file names."""
    if not os.path.exists(path):
        return path
    base, ext = os.path.splitext(path)
    i = 1
    while True:
        candidate = f"{base}__{i}{ext}"
        if not os.path.exists(candidate):
            return candidate
        i += 1


# ============================================================
# PLOTTING UTILITIES (EPS FOR PUBLICATION QUALITY)
# ============================================================

def save_eps(fig, path: str):
    """Save matplotlib figure as EPS format."""
    ensure_dir(os.path.dirname(path))
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight", format="eps")
    plt.close(fig)


def plot_class_distribution(y: np.ndarray, classes: List[str], outpath: str):
    """Plot bar chart showing class distribution."""
    fig, ax = plt.subplots(figsize=(6, 4))
    uniq, cnts = np.unique(y, return_counts=True)
    labels = [classes[i] for i in uniq]
    sns.barplot(x=labels, y=cnts, ax=ax)
    ax.set_ylabel("Count")
    ax.set_xlabel("Class")
    ax.set_title("Class Distribution")
    ax.tick_params(axis="x", rotation=45)
    save_eps(fig, outpath)


def plot_correlation_heatmap(df: pd.DataFrame, outpath: str):
    """Plot correlation heatmap for numeric features."""
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Feature Correlation Heatmap")
    save_eps(fig, outpath)


def plot_feature_histograms(df: pd.DataFrame, outdir: str, max_cols: int = 25):
    """Plot histograms of up to N numeric features."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()[:max_cols]
    n = len(numeric_cols)
    cols = 5
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 2.5))
    axes = axes.flatten()
    for i, c in enumerate(numeric_cols):
        sns.histplot(df[c], bins=30, ax=axes[i])
        axes[i].set_title(c)
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    fig.suptitle("Feature Distributions (subset)")
    save_eps(fig, os.path.join(outdir, "feature_histograms.eps"))


# ============================================================
# MODEL COMPLEXITY (APPROX FLOPS/MACs)
# ============================================================

def estimate_linear_flops(in_f, out_f):
    """Estimate FLOPs for linear layer."""
    return in_f * out_f


def estimate_conv1d_flops(Cin, Cout, k, L):
    """Estimate FLOPs for Conv1D operation."""
    return L * Cout * (Cin * k)


def estimate_lstm_flops(input_size, hidden_size, seq_len):
    """Estimate FLOPs for single-layer LSTM."""
    return 4 * (input_size * hidden_size + hidden_size * hidden_size) * seq_len


def model_complexity_report(model, input_dim: int) -> Dict[str, float]:
    """
    Estimate per-sample complexity for the Parallel CNN + LSTM model.
    This is an approximate count useful for comparing architectures.
    """
    L = input_dim
    conv1 = estimate_conv1d_flops(1, 16, 5, L)
    conv2 = estimate_conv1d_flops(16, 32, 3, L)
    conv3 = estimate_conv1d_flops(32, 64, 3, L)
    conv_total = conv1 + conv2 + conv3

    lstm = estimate_lstm_flops(1, 64, L)
    fc1 = estimate_linear_flops(128, 64)
    fc2 = estimate_linear_flops(64, 32)

    total = conv_total + lstm + fc1 + fc2

    return {
        "conv1d_flops": float(conv_total),
        "lstm_flops": float(lstm),
        "fc_flops": float(fc1 + fc2),
        "total_flops_per_sample": float(total)
    }


# ============================================================
# JSON / CSV SAVE UTILITIES
# ============================================================

def dump_json(obj: dict, path: str):
    """Save dictionary to JSON file."""
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def dump_csv(df: pd.DataFrame, path: str):
    """Save DataFrame to CSV."""
    ensure_dir(os.path.dirname(path))
    df.to_csv(path, index=False)
