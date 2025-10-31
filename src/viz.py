"""
viz.py — Visualization tools for IDS datasets
Author: Dr. Vanlalruata Hnamte
Version: 1.1
"""

import os
import json
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import hsv_to_rgb

from HLC.src.utils import ensure_dir
from HLC.src.config import DIR_OUTPUTS


def _random_dark_colors(n: int):
    """
    Generate n distinct dark vivid colors as hex strings using HSV space.
    """
    cols = []
    for _ in range(n):
        # High saturation, lower value (brightness) to ensure dark colors
        rgb = hsv_to_rgb([random.random(), 0.85, 0.45])
        cols.append('#%02x%02x%02x' % tuple((rgb * 255).astype(int)))
    return cols


def plot_dataset_overview(X, y, feature_names, outdir=None, max_features=20, class_names=None):
    """
    Create dataset overview visualizations:
      - Class distribution (with readable class labels)
      - Feature histograms (limited to max_features)
      - Correlation heatmap (on the plotted features)
     Additionally, writes a viz_guide.json in the same folder with:
      - class labels and counts
      - colors used
      - histogram bins/counts per feature
      - correlation matrix
    """
    # Resolve the default output directory using config
    outdir = outdir or os.path.join(DIR_OUTPUTS, "viz")
    ensure_dir(outdir)

    df = pd.DataFrame(X, columns=feature_names)

    # Map numeric labels to names when provided
    if class_names is not None:
        labels_mapped = []
        for lbl in y:
            try:
                idx = int(lbl)
                if 0 <= idx < len(class_names):
                    labels_mapped.append(str(class_names[idx]))
                else:
                    labels_mapped.append(str(lbl))
            except Exception:
                labels_mapped.append(str(lbl))
        df["Label"] = labels_mapped
    else:
        df["Label"] = y

    # ------------------------------------------------------------------
    # Class balance (random dark palette, explicit label order)
    # ------------------------------------------------------------------
    class_counts = df["Label"].value_counts()
    labels_order = class_counts.index.tolist()
    class_colors = _random_dark_colors(len(labels_order))

    plt.figure(figsize=(6, 4))
    ax = sns.countplot(x="Label", hue="Label", data=df, order=labels_order, palette=class_colors, legend=True)
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    plt.title("Class Distribution")
    # Rotate x-axis labels 90 degrees for readability
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(90)
        lbl.set_ha("center")
    # Extra bottom margin to avoid clipping long labels
    plt.gcf().subplots_adjust(bottom=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "class_distribution.eps"), format="eps", dpi=200, transparent=False)
    plt.close()

    # ------------------------------------------------------------------
    # Feature histograms (limited) with dark random colors
    # ------------------------------------------------------------------
    subset_cols = feature_names[:max_features]
    feat_colors = _random_dark_colors(len(subset_cols))

    axes = df[subset_cols].hist(figsize=(14, 10), bins=30)
    # axes can be a 2D array; flatten safely
    axes_flat = []
    if isinstance(axes, np.ndarray):
        axes_flat = [ax for ax in axes.ravel() if ax is not None]
    else:
        axes_flat = [axes] if axes is not None else []

    # Color each histogram's bars with the assigned color
    for ax, col, c in zip(axes_flat, subset_cols, feat_colors):
        for p in getattr(ax, "patches", []):
            try:
                p.set_facecolor(c)
            except Exception:
                pass
        ax.set_title(col)

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "feature_histograms.eps"), format="eps", dpi=200, transparent=False)
    plt.close()

    # Build histogram data (same bins=30) for guide log
    histograms = {}
    for col, c in zip(subset_cols, feat_colors):
        counts, bins = np.histogram(df[col].values, bins=30)
        histograms[col] = {
            "bins": bins.tolist(),
            "counts": counts.tolist(),
            "color": c
        }

    # ------------------------------------------------------------------
    # Correlation heatmap and capture matrix for guide log
    # ------------------------------------------------------------------
    corr = df[subset_cols].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, cmap="coolwarm", annot=False)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "correlation_heatmap.eps"), format="eps", dpi=200, transparent=False)
    plt.close()

    # ------------------------------------------------------------------
    # Write viz guide JSON log
    # ------------------------------------------------------------------
    guide = {
        "class_names": list(class_names) if class_names is not None else None,
        "class_distribution": {str(k): int(v) for k, v in class_counts.to_dict().items()},
        "class_colors": {str(label): color for label, color in zip(labels_order, class_colors)},
        "features_plotted": subset_cols,
        "histograms": histograms,
        "correlation": {
            "features": subset_cols,
            "matrix": corr.values.tolist()
        }
    }
    with open(os.path.join(outdir, "viz_guide.json"), "w") as f:
        json.dump(guide, f, indent=2)
