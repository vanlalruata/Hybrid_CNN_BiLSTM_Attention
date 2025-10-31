"""
viz.py — Visualization tools for IDS datasets
Author: Dr. Vanlalruata Hnamte
Version: 1.0
"""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from HLC.utils import ensure_dir


def plot_dataset_overview(X, y, feature_names, outdir="outputs/viz", max_features=20):
    ensure_dir(outdir)
    df = pd.DataFrame(X, columns=feature_names)
    df["Label"] = y

    # Class balance
    plt.figure(figsize=(6, 4))
    sns.countplot(x="Label", data=df)
    plt.title("Class Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "class_distribution.eps"), format="eps", dpi=200)
    plt.close()

    # Feature histograms (limited)
    subset_cols = feature_names[:max_features]
    df[subset_cols].hist(figsize=(14, 10), bins=30)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "feature_histograms.eps"), format="eps", dpi=200)
    plt.close()

    # Correlation heatmap
    corr = df[subset_cols].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, cmap="coolwarm", annot=False)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "correlation_heatmap.eps"), format="eps", dpi=200)
    plt.close()
