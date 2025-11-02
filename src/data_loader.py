"""
Data loading and preprocessing utilities
Author: Dr. Vanlalruata Hnamte
Version: 1.0
"""

import os
import glob
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .config import (
    DATASETS, EXPLICIT_FILES, COMMON_DROP, EXTRA_DROP,
    LABEL_COLUMNS, DIR_PROCESSED
)
from HLC.src.utils import ensure_dir, unique_path


# ============================================================
# DATA FILE RESOLUTION
# ============================================================

def find_dataset_csvs(dataset: str) -> List[str]:
    """
    Resolve all CSV paths for a given dataset name.
    Uses EXPLICIT_FILES if specified; otherwise loads all *.csv.
    """
    pattern = DATASETS[dataset]
    parent = os.path.dirname(pattern)
    allow = EXPLICIT_FILES.get(dataset)

    if allow:
        paths = []
        for fname in allow:
            p = os.path.join(parent, fname)
            if os.path.exists(p):
                paths.append(p)
        # Include other CSVs too, deduplicated
        paths.extend(glob.glob(pattern))
        seen = set()
        ordered = []
        for p in paths:
            if p not in seen:
                ordered.append(p)
                seen.add(p)
        return ordered

    return glob.glob(pattern)


def safe_read_csv(path: str, chunksize: int = None) -> pd.DataFrame:
    """
    Safe CSV loader that can handle large files via chunks.
    """
    try:
        if chunksize:
            chunks = []
            for c in pd.read_csv(path, chunksize=chunksize, low_memory=False):
                chunks.append(c)
            return pd.concat(chunks, ignore_index=True)
        else:
            return pd.read_csv(path, low_memory=False)
    except Exception as e:
        print(f"[WARN] Failed reading {path}: {e}")
        return pd.DataFrame()


# ============================================================
# LABEL AND FEATURE HANDLING
# ============================================================

def pick_label_column(df: pd.DataFrame, dataset: str) -> str:
    """Detect which column should be used as a label."""
    for col in LABEL_COLUMNS.get(dataset, []):
        if col in df.columns:
            return col
    for fallback in ["Label", "label", "Attack_type", "subcategory"]:
        if fallback in df.columns:
            return fallback
    raise ValueError(f"No label column found for {dataset}. Sample columns: {list(df.columns)[:30]}")


def drop_identifier_columns(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Drop dataset-specific and common identifier columns."""
    drops = set(COMMON_DROP + EXTRA_DROP.get(dataset, []))
    present = [c for c in drops if c in df.columns]
    if present:
        df = df.drop(columns=present)
    return df


def ensure_numeric(df: pd.DataFrame, exclude: List[str]) -> pd.DataFrame:
    """Convert non-numeric features to integer codes."""
    cols = [c for c in df.columns if c not in exclude]
    for c in cols:
        if df[c].dtype == object:
            df[c] = df[c].astype(str)
            df[c] = pd.factorize(df[c])[0]
    return df


# ============================================================
# MAIN DATA PREPARATION
# ============================================================

def load_raw_dataset(dataset: str) -> Tuple[pd.DataFrame, np.ndarray, List[str], LabelEncoder, StandardScaler, str]:
    """
    Load and preprocess the raw dataset.
    Returns standardized features, encoded labels, encoders, and task types.
    """
    files = find_dataset_csvs(dataset)
    if not files:
        raise FileNotFoundError(f"No CSV files found for {dataset}")

    dfs = []
    for f in files:
        df = safe_read_csv(f)
        if not df.empty:
            dfs.append(df)

    if not dfs:
        raise RuntimeError(f"All CSV reads failed for {dataset}")

    df = pd.concat(dfs, ignore_index=True)

    # Label extraction
    label_col = pick_label_column(df, dataset)
    y_raw = df[label_col].astype(str).values
    df = df.drop(columns=[label_col])

    # Drop identifiers
    df = drop_identifier_columns(df, dataset)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(axis=0)

    # Encode labels
    y_encoder = LabelEncoder()
    y = y_encoder.fit_transform(y_raw[:len(df)])

    # Convert all categorical features to numeric codes
    df = ensure_numeric(df, exclude=[])

    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(df.values)
    feature_cols = df.columns.tolist()

    # Determine a classification type
    task = "binary" if len(np.unique(y)) == 2 else "multiclass"

    return pd.DataFrame(X, columns=feature_cols), y, feature_cols, y_encoder, scaler, task


# ============================================================
# SAVE TRAIN/TEST SPLITS
# ============================================================

def save_processed_split(
    dataset: str,
    task: str,
    Xtr: pd.DataFrame,
    ytr: np.ndarray,
    Xte: pd.DataFrame,
    yte: np.ndarray
) -> Tuple[str, str]:
    """Save training/testing split to CSV with safe naming."""
    ensure_dir(DIR_PROCESSED)
    base_train = os.path.join(DIR_PROCESSED, f"{dataset}_{task}_train.csv")
    base_test = os.path.join(DIR_PROCESSED, f"{dataset}_{task}_test.csv")

    path_train = unique_path(base_train)
    path_test = unique_path(base_test)

    df_tr = Xtr.copy()
    df_tr["__label__"] = ytr
    df_te = Xte.copy()
    df_te["__label__"] = yte

    df_tr.to_csv(path_train, index=False)
    df_te.to_csv(path_test, index=False)

    return path_train, path_test


def list_processed_pairs() -> List[Tuple[str, str, str]]:
    """
    List available processed (train/test) pairs in data/processed.
    Returns: List of (dataset_task, train_path, test_path)
    """
    if not os.path.isdir(DIR_PROCESSED):
        return []

    files = sorted([f for f in os.listdir(DIR_PROCESSED) if f.endswith(".csv")])
    groups = {}

    for f in files:
        if f.endswith("_train.csv"):
            key = f[:-10]
            groups.setdefault(key, [None, None])
            groups[key][0] = os.path.join(DIR_PROCESSED, f)
        elif f.endswith("_test.csv"):
            key = f[:-9]
            groups.setdefault(key, [None, None])
            groups[key][1] = os.path.join(DIR_PROCESSED, f)

    result = []
    for key, (tr, te) in groups.items():
        if tr and te:
            result.append((key, tr, te))
    return result
