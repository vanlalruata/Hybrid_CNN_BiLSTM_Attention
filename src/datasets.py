"""
datasets.py — Dataset loading and preprocessing module
Author: Dr. Vanlalruata Hnamte
Version: 1.0
"""

import os
import glob
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from .utils import ensure_dir, save_scaler, set_all_seeds


AVAILABLE_DATASETS = {
    "EDGE_IIoT": "H:/Datasets/Edge-IIoT/Selected dataset for ML and DL/*.csv",
    "CIC-IoT-2023": "data/raw/CICIoT2023/*.csv",
    "Apose-IoT-23": "data/raw/AposeIoT23/*.csv",
    "CIC-IoMT-2024": "data/raw/CICIoMT2024/*.csv",
    "CIC-IoT-IDAD-2024": "data/raw/CIC-IoT-IDAD-2024/*.csv",
    "CIC-IoT-2025": "data/raw/CICIoT2025/*.csv",
    "BoT-IoT": "data/raw/BoT-IoT/*.csv"
}

COMMON_DROP = [
    "Flow ID", "Src IP", "Dst IP", "Timestamp", "saddr", "daddr",
    "id.orig_h", "id.resp_h", "uid", "frame.time"
]

LABEL_COLS = [
    "Label", "label", "Attack_type", "Attack_label", "attack", "category", "subcategory"
]


def _find_label_column(df: pd.DataFrame):
    for col in LABEL_COLS:
        if col in df.columns:
            return col
    raise ValueError("No label column found.")


def _drop_identifiers(df: pd.DataFrame):
    drop_cols = [c for c in COMMON_DROP if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    return df


def _encode_non_numeric(df: pd.DataFrame):
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype(str)
            df[col] = LabelEncoder().fit_transform(df[col])
    return df


def load_and_prepare_dataset(dataset_name: str, class_type="binary"):
    """Load, preprocess, and return standardized (X, y, feature_names, encoder, scaler)."""
    pattern = AVAILABLE_DATASETS.get(dataset_name)
    if not pattern:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No CSV files for {dataset_name} at {pattern}")

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, low_memory=False)
            dfs.append(df)
        except Exception as e:
            print(f"[WARN] Could not load {f}: {e}")
    df = pd.concat(dfs, ignore_index=True)

    # Find label & drop identifiers
    label_col = _find_label_column(df)
    y_raw = df[label_col].astype(str)
    df = df.drop(columns=[label_col])
    df = _drop_identifiers(df)
    df = _encode_non_numeric(df)

    # Clean numeric matrix and align lengths
    df = df.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
    features = df.columns.tolist()
    X = df.values

    # Classification handling:
    # - If class_type == "binary", collapse labels to {Benign, Attack}
    # - Else (multiclass), retain original labels
    if (class_type or "").lower() == "binary":
        def _to_binary(lbl: str) -> str:
            s = str(lbl).strip().lower()
            # Treat anything that contains "benign" or "normal" as Benign
            if "benign" in s or "normal" in s:
                return "Benign"
            return "Attack"
        y_labels = y_raw.iloc[:len(X)].apply(_to_binary).values if hasattr(y_raw, "iloc") else np.array([_to_binary(v) for v in y_raw[:len(X)]])
    else:
        y_labels = y_raw.iloc[:len(X)].values if hasattr(y_raw, "iloc") else np.array(y_raw[:len(X)])

    y_enc = LabelEncoder()
    y = y_enc.fit_transform(y_labels)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y, features, y_enc, scaler


def split_and_persist(
    dataset_name: str,
    X, y, feature_names, y_encoder, scaler,
    test_size=0.2, random_state=42,
    processed_root="data/processed",
    class_type: str = None
):
    """Split dataset, persist train/test + scaler for reuse.

    Folder naming policy (no timestamps):
      data/processed/<dataset_name>_<binary|multiclass>/
    """
    set_all_seeds(random_state)
    # Determine classification type if not explicitly provided
    inferred_class = "binary" if len(np.unique(y)) == 2 else "multiclass"
    class_type = class_type or inferred_class

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Folder
    suffix = f"{dataset_name}_{class_type}"
    save_dir = os.path.join(processed_root, suffix)
    ensure_dir(save_dir)

    # Persist splits (filenames already signal train/test role)
    np.save(os.path.join(save_dir, "X_train.npy"), X_train)
    np.save(os.path.join(save_dir, "y_train.npy"), y_train)
    np.save(os.path.join(save_dir, "X_test.npy"), X_test)
    np.save(os.path.join(save_dir, "y_test.npy"), y_test)

    save_scaler(scaler, os.path.join(save_dir, "scaler.joblib"))
    meta = {
        "dataset_name": dataset_name,
        "feature_names": feature_names,
        "num_classes": int(len(np.unique(y))),
        "class_type": class_type,
        "input_dim": int(X.shape[1]),
        "class_names": list(y_encoder.classes_)
    }
    with open(os.path.join(save_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    return save_dir


def load_processed_split(split_key: str, processed_root="data/processed"):
    """Load previously split train/test data and metadata."""
    folder = os.path.join(processed_root, split_key)
    if not os.path.exists(folder):
        raise FileNotFoundError(f"Split not found: {folder}")

    X_train = np.load(os.path.join(folder, "X_train.npy"))
    y_train = np.load(os.path.join(folder, "y_train.npy"))
    X_test = np.load(os.path.join(folder, "X_test.npy"))
    y_test = np.load(os.path.join(folder, "y_test.npy"))
    meta = json.load(open(os.path.join(folder, "meta.json")))

    return {"X_train": X_train, "y_train": y_train,
            "X_test": X_test, "y_test": y_test, "meta": meta}
