"""
datasets.py — Dataset loading and preprocessing module
Author: Dr. Vanlalruata Hnamte
Version: 1.0
"""

import os
import glob
import json
import csv
import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from .utils import ensure_dir, save_scaler, set_all_seeds


AVAILABLE_DATASETS = {
    "EDGE_IIoT": "H:/Datasets/Edge-IIoT/Selected dataset for ML and DL/*.csv",
    "CIC-IoT-2023": "H:/Datasets/CIC-IoT-2023/*.csv",
    "Apose-IoT-23": "H:/Datasets/Aposemat-IoT-23/aposemat_iot_23/combined/*.csv",
    "CIC-IoMT-2024": "H:/Datasets/CIC-IoMT-2024/WiFi_MQTT/**/*.csv",
    "CIC-IoT-IDAD-2024": "H:/Datasets/CIC-IoT-IDAD-2024/Flow_Based/*.csv",
    "CIC-IoT-2025": "H:/Datasets/CIC-IoT-2025/all_attack_benign_samples/*.csv",
    "BoT-IoT": "H:/Datasets/BoT-IoT/*.csv"
}

COMMON_DROP = [
    "Flow ID", "Src IP", "Dst IP", "Timestamp", "saddr", "daddr",
    "id.orig_h", "id.resp_h", "uid", "frame.time"
]

LABEL_COLS = [
    "Label", "label", "Attack_type", "Attack_label", "attack", "category", "subcategory", "label2"
]


def _read_csv_auto(path: str) -> pd.DataFrame:
    """Read CSV with auto-detected delimiter (supports ',' and ';')."""
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        sample = fh.read(8192)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=[",", ";"])
            sep = dialect.delimiter
        except Exception:
            # Fallback heuristic
            sep = ";" if sample.count(";") > sample.count(",") else ","
    return pd.read_csv(path, sep=sep, low_memory=False)


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
            df = _read_csv_auto(f)

            # Normalize/rename label column to 'Label' per file
            try:
                label_col = _find_label_column(df)
                if label_col != "Label":
                    df = df.rename(columns={label_col: "Label"})
            except Exception:
                # If no label found here, combined logic will raise later
                pass

            # If label needs manual assignment, derive from filename prefix
            if "Label" in df.columns:
                mask = df["Label"].astype(str).str.strip() == "NeedManualLabel"
                if mask.any():
                    stem = os.path.splitext(os.path.basename(f))[0]
                    # Split on -, _, space, or dot and take the first token
                    tokens = re.split(r"[-_\s.]+", stem)
                    first_token = tokens[0] if tokens and tokens[0] else stem
                    df.loc[mask, "Label"] = first_token

            dfs.append(df)
        except Exception as e:
            print(f"[WARN] Could not load {f}: {e}")
    df = pd.concat(dfs, ignore_index=True)

    # Find label & normalize its name to 'Label'
    label_col = _find_label_column(df)
    if label_col != "Label":
        df = df.rename(columns={label_col: "Label"})
        label_col = "Label"
    y_raw = df[label_col].astype(str)
    df = df.drop(columns=[label_col])
    df = _drop_identifiers(df)
    df = _encode_non_numeric(df)

    # Clean numeric matrix and align lengths
    df = df.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
    # Drop exact duplicate rows in features
    df = df.drop_duplicates()
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

    # Also persist CSV copies with a unified comma delimiter for interoperability
    pd.DataFrame(X_train, columns=feature_names).to_csv(os.path.join(save_dir, "X_train.csv"), index=False)
    pd.DataFrame({"Label": y_train}).to_csv(os.path.join(save_dir, "y_train.csv"), index=False)
    pd.DataFrame(X_test, columns=feature_names).to_csv(os.path.join(save_dir, "X_test.csv"), index=False)
    pd.DataFrame({"Label": y_test}).to_csv(os.path.join(save_dir, "y_test.csv"), index=False)

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
