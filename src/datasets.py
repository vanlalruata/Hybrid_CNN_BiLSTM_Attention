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
from .config import FORCE_DROP, EXTRA_DROP


AVAILABLE_DATASETS = {
    # "EDGE_IIoT": "H:/Datasets/Edge-IIoT/Selected dataset for ML and DL/*.csv",
    # "CIC-IoT-2023": "H:/Datasets/CIC-IoT-2023/*.csv",
    "Apose-IoT-23": "H:/Datasets/Aposemat-IoT-23/aposemat_iot_23/processed/*.csv",
    "BoT-IoT": "H:/Datasets/BoT-IoT_csvs/*.csv",
    "CIC-IoMT-2024": "H:/Datasets/CIC-IoMT-2024/WiFi_MQTT/**/*.csv",
    # "CIC-IoT-IDAD-2024": "H:/Datasets/CIC-IoT-IDAD-2024/Flow_Based/*.csv",
    "CIC-IoT-2025": "H:/Datasets/CIC-IoT-2025/all_attack_benign_samples/*.csv"
}

COMMON_DROP = [
    "Flow ID", "Src IP", "Dst IP", "Timestamp", "saddr", "daddr",
    "id.orig_h", "id.resp_h", "uid", "frame.time"
]

LABEL_COLS = [
    "Label", "label", "Attack_type", "Attack_label", "attack", "category", "subcategory", "label2"
    #"Label", "subcategory"
]


def _read_csv_auto(path: str) -> pd.DataFrame:
    """Read CSV with auto-detected delimiter (supports ',' and ';')."""
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        sample = fh.read(8192)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=[",", ";", "|"])
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
    common_features = None
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

            # Track the intersection of feature columns across files (excluding Label)
            cur_features = [c for c in df.columns if c != "Label"]
            if common_features is None:
                common_features = set(cur_features)
            else:
                common_features &= set(cur_features)

            dfs.append(df)
        except Exception as e:
            print(f"[WARN] Could not load {f}: {e}")

    if not dfs:
        raise FileNotFoundError(f"No valid CSV files could be loaded for {dataset_name} at {pattern}")

    # Align frames to the common feature set to avoid NaNs from column mismatches
    if not common_features:
        raise ValueError("No common feature columns found across CSV files. Please ensure files share a core schema.")
    selected_cols = sorted(list(common_features)) + ["Label"]
    aligned = []
    for d in dfs:
        # Ensure 'Label' is present after earlier normalization
        if "Label" not in d.columns:
            lbl_col = _find_label_column(d)
            if lbl_col != "Label":
                d = d.rename(columns={lbl_col: "Label"})
        aligned.append(d.reindex(columns=selected_cols))
    df = pd.concat(aligned, ignore_index=True)

    # Find a label & normalize its name to 'Label'
    label_col = _find_label_column(df)
    if label_col != "Label":
        df = df.rename(columns={label_col: "Label"})
        label_col = "Label"
    y_raw = df[label_col].astype(str)
    df = df.drop(columns=[label_col])
    df = _drop_identifiers(df)
    # Drop dataset-specific extra columns
    extra_drop = EXTRA_DROP.get(dataset_name, [])
    if extra_drop:
        drop_extra = [c for c in extra_drop if c in df.columns]
        if drop_extra:
            df = df.drop(columns=drop_extra)
            print(f"[CLEAN] Extra-dropped columns for {dataset_name}: {drop_extra}")
    # Force-drop per-dataset configured columns (even if numeric)
    extra_forced = FORCE_DROP.get(dataset_name, [])
    if extra_forced:
        drop_forced = [c for c in extra_forced if c in df.columns]
        if drop_forced:
            df = df.drop(columns=drop_forced)
            print(f"[CLEAN] Force-dropped columns for {dataset_name}: {drop_forced}")
    df = _encode_non_numeric(df)

    # Clean numeric matrix and robustly handle missing values
    df = df.replace([np.inf, -np.inf], np.nan)

    # Impute numeric columns with median; fallback to 0 if median is NaN
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for c in num_cols:
        if df[c].isna().any():
            med = pd.to_numeric(df[c], errors="coerce").median(skipna=True)
            if pd.isna(med) or not np.isfinite(med):
                med = 0
            df[c] = df[c].fillna(med)

    # Final safety: if any NaNs remain (from non-numeric edge cases), fill with 0
    if df.isna().any().any():
        df = df.fillna(0)

    # Drop exact duplicate rows in features
    df = df.drop_duplicates()
    features = df.columns.tolist()
    X = df.values

    # Classification handling:
    # - If class_type == "binary", collapse labels to {Benign, Attack}
    # - Else (multiclass), retain original labels
    # Align labels to the cleaned feature rows using index alignment
    y_series = y_raw.loc[df.index] if hasattr(y_raw, "loc") else np.array(y_raw)[:len(df)]
    if (class_type or "").lower() == "binary":
        def _to_binary(lbl: str) -> str:
            s = str(lbl).strip().lower()
            # Normalize common binary encodings and keywords
            if s in {"0", "false", "no"} or "benign" in s or "normal" in s:
                return "Benign"
            if s in {"1", "true", "yes"} or "attack" in s or "malicious" in s or "ddos" in s or "dos" in s or "scan" in s or "theft" in s or "exfil" in s or "keylog" in s:
                return "Attack"
            # Default fallback: conservative choice
            return "Attack"
        y_labels = y_series.apply(_to_binary).values if hasattr(y_series, "apply") else np.array([_to_binary(v) for v in y_series])
    else:
        y_labels = y_series.values if hasattr(y_series, "values") else np.array(y_series)

    y_enc = LabelEncoder()
    y = y_enc.fit_transform(y_labels)

    if X.shape[0] == 0:
        raise ValueError("No samples remain after preprocessing; please review preprocessing rules.")

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

    # Report class distribution and decide safe stratification
    unique, counts = np.unique(y, return_counts=True)
    try:
        classes = list(y_encoder.classes_)
        distribution = {str(classes[i]): int(np.sum(y == i)) for i in range(len(classes))}
        print(f"[INFO] Class distribution: {distribution}")
    except Exception:
        print(f"[INFO] Class distribution (encoded): {dict(zip(unique.tolist(), counts.tolist()))}")

    def _can_stratify(cnts: np.ndarray, ts: float) -> bool:
        if cnts.size == 0:
            return False
        # sklearn requires at least 2 samples for the least populated class
        if np.min(cnts) < 2:
            return False
        # Ensure each class will have at least one sample in both splits
        if np.any(cnts * ts < 1) or np.any(cnts * (1 - ts) < 1):
            return False
        return True

    if _can_stratify(counts, test_size):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=random_state
        )
    else:
        min_count = int(counts.min()) if counts.size else 0
        print(f"[WARN] Stratified split not feasible (min class count={min_count}, test_size={test_size}). Falling back to non-stratified split.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=None, random_state=random_state
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
    # pd.DataFrame(X_train, columns=feature_names).to_csv(os.path.join(save_dir, "X_train.csv"), index=False)
    # pd.DataFrame({"Label": y_train}).to_csv(os.path.join(save_dir, "y_train.csv"), index=False)
    # pd.DataFrame(X_test, columns=feature_names).to_csv(os.path.join(save_dir, "X_test.csv"), index=False)
    # pd.DataFrame({"Label": y_test}).to_csv(os.path.join(save_dir, "y_test.csv"), index=False)

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
