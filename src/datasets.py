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
from .config import FORCE_DROP, EXTRA_DROP, LABEL_COLUMNS, DATASETS, COMMON_DROP


AVAILABLE_DATASETS = DATASETS

COMMON_DROP = [c for c in COMMON_DROP if c] # Use shared list from config

LABEL_COLS = [
    "Label", "label", "Attack_type", "Attack_label", "attack", "category", "subcategory", "label2"
    #"Label", "subcategory"
]


def _read_csv_auto(path: str, nrows: int = None) -> pd.DataFrame:
    """Read CSV with auto-detected delimiter (supports ',' and ';')."""
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        sample = fh.read(8192)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=[",", ";", "|"])
            sep = dialect.delimiter
        except Exception:
            # Fallback heuristic
            sep = ";" if sample.count(";") > sample.count(",") else ","
    return pd.read_csv(path, sep=sep, low_memory=False, nrows=nrows)


def _find_label_column(df: pd.DataFrame, dataset_name: str = None):
    # Try per-dataset overrides from config first
    if dataset_name and dataset_name in LABEL_COLUMNS:
        for col in LABEL_COLUMNS[dataset_name]:
            if col in df.columns:
                return col
    # Fallback to general list
    for col in LABEL_COLS:
        if col in df.columns:
            return col
    raise ValueError(f"No label column found. Checked columns: {LABEL_COLS}")


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
            # If the file is extremely large (> 1 GB), load only a subset of rows to avoid memory overhead and long load times
            nrows = 1000000 if os.path.getsize(f) > 1024 * 1024 * 1024 * 10 else None
            df = _read_csv_auto(f, nrows=nrows)

            # Normalize/rename label column to 'Label' per file
            try:
                label_col = _find_label_column(df, dataset_name=dataset_name)
                if label_col != "Label":
                    # If 'Label' already exists, drop it to avoid duplicate columns after rename
                    if "Label" in df.columns:
                        df = df.drop(columns=["Label"])
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
            lbl_col = _find_label_column(d, dataset_name=dataset_name)
            if lbl_col != "Label":
                d = d.rename(columns={lbl_col: "Label"})
        aligned.append(d.reindex(columns=selected_cols))
    df = pd.concat(aligned, ignore_index=True)

    # Find a label & normalize its name to 'Label'
    label_col = _find_label_column(df, dataset_name=dataset_name)
    if label_col != "Label":
        if "Label" in df.columns:
            df = df.drop(columns=["Label"])
        df = df.rename(columns={label_col: "Label"})
        label_col = "Label"
    
    # Ensure y_raw is a 1D Series even if duplicate columns somehow persisted
    raw_labels = df[label_col]
    if isinstance(raw_labels, pd.DataFrame):
        # Take the first one if multiple exist
        y_raw = raw_labels.iloc[:, 0].astype(str)
    else:
        y_raw = raw_labels.astype(str)

    df = df.drop(columns=[label_col])
    df = _drop_identifiers(df)
    # Drop dataset-specific extra columns
    extra_drop = EXTRA_DROP.get(dataset_name, [])
    if extra_drop:
        # Use info instead of print to avoid confusing user
        # print(f"[CLEAN] Extra-dropped columns for {dataset_name}: {drop_extra}")
        drop_extra = [c for c in extra_drop if c in df.columns]
        if drop_extra:
            df = df.drop(columns=drop_extra)
    # Force-drop per-dataset configured columns (even if numeric)
    extra_forced = FORCE_DROP.get(dataset_name, [])
    if extra_forced:
        # print(f"[CLEAN] Force-dropped columns for {dataset_name}: {drop_forced}")
        drop_forced = [c for c in extra_forced if c in df.columns]
        if drop_forced:
            df = df.drop(columns=drop_forced)
    df = _encode_non_numeric(df)

    # Clean numeric matrix and robustly handle missing values
    df = df.replace([np.inf, -np.inf], np.nan)

    # Impute numeric columns with median; fallback to 0 if median is NaN
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    import warnings
    for c in num_cols:
        if df[c].isna().any():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                # Check if the column is all NaNs to avoid the Mean of empty slice warning
                if df[c].isna().all():
                    med = 0
                else:
                    med = pd.to_numeric(df[c], errors="coerce").median(skipna=True)
            
            if pd.isna(med) or not np.isfinite(med):
                med = 0
            df[c] = df[c].fillna(med)

    # Final safety: if any NaNs remain (from non-numeric edge cases), fill with 0
    if df.isna().any().any():
        df = df.fillna(0)

    # Drop exact duplicate rows in features
    df = df.drop_duplicates()

    # Stratified downsampling if the dataset is too large to prevent extremely long training times
    max_samples = 10000000
    if len(df) > max_samples:
        print(f"[INFO] Dataset has {len(df)} rows after removing duplicates. Downsampling to {max_samples} for efficiency.")
        y_temp = y_raw.loc[df.index] if hasattr(y_raw, "loc") else np.array(y_raw)[:len(df)]
        y_temp = pd.Series(y_temp).astype(str)
        unique_labels, label_counts = np.unique(y_temp, return_counts=True)
        if len(unique_labels) > 1 and np.min(label_counts) >= 2:
            try:
                _, df = train_test_split(
                    df, test_size=max_samples, stratify=y_temp, random_state=42
                )
            except Exception as e:
                print(f"[WARN] Stratified downsampling failed: {e}. Falling back to random sampling.")
                df = df.sample(n=max_samples, random_state=42)
        else:
            df = df.sample(n=max_samples, random_state=42)

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

    return X, y, features, y_enc, None


def split_and_persist(
    dataset_name: str,
    X, y, feature_names, y_encoder, scaler,
    test_size=0.15, random_state=42,
    processed_root="data/processed",
    class_type: str = None
):
    """Split dataset into train (70%), validation (15%), and test (15%) and persist.

    Folder naming policy (no timestamps):
      data/processed/<dataset_name>_<binary|multiclass>/
    """
    set_all_seeds(random_state)
    inferred_class = "binary" if len(np.unique(y)) == 2 else "multiclass"
    class_type = class_type or inferred_class

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
        if np.min(cnts) < 2:
            return False
        if np.any(cnts * ts < 1) or np.any(cnts * (1 - ts) < 1):
            return False
        return True

    # 1. Split off test set (15%)
    if _can_stratify(counts, 0.15):
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.15, stratify=y, random_state=random_state
        )
    else:
        print("[WARN] Stratified split for test set not feasible. Falling back to non-stratified.")
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.15, stratify=None, random_state=random_state
        )

    # 2. Split remainder into train (70%) and validation (15%)
    # 0.15 / 0.85 = 0.17647
    temp_unique, temp_counts = np.unique(y_temp, return_counts=True)
    if _can_stratify(temp_counts, 0.17647):
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.17647, stratify=y_temp, random_state=random_state
        )
    else:
        print("[WARN] Stratified split for validation set not feasible. Falling back to non-stratified.")
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.17647, stratify=None, random_state=random_state
        )

    # 3. Fit scaler ONLY on training data, then transform all partitions
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Folder
    suffix = f"{dataset_name}_{class_type}"
    save_dir = os.path.join(processed_root, suffix)
    ensure_dir(save_dir)

    # Persist splits
    np.save(os.path.join(save_dir, "X_train.npy"), X_train)
    np.save(os.path.join(save_dir, "y_train.npy"), y_train)
    np.save(os.path.join(save_dir, "X_val.npy"), X_val)
    np.save(os.path.join(save_dir, "y_val.npy"), y_val)
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
    """Load previously split train/validation/test data and metadata."""
    folder = os.path.join(processed_root, split_key)
    if not os.path.exists(folder):
        raise FileNotFoundError(f"Split not found: {folder}")

    X_train = np.load(os.path.join(folder, "X_train.npy"))
    y_train = np.load(os.path.join(folder, "y_train.npy"))
    X_val = np.load(os.path.join(folder, "X_val.npy"))
    y_val = np.load(os.path.join(folder, "y_val.npy"))
    X_test = np.load(os.path.join(folder, "X_test.npy"))
    y_test = np.load(os.path.join(folder, "y_test.npy"))
    meta = json.load(open(os.path.join(folder, "meta.json")))

    return {"X_train": X_train, "y_train": y_train,
            "X_val": X_val, "y_val": y_val,
            "X_test": X_test, "y_test": y_test, "meta": meta}
