"""
Main Orchestrator CLI for IoT/IIoT IDS Experiments
Author: Dr. Vanlalruata Hnamte
Version: 1.0

Features
--------
1) Load & split datasets (persist splits with dataset-prefixed names)
2) Load existing splits
3) Visualize features (histograms, KDEs, heatmaps) → EPS
4) Train models (Binary or Multiclass) with CNN+LSTM Fusion (or fallback MLP)
   - Saves checkpoints, learning curves, per-epoch CSV history
   - Logs timing & estimated complexity proxies
5) Evaluate (Confusion matrix, ROC-AUC, PR curves, metrics tables)
6) XAI (SHAP, LIME, ANOVA)
7) Ablation (user-selected feature subsets)
8) Inference (load saved model & run on new/live data)
9) Non-interactive subcommands with argparse (automation-friendly)

Notes
-----
- GPU is auto-detected via torch.cuda.is_available()
- All plots are saved as .eps for publication-quality figures
- Energy/Resource efficiency is reported via a transparent power proxy
- Time complexity is recorded as: params, forward MACs estimates (best-effort)
"""

import os
import sys
import json
import glob
import argparse
import time
import numpy as np
import pandas as pd

# Matplotlib (headless EPS saving)
import matplotlib
matplotlib.use("Agg")

import torch

# -----------------------------
# Local imports from /src
# -----------------------------
try:
    from HLC.src.datasets import (
        AVAILABLE_DATASETS,
        load_and_prepare_dataset,
        split_and_persist,
        load_processed_split
    )
    from HLC.src.models import CNN_LSTM_Fusion, SimpleMLP
    from HLC.src.train import train_experiment                        # returns ckpt_path, history_csv, artifacts_dir
    from HLC.src.evaluate import evaluate_on_test                     # classification/regression entrypoint
    from HLC.src.inference import load_model, predict_live
    from HLC.src.xai import run_xai                                   # shap/lime/anova
    from HLC.src.ablation import run_ablation_user_selected           # ablation on a given feature subset
    from HLC.src.viz import plot_dataset_overview                     # saves EPS figs for distributions/heatmaps
    from HLC.src.utils import set_all_seeds, ensure_dir
    from HLC.src.config import DIR_OUTPUTS as CFG_OUTPUTS, DIR_PROCESSED as CFG_PROCESSED
except Exception as e:
    print("[ERROR] Ensure you have all required src modules. Import failed:", e)
    sys.exit(1)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
POWER_WATTS = float(os.environ.get("POWER_WATTS", "120"))
# Use configured root paths from config.py
DEFAULT_OUTROOT = CFG_OUTPUTS
# DEFAULT_DATAROOT is the parent of the configured processed dir (…/data)
DEFAULT_DATAROOT = os.path.dirname(CFG_PROCESSED)

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _prompt(msg, default=None, choices=None):
    # Flush sys.stdin to clear any stray input before prompting the user
    if sys.stdin.isatty():
        import msvcrt
        while msvcrt.kbhit():
            msvcrt.getch()
    while True:
        try:
            val = input(f"{msg} " + (f"[default: {default}] " if default else "") + (f"choices={choices} " if choices else "")).strip()
        except EOFError:
            return default if default is not None else ""
        if not val and default is not None:
            return default
        if choices and val not in choices:
            print(f"Invalid choice. Allowed: {choices}")
            continue
        return val

def list_processed_splits(processed_root=f"{DEFAULT_DATAROOT}/processed"):
    if not os.path.isdir(processed_root):
        return []
    cands = sorted([d for d in os.listdir(processed_root) if os.path.isdir(os.path.join(processed_root, d))])
    return cands

def _safe_json_dump(obj, path):
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def estimate_model_complexity(model, input_dim: int, seq_len: int = 1):
    """
    Best-effort lightweight complexity proxy:
    - number of parameters
    - rough Macs per forward pass (dense + conv + lstm estimates)
    - does not account for activation/batchnorm/dropout ops
    """
    params = sum(p.numel() for p in model.parameters())
    macs = 0

    # Try introspecting layers
    for name, module in model.named_modules():
        # Dense
        if hasattr(module, "weight") and module.__class__.__name__.lower().startswith("linear"):
            w = module.weight.shape  # [out, in]
            macs += 2 * w[0] * w[1]
        # Conv1d
        if module.__class__.__name__.lower() == "conv1d":
            # approx: 2 * Cout * Cin * K * L
            c_out, c_in, k = module.weight.shape[:3]
            L = seq_len
            macs += 2 * c_out * c_in * k * L
        # LSTM
        if module.__class__.__name__.lower() == "lstm":
            # macs ~ 4 * (I*H + H*H + H) * T   (I: input size, H: hidden size, T: seq_len)
            i = module.input_size
            h = module.hidden_size
            T = seq_len
            macs += 4 * (i * h + h * h + h) * T

    return {"params": int(params), "est_macs_per_sample": int(macs)}


def _measure_infer_time(model, X: np.ndarray, batch_size: int = 512):
    """
    Measure end-to-end inference time on a feature matrix X using current DEVICE.
    Returns timing metrics for complexity reporting.
    """
    from torch.utils.data import DataLoader, TensorDataset
    model.eval()
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, pin_memory=torch.cuda.is_available())

    if torch.cuda.is_available(): torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(DEVICE)
            _ = model(xb)
    if torch.cuda.is_available(): torch.cuda.synchronize()
    t1 = time.perf_counter()

    total_time = t1 - t0
    batches = max(1, len(loader))
    n_samples = int(X.shape[0])
    ms_per_batch = (total_time / batches) * 1000.0
    ms_per_sample = (total_time / max(1, n_samples)) * 1000.0
    samples_per_sec = (n_samples / total_time) if total_time > 0 else float("inf")
    energy_j = total_time * POWER_WATTS

    return {
        "total_time_sec": float(total_time),
        "batches": int(batches),
        "samples": n_samples,
        "ms_per_batch": float(ms_per_batch),
        "ms_per_sample": float(ms_per_sample),
        "samples_per_sec": float(samples_per_sec),
        "energy_j": float(energy_j),
    }

# ---------------------------------------------------------------------
# Option 1: Load & split dataset
# ---------------------------------------------------------------------

def opt_load_and_split():
    print("\n[1] Load & split a dataset")
    print("Available datasets:", list(AVAILABLE_DATASETS.keys()))
    dname = _prompt("Dataset name?", choices=list(AVAILABLE_DATASETS.keys()))
    class_type = _prompt("Classification type? (binary/multiclass)", default="binary", choices=["binary","multiclass"])

    print(f"[INFO] Loading {dname} ({class_type})...")
    # Show all CSV files that will be loaded for transparency (names with extension)
    pattern = AVAILABLE_DATASETS.get(dname)
    csv_files = sorted(glob.glob(pattern)) if pattern else []
    if not csv_files:
        print(f"[WARN] No CSV files matched pattern: {pattern}")
    else:
        print("[INFO] Files to load:")
        for p in csv_files:
            print(f"  - {os.path.basename(p)}")
    X, y, feat_cols, y_enc, scaler = load_and_prepare_dataset(dname, class_type=class_type)

    # Persist split with a dataset prefix
    print("\n--- Split configuration ---")
    test_size = float(_prompt("Test size (e.g., 0.2)?", default="0.2"))
    seed = int(_prompt("Random seed?", default="42"))
    split_info = split_and_persist(
        dataset_name=dname,
        X=X, y=y, feature_names=feat_cols,
        y_encoder=y_enc, scaler=scaler,
        test_size=test_size, random_state=seed,
        processed_root=f"{DEFAULT_DATAROOT}/processed",
        class_type=class_type
    )
    print("[DONE] Split saved:", split_info)
    return 0

# ---------------------------------------------------------------------
# Option 2: Load existing split (no action, just validation)
# ---------------------------------------------------------------------

def opt_load_existing_split():
    print("\n[2] Load an existing processed split")
    splits = list_processed_splits()
    if not splits:
        print("[WARN] No processed splits found. Run option 1 first.")
        return 0

    print("Available processed splits:", splits)
    key = _prompt("Enter split folder name (from above):")
    bundle = load_processed_split(key, processed_root=f"{DEFAULT_DATAROOT}/processed")
    Xtr, ytr, Xte, yte, meta = bundle["X_train"], bundle["y_train"], bundle["X_test"], bundle["y_test"], bundle["meta"]

    print(f"[OK] Loaded split '{key}' → shapes: Xtr={Xtr.shape}, Xte={Xte.shape}, classes={len(np.unique(ytr))}")
    return 0

# ---------------------------------------------------------------------
# Option 3: Visualize dataset features (EPS)
# ---------------------------------------------------------------------

def opt_visualize_dataset():
    print("\n[3] Visualize dataset features")
    mode = _prompt("Use (raw) dataset or (processed) split? (raw/processed)", default="processed", choices=["raw","processed"])

    if mode == "raw":
        print("Available datasets:", list(AVAILABLE_DATASETS.keys()))
        dname = _prompt("Dataset name?", choices=list(AVAILABLE_DATASETS.keys()))
        class_type = _prompt("Classification type? (binary/multiclass)", default="binary", choices=["binary","multiclass"])
        X, y, feat_cols, y_enc, scaler = load_and_prepare_dataset(dname, class_type=class_type)
        outdir = os.path.join(DEFAULT_OUTROOT, f"{dname}_{class_type}", "viz")
        # Pass class_names so class_distribution shows human-readable labels (not 0,1,2,...)
        plot_dataset_overview(X, y, feat_cols, outdir=outdir, class_names=list(y_enc.classes_))  # saves EPS
        print(f"[DONE] Visualizations saved to {outdir}")
    else:
        splits = list_processed_splits()
        if not splits:
            print("[WARN] No processed splits found.")
            return 0
        print("Available processed splits:", splits)
        key = _prompt("Split key:")
        bundle = load_processed_split(key, processed_root=f"{DEFAULT_DATAROOT}/processed")
        Xtr, ytr, Xte, yte, meta = bundle["X_train"], bundle["y_train"], bundle["X_test"], bundle["y_test"], bundle["meta"]
        outdir = os.path.join(DEFAULT_OUTROOT, f"{key}", "viz")
        # Use meta['class_names'] for readable class labels in plots
        plot_dataset_overview(
            np.vstack([Xtr, Xte]),
            np.hstack([ytr, yte]),
            meta["feature_names"],
            outdir=outdir,
            class_names=meta.get("class_names")
        )
        print(f"[DONE] Visualizations saved to {outdir}")
    return 0

# ---------------------------------------------------------------------
# Option 4: Train Model (Binary/Multiclass)
# ---------------------------------------------------------------------

def opt_train_model():
    print("\n[4] Train a model")
    splits = list_processed_splits()
    if not splits:
        print("[WARN] No processed splits found. Run option 1 first.")
        return 0
    print("Available splits:", splits)
    key = _prompt("Split key (e.g., EDGE_IIoT_binary_2025-10-29_08-30-00):")
    bundle = load_processed_split(key, processed_root=f"{DEFAULT_DATAROOT}/processed")
    Xtr, ytr, Xte, yte, meta = bundle["X_train"], bundle["y_train"], bundle["X_test"], bundle["y_test"], bundle["meta"]

    # Confirm a classification basis
    ncls = len(np.unique(ytr))
    class_type = "binary" if ncls == 2 else "multiclass"
    print(f"[INFO] Detected classes={ncls} → {class_type}")

    # Train
    epochs = int(_prompt("Epochs?", default="50"))
    batch = int(_prompt("Batch size?", default="256"))
    lr = float(_prompt("Learning rate?", default="0.001"))
    patience = int(_prompt("Early stop patience?", default="8"))
    model_choice = _prompt("Model (fusion/dnn)?", default="fusion", choices=["fusion","dnn"])
    suffix = "fusion" if model_choice == "fusion" else "dnn"

    ckpt_path, hist_csv, artifacts_dir, model = train_experiment(
        Xtr, ytr, Xte, yte,
        model_name=("cnn_lstm_fusion" if model_choice=="fusion" else "dnn"),
        dataset_key=key,
        out_root=DEFAULT_OUTROOT,
        epochs=epochs, batch_size=batch, lr=lr, early_stop_patience=patience,
        class_names=meta["class_names"]
    )

    # Complexity snapshot
    complexity = estimate_model_complexity(model, input_dim=meta["input_dim"], seq_len=meta.get("seq_len", 1))

    # Measure inference/detection time on validation (test) split for complexity report
    try:
        infer = _measure_infer_time(model, Xte, batch_size=batch)
        complexity.update({
            "measured_infer_total_sec": infer["total_time_sec"],
            "measured_ms_per_batch": infer["ms_per_batch"],
            "measured_ms_per_sample": infer["ms_per_sample"],
            "measured_samples_per_sec": infer["samples_per_sec"],
            "measured_energy_j_proxy": infer["energy_j"],
            "measured_batches": infer["batches"],
            "measured_samples": infer["samples"],
            # Alias emphasizing detection time on validation (same pass)
            "val_detect_total_sec": infer["total_time_sec"],
            "val_detect_ms_per_sample": infer["ms_per_sample"]
        })
    except Exception as e:
        print(f"[WARN] Failed to measure inference time for complexity: {e}")

    _safe_json_dump(complexity, os.path.join(artifacts_dir, f"complexity_{suffix}.json"))
    print(f"[DONE] Checkpoint: {ckpt_path}")
    print(f"       History CSV: {hist_csv}")
    print(f"       Complexity: {complexity}")
    return 0

# ---------------------------------------------------------------------
# Option 5: Plot output performance (Evaluation figs)
# ---------------------------------------------------------------------

def opt_plot_performance():
    print("\n[5] Evaluate & plot performance")
    splits = list_processed_splits()
    if not splits:
        print("[WARN] No processed splits found.")
        return 0
    print("Available splits:", splits)
    key = _prompt("Split key:")
    model_choice = _prompt("Model (fusion/dnn)?", default="fusion", choices=["fusion","dnn"])
    suffix = "fusion" if model_choice == "fusion" else "dnn"
    model_folder = "cnn_lstm_fusion" if model_choice == "fusion" else "dnn"
    bundle = load_processed_split(key, processed_root=f"{DEFAULT_DATAROOT}/processed")
    Xtr, ytr, Xte, yte, meta = bundle["X_train"], bundle["y_train"], bundle["X_test"], bundle["y_test"], bundle["meta"]

    # Locate a checkpoint
    ckdir = os.path.join(DEFAULT_OUTROOT, model_folder, key, "checks")
    ckpts = sorted(glob.glob(os.path.join(ckdir, f"*_{suffix}.pt")))
    if not ckpts:
        print(f"[WARN] No checkpoints found for {model_choice} model at {ckdir}. Train first (option 4).")
        return 0
    ckpt = ckpts[-1]
    outdir = os.path.join(DEFAULT_OUTROOT, model_folder, key, "eval")

    task = "binary" if len(np.unique(ytr))==2 else "multiclass"
    metrics = evaluate_on_test(Xte, yte, ckpt, outdir, task_type=task)
    print("[DONE] Metrics written to", outdir)
    print(json.dumps(metrics, indent=2))
    return 0

# ---------------------------------------------------------------------
# Option 6: XAI (SHAP, LIME, ANOVA)
# ---------------------------------------------------------------------

def opt_xai():
    print("\n[6] XAI (SHAP / LIME / ANOVA)")
    splits = list_processed_splits()
    if not splits:
        print("[WARN] No processed splits found.")
        return 0
    print("Available splits:", splits)
    key = _prompt("Split key:")
    model_choice = _prompt("Model (fusion/dnn)?", default="fusion", choices=["fusion","dnn"])
    suffix = "fusion" if model_choice == "fusion" else "dnn"
    model_folder = "cnn_lstm_fusion" if model_choice == "fusion" else "dnn"
    bundle = load_processed_split(key, processed_root=f"{DEFAULT_DATAROOT}/processed")
    Xtr, ytr, Xte, yte, meta = bundle["X_train"], bundle["y_train"], bundle["X_test"], bundle["y_test"], bundle["meta"]

    ckdir = os.path.join(DEFAULT_OUTROOT, model_folder, key, "checks")
    ckpts = sorted(glob.glob(os.path.join(ckdir, f"*_{suffix}.pt")))
    if not ckpts:
        print(f"[WARN] No checkpoints found for {model_choice} model at {ckdir}. Train first.")
        return 0
    ckpt = ckpts[-1]

    methods = _prompt("Which explainers? (comma sep: shap,lime,anova)", default="shap,lime,anova")
    methods = [m.strip().lower() for m in methods.split(",") if m.strip()]
    outdir = os.path.join(DEFAULT_OUTROOT, model_folder, key, "xai")

    run_xai(
        X_train=Xtr, X_test=Xte, y_train=ytr, y_test=yte,
        checkpoint=ckpt, feature_names=meta["feature_names"],
        class_names=meta["class_names"],
        methods=methods, outdir=outdir
    )
    print(f"[DONE] XAI artifacts saved to {outdir}")
    return 0

# ---------------------------------------------------------------------
# Option 7: Ablation (user-selected features)
# ---------------------------------------------------------------------

def opt_ablation():
    print("\n[7] Ablation (user-selected features)")
    splits = list_processed_splits()
    if not splits:
        print("[WARN] No processed splits found.")
        return 0
    print("Available splits:", splits)
    key = _prompt("Split key:")
    bundle = load_processed_split(key, processed_root=f"{DEFAULT_DATAROOT}/processed")
    Xtr, ytr, Xte, yte, meta = bundle["X_train"], bundle["y_train"], bundle["X_test"], bundle["y_test"], bundle["meta"]

    feat_names = meta["feature_names"]
    print(f"[INFO] There are {len(feat_names)} features.")
    print("Enter a comma-separated list of feature names to KEEP (subset).")
    print("Tip: You can copy from XAI top-features CSV.")
    subset_raw = _prompt("Feature subset (comma-separated):")
    subset = [s.strip() for s in subset_raw.split(",") if s.strip()]
    model_choice = _prompt("Model (fusion/dnn)?", default="fusion", choices=["fusion","dnn"])
    model_folder = "cnn_lstm_fusion" if model_choice == "fusion" else "dnn"
    outdir = os.path.join(DEFAULT_OUTROOT, model_folder, key, "ablation")

    run_ablation_user_selected(
        X_train=Xtr, y_train=ytr, X_test=Xte, y_test=yte,
        feature_names=feat_names, keep_features=subset,
        dataset_key=key, outdir=outdir,
        model_name=("cnn_lstm_fusion" if model_choice=="fusion" else "dnn")
    )
    print(f"[DONE] Ablation results saved to {outdir}")
    return 0

# ---------------------------------------------------------------------
# Option 8: Evaluate (explicit)
# ---------------------------------------------------------------------

def opt_evaluate():
    print("\n[8] Evaluate (explicit)")
    splits = list_processed_splits()
    if not splits:
        print("[WARN] No processed splits found.")
        return 0
    print("Available splits:", splits)
    key = _prompt("Split key:")
    model_choice = _prompt("Model (fusion/dnn)?", default="fusion", choices=["fusion","dnn"])
    suffix = "fusion" if model_choice == "fusion" else "dnn"
    model_folder = "cnn_lstm_fusion" if model_choice == "fusion" else "dnn"
    bundle = load_processed_split(key, processed_root=f"{DEFAULT_DATAROOT}/processed")
    Xtr, ytr, Xte, yte, meta = bundle["X_train"], bundle["y_train"], bundle["X_test"], bundle["y_test"], bundle["meta"]

    ckdir = os.path.join(DEFAULT_OUTROOT, model_folder, key, "checks")
    ckpts = sorted(glob.glob(os.path.join(ckdir, f"*_{suffix}.pt")))
    if not ckpts:
        print(f"[WARN] No checkpoints found for {model_choice} model at {ckdir}. Train first.")
        return 0
    ckpt = ckpts[-1]
    outdir = os.path.join(DEFAULT_OUTROOT, model_folder, key, "eval")

    task = "binary" if len(np.unique(ytr))==2 else "multiclass"
    metrics = evaluate_on_test(Xte, yte, ckpt, outdir, task_type=task)
    print("[DONE] Metrics written to", outdir)
    print(json.dumps(metrics, indent=2))
    return 0

# ---------------------------------------------------------------------
# Option 9: Inference (live/new data)
# ---------------------------------------------------------------------

def opt_inference():
    print("\n[9] Inference on new/live data")
    splits = list_processed_splits()
    if not splits:
        print("[WARN] No processed splits found.")
        return 0
    print("Available splits:", splits)
    key = _prompt("Split key:")
    bundle = load_processed_split(key, processed_root=f"{DEFAULT_DATAROOT}/processed")
    meta = bundle["meta"]
    dataset_name = meta["dataset_name"]
    class_type = meta["class_type"]
    model_choice = _prompt("Model (fusion/dnn)?", default="fusion", choices=["fusion","dnn"])
    suffix = "fusion" if model_choice == "fusion" else "dnn"

    # load model by dataset & class type
    model, state = load_model(dataset_name, class_type, model_root=DEFAULT_OUTROOT, model_type=model_choice)
    in_dim = int(state["input_dim"])

    # Provide a CSV path of *already preprocessed & scaled* features or raw?
    # For safety, we accept raw numeric CSV with the same columns as training features and will scale with stored scaler.
    csv_path = _prompt("Path to CSV with NEW flows (columns must match training features):")
    if not os.path.exists(csv_path):
        print("[ERROR] CSV not found.")
        return 0
    df_new = pd.read_csv(csv_path)
    # Ensure column order matches training
    train_feats = state.get("feature_names") or meta["feature_names"]
    missing = [c for c in train_feats if c not in df_new.columns]
    if missing:
        print("[ERROR] Missing columns in provided CSV:", missing[:10], "...")
        return 0
    df_new = df_new[train_feats]
    X_new = df_new.values.astype(np.float32)

    # Scale if scaler.pkl exists in split meta
    scaler = meta.get("scaler_obj", None)
    if scaler is None and os.path.exists(os.path.join(DEFAULT_DATAROOT, "processed", key, "scaler.joblib")):
        # If utils implemented joblib dump/load, datasets.py should attach scaler to meta already.
        pass

    result = predict_live(model, X_new, scaler=None, return_proba=True)
    model_folder = "cnn_lstm_fusion" if model_choice == "fusion" else "dnn"
    outdir = os.path.join(DEFAULT_OUTROOT, model_folder, key, "inference")
    ensure_dir(outdir)
    np.save(os.path.join(outdir, f"y_pred_{suffix}.npy"), result["y_pred"])
    if result["y_proba"] is not None:
        np.save(os.path.join(outdir, f"y_proba_{suffix}.npy"), result["y_proba"])
    _safe_json_dump(
        {"infer_time_s": result["infer_time_s"], "energy_j_proxy": result["energy_j_proxy"]},
        os.path.join(outdir, f"inference_meta_{suffix}.json")
    )
    print(f"[DONE] Inference saved to {outdir}")
    return 0

# ---------------------------------------------------------------------
# Interactive menu
# ---------------------------------------------------------------------

MENU = """
==================== IDS EXPERIMENT CLI ====================
(1) Load & split dataset (persist)
(2) Load existing split (validate)
(3) Visualize features (EPS)
(4) Train model (save ckpt + curves + history CSV)
(5) Evaluate & plot performance (EPS + CSV/JSON)
(6) XAI (SHAP/LIME/ANOVA)
(7) Ablation (user-selected feature subset)
(8) Evaluate (explicit)
(9) Inference (load_model → predict live CSV)
(q) Quit
============================================================
"""

def interactive_menu():
    while True:
        print(MENU)
        choice = input("Select option: ").strip().lower()
        if choice == "1": opt_load_and_split()
        elif choice == "2": opt_load_existing_split()
        elif choice == "3": opt_visualize_dataset()
        elif choice == "4": opt_train_model()
        elif choice == "5": opt_plot_performance()
        elif choice == "6": opt_xai()
        elif choice == "7": opt_ablation()
        elif choice == "8": opt_evaluate()
        elif choice == "9": opt_inference()
        elif choice in ["q","quit","exit"]: break
        else:
            print("Unknown option. Please choose 1-9 or q.")

# ---------------------------------------------------------------------
# Argparse subcommands (automation)
# ---------------------------------------------------------------------

def build_argparser():
    p = argparse.ArgumentParser(description="IDS Experiment Orchestrator (Fusion CNN+LSTM)")
    sub = p.add_subparsers(dest="cmd")

    # load & split
    ps = sub.add_parser("split", help="Load raw dataset, split & persist")
    ps.add_argument("--dataset", required=True, choices=list(AVAILABLE_DATASETS.keys()))
    ps.add_argument("--class_type", default="binary", choices=["binary","multiclass"])
    ps.add_argument("--test_size", type=float, default=0.2)
    ps.add_argument("--seed", type=int, default=42)

    # visualize
    pv = sub.add_parser("viz", help="Visualize dataset or split")
    pv.add_argument("--mode", default="processed", choices=["raw","processed"])
    pv.add_argument("--dataset", choices=list(AVAILABLE_DATASETS.keys()))
    pv.add_argument("--class_type", choices=["binary","multiclass"], default="binary")
    pv.add_argument("--split_key")

    # train
    pt = sub.add_parser("train", help="Train a model on a processed split")
    pt.add_argument("--split_key", required=True)
    pt.add_argument("--model", default="fusion", choices=["fusion","dnn"])
    pt.add_argument("--epochs", type=int, default=50)
    pt.add_argument("--batch", type=int, default=256)
    pt.add_argument("--lr", type=float, default=1e-3)
    pt.add_argument("--patience", type=int, default=8)

    # evaluate
    pe = sub.add_parser("eval", help="Evaluate a trained model on its split")
    pe.add_argument("--split_key", required=True)

    # xai
    px = sub.add_parser("xai", help="Run SHAP/LIME/ANOVA")
    px.add_argument("--split_key", required=True)
    px.add_argument("--methods", default="shap,lime,anova")

    # ablation
    pa = sub.add_parser("ablate", help="User-selected feature subset ablation")
    pa.add_argument("--split_key", required=True)
    pa.add_argument("--keep_feats", required=True, help="Comma-separated feature names to keep")

    # inference
    pi = sub.add_parser("infer", help="Run inference on new/live CSV")
    pi.add_argument("--split_key", required=True)
    pi.add_argument("--csv", required=True)

    return p


def main():
    # If no arguments → interactive TUI
    if len(sys.argv) == 1:
        interactive_menu()
        return

    parser = build_argparser()
    args = parser.parse_args()

    if args.cmd == "split":
        X, y, feat_cols, y_enc, scaler = load_and_prepare_dataset(args.dataset, class_type=args.class_type)
        info = split_and_persist(
            dataset_name=args.dataset, X=X, y=y, feature_names=feat_cols,
            y_encoder=y_enc, scaler=scaler, test_size=args.test_size,
            random_state=args.seed, processed_root=f"{DEFAULT_DATAROOT}/processed",
            class_type=args.class_type
        )
        print(json.dumps(info, indent=2))

    elif args.cmd == "viz":
        if args.mode == "raw":
            if not args.dataset:
                print("[ERROR] --dataset required for raw viz")
                sys.exit(2)
            X, y, feat_cols, y_enc, scaler = load_and_prepare_dataset(args.dataset, class_type=args.class_type)
            outdir = os.path.join(DEFAULT_OUTROOT, f"{args.dataset}_{args.class_type}", "viz")
            plot_dataset_overview(X, y, feat_cols, outdir=outdir)
            print(f"[DONE] {outdir}")
        else:
            if not args.split_key:
                print("[ERROR] --split_key required for processed viz")
                sys.exit(2)
            bundle = load_processed_split(args.split_key, processed_root=f"{DEFAULT_DATAROOT}/processed")
            Xtr, ytr, Xte, yte, meta = bundle["X_train"], bundle["y_train"], bundle["X_test"], bundle["y_test"], bundle["meta"]
            outdir = os.path.join(DEFAULT_OUTROOT, f"{args.split_key}", "viz")
            plot_dataset_overview(np.vstack([Xtr, Xte]), np.hstack([ytr, yte]), meta["feature_names"], outdir=outdir)
            print(f"[DONE] {outdir}")

    elif args.cmd == "train":
        bundle = load_processed_split(args.split_key, processed_root=f"{DEFAULT_DATAROOT}/processed")
        Xtr, ytr, Xte, yte, meta = bundle["X_train"], bundle["y_train"], bundle["X_test"], bundle["y_test"], bundle["meta"]
        ckpt_path, hist_csv, artifacts_dir, model = train_experiment(
            Xtr, ytr, Xte, yte,
            model_name=("cnn_lstm_fusion" if args.model=="fusion" else "dnn"),
            dataset_key=args.split_key,
            out_root=DEFAULT_OUTROOT,
            epochs=args.epochs, batch_size=args.batch, lr=args.lr,
            early_stop_patience=args.patience,
            class_names=meta["class_names"]
        )
        complexity = estimate_model_complexity(model, input_dim=meta["input_dim"], seq_len=meta.get("seq_len", 1))
        _safe_json_dump(complexity, os.path.join(artifacts_dir, "complexity.json"))
        print(json.dumps({"ckpt": ckpt_path, "history_csv": hist_csv, "complexity": complexity}, indent=2))

    elif args.cmd == "eval":
        bundle = load_processed_split(args.split_key, processed_root=f"{DEFAULT_DATAROOT}/processed")
        Xtr, ytr, Xte, yte, meta = bundle["X_train"], bundle["y_train"], bundle["X_test"], bundle["y_test"], bundle["meta"]
        ckdir = os.path.join(DEFAULT_OUTROOT, args.split_key, "checks")
        ckpts = sorted(glob.glob(os.path.join(ckdir, "*.pt")))
        if not ckpts:
            print("[ERROR] no checkpoints under", ckdir)
            sys.exit(2)
        ckpt = ckpts[-1]
        task = "binary" if len(np.unique(ytr))==2 else "multiclass"
        outdir = os.path.join(DEFAULT_OUTROOT, args.split_key, "eval")
        metrics = evaluate_on_test(Xte, yte, ckpt, outdir, task_type=task)
        print(json.dumps(metrics, indent=2))

    elif args.cmd == "xai":
        bundle = load_processed_split(args.split_key, processed_root=f"{DEFAULT_DATAROOT}/processed")
        Xtr, ytr, Xte, yte, meta = bundle["X_train"], bundle["y_train"], bundle["X_test"], bundle["y_test"], bundle["meta"]
        ckdir = os.path.join(DEFAULT_OUTROOT, args.split_key, "checks")
        ckpts = sorted(glob.glob(os.path.join(ckdir, "*.pt")))
        if not ckpts:
            print("[ERROR] no checkpoints under", ckdir)
            sys.exit(2)
        ckpt = ckpts[-1]
        methods = [m.strip().lower() for m in args.methods.split(",") if m.strip()]
        outdir = os.path.join(DEFAULT_OUTROOT, args.split_key, "xai")
        run_xai(
            X_train=Xtr, X_test=Xte, y_train=ytr, y_test=yte,
            checkpoint=ckpt, feature_names=meta["feature_names"],
            class_names=meta["class_names"], methods=methods, outdir=outdir
        )
        print(f"[DONE] {outdir}")

    elif args.cmd == "ablate":
        bundle = load_processed_split(args.split_key, processed_root=f"{DEFAULT_DATAROOT}/processed")
        Xtr, ytr, Xte, yte, meta = bundle["X_train"], bundle["y_train"], bundle["X_test"], bundle["y_test"], bundle["meta"]
        keep = [s.strip() for s in args.keep_feats.split(",") if s.strip()]
        outdir = os.path.join(DEFAULT_OUTROOT, args.split_key, "ablation")
        run_ablation_user_selected(
            X_train=Xtr, y_train=ytr, X_test=Xte, y_test=yte,
            feature_names=meta["feature_names"], keep_features=keep,
            dataset_key=args.split_key, outdir=outdir
        )
        print(f"[DONE] {outdir}")

    elif args.cmd == "infer":
        bundle = load_processed_split(args.split_key, processed_root=f"{DEFAULT_DATAROOT}/processed")
        meta = bundle["meta"]
        model, state = load_model(meta["dataset_name"], meta["class_type"], model_root=DEFAULT_OUTROOT)
        df_new = pd.read_csv(args.csv)
        train_feats = state.get("feature_names") or meta["feature_names"]
        missing = [c for c in train_feats if c not in df_new.columns]
        if missing:
            print("[ERROR] CSV missing training features:", missing[:10], "...")
            sys.exit(2)
        X_new = df_new[train_feats].values.astype(np.float32)
        result = predict_live(model, X_new, scaler=None, return_proba=True)
        outdir = os.path.join(DEFAULT_OUTROOT, args.split_key, "inference")
        ensure_dir(outdir)
        np.save(os.path.join(outdir, "y_pred.npy"), result["y_pred"])
        if result["y_proba"] is not None:
            np.save(os.path.join(outdir, "y_proba.npy"), result["y_proba"])
        with open(os.path.join(outdir, "inference_meta.json"), "w") as f:
            json.dump({"infer_time_s": result["infer_time_s"], "energy_j_proxy": result["energy_j_proxy"]}, f, indent=2)
        print(f"[DONE] {outdir}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
