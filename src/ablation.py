"""
ablation.py — User-selected feature ablation for IDS models
Author: Dr. Vanlalruata Hnamte
Version: 1.0
"""

import os
import json
import time
import math
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score,
    precision_recall_fscore_support, mean_squared_error, mean_absolute_error
)
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------
# Local project imports (trainer, models, utils).  These must exist.
# If your trainer API differs, adapt the adapter function below.
# ---------------------------------------------------------------------
try:
    # Preferred: your modular trainer core
    from .models import CNN_LSTM_Fusion, SimpleMLP
    from .utils import set_all_seeds, ensure_dir
    from .train import train_once  # expected signature documented below (may not exist; fallback below)
except Exception:
    # Minimal fallbacks (only shapes used; adapt to your real project layout)
    from .models import CNN_LSTM_Fusion, SimpleMLP  # must exist from previous files

    def set_all_seeds(seed: int):
        import random
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def ensure_dir(path: str):
        os.makedirs(path, exist_ok=True)

    # Fallback train_once adapter: a very light wrapper that trains quickly.
    # Replace this with your project's train_once if available.
    def train_once(
        Xtr: np.ndarray,
        ytr: np.ndarray,
        Xte: np.ndarray,
        yte: np.ndarray,
        feature_names: List[str],
        outdir: str,
        run_name: str,
        model_name: str = "cnn_lstm_fusion",
        task_type: str = "classification",
        seed: int = 42,
        epochs: int = 40,
        batch_size: int = 256,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        early_stop_patience: int = 5,
        class_names: Optional[List[str]] = None,
        device: Optional[torch.device] = None,
    ) -> Dict:
        """
        Returns:
          {
            "ckpt_path": ".../best_{run_name}.pt",
            "metrics": {... classification & runtime ...},
            "history_csv": ".../hist_{run_name}.csv",
          }
        """
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        set_all_seeds(seed)

        # Choose model
        input_dim = Xtr.shape[1]
        num_classes = len(np.unique(ytr)) if task_type == "classification" else 1
        if model_name.lower() in ["mlp", "simplemlp", "dnn"]:
            model = SimpleMLP(input_dim, num_classes)
        else:
            model = CNN_LSTM_Fusion(input_dim, num_classes, hidden_dim=64)
        model = model.to(device)

        # Data
        tens = lambda a: torch.tensor(a, dtype=torch.float32)
        train_ds = torch.utils.data.TensorDataset(tens(Xtr), torch.tensor(ytr, dtype=torch.long if task_type=="classification" else torch.float32))
        test_ds  = torch.utils.data.TensorDataset(tens(Xte), torch.tensor(yte, dtype=torch.long if task_type=="classification" else torch.float32))
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=torch.cuda.is_available())
        test_loader  = torch.utils.data.DataLoader(test_ds,  batch_size=batch_size, shuffle=False, pin_memory=torch.cuda.is_available())

        # Loss/optim
        if task_type == "classification":
            criterion = torch.nn.CrossEntropyLoss()
        else:
            criterion = torch.nn.MSELoss()
        optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        history = []
        best_loss = float("inf")
        patience = early_stop_patience

        t0 = time.perf_counter()
        for ep in range(1, epochs + 1):
            model.train()
            losses, n = 0.0, 0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optim.zero_grad(set_to_none=True)
                out = model(xb)
                if task_type == "classification":
                    loss = criterion(out, yb)
                else:
                    loss = criterion(out.squeeze(), yb.squeeze())
                loss.backward()
                optim.step()
                bs = yb.size(0)
                losses += loss.item() * bs
                n += bs
            train_loss = losses / max(n, 1)

            # quick validation on test_loader
            model.eval()
            with torch.no_grad():
                losses, n = 0.0, 0
                y_true, y_pred = [], []
                for xb, yb in test_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    out = model(xb)
                    if task_type == "classification":
                        loss = criterion(out, yb)
                        pred = out.argmax(1)
                        y_pred.extend(pred.detach().cpu().numpy().tolist())
                        y_true.extend(yb.detach().cpu().numpy().tolist())
                    else:
                        loss = criterion(out.squeeze(), yb.squeeze())
                    bs = yb.size(0)
                    losses += loss.item() * bs
                    n += bs
                val_loss = losses / max(n, 1)

            history.append({"epoch": ep, "train_loss": train_loss, "val_loss": val_loss})

            if val_loss + 1e-6 < best_loss:
                best_loss = val_loss
                patience = early_stop_patience
                ensure_dir(outdir)
                ckpt_path = os.path.join(outdir, f"best_{run_name}.pt")
                torch.save({
                    "model_name": "cnn_lstm_fusion" if model_name.lower() not in ["mlp", "simplemlp"] else "simplemlp",
                    "state_dict": model.state_dict(),
                    "input_dim": input_dim,
                    "num_classes": num_classes,
                    "task_type": task_type,
                    "class_names": class_names
                }, ckpt_path)
            else:
                patience -= 1
                if patience == 0:
                    break

        t1 = time.perf_counter()
        train_time_s = t1 - t0

        # Final eval
        model.eval()
        y_true, y_pred, y_prob = [], [], []
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device)
                out = model(xb)
                if task_type == "classification":
                    prob = torch.softmax(out, dim=1).cpu().numpy()
                    pred = prob.argmax(axis=1)
                    y_prob.extend(prob.tolist())
                    y_pred.extend(pred.tolist())
                    y_true.extend(yb.numpy().tolist())
                else:
                    y_pred.extend(out.squeeze().cpu().numpy().tolist())
                    y_true.extend(yb.numpy().tolist())

        # Pack metrics
        metrics = {"task_type": task_type, "train_time_s": train_time_s}
        if task_type == "classification":
            pr, rc, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
            acc = (np.array(y_true) == np.array(y_pred)).mean()
            metrics.update({"precision_macro": pr, "recall_macro": rc, "f1_macro": f1, "accuracy": acc})
            # ROC-AUC (macro one-vs-rest if possible)
            try:
                y_true_ovr = np.eye(len(np.unique(y_true)))[np.array(y_true, dtype=int)]
                metrics["roc_auc_ovr_macro"] = roc_auc_score(y_true_ovr, np.array(y_prob), average="macro", multi_class="ovr")
            except Exception:
                metrics["roc_auc_ovr_macro"] = None
        else:
            y_true_arr, y_pred_arr = np.array(y_true), np.array(y_pred)
            mse = mean_squared_error(y_true_arr, y_pred_arr)
            rmse = math.sqrt(mse)
            mae = mean_absolute_error(y_true_arr, y_pred_arr)
            metrics.update({"mse": mse, "rmse": rmse, "mae": mae})

        # Save history CSV
        hist_path = os.path.join(outdir, f"hist_{run_name}.csv")
        pd.DataFrame(history).to_csv(hist_path, index=False)

        return {"ckpt_path": ckpt_path, "metrics": metrics, "history_csv": hist_path}


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------
# Utility: index mapping for user-selected feature names
# ---------------------------------------------------------------------
def map_features_to_indices(all_features: List[str], selected_features: List[str]) -> List[int]:
    name_to_idx = {n: i for i, n in enumerate(all_features)}
    missing = [n for n in selected_features if n not in name_to_idx]
    if missing:
        raise ValueError(f"The following selected features are not found in dataset features: {missing[:10]}{' ...' if len(missing) > 10 else ''}")
    return [name_to_idx[n] for n in selected_features]


# ---------------------------------------------------------------------
# Rough time complexity proxies
# ---------------------------------------------------------------------
def count_trainable_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def estimate_linear_macs(model: torch.nn.Module, input_dim: int) -> int:
    """
    Very rough MACs estimate for linear/conv layers in a single forward pass
    with tabular input shaped [batch, input_dim].
    This is a lower bound; CNN+LSTM fusion adds extra ops on the temporal path.
    """
    macs = 0
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            macs += m.in_features * m.out_features
        # (Optional) add rules for Conv1d, LSTM, etc., if desired
    return macs


# ---------------------------------------------------------------------
# Core: run ablation for user-defined subsets
# ---------------------------------------------------------------------
def run_ablation(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    dataset_name: str,
    classification_type: str,  # "binary" or "multiclass" (or "regression" if needed)
    subsets: Dict[str, List[str]],  # {"subset_name": [feature, ...], ...}
    out_root: str,
    model_name: str = "cnn_lstm_fusion",
    test_size: float = 0.2,
    seed: int = 42,
    epochs: int = 60,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    early_stop_patience: int = 6,
    class_names: Optional[List[str]] = None
) -> Dict:
    """
    Executes user-selected ablation. For each named subset:
      1) Slice X to those features
      2) train_once(...)
      3) Collect metrics, params, MACs, wall time
      4) Aggregate into CSV + plots

    Returns a dict with a manifest and results path.
    """
    ensure_dir(out_root)
    task_folder = f"{dataset_name}_{classification_type}"
    outdir = os.path.join(out_root, task_folder, "ablation")
    ensure_dir(outdir)

    # Persist the manifest for exact reproducibility
    manifest = {
        "dataset": dataset_name,
        "classification_type": classification_type,
        "model_name": model_name,
        "seed": seed,
        "subsets": subsets,
        "feature_pool_size": len(feature_names),
        "feature_pool": feature_names,
        "trainer_hparams": {
            "epochs": epochs, "batch_size": batch_size,
            "lr": lr, "weight_decay": weight_decay,
            "early_stop_patience": early_stop_patience
        }
    }
    with open(os.path.join(outdir, "ablation_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    # Prepare train/test split once to keep comparisons fair
    set_all_seeds(seed)
    if classification_type in ["binary", "multiclass"]:
        Xtr_all, Xte_all, ytr, yte = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=y
        )
        task_type = "classification"
    else:
        Xtr_all, Xte_all, ytr, yte = train_test_split(
            X, y, test_size=test_size, random_state=seed
        )
        task_type = "regression"

    # Assemble results table
    rows = []

    for subset_name, feat_list in subsets.items():
        print(f"[Ablation] Running subset '{subset_name}' with {len(feat_list)} features…")

        # Map names → indices → slice
        idxs = map_features_to_indices(feature_names, feat_list)
        Xtr = Xtr_all[:, idxs]
        Xte = Xte_all[:, idxs]

        run_name = f"ablate_{subset_name}_{len(idxs)}f"
        run_out = os.path.join(outdir, subset_name)
        ensure_dir(run_out)

        # Train once (uses your project trainer if available)
        result = train_once(
            Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte,
            feature_names=feat_list,
            outdir=run_out,
            run_name=run_name,
            model_name=model_name,
            task_type=task_type,
            seed=seed,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            early_stop_patience=early_stop_patience,
            class_names=class_names,
            device=DEVICE
        )

        # Load checkpoint and estimate params/MACs
        state = torch.load(result["ckpt_path"], map_location=DEVICE)
        input_dim = Xtr.shape[1]
        num_classes = int(state.get("num_classes", 2))
        if (state.get("model_name") or "").lower() in ["simplemlp", "mlp"]:
            model = SimpleMLP(input_dim, num_classes)
        else:
            model = CNN_LSTM_Fusion(input_dim, num_classes, hidden_dim=int(state.get("hidden_dim", 64)))
        model.load_state_dict(state["state_dict"])
        model.to(DEVICE).eval()

        params = count_trainable_params(model)
        macs = estimate_linear_macs(model, input_dim)

        m = result["metrics"]
        row = {
            "subset": subset_name,
            "n_features": len(feat_list),
            "train_time_s": m.get("train_time_s", None),
            "params": params,
            "estimated_macs_per_forward": macs,
        }

        if task_type == "classification":
            row.update({
                "accuracy": m.get("accuracy"),
                "precision_macro": m.get("precision_macro"),
                "recall_macro": m.get("recall_macro"),
                "f1_macro": m.get("f1_macro"),
                "roc_auc_ovr_macro": m.get("roc_auc_ovr_macro"),
            })
        else:
            row.update({
                "mse": m.get("mse"),
                "rmse": m.get("rmse"),
                "mae": m.get("mae"),
            })

        rows.append(row)

    df = pd.DataFrame(rows).sort_values(by=("f1_macro" if task_type=="classification" else "rmse"),
                                        ascending=(False if task_type=="classification" else True))
    csv_path = os.path.join(outdir, "ablation_results.csv")
    df.to_csv(csv_path, index=False)

    # ---- Plots (EPS) ---------------------------------------------------
    ensure_dir(os.path.join(outdir, "plots"))

    # Performance vs. #features
    plt.figure(figsize=(8, 5))
    if task_type == "classification":
        plt.plot(df["n_features"], df["f1_macro"], marker="o")
        plt.xlabel("Number of features")
        plt.ylabel("F1 (macro)")
        plt.title(f"Ablation: F1 vs #Features — {dataset_name} ({classification_type})")
    else:
        plt.plot(df["n_features"], df["rmse"], marker="o")
        plt.xlabel("Number of features")
        plt.ylabel("RMSE")
        plt.title(f"Ablation: RMSE vs #Features — {dataset_name} (regression)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "plots", "perf_vs_nfeatures.eps"), format="eps", dpi=200)
    plt.close()

    # Bar chart per subset (F1 or RMSE)
    plt.figure(figsize=(10, 6))
    labels = df["subset"].tolist()
    x = np.arange(len(labels))
    if task_type == "classification":
        vals = df["f1_macro"].values
        plt.bar(x, vals)
        plt.ylabel("F1 (macro)")
        plt.title("Ablation — F1 by subset")
    else:
        vals = df["rmse"].values
        plt.bar(x, vals)
        plt.ylabel("RMSE")
        plt.title("Ablation — RMSE by subset")
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "plots", "perf_by_subset.eps"), format="eps", dpi=200)
    plt.close()

    # Params/MACs scatter
    plt.figure(figsize=(8, 5))
    plt.scatter(df["params"], df["estimated_macs_per_forward"])
    for i, txt in enumerate(df["subset"].tolist()):
        plt.annotate(txt, (df["params"].iloc[i], df["estimated_macs_per_forward"].iloc[i]), fontsize=7, alpha=0.8)
    plt.xlabel("Trainable parameters")
    plt.ylabel("Estimated MACs per forward (lower bound)")
    plt.title("Model complexity across subsets")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "plots", "complexity_scatter.eps"), format="eps", dpi=200)
    plt.close()

    return {
        "manifest": os.path.join(outdir, "ablation_manifest.json"),
        "results_csv": csv_path,
        "plots_dir": os.path.join(outdir, "plots"),
        "results_df": df
    }


# ---------------------------------------------------------------------
# Convenience: build subsets from a list of lists (already names)
# ---------------------------------------------------------------------
def build_subsets_from_lists(named_feature_lists: List[Tuple[str, List[str]]]) -> Dict[str, List[str]]:
    """
    Helper to convert [("top5", [...]), ("top10", [...])] to dict.
    """
    d = {}
    for name, feats in named_feature_lists:
        if name in d:
            raise ValueError(f"Duplicate subset name: {name}")
        d[name] = list(dict.fromkeys(feats))  # dedupe while preserving order
    return d


# -----------------------------------------------------------------------------
# Convenience wrapper used by main.py:
#   run_ablation_user_selected(X_train, y_train, X_test, y_test,
#                              feature_names, keep_features, dataset_key, outdir, ...)
# -----------------------------------------------------------------------------
def run_ablation_user_selected(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
    keep_features: List[str],
    dataset_key: str,
    outdir: str,
    model_name: str = "cnn_lstm_fusion",
    seed: int = 42,
    epochs: int = 60,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    early_stop_patience: int = 6,
    class_names: Optional[List[str]] = None
) -> Dict:
    """
    Slice the provided train/test arrays to a user-selected feature subset and train once.
    Produces a compact result JSON and preserves the history CSV and checkpoint under 'outdir'.
    """
    ensure_dir(outdir)
    set_all_seeds(seed)

    # Map feature names -> indices and slice
    idxs = map_features_to_indices(feature_names, keep_features)
    Xtr = X_train[:, idxs]
    Xte = X_test[:, idxs]

    subset_name = "user_selected"
    run_name = f"{subset_name}_{len(idxs)}f"

    # Train using the local lightweight 'train_once' (fallback defined above)
    result = train_once(
        Xtr=Xtr, ytr=y_train, Xte=Xte, yte=y_test,
        feature_names=keep_features,
        outdir=outdir,
        run_name=run_name,
        model_name=model_name,
        task_type="classification",  # as per project spec (all datasets treated as classification)
        seed=seed,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        early_stop_patience=early_stop_patience,
        class_names=class_names,
        device=DEVICE
    )

    # Complexity snapshot for the trained subset model
    state = torch.load(result["ckpt_path"], map_location=DEVICE)
    input_dim = Xtr.shape[1]
    num_classes = int(state.get("num_classes", len(np.unique(y_train))))
    if (state.get("model_name") or "").lower() in ["simplemlp", "mlp"]:
        model = SimpleMLP(input_dim, num_classes)
    else:
        model = CNN_LSTM_Fusion(input_dim, num_classes, hidden_dim=int(state.get("hidden_dim", 64)))
    model.load_state_dict(state["state_dict"])
    model.to(DEVICE).eval()

    params = count_trainable_params(model)
    macs = estimate_linear_macs(model, input_dim)

    summary = {
        "dataset_key": dataset_key,
        "subset": subset_name,
        "n_features": len(keep_features),
        "kept_features": keep_features,
        "ckpt_path": result["ckpt_path"],
        "history_csv": result["history_csv"],
        "metrics": result["metrics"],
        "params": int(params),
        "estimated_macs_per_forward": int(macs),
    }

    model_suffix = "fusion" if "fusion" in model_name.lower() else "dnn"
    with open(os.path.join(outdir, f"result_{model_suffix}.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Also save a one-row CSV for a quick glance
    df = pd.DataFrame([{
        "subset": subset_name,
        "n_features": len(keep_features),
        **({k: v for k, v in result["metrics"].items() if isinstance(v, (int, float))})
    }])
    df.to_csv(os.path.join(outdir, f"ablation_user_selected_{model_suffix}.csv"), index=False)

    return summary
