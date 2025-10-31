"""
train.py — Model training pipeline for IDS experiments
Author: Dr. Vanlalruata Hnamte
Version: 1.0
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import math

from .models import CNN_LSTM_Fusion, SimpleMLP
from .utils import ensure_dir, set_all_seeds


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
POWER_WATTS = float(os.environ.get("POWER_WATTS", "120"))


def train_experiment(
    X_train, y_train, X_test, y_test,
    model_name: str, dataset_key: str,
    out_root="outputs", epochs=50, batch_size=256,
    lr=1e-3, early_stop_patience=8, class_names=None
):
    """Train CNN+LSTM Fusion (or MLP) model and save artifacts."""

    set_all_seeds(42)
    input_dim = X_train.shape[1]
    num_classes = len(np.unique(y_train))
    outdir = os.path.join(out_root, dataset_key)
    ensure_dir(outdir)
    ckpt_dir = os.path.join(outdir, "checks")
    ensure_dir(ckpt_dir)

    if model_name.lower() in ["cnn_lstm_fusion", "fusion"]:
        model = CNN_LSTM_Fusion(input_dim, num_classes)
    else:
        model = SimpleMLP(input_dim, num_classes)
    model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.AdamW(model.parameters(), lr=lr)

    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    test_ds = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    best_loss = float("inf")
    patience = early_stop_patience
    hist = []

    if torch.cuda.is_available(): torch.cuda.synchronize()
    t0 = time.perf_counter()

    for ep in range(1, epochs + 1):
        model.train()
        total_loss, total_correct, total_samples = 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optim.zero_grad(set_to_none=True)
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optim.step()
            total_loss += loss.item() * len(yb)
            total_correct += (out.argmax(1) == yb).sum().item()
            total_samples += len(yb)

        train_loss = total_loss / total_samples
        train_acc = total_correct / total_samples

        # Validation
        model.eval()
        with torch.no_grad():
            val_loss, val_correct, val_total = 0.0, 0, 0
            for xb, yb in test_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                out = model(xb)
                loss = criterion(out, yb)
                val_loss += loss.item() * len(yb)
                val_correct += (out.argmax(1) == yb).sum().item()
                val_total += len(yb)
        val_loss /= val_total
        val_acc = val_correct / val_total

        hist.append({"epoch": ep, "train_loss": train_loss, "val_loss": val_loss,
                     "train_acc": train_acc, "val_acc": val_acc})
        print(f"[EPOCH {ep}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} acc={val_acc:.4f}")

        if val_loss < best_loss - 1e-6:
            best_loss = val_loss
            patience = early_stop_patience
            torch.save({
                "state_dict": model.state_dict(),
                "input_dim": input_dim,
                "num_classes": num_classes,
                "model_name": model_name,
                "class_names": class_names
            }, os.path.join(ckpt_dir, f"best_ep{ep}.pt"))
        else:
            patience -= 1
            if patience == 0:
                break

    if torch.cuda.is_available(): torch.cuda.synchronize()
    t1 = time.perf_counter()
    total_time = t1 - t0

    # Save learning curve
    hist_df = torch.tensor([(h["train_loss"], h["val_loss"]) for h in hist])
    plt.figure()
    plt.plot([h["epoch"] for h in hist], [h["train_loss"] for h in hist], label="Train")
    plt.plot([h["epoch"] for h in hist], [h["val_loss"] for h in hist], label="Val")
    plt.xlabel("Epochs"); plt.ylabel("Loss")
    plt.title("Learning Curve")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(outdir, "learning_curve.eps"), format="eps", dpi=200)
    plt.close()

    hist_csv = os.path.join(outdir, "history.csv")
    import pandas as pd
    pd.DataFrame(hist).to_csv(hist_csv, index=False)

    # Efficiency snapshot (training)
    epochs_run = len(hist)
    batches_per_epoch = math.ceil(len(X_train) / batch_size) if batch_size > 0 else 0
    ms_per_batch = (total_time / max(1, epochs_run * batches_per_epoch)) * 1000.0
    efficiency = {
        "epochs_run": int(epochs_run),
        "train_total_sec": float(total_time),
        "ms_per_batch_est": float(ms_per_batch),
        "power_watts_assumed": float(POWER_WATTS),
        "energy_j_proxy": float(total_time * POWER_WATTS),
        "device": str(DEVICE),
        "batch_size": int(batch_size),
        "lr": float(lr)
    }
    with open(os.path.join(outdir, "efficiency.json"), "w") as f:
        import json as _json
        _json.dump(efficiency, f, indent=2)

    return os.path.join(ckpt_dir, f"best_ep{ep}.pt"), hist_csv, outdir, model
