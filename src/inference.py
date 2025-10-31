"""
inference.py — Real-time inference for IDS models
Author: Dr. Vanlalruata Hnamte
Version: 1.0
"""

import os
import time
import numpy as np
import torch
from HLC.src.models import CNN_LSTM_Fusion, SimpleMLP

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
POWER_WATTS = float(os.environ.get("POWER_WATTS", "120"))


def load_model(dataset_name: str, classification_type: str, model_root="outputs"):
    """
    Loads a trained model for a specific dataset and classification type.
    Example:
        model, meta = load_model("EDGE_IIoT", "binary")
    """
    base = os.path.join(model_root, f"{dataset_name}_{classification_type}", "checks")
    ckpts = [f for f in os.listdir(base) if f.endswith(".pt")]
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint found in {base}")
    best = sorted(ckpts)[-1]  # assume last saved is best
    path = os.path.join(base, best)
    state = torch.load(path, map_location=DEVICE)

    name = (state.get("model_name") or "").lower()
    in_dim = int(state.get("input_dim"))
    n_cls = int(state.get("num_classes", 1))
    hid = int(state.get("hidden_dim", 64))

    if name in ["cnn_lstm_fusion", "fusion"]:
        model = CNN_LSTM_Fusion(in_dim, n_cls, hidden_dim=hid)
    else:
        model = SimpleMLP(in_dim, n_cls)

    model.load_state_dict(state["state_dict"])
    model.to(DEVICE).eval()
    print(f"[Inference] Loaded {name.upper()} ({n_cls} classes) for {dataset_name} [{classification_type}]")
    return model, state


@torch.no_grad()
def predict_live(model, X_new: np.ndarray, scaler=None, return_proba=True):
    """
    Runs fast batched inference on live feature vectors.
    - X_new: np.ndarray [N, D]
    - scaler: (optional) StandardScaler used during training
    Returns: dict with y_pred, (optionally y_proba), timing, and energy proxy.
    """
    if scaler is not None:
        X_new = scaler.transform(X_new)

    xb = torch.tensor(X_new, dtype=torch.float32).to(DEVICE)
    if torch.cuda.is_available(): torch.cuda.synchronize()
    t0 = time.perf_counter()
    logits = model(xb)
    if torch.cuda.is_available(): torch.cuda.synchronize()
    t1 = time.perf_counter()

    if logits.shape[1] > 1:
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        y_pred = probs.argmax(1)
    else:
        probs = None
        y_pred = logits.squeeze().cpu().numpy()

    return {
        "y_pred": y_pred,
        "y_proba": probs if return_proba else None,
        "infer_time_s": t1 - t0,
        "energy_j_proxy": (t1 - t0) * POWER_WATTS
    }
