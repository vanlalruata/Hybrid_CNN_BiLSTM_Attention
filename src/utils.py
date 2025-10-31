"""
utils.py — Utility functions shared across IDS experiment modules
Author: Dr. Vanlalruata Hnamte
Version: 1.0
"""

import os
import random
import numpy as np
import torch
import joblib
import time
from contextlib import contextmanager


def ensure_dir(path: str):
    """Create a directory recursively if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def set_all_seeds(seed: int = 42):
    """Make all RNGs deterministic for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_scaler(scaler, path: str):
    """Save StandardScaler or similar preprocessing object."""
    ensure_dir(os.path.dirname(path))
    joblib.dump(scaler, path)


def load_scaler(path: str):
    """Load StandardScaler from disk if exists."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Scaler not found: {path}")
    return joblib.load(path)


@contextmanager
def timer(msg: str = None):
    """Simple timing context manager for benchmarking."""
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    print(f"[TIMER] {msg or ''} took {end - start:.4f} seconds.")


def unique_path(base_path: str) -> str:
    """
    Generate a unique file path by appending '(n)' before the file extension if the path exists.
    Examples:
      - reports/metrics.csv -> reports/metrics.csv (if free)
      - if taken, returns reports/metrics (1).csv, then (2), etc.
    """
    # Ensure the parent directory exists (no-op if base_path is just a filename)
    parent = os.path.dirname(base_path)
    if parent:
        ensure_dir(parent)

    # If the base path is available, use it directly
    if not os.path.exists(base_path):
        return base_path

    root, ext = os.path.splitext(base_path)
    n = 1
    while True:
        candidate = f"{root} ({n}){ext}"
        if not os.path.exists(candidate):
            return candidate
        n += 1
