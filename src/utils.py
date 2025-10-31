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
    """Create directory recursively if it doesn't exist."""
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
