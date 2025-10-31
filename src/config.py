"""
config.py — Configuration constants for IoT/IIoT IDS framework
Author: Dr. Vanlalruata Hnamte
Version: 1.0
"""

import os
from typing import Dict, List

# ============================================================
# DATASET CONFIGURATION
# ============================================================

# Raw dataset root glob patterns
DATASETS: Dict[str, str] = {
    "EDGE_IIoT": "H:/Datasets/Edge-IIoT/Selected dataset for ML and DL/*.csv",
    "CIC-IoT-2023": "data/raw/CICIoT2023/*.csv",
    "Apose-IoT-23": "data/raw/AposeIoT23/*.csv",
    "BoT-IoT": "data/raw/BoT-IoT/*.csv",
    "CIC-IoMT-2024": "data/raw/CICIoMT2024/*.csv",
    "CIC-IoT-IDAD-2024": "data/raw/CIC-IoT-IDAD-2024/*.csv",
    "CIC-IoT-2025": "data/raw/CICIoT2025/*.csv",
}

# Datasets with explicit allowed files (others load all .csv)
EXPLICIT_FILES: Dict[str, List[str]] = {
    # "EDGE_IIoT": [
    #     "backdoor_attack.csv",
    #     "ddos_http_flood_attack.csv",
    #     "os_fingerprinting_attack.csv",
    # ],
}

# Common identifier columns to drop across datasets
COMMON_DROP = [
    'Flow ID', 'Src IP', 'Dst IP', 'Timestamp', 'id.orig_h', 'id.resp_h',
    'frame.time', 'stime', 'ltime', 'saddr', 'daddr', 'uid', 'srcid'
]

# Extra per-dataset columns to drop
EXTRA_DROP = {
    "EDGE_IIoT": ['ip.src_host', 'ip.dst_host', 'Attack_label'],
    "CIC-IoT-2023": [],
    "Apose-IoT-23": [],
    "BoT-IoT": [],
    "CIC-IoMT-2024": [],
    "CIC-IoT-IDAD-2024": ['Flow ID', 'Src IP', 'Dst IP', 'Timestamp'],
    "CIC-IoT-2025": [],
}

# Label columns (checked in order)
LABEL_COLUMNS = {
    "EDGE_IIoT": ["Attack_type", "Attack_label"],
    "CIC-IoT-2023": ["Label"],
    "Apose-IoT-23": ["label"],
    "BoT-IoT": ["attack", "category", "subcategory"],
    "CIC-IoMT-2024": ["Label"],
    "CIC-IoT-IDAD-2024": ["Label"],
    "CIC-IoT-2025": ["label1", "label_full"],
}

# ============================================================
# EXPERIMENT SETTINGS
# ============================================================

# Training hyperparameters
EPOCHS = 100
BATCH_SIZE = 256
LR = 1e-3
WEIGHT_DECAY = 1e-4
EARLY_STOP_PATIENCE = 8

# Multi-run evaluation
N_RUNS = 5
BASE_SEEDS = list(range(N_RUNS))

# Power consumption proxy (Watts)
POWER_WATTS = float(os.environ.get("POWER_WATTS", "120"))

# ============================================================
# PATHS
# ============================================================

# Optional root for all experiment outputs. If set, all processed and outputs
# directories will be nested under:
#   <OUTPUT_ROOT>/data/processed
#   <OUTPUT_ROOT>/data/outputs
# Configure via environment variable OUTPUT_ROOT or HLC_OUTPUT_ROOT,
# or by editing OUTPUT_ROOT below.
# OUTPUT_ROOT = os.environ.get("OUTPUT_ROOT", "").strip() or os.environ.get("HLC_OUTPUT_ROOT", "").strip()
OUTPUT_ROOT = "E:/HLC"

def _pjoin(*parts) -> str:
    return os.path.normpath(os.path.join(*parts))

if OUTPUT_ROOT:
    DIR_PROCESSED = _pjoin(OUTPUT_ROOT, "data", "processed")
    DIR_OUTPUTS   = _pjoin(OUTPUT_ROOT, "data", "outputs")
else:
    DIR_PROCESSED = "data/processed"
    DIR_OUTPUTS   = "outputs"

# Optional: place models under outputs for consistency when using OUTPUT_ROOT
DIR_MODELS = _pjoin(DIR_OUTPUTS, "models") if OUTPUT_ROOT else "models"

# ============================================================
# XAI / ANALYSIS CONFIG
# ============================================================

# SHAP & LIME sampling parameters
SHAP_SAMPLE_SIZE = 1024
LIME_SAMPLE_SIZE = 32

# ANOVA top-K features to plot
ANOVA_TOPK = 20

# Plot DPI
FIG_DPI = 160
