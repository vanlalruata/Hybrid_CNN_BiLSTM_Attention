"""
models.py — Model architectures for IDS
Author: Dr. Vanlalruata Hnamte
Version: 1.1

Implements:
- CNN_LSTM_Fusion: parallel feature extraction via CNN and LSTM branches
- DNN: 4-layer deep neural network baseline for ablation and benchmark
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1️⃣ Parallel CNN + LSTM Fusion
# ============================================================

class CNN_LSTM_Fusion(nn.Module):
    """
    Parallel Fusion Architecture combining CNN and LSTM encoders.

    Architecture:
      ├─ Input: [batch, features]
      ├─ Branch 1 (CNN):   Linear → reshape → Conv1D → Flatten → Dense
      ├─ Branch 2 (LSTM):  Reshape → LSTM(hidden=64) → Last hidden
      ├─ Fusion: Concatenate([CNN_out, LSTM_out])
      └─ Classifier: Dense → Dropout → Dense(num_classes)
    """

    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 64, cnn_only: bool = False, lstm_only: bool = False, no_gating: bool = False):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.cnn_only = cnn_only
        self.lstm_only = lstm_only
        self.no_gating = no_gating

        # CNN branch
        self.cnn_fc = nn.Linear(input_dim, hidden_dim * 2)
        self.cnn_conv = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.cnn_bn = nn.BatchNorm1d(32)
        self.cnn_pool = nn.AdaptiveMaxPool1d(16)
        self.cnn_fc_out = nn.Linear(32 * 16, hidden_dim)

        # LSTM branch
        self.lstm_fc = nn.Linear(input_dim, hidden_dim * 2)
        self.lstm = nn.LSTM(hidden_dim * 2, hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.lstm_fc_out = nn.Linear(hidden_dim * 2, hidden_dim)

        # Feature gating layer (dynamic feature refinement / gating mechanism)
        self.gate_fc = nn.Linear(hidden_dim * 2, hidden_dim * 2)

        # Fusion & Classifier
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.drop = nn.Dropout(0.25)
        self.out = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # Ensure [B, input_dim]
        if x.dim() == 1:
            x = x.unsqueeze(0)

        # CNN branch
        if not self.lstm_only:
            cnn_in = F.relu(self.cnn_fc(x)).unsqueeze(1)  # [B, 1, 2H]
            cnn_feat = self.cnn_pool(F.relu(self.cnn_bn(self.cnn_conv(cnn_in))))  # [B, 32, 16]
            cnn_feat = cnn_feat.flatten(1)
            cnn_feat = F.relu(self.cnn_fc_out(cnn_feat))  # [B, H]
        else:
            cnn_feat = torch.zeros(x.shape[0], self.hidden_dim, device=x.device, dtype=x.dtype)

        # LSTM branch
        if not self.cnn_only:
            lstm_in = F.relu(self.lstm_fc(x)).unsqueeze(1)  # [B, 1, 2H]
            lstm_out, _ = self.lstm(lstm_in)
            lstm_feat = F.relu(self.lstm_fc_out(lstm_out[:, -1, :]))  # [B, H]
        else:
            lstm_feat = torch.zeros(x.shape[0], self.hidden_dim, device=x.device, dtype=x.dtype)

        # Fusion
        f = torch.cat([cnn_feat, lstm_feat], dim=1)  # [B, hidden_dim * 2]
        
        # Gating / Dynamic Feature Refinement
        if not self.no_gating:
            e = torch.tanh(self.gate_fc(f))
            alpha = torch.softmax(e, dim=1)
            f_prime = alpha * f
        else:
            f_prime = f
        
        fused = F.relu(self.fc1(f_prime))
        fused = self.drop(fused)
        return self.out(fused)


# ============================================================
# 2️⃣ DNN Baseline (4 hidden layers)
# ============================================================

class DNN(nn.Module):
    """
    4-layer Deep Neural Network baseline for tabular IDS:
      Input -> [BN] -> Dense -> ReLU -> Dropout
                   -> Dense -> ReLU -> Dropout
                   -> Dense -> ReLU -> Dropout
                   -> Dense -> ReLU -> Dropout
                   -> Output(num_classes)
    Notes:
      - Uses progressively decreasing hidden sizes by default.
      - BatchNorm on input to stabilize training on standardized features.
    """
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        h1: int = 512,
        h2: int = 256,
        h3: int = 128,
        h4: int = 64,
        p1: float = 0.30,
        p2: float = 0.25,
        p3: float = 0.20,
        p4: float = 0.15
    ):
        super().__init__()
        self.bn0 = nn.BatchNorm1d(input_dim)
        self.fc1 = nn.Linear(input_dim, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, h3)
        self.fc4 = nn.Linear(h3, h4)
        self.out = nn.Linear(h4, num_classes)
        self.drop1 = nn.Dropout(p1)
        self.drop2 = nn.Dropout(p2)
        self.drop3 = nn.Dropout(p3)
        self.drop4 = nn.Dropout(p4)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = self.bn0(x)
        x = F.relu(self.fc1(x)); x = self.drop1(x)
        x = F.relu(self.fc2(x)); x = self.drop2(x)
        x = F.relu(self.fc3(x)); x = self.drop3(x)
        x = F.relu(self.fc4(x)); x = self.drop4(x)
        return self.out(x)

# Backward compatibility: keep the previous name working until callers migrate
SimpleMLP = DNN
