"""
models.py — Model architectures for IDS
Author: Dr. Vanlalruata Hnamte
Version: 1.0

Implements:
- CNN_LSTM_Fusion: parallel feature extraction via CNN and LSTM branches
- SimpleMLP: baseline dense network for ablation and benchmark
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

    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 64):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim

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

        # Fusion & Classifier
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.drop = nn.Dropout(0.25)
        self.out = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # Ensure [B, input_dim]
        if x.dim() == 1:
            x = x.unsqueeze(0)

        # CNN branch
        cnn_in = F.relu(self.cnn_fc(x)).unsqueeze(1)  # [B, 1, 2H]
        cnn_feat = self.cnn_pool(F.relu(self.cnn_bn(self.cnn_conv(cnn_in))))  # [B, 32, 16]
        cnn_feat = cnn_feat.flatten(1)
        cnn_feat = F.relu(self.cnn_fc_out(cnn_feat))  # [B, H]

        # LSTM branch
        lstm_in = F.relu(self.lstm_fc(x)).unsqueeze(1)  # [B, 1, 2H]
        lstm_out, _ = self.lstm(lstm_in)
        lstm_feat = F.relu(self.lstm_fc_out(lstm_out[:, -1, :]))  # [B, H]

        # Fusion
        fused = torch.cat([cnn_feat, lstm_feat], dim=1)
        fused = F.relu(self.fc1(fused))
        fused = self.drop(fused)
        return self.out(fused)


# ============================================================
# 2️⃣ Simple MLP Baseline
# ============================================================

class SimpleMLP(nn.Module):
    """A minimal baseline dense network."""
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x):
        return self.net(x)
