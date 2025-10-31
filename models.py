"""
models.py — Hybrid Deep Learning Models for IoT/IIoT IDS
Author: Dr. Vanlalruata Hnamte
Version: 1.0
"""

import torch
import torch.nn as nn


# ============================================================
# PARALLEL FUSION CNN + LSTM MODEL
# ============================================================

class CNN_LSTM_Fusion(nn.Module):
    """
    Parallel CNN + LSTM hybrid for intrusion detection.

    The model splits the feature vector into two branches:
      • CNN branch learns spatial correlations among features.
      • LSTM branch models sequential/temporal relationships.
    Both embeddings are concatenated before dense layers.
    """

    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 64):
        super().__init__()

        # CNN branch
        self.cnn_branch = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1)   # global average pooling
        )

        # LSTM branch
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        # Fusion + classification layers
        fusion_input = 64 + (hidden_dim * 2)
        self.fc_layers = nn.Sequential(
            nn.Linear(fusion_input, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        """
        x: [batch_size, input_dim]
        Returns logits: [batch_size, num_classes]
        """
        # Reshape for Conv1d: [B, C=1, L=input_dim]
        x_cnn = x.unsqueeze(1)
        x_cnn = self.cnn_branch(x_cnn)
        x_cnn = x_cnn.view(x_cnn.size(0), -1)

        # Reshape for LSTM: [B, seq_len=input_dim, features=1]
        x_lstm = x.unsqueeze(-1)
        _, (h_n, _) = self.lstm(x_lstm)
        x_lstm = torch.cat((h_n[0], h_n[1]), dim=1)  # bidirectional concat

        # Fusion
        fused = torch.cat((x_cnn, x_lstm), dim=1)
        out = self.fc_layers(fused)
        return out


# ============================================================
# SIMPLE BASELINE (OPTIONAL)
# ============================================================

class SimpleMLP(nn.Module):
    """
    Simple multilayer perceptron baseline for comparison.
    """

    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)
