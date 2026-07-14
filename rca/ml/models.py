from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MetricAutoencoder(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 16, bottleneck: int = 4) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, bottleneck), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, hidden), nn.ReLU(), nn.Linear(hidden, in_dim)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


class DeepLog(nn.Module):
    def __init__(self, num_keys: int, hidden_size: int = 64, num_layers: int = 2, input_size: int = 1) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_keys)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])


class RCAScorer(nn.Module):
    def __init__(self, in_channels: int, hidden: int = 64, dropout: float = 0.3) -> None:
        super().__init__()
        from torch_geometric.nn import GCNConv

        self.dropout = dropout
        self.conv1 = GCNConv(in_channels, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.relu(self.conv2(h, edge_index))
        return self.head(h).squeeze(-1)
