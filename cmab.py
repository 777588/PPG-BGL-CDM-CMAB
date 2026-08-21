"""CMAB regression network with Conv1D encoding, multi-view attention, three cascaded BiLSTMs, cross-layer Hadamard interactions, and a fully connected regression head."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvEncoder1D(nn.Module):
    def __init__(self, dropout: float = 0.3):
        super().__init__()
        channels = [1, 64, 128, 256, 512]
        blocks = []
        for cin, cout in zip(channels[:-1], channels[1:]):
            blocks.extend([
                nn.Conv1d(cin, cout, kernel_size=5, stride=1, padding=2),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=2, stride=2),
                nn.Dropout(dropout),
            ])
        self.net = nn.Sequential(*blocks)

    def forward(self, x):
        return self.net(x)


class LocalChannelAttention(nn.Module):
    """Apply global average pooling over time, followed by a kernel-size-3 Conv1D over the channel descriptor."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=True)

    def forward(self, x):
        gap = x.mean(dim=-1)
        a = torch.sigmoid(self.conv(gap.unsqueeze(1))).squeeze(1).unsqueeze(-1)
        return a


class TemporalSelfAttention(nn.Module):
    def __init__(self, channels: int = 512, heads: int = 4):
        super().__init__()
        if channels % heads != 0:
            raise ValueError("channels must be divisible by heads")
        self.attn = nn.MultiheadAttention(channels, heads, batch_first=True)

    def forward(self, x):
        seq = x.transpose(1, 2)
        y, _ = self.attn(seq, seq, seq, need_weights=False)
        y = y.transpose(1, 2)
        return y.mean(dim=1, keepdim=True)  # Do not apply a separate sigmoid to temporal attention before fusion.


class MultiViewAttention(nn.Module):
    def __init__(self, channels: int = 512, heads: int = 4, lambda_att: float = 0.4):
        super().__init__()
        self.channel_att = LocalChannelAttention()
        self.temporal_att = TemporalSelfAttention(channels, heads)
        self.lambda_att = float(lambda_att)

    def forward(self, x):
        ac = self.channel_att(x)
        at = self.temporal_att(x)
        joint = ac * at  # Broadcast and multiply the channel and temporal attention responses element-wise.
        a = torch.sigmoid(self.lambda_att * joint)
        return x * a


class CMABRegressor(nn.Module):
    def __init__(
        self,
        h1: int = 12,
        h2: int = 16,
        h3: int = 20,
        fc_dropout: float = 0.32,
        lambda_att: float = 0.45,
        projection_dim: int = 128,
    ):
        super().__init__()
        self.encoder = ConvEncoder1D(dropout=0.3)
        self.attention = MultiViewAttention(channels=512, heads=4, lambda_att=lambda_att)

        self.lstm1 = nn.LSTM(512, h1, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(2 * h1, h2, batch_first=True, bidirectional=True)
        self.lstm3 = nn.LSTM(2 * h2, h3, batch_first=True, bidirectional=True)

        d = projection_dim
        self.proj1 = nn.Linear(2 * h1, d)
        self.proj2 = nn.Linear(2 * h2, d)
        self.proj3 = nn.Linear(2 * h3, d)

        self.fc1 = nn.Linear(6 * d, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)
        self.drop = nn.Dropout(fc_dropout)

    def forward(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(1)
        x = self.encoder(x)
        x = self.attention(x)
        seq = x.transpose(1, 2)

        h1_seq, _ = self.lstm1(seq)
        h2_seq, _ = self.lstm2(h1_seq)
        h3_seq, _ = self.lstm3(h2_seq)

        z1 = self.proj1(h1_seq.mean(dim=1))
        z2 = self.proj2(h2_seq.mean(dim=1))
        z3 = self.proj3(h3_seq.mean(dim=1))

        p12 = z1 * z2
        p23 = z2 * z3
        p31 = z3 * z1
        fused = torch.cat([z1, z2, z3, p12, p23, p31], dim=-1)

        x = self.drop(F.relu(self.fc1(fused)))
        x = self.drop(F.relu(self.fc2(x)))
        x = self.drop(F.relu(self.fc3(x)))
        return self.fc4(x).squeeze(-1)


if __name__ == "__main__":
    model = CMABRegressor()
    dummy = torch.randn(2, 1, 10875)
    with torch.no_grad():
        y = model(dummy)
    print("output shape:", y.shape)
    print("parameters:", sum(p.numel() for p in model.parameters()))
