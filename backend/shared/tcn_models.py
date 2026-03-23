from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class TemporalBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        k: int = 3,
        dilation: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        pad = (k - 1) * dilation
        self.conv1 = nn.Conv1d(
            in_ch, out_ch, kernel_size=k, dilation=dilation, padding=pad
        )
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            out_ch, out_ch, kernel_size=k, dilation=dilation, padding=pad
        )
        self.act2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)

        self.down = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else None
        self.pad = pad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(x)
        y = y[..., : -self.pad] if self.pad > 0 else y
        y = self.drop1(self.act1(y))

        y = self.conv2(y)
        y = y[..., : -self.pad] if self.pad > 0 else y
        y = self.drop2(self.act2(y))

        residual = x if self.down is None else self.down(x)
        residual = residual[..., -y.shape[-1] :]
        return y + residual


class SimpleTCN(nn.Module):
    def __init__(
        self,
        in_dim: int,
        num_classes: int = 2,
        channels: Tuple[int, ...] = (128, 128, 128),
        k: int = 3,
        dropout: float = 0.1,
        use_attention: bool = True,
    ) -> None:
        super().__init__()
        layers = []
        ch_in = in_dim
        dilation = 1
        for ch_out in channels:
            layers.append(
                TemporalBlock(ch_in, ch_out, k=k, dilation=dilation, dropout=dropout)
            )
            ch_in = ch_out
            dilation *= 2
        self.tcn = nn.Sequential(*layers)
        self.use_attention = bool(use_attention)
        self.pool = None if self.use_attention else nn.AdaptiveAvgPool1d(1)
        self.attention = nn.Linear(channels[-1], 1) if self.use_attention else None
        self.fc = nn.Linear(channels[-1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        y = self.tcn(x)
        if self.use_attention:
            y = y.transpose(1, 2)
            scores = self.attention(y).squeeze(-1)
            weights = torch.softmax(scores, dim=-1)
            context = (y * weights.unsqueeze(-1)).sum(dim=1)
        else:
            context = self.pool(y).squeeze(-1)
        return self.fc(context)


class PhaseTemporalBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        k: int = 3,
        d: int = 1,
    ) -> None:
        super().__init__()
        pad = (k - 1) * d
        self.conv1 = nn.Conv1d(in_ch, out_ch, k, padding=pad, dilation=d)
        self.conv2 = nn.Conv1d(out_ch, out_ch, k, padding=pad, dilation=d)
        self.relu = nn.ReLU()
        self.down = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.relu(self.conv1(x))
        y = self.relu(self.conv2(y))
        if self.down is not None:
            x = self.down(x)
        return y[..., : x.size(-1)] + x


class PhaseTCN(nn.Module):
    def __init__(self, in_dim: int = 10, num_classes: int = 2) -> None:
        super().__init__()
        self.tcn = nn.Sequential(
            PhaseTemporalBlock(in_dim, 64, d=1),
            PhaseTemporalBlock(64, 64, d=2),
            PhaseTemporalBlock(64, 64, d=4),
        )
        self.fc = nn.Conv1d(64, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.tcn(x)
        x = self.fc(x)
        return x.transpose(1, 2)
