from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# -----------------------------
# Dataset
# -----------------------------


class PhaseDataset(Dataset):
    def __init__(self, data_dir: str, window: int = 30, stride: int = 5):
        self.samples = []

        files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        assert files, f"No .npz files found in {data_dir}"

        for f in files:
            d = np.load(f)
            X = d["features"]  # (T, 9)
            y = d["labels"]  # (T,) → values {0,1}
            m = d["mask"]  # (T,)

            T = len(X)
            for i in range(0, T - window + 1, stride):
                xs = X[i : i + window]
                ys = y[i : i + window]
                ms = m[i : i + window]

                if ms.sum() == 0:
                    continue

                self.samples.append((xs, ys))

        print(f"Loaded {len(self.samples)} samples from {len(files)} files")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return (
            torch.from_numpy(x).float(),  # (W, 9)
            torch.from_numpy(y).long(),  # (W,)
        )


# -----------------------------
# TCN Model
# -----------------------------


class TemporalBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, d: int = 1):
        super().__init__()
        pad = (k - 1) * d

        self.conv1 = nn.Conv1d(in_ch, out_ch, k, padding=pad, dilation=d)
        self.conv2 = nn.Conv1d(out_ch, out_ch, k, padding=pad, dilation=d)
        self.relu = nn.ReLU()

        self.down = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.relu(self.conv1(x))
        y = self.relu(self.conv2(y))

        if self.down:
            x = self.down(x)

        # Ensure same temporal length
        return y[..., : x.size(-1)] + x


class SimpleTCN(nn.Module):
    def __init__(self, in_dim: int = 9, num_classes: int = 2):
        super().__init__()

        self.tcn = nn.Sequential(
            TemporalBlock(in_dim, 64, d=1),
            TemporalBlock(64, 64, d=2),
            TemporalBlock(64, 64, d=4),
        )

        self.fc = nn.Conv1d(64, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, W, F)
        x = x.transpose(1, 2)  # (B, F, W)
        x = self.tcn(x)
        x = self.fc(x)
        return x.transpose(1, 2)  # (B, W, C)


# -----------------------------
# Train
# -----------------------------


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = PhaseDataset(args.data, args.window, args.stride)
    dl = DataLoader(ds, batch_size=args.bs, shuffle=True)

    model = SimpleTCN(in_dim=9, num_classes=2).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        for x, y in dl:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)  # (B, W, 2)

            loss = crit(
                logits.reshape(-1, 2),
                y.reshape(-1),
            )

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()

        print(
            f"Epoch {epoch:03d} | "
            f"loss={total_loss / max(len(dl), 1):.4f}"
        )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "in_dim": 9,
            "num_classes": 2,
            "window": args.window,
        },
        args.out,
    )
    print("Saved model to:", args.out)


# -----------------------------
# CLI
# -----------------------------


def main():
    ap = argparse.ArgumentParser(
        description="Train lunge side-view phase TCN"
    )
    ap.add_argument("--data", required=True, help="Directory of .npz phase files")
    ap.add_argument("--out", required=True, help="Output .pt model path")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--window", type=int, default=30)
    ap.add_argument("--stride", type=int, default=5)
    args = ap.parse_args()

    train(args)


if __name__ == "__main__":
    main()

