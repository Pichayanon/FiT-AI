from __future__ import annotations

if __package__ in {None, ""}:
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import argparse
import glob
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from shared.tcn_models import PhaseTCN


# -----------------------------
# Dataset
# -----------------------------


class PhaseDataset(Dataset):
    def __init__(self, data_dir: str, window: int = 30, stride: int = 5):
        self.samples = []
        self.in_dim = None

        npz_paths = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        assert npz_paths, f"No .npz files found in {data_dir}"

        for npz_path in npz_paths:
            dataset = np.load(npz_path)
            feature_matrix = dataset["features"]  # (T, F)
            label_array = dataset["labels"]  # (T,) → values {0,1}
            mask_array = dataset["mask"]  # (T,)

            if self.in_dim is None:
                self.in_dim = int(feature_matrix.shape[1])
            elif int(feature_matrix.shape[1]) != int(self.in_dim):
                raise ValueError(
                    f"Inconsistent feature dimension in {npz_path}: "
                    f"got {feature_matrix.shape[1]}, expected {self.in_dim}"
                )

            sequence_length = len(feature_matrix)
            for start_index in range(0, sequence_length - window + 1, stride):
                window_features = feature_matrix[start_index : start_index + window]
                window_labels = label_array[start_index : start_index + window]
                window_mask = mask_array[start_index : start_index + window]

                if window_mask.sum() == 0:
                    continue

                self.samples.append((window_features, window_labels))

        print(f"Loaded {len(self.samples)} samples from {len(npz_paths)} files")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        window_features, window_labels = self.samples[idx]
        return (
            torch.from_numpy(window_features).float(),  # (W, F)
            torch.from_numpy(window_labels).long(),  # (W,)
        )


# -----------------------------
# Train
# -----------------------------


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    phase_dataset = PhaseDataset(args.data, args.window, args.stride)
    data_loader = DataLoader(phase_dataset, batch_size=args.bs, shuffle=True)

    if phase_dataset.in_dim is None:
        raise RuntimeError("Dataset appears empty; cannot infer feature dimension.")

    model = PhaseTCN(in_dim=phase_dataset.in_dim, num_classes=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        for feature_window, label_window in data_loader:
            feature_window = feature_window.to(device)
            label_window = label_window.to(device)

            logits = model(feature_window)  # (B, W, 2)

            loss = criterion(
                logits.reshape(-1, 2),
                label_window.reshape(-1),
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(
            f"Epoch {epoch:03d} | "
            f"loss={total_loss / max(len(data_loader), 1):.4f}"
        )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "in_dim": int(phase_dataset.in_dim),
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
