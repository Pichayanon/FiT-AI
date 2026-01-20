"""
  python train_feet_too_close_squat.py --data dataset_standing --out models/feet_too_close_tcn.pt
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


# -----------------------------
# Utilities
# -----------------------------

def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducible training.

    :param seed: random seed
    :returns: None
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resample_time(x: np.ndarray, target_T: int) -> np.ndarray:
    """Resample a time series to a fixed length.

    Uses linear interpolation per feature.

    :param x: input array (T, D)
    :param target_T: target time length
    :returns: resampled array (target_T, D)
    """
    T, D = x.shape
    if T == target_T:
        return x.astype(np.float32)
    if T < 2:
        return np.repeat(x, target_T, axis=0)[:target_T].astype(np.float32)

    src = np.linspace(0, 1, T)
    dst = np.linspace(0, 1, target_T)
    out = np.zeros((target_T, D), dtype=np.float32)
    for j in range(D):
        out[:, j] = np.interp(dst, src, x[:, j])
    return out


def normalize_per_sample(x: np.ndarray) -> np.ndarray:
    """Normalize a sample across time (per feature).

    :param x: input array (T, D)
    :returns: normalized array (T, D)
    """
    mu = x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True) + 1e-6
    return ((x - mu) / sd).astype(np.float32)


def load_npz(path: str) -> Tuple[np.ndarray, np.ndarray, str]:
    """Load one .npz training file.

    :param path: path to .npz file
    :returns: (keypoints, mask, label)
    """
    z = np.load(path, allow_pickle=True)
    keypoints = z["keypoints"].astype(np.float32)  # (T,33,4)
    mask = z["mask"].astype(np.float32)            # (T,)
    label = str(z["label"])
    return keypoints, mask, label


# -----------------------------
# Feature Set A
# -----------------------------
# MediaPipe Pose indices (33)
L_SHO, R_SHO = 11, 12
L_HIP, R_HIP = 23, 24
L_KNE, R_KNE = 25, 26
L_ANK, R_ANK = 27, 28


def _safe_norm(v: np.ndarray, eps: float = 1e-6) -> float:
    """Safe norm for a vector.

    :param v: input vector
    :param eps: small epsilon
    :returns: norm + eps
    """
    return float(np.sqrt(np.sum(v * v)) + eps)


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    """Distance between two points.

    :param a: point a
    :param b: point b
    :returns: euclidean distance
    """
    return float(np.sqrt(np.sum((a - b) ** 2)))


def _angle_3pts(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle ABC in degrees.

    :param a: point a
    :param b: vertex b
    :param c: point c
    :returns: angle in degrees
    """
    ba = a - b
    bc = c - b
    denom = (_safe_norm(ba) * _safe_norm(bc)) + 1e-6
    cosang = float(np.clip(np.dot(ba, bc) / denom, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosang)))


def extract_features_A(kp: np.ndarray) -> np.ndarray:
    """Extract Feature Set A for each frame.

    :param kp: keypoints (T,33,4) (x,y,z,vis)
    :returns: features (T,111)
    """
    T = kp.shape[0]
    xyz = kp[..., :3].astype(np.float32)  # (T,33,3)

    out_list: List[np.ndarray] = []
    for t in range(T):
        p = xyz[t]

        lhip = p[L_HIP]
        rhip = p[R_HIP]
        lsho = p[L_SHO]
        rsho = p[R_SHO]
        lkne = p[L_KNE]
        rkne = p[R_KNE]
        lank = p[L_ANK]
        rank = p[R_ANK]

        mid_hip = 0.5 * (lhip + rhip)
        hip_w = _dist(lhip, rhip)

        sho_w = _dist(lsho, rsho)
        scale = hip_w if hip_w > 1e-4 else (sho_w if sho_w > 1e-4 else 1.0)

        # A1: body-centric xyz normalized (99)
        p_norm = (p - mid_hip) / (scale + 1e-6)
        feat_xyz = p_norm.reshape(-1).astype(np.float32)

        # A2: distances + ratios (7)
        ankle_w = _dist(lank, rank)
        knee_w = _dist(lkne, rkne)

        ankle_hip = ankle_w / (scale + 1e-6)
        knee_hip = knee_w / (scale + 1e-6)
        sho_hip = sho_w / (scale + 1e-6)

        feat_dist = np.array(
            [hip_w, sho_w, ankle_w, knee_w, ankle_hip, knee_hip, sho_hip],
            dtype=np.float32
        )

        # A3: angles (5)
        knee_L = _angle_3pts(lhip, lkne, lank)
        knee_R = _angle_3pts(rhip, rkne, rank)
        hip_L = _angle_3pts(lsho, lhip, lkne)
        hip_R = _angle_3pts(rsho, rhip, rkne)

        mid_sho = 0.5 * (lsho + rsho)
        v = (mid_sho - mid_hip).astype(np.float32)
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        denom = (_safe_norm(v) * _safe_norm(up)) + 1e-6
        cosang = float(np.clip(np.dot(v, up) / denom, -1.0, 1.0))
        torso_tilt = float(np.degrees(np.arccos(cosang)))

        feat_ang = np.array([knee_L, knee_R, hip_L, hip_R, torso_tilt], dtype=np.float32)

        out = np.concatenate([feat_xyz, feat_dist, feat_ang], axis=0)
        out_list.append(out)

    return np.stack(out_list, axis=0).astype(np.float32)


# -----------------------------
# Dataset
# -----------------------------

@dataclass
class Sample:
    """A single dataset item.

    :param path: path to .npz
    :param label: label string
    """
    path: str
    label: str


class StandingNPZDataset(Dataset):
    """Dataset that reads standing segments from .npz.

    :param samples: list of Sample
    :param label_map: label -> class index
    :param T: resample length
    :param feature_set: "A" or "RAW"
    """

    def __init__(self, samples: List[Sample], label_map: Dict[str, int], T: int, feature_set: str = "A"):
        self.samples = samples
        self.label_map = label_map
        self.T = int(T)
        self.feature_set = str(feature_set)

        if self.feature_set == "RAW":
            self.in_dim = 33 * 4  # 132
        elif self.feature_set == "A":
            self.in_dim = 111
        else:
            raise ValueError(f"Unknown feature_set={self.feature_set}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        kp, mask, label = load_npz(s.path)

        if self.feature_set == "RAW":
            T0 = kp.shape[0]
            x = kp.reshape(T0, -1).astype(np.float32)
        else:
            x = extract_features_A(kp)

        x = resample_time(x, self.T)
        x = normalize_per_sample(x)

        y = self.label_map[label]
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)


def build_splits(npz_paths: List[str], val_ratio: float = 0.2, seed: int = 42) -> Tuple[List[str], List[str]]:
    """Split file paths into train/val.

    :param npz_paths: all .npz paths
    :param val_ratio: validation ratio
    :param seed: random seed
    :returns: (train_paths, val_paths)
    """
    random.Random(seed).shuffle(npz_paths)
    n = len(npz_paths)
    n_val = int(round(n * val_ratio))
    val = npz_paths[:n_val]
    tr = npz_paths[n_val:]
    return tr, val


def infer_labels(npz_paths: List[str]) -> List[str]:
    """Infer labels by reading each .npz.

    :param npz_paths: list of .npz paths
    :returns: list of label strings
    """
    labels: List[str] = []
    for p in npz_paths:
        _, _, lab = load_npz(p)
        labels.append(lab)
    return labels


def make_label_map(labels: List[str]) -> Dict[str, int]:
    """Make label mapping (sorted).

    :param labels: list of label strings
    :returns: mapping label -> int
    """
    uniq = sorted(list(set(labels)))
    return {name: i for i, name in enumerate(uniq)}


# -----------------------------
# TCN
# -----------------------------

class TemporalBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        pad = (k - 1) * dilation
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=k, dilation=dilation, padding=pad)
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=k, dilation=dilation, padding=pad)
        self.act2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)

        self.down = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else None
        self.pad = pad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(x)
        y = y[..., :-self.pad] if self.pad > 0 else y
        y = self.drop1(self.act1(y))

        y = self.conv2(y)
        y = y[..., :-self.pad] if self.pad > 0 else y
        y = self.drop2(self.act2(y))

        res = x if self.down is None else self.down(x)
        res = res[..., -y.shape[-1]:]
        return y + res


class SimpleTCN(nn.Module):
    def __init__(
        self,
        in_dim: int,
        num_classes: int = 2,
        channels: Tuple[int, int, int] = (128, 128, 128),
        k: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        layers: List[nn.Module] = []
        ch_in = in_dim
        dilation = 1
        for ch_out in channels:
            layers.append(TemporalBlock(ch_in, ch_out, k=k, dilation=dilation, dropout=dropout))
            ch_in = ch_out
            dilation *= 2

        self.tcn = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(channels[-1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,T,D) -> (B,D,T)
        x = x.transpose(1, 2)
        y = self.tcn(x)
        y = self.pool(y).squeeze(-1)
        return self.fc(y)


# -----------------------------
# Train / Eval
# -----------------------------

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    """Evaluate model.

    :param model: pytorch model
    :param loader: dataloader
    :param device: cpu/cuda
    :returns: (loss_avg, acc)
    """
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    ce = nn.CrossEntropyLoss()

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = ce(logits, y)
        loss_sum += float(loss.item()) * x.size(0)

        pred = torch.argmax(logits, dim=1)
        correct += int((pred == y).sum().item())
        total += x.size(0)

    acc = correct / max(1, total)
    loss_avg = loss_sum / max(1, total)
    return loss_avg, acc


def train(args: argparse.Namespace) -> None:
    """Train a TCN model for feet-too-close classification.

    :param args: cli args
    :returns: None
    :raises: RuntimeError if no .npz found
    """
    set_seed(args.seed)

    npz_paths = sorted(glob.glob(os.path.join(args.data, "*.npz")))
    if len(npz_paths) == 0:
        raise RuntimeError(f"No .npz found in: {args.data}")

    labels = infer_labels(npz_paths)
    label_map = make_label_map(labels)

    print("[DATA] total:", len(npz_paths))
    print("[DATA] labels:", label_map)
    print("[FEAT] feature_set:", args.feature_set)

    tr_paths, va_paths = build_splits(npz_paths, val_ratio=args.val_ratio, seed=args.seed)

    tr_samples = [Sample(p, load_npz(p)[2]) for p in tr_paths]
    va_samples = [Sample(p, load_npz(p)[2]) for p in va_paths]

    train_ds = StandingNPZDataset(tr_samples, label_map, T=args.T, feature_set=args.feature_set)
    val_ds = StandingNPZDataset(va_samples, label_map, T=args.T, feature_set=args.feature_set)

    train_loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.bs, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print("[DEVICE]", device)

    model = SimpleTCN(
        in_dim=int(train_ds.in_dim),
        num_classes=len(label_map),
        channels=(args.ch, args.ch, args.ch),
        dropout=float(args.dropout),
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.wd))
    ce = nn.CrossEntropyLoss()

    best_val_acc = -1.0
    best_state = None

    for ep in range(1, int(args.epochs) + 1):
        model.train()
        total = 0
        correct = 0
        loss_sum = 0.0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            opt.zero_grad()
            logits = model(x)
            loss = ce(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            loss_sum += float(loss.item()) * x.size(0)
            pred = torch.argmax(logits, dim=1)
            correct += int((pred == y).sum().item())
            total += x.size(0)

        tr_loss = loss_sum / max(1, total)
        tr_acc = correct / max(1, total)

        va_loss, va_acc = evaluate(model, val_loader, device)

        print(
            f"Epoch {ep:03d}/{args.epochs} | "
            f"train loss={tr_loss:.4f} acc={tr_acc:.3f} | "
            f"val loss={va_loss:.4f} acc={va_acc:.3f}"
        )

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print("  -> best updated")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    feat_dim_detail: Dict[str, int] = {}
    if args.feature_set == "RAW":
        feat_dim_detail = {"raw_xyzw": 132}
    elif args.feature_set == "A":
        feat_dim_detail = {"bodycentric_xyz": 99, "dist_ratio": 7, "angles": 5, "total": 111}

    ckpt = {
        "model_state": best_state if best_state is not None else model.state_dict(),
        "in_dim": int(train_ds.in_dim),
        "T": int(args.T),
        "label_map": label_map,
        "meta": {
            "task": "feet_too_close_squat",
            "feature_set": str(args.feature_set),
            "feat_dim_detail": feat_dim_detail,
            "data_dir": os.path.abspath(args.data),
            "val_ratio": float(args.val_ratio),
            "epochs": int(args.epochs),
            "batch_size": int(args.bs),
            "lr": float(args.lr),
            "wd": float(args.wd),
            "channels": [int(args.ch), int(args.ch), int(args.ch)],
            "dropout": float(args.dropout),
            "seed": int(args.seed),
            "best_val_acc": float(best_val_acc),
        },
    }

    torch.save(ckpt, args.out)
    print("[SAVE] checkpoint:", args.out)
    print("[SAVE] best_val_acc:", best_val_acc)
    print("[SAVE] label_map:", json.dumps(label_map, ensure_ascii=False))
    print("[SAVE] in_dim:", train_ds.in_dim)


def main() -> None:
    """CLI entrypoint.

    :returns: None
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="folder with *.npz (dataset_standing)")
    ap.add_argument("--out", required=True, help="output .pt path e.g. models/feet_too_close_tcn.pt")

    ap.add_argument("--T", type=int, default=30, help="resample length")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--ch", type=int, default=128)

    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cpu", action="store_true")

    ap.add_argument(
        "--feature_set",
        type=str,
        default="A",
        choices=["A", "RAW"],
        help="RAW=132 (x,y,z,vis), A=111 (body-centric xyz + distances + angles)",
    )

    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()