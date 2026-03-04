"""
train_squat_stand.py

Train a TCN to classify standing stance:
  good_stand | stand_too_narrow | stand_too_wide

Focused Feature Set (13 dims per frame):
  - 4 width ratios   : ankle/hip, ankle/sho, knee/hip, knee/sho
  - 6 X-positions     : L/R ankle, knee, hip (normalized by hip_width)
  - 3 angles          : knee_L, knee_R, torso_tilt (all / 180)

Usage:
  python squat/train_squat_stand.py \
    --data dataset/squat/dataset_standing \
    --out  squat/models/squat_stand_tcn.pt
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


def _confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true.tolist(), y_pred.tolist()):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1
    return cm


def _classification_report(cm: np.ndarray) -> Dict[str, Any]:
    """Return per-class precision/recall/f1/support + macro/weighted averages."""
    num_classes = int(cm.shape[0])
    support = cm.sum(axis=1).astype(np.int64)
    pred_sum = cm.sum(axis=0).astype(np.int64)
    tp = np.diag(cm).astype(np.int64)
    fp = (pred_sum - tp).astype(np.int64)
    fn = (support - tp).astype(np.int64)

    prec = np.zeros((num_classes,), dtype=np.float64)
    rec = np.zeros((num_classes,), dtype=np.float64)
    f1 = np.zeros((num_classes,), dtype=np.float64)

    for i in range(num_classes):
        p_den = tp[i] + fp[i]
        r_den = tp[i] + fn[i]
        prec[i] = float(tp[i] / p_den) if p_den > 0 else 0.0
        rec[i] = float(tp[i] / r_den) if r_den > 0 else 0.0
        denom = prec[i] + rec[i]
        f1[i] = float(2.0 * prec[i] * rec[i] / denom) if denom > 0 else 0.0

    total = int(cm.sum())
    acc = float(tp.sum() / total) if total > 0 else 0.0
    macro_f1 = float(f1.mean()) if num_classes > 0 else 0.0
    weighted_f1 = float((f1 * support).sum() / max(1, support.sum()))

    return {
        "accuracy": acc,
        "per_class": {
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "support": support,
        },
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }


def _print_eval_report(cm: np.ndarray, idx_to_label: Dict[int, str]) -> None:
    rep = _classification_report(cm)
    prec = rep["per_class"]["precision"]
    rec = rep["per_class"]["recall"]
    f1 = rep["per_class"]["f1"]
    sup = rep["per_class"]["support"]

    print("\n[VAL] Confusion matrix (rows=true, cols=pred):")
    print(cm)
    print("\n[VAL] Per-class metrics:")
    print(f"{'class':<22} {'support':>7} {'precision':>9} {'recall':>7} {'f1':>7}")
    for i in range(cm.shape[0]):
        name = idx_to_label.get(int(i), str(i))
        print(f"{name:<22} {int(sup[i]):>7} {prec[i]:>9.3f} {rec[i]:>7.3f} {f1[i]:>7.3f}")

    print("\n[VAL] Summary:")
    print(f"  accuracy   : {rep['accuracy']:.4f}")
    print(f"  macro_f1   : {rep['macro_f1']:.4f}")
    print(f"  weighted_f1: {rep['weighted_f1']:.4f}")


# -----------------------------
# Constants
# -----------------------------
FEATURE_DIM = 16

# MediaPipe Pose indices (only the ones we need)
L_SHO, R_SHO = 11, 12
L_HIP, R_HIP = 23, 24
L_KNE, R_KNE = 25, 26
L_ANK, R_ANK = 27, 28


# -----------------------------
# Utilities
# -----------------------------

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resample_time(x: np.ndarray, target_T: int) -> np.ndarray:
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
    mu = x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True) + 1e-6
    return ((x - mu) / sd).astype(np.float32)


def load_npz(path: str) -> Tuple[np.ndarray, np.ndarray, str]:
    z = np.load(path, allow_pickle=True)
    keypoints = z["keypoints"].astype(np.float32)  # (T,33,4)
    mask = z["mask"].astype(np.float32)            # (T,)
    label = str(z["label"])
    return keypoints, mask, label


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.sum((a - b) ** 2)))


def _safe_norm(v: np.ndarray, eps: float = 1e-6) -> float:
    return float(np.sqrt(np.sum(v * v)) + eps)


def _angle_3pts(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ba = a - b
    bc = c - b
    denom = (_safe_norm(ba) * _safe_norm(bc)) + 1e-6
    cosang = float(np.clip(np.dot(ba, bc) / denom, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosang)))


# -----------------------------
# Feature Extraction (16 dims)
# -----------------------------

def extract_stand_features(kp: np.ndarray) -> np.ndarray:
    """Extract stance-focused features.

    Per frame (16 dims):
      [0]  ankle_w / hip_w
      [1]  ankle_w / sho_w
      [2]  knee_w  / hip_w
      [3]  knee_w  / sho_w
      [4]  L ankle X (norm)
      [5]  R ankle X (norm)
      [6]  L knee  X (norm)
      [7]  R knee  X (norm)
      [8]  L hip   X (norm)
      [9]  R hip   X (norm)
      [10] knee angle L  / 180
      [11] knee angle R  / 180
      [12] torso tilt    / 180
      [13] feet distance (ankle_w / scale)
      [14] shoulder distance (sho_w / scale)
      [15] feet/shoulder ratio (ankle_w / sho_w)

    :param kp: (T, 33, 4)
    :returns: (T, 16)
    """
    T = kp.shape[0]
    xyz = kp[..., :3].astype(np.float32)
    out = np.zeros((T, FEATURE_DIM), dtype=np.float32)

    for t in range(T):
        p = xyz[t]
        lhip, rhip = p[L_HIP], p[R_HIP]
        lsho, rsho = p[L_SHO], p[R_SHO]
        lkne, rkne = p[L_KNE], p[R_KNE]
        lank, rank = p[L_ANK], p[R_ANK]

        mid_hip = 0.5 * (lhip + rhip)
        hip_w = _dist(lhip, rhip)
        sho_w = _dist(lsho, rsho)
        ankle_w = _dist(lank, rank)
        knee_w = _dist(lkne, rkne)
        scale = hip_w if hip_w > 1e-4 else (sho_w if sho_w > 1e-4 else 1.0)

        # Width ratios (the core signal for narrow/wide)
        out[t, 0] = ankle_w / (hip_w + 1e-6)
        out[t, 1] = ankle_w / (sho_w + 1e-6)
        out[t, 2] = knee_w / (hip_w + 1e-6)
        out[t, 3] = knee_w / (sho_w + 1e-6)

        # Normalized X positions (left-right spread)
        out[t, 4] = (lank[0] - mid_hip[0]) / (scale + 1e-6)
        out[t, 5] = (rank[0] - mid_hip[0]) / (scale + 1e-6)
        out[t, 6] = (lkne[0] - mid_hip[0]) / (scale + 1e-6)
        out[t, 7] = (rkne[0] - mid_hip[0]) / (scale + 1e-6)
        out[t, 8] = (lhip[0] - mid_hip[0]) / (scale + 1e-6)
        out[t, 9] = (rhip[0] - mid_hip[0]) / (scale + 1e-6)

        # Angles (sanity: standing knees should be ~170-180)
        out[t, 10] = _angle_3pts(lhip, lkne, lank) / 180.0
        out[t, 11] = _angle_3pts(rhip, rkne, rank) / 180.0

        # Torso tilt
        mid_sho = 0.5 * (lsho + rsho)
        v = (mid_sho - mid_hip).astype(np.float32)
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        denom = (_safe_norm(v) * _safe_norm(up)) + 1e-6
        cosang = float(np.clip(np.dot(v, up) / denom, -1.0, 1.0))
        out[t, 12] = float(np.degrees(np.arccos(cosang))) / 180.0

        out[t, 13] = ankle_w / (scale + 1e-6)
        out[t, 14] = sho_w / (scale + 1e-6)
        out[t, 15] = ankle_w / (sho_w + 1e-6)

    return out


# -----------------------------
# Dataset
# -----------------------------

@dataclass
class Sample:
    path: str
    label: str


class StandingNPZDataset(Dataset):
    def __init__(self, samples: List[Sample], label_map: Dict[str, int], T: int):
        self.samples = samples
        self.label_map = label_map
        self.T = int(T)
        self.in_dim = FEATURE_DIM

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        kp, mask, label = load_npz(s.path)
        x = extract_stand_features(kp)
        x = resample_time(x, self.T)
        # REMOVED: x = normalize_per_sample(x)  (destroys spatial proportions)
        y = self.label_map[label]
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)


def build_splits(npz_paths: List[str], val_ratio: float = 0.2, seed: int = 42) -> Tuple[List[str], List[str]]:
    random.Random(seed).shuffle(npz_paths)
    n = len(npz_paths)
    n_val = int(round(n * val_ratio))
    return npz_paths[n_val:], npz_paths[:n_val]


def infer_labels(npz_paths: List[str]) -> List[str]:
    return [load_npz(p)[2] for p in npz_paths]


def make_label_map(labels: List[str]) -> Dict[str, int]:
    uniq = sorted(set(labels))
    return {name: i for i, name in enumerate(uniq)}


def count_label_dist(npz_paths: List[str]) -> Dict[str, int]:
    c: Dict[str, int] = {}
    for p in npz_paths:
        lab = load_npz(p)[2]
        c[lab] = c.get(lab, 0) + 1
    return c


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


@torch.no_grad()
def evaluate_with_preds(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Evaluate model and also return (y_true, y_pred) for reporting."""
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    ce = nn.CrossEntropyLoss()

    y_true: List[int] = []
    y_pred: List[int] = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = ce(logits, y)
        loss_sum += float(loss.item()) * x.size(0)

        pred = torch.argmax(logits, dim=1)
        correct += int((pred == y).sum().item())
        total += x.size(0)

        y_true.extend(y.detach().cpu().numpy().tolist())
        y_pred.extend(pred.detach().cpu().numpy().tolist())

    acc = correct / max(1, total)
    loss_avg = loss_sum / max(1, total)
    return loss_avg, acc, np.asarray(y_true, dtype=np.int64), np.asarray(y_pred, dtype=np.int64)


def train(args: argparse.Namespace) -> None:
    """Train a TCN for squat standing classification."""
    set_seed(args.seed)

    npz_paths = sorted(glob.glob(os.path.join(args.data, "*.npz")))
    if len(npz_paths) == 0:
        raise RuntimeError(f"No .npz found in: {args.data}")

    labels = infer_labels(npz_paths)
    label_map = make_label_map(labels)
    idx_to_label = {i: name for name, i in label_map.items()}

    print("[DATA] total:", len(npz_paths))
    print("[DATA] labels:", label_map)
    print(f"[FEAT] stand-focused {FEATURE_DIM} dims")

    tr_paths, va_paths = build_splits(npz_paths, val_ratio=args.val_ratio, seed=args.seed)
    print(f"[SPLIT] train: {len(tr_paths)} | val: {len(va_paths)}")
    print("[SPLIT] train dist:", count_label_dist(tr_paths))
    print("[SPLIT] val   dist:", count_label_dist(va_paths))

    tr_samples = [Sample(p, load_npz(p)[2]) for p in tr_paths]
    va_samples = [Sample(p, load_npz(p)[2]) for p in va_paths]

    train_ds = StandingNPZDataset(tr_samples, label_map, T=args.T)
    val_ds = StandingNPZDataset(va_samples, label_map, T=args.T)

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
    best_ep = -1

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
            best_ep = ep
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"  -> best @epoch {ep} val_acc={va_acc:.3f}")

            # Print detailed evaluation for the current best checkpoint.
            _, _, y_t, y_p = evaluate_with_preds(model, val_loader, device)
            cm = _confusion_matrix(y_t, y_p, num_classes=len(label_map))
            _print_eval_report(cm, idx_to_label)

    print(f"\n=== SUMMARY ===")
    print(f"Best epoch: {best_ep} | Best val acc: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")

    # Print final report for the (current) best weights if we have them.
    if best_state is not None:
        model.load_state_dict(best_state)
        _, _, y_t, y_p = evaluate_with_preds(model, val_loader, device)
        cm = _confusion_matrix(y_t, y_p, num_classes=len(label_map))
        _print_eval_report(cm, idx_to_label)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    ckpt = {
        "model_state": best_state if best_state is not None else model.state_dict(),
        "in_dim": FEATURE_DIM,
        "T": int(args.T),
        "label_map": label_map,
        "meta": {
            "task": "squat_stand",
            "feature_set": "stand_focused_16d",
            "feat_dim_detail": {
                "width_ratios": 4, "x_positions": 6, "angles": 3, "distances": 2, "feet_over_shoulder": 1, "total": FEATURE_DIM
            },
            "data_dir": os.path.abspath(args.data),
            "val_ratio": float(args.val_ratio),
            "T": int(args.T),
            "lr": float(args.lr),
            "wd": float(args.wd),
            "epochs": int(args.epochs),
            "bs": int(args.bs),
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
    ap.add_argument("--data", required=True, help="folder with *.npz (dataset/squat/dataset_standing)")
    ap.add_argument("--out", required=True, help="output .pt path e.g. squat/models/squat_stand_tcn.pt")

    ap.add_argument("--T", type=int, default=30, help="resample length")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--ch", type=int, default=128)

    ap.add_argument("--val_ratio", type=float, default=0.35)
    ap.add_argument("--seed", type=int, default=219)
    ap.add_argument("--cpu", action="store_true")

    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()