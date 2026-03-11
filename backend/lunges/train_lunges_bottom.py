"""
train_lunges_bottom.py

Train a TCN to classify lunge errors:
  correct | torso_lean_forward | knee_over_toe | not_deep_enough

Focused Feature Set (42 dims per frame):
  - 30 dims: 10 key joints (L/R: Ear, Sho, Hip, Kne, Ank) xyz (body-centric, normalized)
  - 12 dims: Computed scalars
      - knee_angle_l, knee_angle_r
      - hip_angle_l, hip_angle_r
      - torso_tilt
      - stride_length_ratio (dist(ank_l, ank_r) / height)
      - knee_over_toe_l (signed dist), knee_over_toe_r
      - rear_knee_dist_ground_l, rear_knee_dist_ground_r (y dist to lowest ankle)
      - spine_angle
      - hip_drop (vertical movement)

Usage:
  python lunges/train_lunges_bottom.py --data dataset/lunges/dataset_bottom --out lunges/models/lunges_bottom_tcn.pt
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
import matplotlib.pyplot as plt


# -----------------------------
# Utils & Math
# -----------------------------

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

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

def load_npz(path: str) -> Tuple[np.ndarray, np.ndarray, str]:
    z = np.load(path, allow_pickle=True)
    keypoints = z["keypoints"].astype(np.float32)
    mask = z["mask"].astype(np.float32)
    label = str(z["label"])
    return keypoints, mask, label


# -----------------------------
# Evaluation Helpers
# -----------------------------

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

    print("\n[VAL] Per-class Accuracy:")
    cm_diag = cm.diagonal()
    cm_sum = cm.sum(axis=1)
    for i in range(cm.shape[0]):
        name = idx_to_label.get(int(i), str(i))
        cls_acc = cm_diag[i] / cm_sum[i] if cm_sum[i] > 0 else 0.0
        print(f"  {name:<20}: {cls_acc:.4f} ({cm_diag[i]}/{cm_sum[i]})")

    print("\n[VAL] Summary:")
    print(f"  accuracy   : {rep['accuracy']:.4f}")
    print(f"  macro_f1   : {rep['macro_f1']:.4f}")
    print(f"  weighted_f1: {rep['weighted_f1']:.4f}")


# -----------------------------
# Feature Extraction
# -----------------------------

FEATURE_DIM = 42
L_EAR, R_EAR = 7, 8
L_SHO, R_SHO = 11, 12
L_HIP, R_HIP = 23, 24
L_KNE, R_KNE = 25, 26
L_ANK, R_ANK = 27, 28
KEY_JOINTS = [L_EAR, R_EAR, L_SHO, R_SHO, L_HIP, R_HIP, L_KNE, R_KNE, L_ANK, R_ANK]

def extract_lunge_features(kp: np.ndarray) -> np.ndarray:
    """
    Extracts features relevant to lunges.
    Input: (T, 33, 4)
    Output: (T, 42)
    """
    T = kp.shape[0]
    xyz = kp[..., :3].astype(np.float32)
    out = np.zeros((T, FEATURE_DIM), dtype=np.float32)

    # Indices
    L_HEEL, R_HEEL = 29, 30
    L_FOOT, R_FOOT = 31, 32

    for t in range(T):
        p = xyz[t]  # (33, 3)

        # -----------------------------
        # 1. Detect Facing & Normalize X
        # -----------------------------
        # Vector heel->toe indicates direction
        # We use average of both feet to be robust
        l_dir = p[L_FOOT][0] - p[L_HEEL][0]
        r_dir = p[R_FOOT][0] - p[R_HEEL][0]
        avg_dir = l_dir + r_dir
        
        # If avg_dir < 0, facing Left. We want to normalize to facing Right (+X).
        facing_right = (avg_dir >= 0)
        facing_mult = 1.0 if facing_right else -1.0

        # Create a working copy of points where X is normalized to face Right
        # If facing left, we negate X. (We'll handle relative translation later)
        p_norm = p.copy()
        if not facing_right:
            p_norm[:, 0] = -p_norm[:, 0]

        # -----------------------------
        # 2. Identify Front vs Back Leg
        # -----------------------------
        # Now that we are 'facing right', the Front leg should have larger X than Back leg.
        l_ank_x = p_norm[L_ANK][0]
        r_ank_x = p_norm[R_ANK][0]
        
        is_l_front = l_ank_x > r_ank_x
        
        # Define indices for Front (F) and Back (B)
        # If L is front, map F->L, B->R.
        # If R is front, map F->R, B->L.
        if is_l_front:
            IDX_F_EAR, IDX_B_EAR = L_EAR, R_EAR
            IDX_F_SHO, IDX_B_SHO = L_SHO, R_SHO
            IDX_F_HIP, IDX_B_HIP = L_HIP, R_HIP
            IDX_F_KNE, IDX_B_KNE = L_KNE, R_KNE
            IDX_F_ANK, IDX_B_ANK = L_ANK, R_ANK
        else:
            IDX_F_EAR, IDX_B_EAR = R_EAR, L_EAR
            IDX_F_SHO, IDX_B_SHO = R_SHO, L_SHO
            IDX_F_HIP, IDX_B_HIP = R_HIP, L_HIP
            IDX_F_KNE, IDX_B_KNE = R_KNE, L_KNE
            IDX_F_ANK, IDX_B_ANK = R_ANK, L_ANK

        # Get landmarks for Front/Back
        f_ear, b_ear = p_norm[IDX_F_EAR], p_norm[IDX_B_EAR]
        f_sho, b_sho = p_norm[IDX_F_SHO], p_norm[IDX_B_SHO]
        f_hip, b_hip = p_norm[IDX_F_HIP], p_norm[IDX_B_HIP]
        f_kne, b_kne = p_norm[IDX_F_KNE], p_norm[IDX_B_KNE]
        f_ank, b_ank = p_norm[IDX_F_ANK], p_norm[IDX_B_ANK]

        mid_hip = 0.5 * (f_hip + b_hip)
        mid_sho = 0.5 * (f_sho + b_sho)
        mid_ear = 0.5 * (f_ear + b_ear)

        # Scale normalization
        torso_len = _dist(mid_hip, mid_sho)
        scale = torso_len if torso_len > 1e-4 else 1.0

        # -----------------------------
        # 3. Features Construction
        # -----------------------------
        
        # List of joints in order: F_EAR, B_EAR, F_SHO, B_SHO, ...
        SORTED_JOINTS = [
            IDX_F_EAR, IDX_B_EAR, 
            IDX_F_SHO, IDX_B_SHO, 
            IDX_F_HIP, IDX_B_HIP, 
            IDX_F_KNE, IDX_B_KNE, 
            IDX_F_ANK, IDX_B_ANK
        ]

        # [0-29] Body-centric XYZ (Front/Back sorted)
        for i, j_idx in enumerate(SORTED_JOINTS):
            # p_norm is already X-flipped if needed.
            normed = (p_norm[j_idx] - mid_hip) / scale
            out[t, i*3:(i+1)*3] = normed
            
        # [30-33] Angles (F/B)
        out[t, 30] = _angle_3pts(f_hip, f_kne, f_ank) / 180.0 # Front Knee
        out[t, 31] = _angle_3pts(b_hip, b_kne, b_ank) / 180.0 # Back Knee
        out[t, 32] = _angle_3pts(f_sho, f_hip, f_kne) / 180.0 # Front Hip
        out[t, 33] = _angle_3pts(b_sho, b_hip, b_kne) / 180.0 # Back Hip
        
        # [34] Torso tilt (vertical alignment)
        spine_vec = mid_sho - mid_hip
        # We want angle with vertical UP (0,1,0) or (-Y)? 
        # In MP, Y is down. Vertical UP is (0,-1,0).
        # But we are in "Right Facing" normalized space.
        # Vertical is still Y axis. 
        vertical = np.array([0.0, -1.0, 0.0], dtype=np.float32)
        denom = (_safe_norm(spine_vec) * _safe_norm(vertical)) + 1e-6
        cosang = float(np.clip(np.dot(spine_vec, vertical) / denom, -1.0, 1.0))
        out[t, 34] = float(np.degrees(np.arccos(cosang))) / 180.0
        
        # [35] Stride Length Ratio
        stride_dist = _dist(f_ank, b_ank)
        out[t, 35] = stride_dist / scale
        
        # [36-37] Knee Over Toe (Signed X difference)
        # Since we normalized to Face Right, 'Over Toe' means Knee.x > Ankle.x
        out[t, 36] = (f_kne[0] - f_ank[0]) # Front
        out[t, 37] = (b_kne[0] - b_ank[0]) # Back
        
        # [38-39] Knee Height (Depth)
        ground_y = max(f_ank[1], b_ank[1]) # Lowest point (max Y in MP)
        out[t, 38] = (ground_y - f_kne[1]) # Front knee height
        out[t, 39] = (ground_y - b_kne[1]) # Back knee height (Target for depth)
        
        # [40] Spine Angle (Ear-Sho-Hip)
        out[t, 40] = _angle_3pts(mid_ear, mid_sho, mid_hip) / 180.0
        
        # [41] Hip Drop
        out[t, 41] = (ground_y - mid_hip[1]) / scale

    return out

    return out

# -----------------------------
# Dataset & Model
# -----------------------------

@dataclass
class Sample:
    path: str
    label: str

class LungeNPZDataset(Dataset):
    def __init__(self, samples: List[Sample], label_map: Dict[str, int], T: int):
        self.samples = samples
        self.label_map = label_map
        self.T = int(T)
        self.in_dim = FEATURE_DIM

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        kp, _, label = load_npz(s.path)
        x = extract_lunge_features(kp)
        x = resample_time(x, self.T)
        y = self.label_map[label]
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)

class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, dilation=1, dropout=0.1):
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

    def forward(self, x):
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
    def __init__(self, in_dim, num_classes, channels=(128, 128, 128), k=3, dropout=0.1):
        super().__init__()
        layers = []
        ch_in = in_dim
        dilation = 1
        for ch_out in channels:
            layers.append(TemporalBlock(ch_in, ch_out, k=k, dilation=dilation, dropout=dropout))
            ch_in = ch_out
            dilation *= 2
        self.tcn = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(channels[-1], num_classes)

    def forward(self, x):
        x = x.transpose(1, 2)
        y = self.tcn(x)
        y = self.pool(y).squeeze(-1)
        return self.fc(y)

# -----------------------------
# Train / Eval
# -----------------------------

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
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

        bs = x.size(0)
        loss_sum += float(loss.item()) * bs
        pred = torch.argmax(logits, dim=1)
        correct += int((pred == y).sum().item())
        total += bs

    acc = correct / max(1, total)
    loss_avg = loss_sum / max(1, total)
    return loss_avg, acc


@torch.no_grad()
def evaluate_with_preds(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> Tuple[float, float, np.ndarray, np.ndarray]:
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

        bs = x.size(0)
        loss_sum += float(loss.item()) * bs
        pred = torch.argmax(logits, dim=1)
        correct += int((pred == y).sum().item())
        total += bs

        y_true.extend(y.detach().cpu().numpy().tolist())
        y_pred.extend(pred.detach().cpu().numpy().tolist())

    acc = correct / max(1, total)
    loss_avg = loss_sum / max(1, total)
    return loss_avg, acc, np.asarray(y_true, dtype=np.int64), np.asarray(y_pred, dtype=np.int64)


# -----------------------------
# Training Loop
# -----------------------------

def train(args):
    set_seed(args.seed)
    
    # Simple glob
    all_paths = sorted(glob.glob(os.path.join(args.data, "*.npz")))
    if not all_paths:
        raise ValueError(f"No .npz found in {args.data}")
    
    # Split
    random.shuffle(all_paths)
    n_val = int(len(all_paths) * args.val_ratio)
    tr_paths = all_paths[n_val:]
    va_paths = all_paths[:n_val]
    
    labels = [load_npz(p)[2] for p in all_paths]
    label_map = {name: i for i, name in enumerate(sorted(set(labels)))}
    idx_to_label = {i: name for name, i in label_map.items()}
    print("Labels:", label_map)
    
    train_ds = LungeNPZDataset([Sample(p, load_npz(p)[2]) for p in tr_paths], label_map, args.T)
    val_ds = LungeNPZDataset([Sample(p, load_npz(p)[2]) for p in va_paths], label_map, args.T)
    
    train_loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.bs, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = SimpleTCN(train_ds.in_dim, len(label_map), dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    ce = nn.CrossEntropyLoss()
    
    best_val_acc = -1.0
    best_state = None
    best_ep = -1
    
    # Track history
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    
    for ep in range(1, args.epochs + 1):
        model.train()
        loss_sum, corr, total = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = ce(out, y)
            loss.backward()
            opt.step()
            loss_sum += float(loss.item()) * x.size(0)
            corr += (out.argmax(1) == y).sum().item()
            total += x.size(0)
        
        tr_acc = corr / max(1, total)
        tr_loss = loss_sum / max(1, total)
        
        # Val
        va_loss, va_acc = evaluate(model, val_loader, device)
        
        # Record history
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)
        
        print(f"Ep {ep}: TrLoss={tr_loss:.4f} TrAcc={tr_acc:.3f} ValLoss={va_loss:.4f} ValAcc={va_acc:.3f}")
        
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_ep = ep
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"  -> best updated @epoch {best_ep} | best_val_acc={best_val_acc:.3f}")
            
            # Detailed evaluation
            _, _, y_t, y_p = evaluate_with_preds(model, val_loader, device)
            cm = _confusion_matrix(y_t, y_p, num_classes=len(label_map))
            _print_eval_report(cm, idx_to_label)
            
    print("\n=== SUMMARY ===")
    print(f"Best epoch   : {best_ep}")
    print(f"Best val acc : {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")

    # Plot history
    try:
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(history["train_loss"], label="Train Loss")
        plt.plot(history["val_loss"], label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss over Epochs")
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(history["train_acc"], label="Train Acc")
        plt.plot(history["val_acc"], label="Val Acc")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Accuracy over Epochs")
        plt.legend()
        plt.grid(True)
        
        plot_path = args.out.replace(".pt", "_history.png")
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        print(f"[PLOT] Saved training history to: {plot_path}")
    except Exception as e:
        print(f"[WARN] Could not save plot: {e}")

    # Save Best Model
    if best_state is not None:
        torch.save({
            "model_state": best_state,
            "in_dim": FEATURE_DIM,
            "label_map": label_map,
            "T": args.T,
            "meta": {
                "val_acc": best_val_acc,
                "epoch": best_ep
            }
        }, args.out)
        print(f"[SAVE] Saved best model to {args.out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=921)
    ap.add_argument("--T", type=int, default=30)
    ap.add_argument("--cpu", action="store_true")
    train(ap.parse_args())
