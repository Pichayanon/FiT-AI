"""
train_squat_stand.py

Train a TCN to classify standing stance:
  good_stand | stand_too_narrow | stand_too_wide

Focused Feature Set (16 dims per frame):
  - 4 width ratios   : ankle/hip, ankle/sho, knee/hip, knee/sho
  - 6 X-positions     : L/R ankle, knee, hip (normalized by hip_width)
  - 3 angles          : knee_L, knee_R, torso_tilt (all / 180)

Usage:
  python squat/train_squat_stand.py \
    --data dataset/squat/dataset_standing \
    --out  squat/models/squat_stand_tcn.pt
"""

from __future__ import annotations

if __package__ in {None, ""}:
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

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

from shared.tcn_models import SimpleTCN
from squat.features import STAND_FEATURE_DIM, extract_stand_features


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
# Utilities
# -----------------------------

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resample_time(feature_sequence: np.ndarray, target_T: int) -> np.ndarray:
    time_steps, feature_dim = feature_sequence.shape
    if time_steps == target_T:
        return feature_sequence.astype(np.float32)
    if time_steps < 2:
        return np.repeat(feature_sequence, target_T, axis=0)[:target_T].astype(np.float32)
    src = np.linspace(0, 1, time_steps)
    dst = np.linspace(0, 1, target_T)
    resampled_sequence = np.zeros((target_T, feature_dim), dtype=np.float32)
    for feature_index in range(feature_dim):
        resampled_sequence[:, feature_index] = np.interp(
            dst,
            src,
            feature_sequence[:, feature_index],
        )
    return resampled_sequence


def normalize_per_sample(feature_sequence: np.ndarray) -> np.ndarray:
    mean_values = feature_sequence.mean(axis=0, keepdims=True)
    std_values = feature_sequence.std(axis=0, keepdims=True) + 1e-6
    return ((feature_sequence - mean_values) / std_values).astype(np.float32)


def load_npz(path: str) -> Tuple[np.ndarray, np.ndarray, str]:
    z = np.load(path, allow_pickle=True)
    keypoints = z["keypoints"].astype(np.float32)  # (T,33,4)
    mask = z["mask"].astype(np.float32)            # (T,)
    label = str(z["label"])
    return keypoints, mask, label


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
        self.in_dim = STAND_FEATURE_DIM

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        keypoints, mask, label_name = load_npz(sample.path)
        feature_sequence = extract_stand_features(keypoints)
        feature_sequence = resample_time(feature_sequence, self.T)
        # REMOVED: normalize_per_sample(feature_sequence) (destroys spatial proportions)
        label_id = self.label_map[label_name]
        return torch.from_numpy(feature_sequence), torch.tensor(label_id, dtype=torch.long)


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
    label_counts: Dict[str, int] = {}
    for npz_path in npz_paths:
        label_name = load_npz(npz_path)[2]
        label_counts[label_name] = label_counts.get(label_name, 0) + 1
    return label_counts


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
    criterion = nn.CrossEntropyLoss()

    for feature_batch, label_batch in loader:
        feature_batch = feature_batch.to(device)
        label_batch = label_batch.to(device)
        logits = model(feature_batch)
        loss = criterion(logits, label_batch)
        loss_sum += float(loss.item()) * feature_batch.size(0)

        predicted_labels = torch.argmax(logits, dim=1)
        correct += int((predicted_labels == label_batch).sum().item())
        total += feature_batch.size(0)

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
    criterion = nn.CrossEntropyLoss()

    y_true: List[int] = []
    y_pred: List[int] = []

    for feature_batch, label_batch in loader:
        feature_batch = feature_batch.to(device)
        label_batch = label_batch.to(device)
        logits = model(feature_batch)
        loss = criterion(logits, label_batch)
        loss_sum += float(loss.item()) * feature_batch.size(0)

        predicted_labels = torch.argmax(logits, dim=1)
        correct += int((predicted_labels == label_batch).sum().item())
        total += feature_batch.size(0)

        y_true.extend(label_batch.detach().cpu().numpy().tolist())
        y_pred.extend(predicted_labels.detach().cpu().numpy().tolist())

    acc = correct / max(1, total)
    loss_avg = loss_sum / max(1, total)
    return loss_avg, acc, np.asarray(y_true, dtype=np.int64), np.asarray(y_pred, dtype=np.int64)


def train(args: argparse.Namespace) -> None:
    """Train a TCN for squat standing classification."""
    set_seed(args.seed)

    # Check for explicit train/val/test folders
    train_dir = os.path.join(args.data, "train")
    val_dir = os.path.join(args.data, "val")
    test_dir = os.path.join(args.data, "test")

    # Handle "validation" alias
    if not os.path.isdir(val_dir) and os.path.isdir(os.path.join(args.data, "validation")):
        val_dir = os.path.join(args.data, "validation")

    has_subfolders = os.path.isdir(train_dir) and os.path.isdir(val_dir)

    train_paths: List[str] = []
    val_paths: List[str] = []
    test_paths: List[str] = []

    if has_subfolders:
        print(f"[DATA] Detected subfolders in {args.data}")
        train_paths = sorted(glob.glob(os.path.join(train_dir, "*.npz")))
        val_paths = sorted(glob.glob(os.path.join(val_dir, "*.npz")))
        if os.path.isdir(test_dir):
            test_paths = sorted(glob.glob(os.path.join(test_dir, "*.npz")))

        all_paths = train_paths + val_paths + test_paths
        if len(all_paths) == 0:
            raise RuntimeError(f"No .npz found in subfolders of: {args.data}")
    else:
        # Fallback to flat directory + random split
        print(f"[DATA] No train/val subfolders found. Using random split from {args.data}")
        all_paths = sorted(glob.glob(os.path.join(args.data, "*.npz")))
        if len(all_paths) == 0:
            raise RuntimeError(f"No .npz found in: {args.data}")
        train_paths, val_paths = build_splits(
            all_paths,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )
        # No implicit test set in this mode

    labels = infer_labels(all_paths)
    label_map = make_label_map(labels)
    idx_to_label = {i: name for name, i in label_map.items()}

    print("[DATA] total found:", len(all_paths))
    print("[DATA] labels:", label_map)
    print(f"[FEAT] stand-focused {STAND_FEATURE_DIM} dims")

    print(f"[SPLIT] train: {len(train_paths)} | val: {len(val_paths)} | test: {len(test_paths)}")
    print("[SPLIT] train dist:", count_label_dist(train_paths))
    print("[SPLIT] val   dist:", count_label_dist(val_paths))
    if test_paths:
        print("[SPLIT] test  dist:", count_label_dist(test_paths))

    train_samples = [Sample(path, load_npz(path)[2]) for path in train_paths]
    val_samples = [Sample(path, load_npz(path)[2]) for path in val_paths]
    test_samples = [Sample(path, load_npz(path)[2]) for path in test_paths]

    train_dataset = StandingNPZDataset(train_samples, label_map, T=args.T)
    val_dataset = StandingNPZDataset(val_samples, label_map, T=args.T)
    test_dataset = StandingNPZDataset(test_samples, label_map, T=args.T) if test_samples else None

    train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.bs, shuffle=False, num_workers=0)
    test_loader = (
        DataLoader(test_dataset, batch_size=args.bs, shuffle=False, num_workers=0)
        if test_dataset
        else None
    )

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print("[DEVICE]", device)

    model = SimpleTCN(
        in_dim=int(train_dataset.in_dim),
        num_classes=len(label_map),
        channels=(args.ch, args.ch, args.ch),
        dropout=float(args.dropout),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.wd),
    )
    criterion = nn.CrossEntropyLoss()

    best_val_acc = -1.0
    best_state = None
    best_epoch = -1
    
    # Track history
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch_index in range(1, int(args.epochs) + 1):
        model.train()
        total = 0
        correct = 0
        loss_sum = 0.0

        for feature_batch, label_batch in train_loader:
            feature_batch = feature_batch.to(device)
            label_batch = label_batch.to(device)

            optimizer.zero_grad()
            logits = model(feature_batch)
            loss = criterion(logits, label_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            batch_size = feature_batch.size(0)
            loss_sum += float(loss.item()) * batch_size
            predicted_labels = torch.argmax(logits, dim=1)
            correct += int((predicted_labels == label_batch).sum().item())
            total += batch_size

        tr_loss = loss_sum / max(1, total)
        tr_acc = correct / max(1, total)

        va_loss, va_acc = evaluate(model, val_loader, device)

        # Record history
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)

        print(
            f"Epoch {epoch_index:03d}/{args.epochs} | "
            f"train loss={tr_loss:.4f} acc={tr_acc:.3f} | "
            f"val loss={va_loss:.4f} acc={va_acc:.3f}"
        )

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_epoch = epoch_index
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"  -> best updated @epoch {best_epoch} | best_val_acc={best_val_acc:.3f}")

            # Print detailed evaluation for the current best checkpoint.
            _, _, y_t, y_p = evaluate_with_preds(model, val_loader, device)
            cm = _confusion_matrix(y_t, y_p, num_classes=len(label_map))
            _print_eval_report(cm, idx_to_label)

    print("\n=== SUMMARY ===")
    print(f"Train samples: {len(train_paths)}")
    print(f"Val samples  : {len(val_paths)}")
    print(f"Best epoch   : {best_epoch}")
    print(f"Best val acc : {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")

    # Plot history
    try:
        import matplotlib.pyplot as plt

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

    # Print final report for the (current) best weights if we have them.
    if best_state is not None:
        model.load_state_dict(best_state)
        print("\n[VAL] Best Validation Evaluation:")
        _, _, y_t, y_p = evaluate_with_preds(model, val_loader, device)
        cm = _confusion_matrix(y_t, y_p, num_classes=len(label_map))
        _print_eval_report(cm, idx_to_label)

        # Evaluate on Test set if available
        if test_loader is not None:
            print("\n[TEST] Test Set Evaluation:")
            test_loss, test_acc, y_t_test, y_p_test = evaluate_with_preds(model, test_loader, device)
            print(f"  test loss={test_loss:.4f} acc={test_acc:.4f}")
            cm_test = _confusion_matrix(y_t_test, y_p_test, num_classes=len(label_map))
            _print_eval_report(cm_test, idx_to_label)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    checkpoint = {
        "model_state": best_state if best_state is not None else model.state_dict(),
        "in_dim": STAND_FEATURE_DIM,
        "T": int(args.T),
        "label_map": label_map,
        "meta": {
            "task": "squat_stand",
            "feature_set": "stand_focused_16d",
            "feat_dim_detail": {
                "width_ratios": 4, "x_positions": 6, "angles": 3, "distances": 2, "feet_over_shoulder": 1, "total": STAND_FEATURE_DIM
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
            "best_epoch": int(best_epoch),
        },
    }

    torch.save(checkpoint, args.out)
    print("[SAVE] checkpoint:", args.out)
    print("[SAVE] best_val_acc:", best_val_acc)
    print("[SAVE] best_epoch:", best_epoch)
    print("[SAVE] label_map:", json.dumps(label_map, ensure_ascii=False))
    print("[SAVE] in_dim:", train_dataset.in_dim)


def main() -> None:
    """CLI entrypoint.

    :returns: None
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="folder with *.npz (dataset/squat/dataset_standing)")
    ap.add_argument("--out", required=True, help="output .pt path e.g. squat/models/squat_stand_tcn.pt")

    ap.add_argument("--T", type=int, default=30, help="resample length")
    ap.add_argument("--epochs", type=int, default=75)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--ch", type=int, default=64)

    ap.add_argument("--val_ratio", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=921)
    ap.add_argument("--cpu", action="store_true")

    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
