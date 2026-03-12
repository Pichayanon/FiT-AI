"""
TCN model loading and prediction service.

Provides functions to load TCN checkpoints and run predictions,
used by squat and lunges streaming backends.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from .tcn_models import SimpleTCN, PhaseTCN
from .video_utils import resample_time


def load_tcn(path: str) -> Tuple[
    Optional[SimpleTCN],
    Optional[int],
    Optional[Dict[int, str]],
    Optional[int],
]:
    """Load a SimpleTCN checkpoint.

    Expected checkpoint keys: in_dim, T, label_map, model_state.

    Args:
        path: Path to the .pt checkpoint file.

    Returns:
        Tuple of (model, T, inv_labels, in_dim).
        All None if loading fails.
    """
    try:
        ckpt = torch.load(path, map_location="cpu")
        in_dim = int(ckpt["in_dim"])
        t_val = int(ckpt["T"])
        label_map = ckpt["label_map"]
        inv = {v: k for k, v in label_map.items()}
        model = SimpleTCN(
            in_dim=in_dim,
            num_classes=len(inv),
            channels=(128, 128, 128),
            dropout=0.1,
        )
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        print(f"[MODEL] Loaded: {path} in_dim={in_dim} T={t_val} classes={inv}")
        return model, t_val, inv, in_dim
    except Exception as e:  # pylint: disable=broad-except
        print(f"[MODEL] Cannot load: {path} err={e}")
        return None, None, None, None


def load_phase_tcn(
    path: str,
) -> Tuple[Optional[PhaseTCN], Optional[int], Optional[int]]:
    """Load a PhaseTCN checkpoint for phase prediction.

    Expected checkpoint keys: state_dict, in_dim, num_classes, window.

    Args:
        path: Path to the .pt checkpoint file.

    Returns:
        Tuple of (model, window_size, in_dim).
        All None if loading fails.
    """
    try:
        ckpt = torch.load(path, map_location="cpu")
        in_dim = int(ckpt.get("in_dim", 10))
        num_classes = int(ckpt.get("num_classes", 2))
        window = int(ckpt.get("window", 30))
        model = PhaseTCN(in_dim=in_dim, num_classes=num_classes)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        print(
            f"[MODEL] Phase TCN loaded: {path} "
            f"in_dim={in_dim} window={window} num_classes={num_classes}"
        )
        return model, window, in_dim
    except Exception as e:  # pylint: disable=broad-except
        print(f"[MODEL] Cannot load phase TCN: {path} err={e}")
        return None, None, None


def tcn_predict(
    model: Any,
    inv_labels: Dict[int, str],
    target_t: int,
    x_win: np.ndarray,
) -> Tuple[str, float, np.ndarray]:
    """Run prediction with a SimpleTCN model.

    Resamples the input window to the model's expected time dimension,
    runs inference, and returns the predicted label with confidence.

    Args:
        model: Loaded SimpleTCN model.
        inv_labels: Mapping from class index to label string.
        target_t: Target time dimension for resampling.
        x_win: Input feature window of shape (T_raw, D).

    Returns:
        Tuple of (predicted_label, confidence, probability_array).
    """
    x = resample_time(x_win.astype(np.float32), int(target_t))
    xt = torch.from_numpy(x).unsqueeze(0)  # (1, T, D)
    with torch.no_grad():
        logits = model(xt)
        prob = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred = int(np.argmax(prob))
        conf = float(prob[pred])
        pred_label = inv_labels.get(pred, str(pred))
    return pred_label, conf, prob
