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


def _get_simple_tcn_config(
    checkpoint: Dict[str, Any],
) -> Tuple[Tuple[int, ...], float]:
    """Extract SimpleTCN architecture config from checkpoint metadata."""
    metadata = checkpoint.get("meta", {})

    channels_raw = metadata.get("channels", (128, 128, 128))
    if isinstance(channels_raw, (list, tuple)) and channels_raw:
        channels = tuple(int(ch) for ch in channels_raw)
    else:
        channels = (128, 128, 128)

    dropout = float(metadata.get("dropout", 0.1))
    return channels, dropout


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
        checkpoint = torch.load(path, map_location="cpu")
        in_dim = int(checkpoint["in_dim"])
        target_window_size = int(checkpoint["T"])
        label_map = checkpoint["label_map"]
        inverse_label_map = {value: key for key, value in label_map.items()}
        channels, dropout = _get_simple_tcn_config(checkpoint)
        model = SimpleTCN(
            in_dim=in_dim,
            num_classes=len(inverse_label_map),
            channels=channels,
            dropout=dropout,
        )
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        print(
            f"[MODEL] Loaded: {path} in_dim={in_dim} T={target_window_size} "
            f"classes={inverse_label_map} channels={channels}"
        )
        return model, target_window_size, inverse_label_map, in_dim
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
        checkpoint = torch.load(path, map_location="cpu")
        in_dim = int(checkpoint.get("in_dim", 10))
        num_classes = int(checkpoint.get("num_classes", 2))
        window_size = int(checkpoint.get("window", 30))
        model = PhaseTCN(in_dim=in_dim, num_classes=num_classes)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        print(
            f"[MODEL] Phase TCN loaded: {path} "
            f"in_dim={in_dim} window={window_size} num_classes={num_classes}"
        )
        return model, window_size, in_dim
    except Exception as e:  # pylint: disable=broad-except
        print(f"[MODEL] Cannot load phase TCN: {path} err={e}")
        return None, None, None


def tcn_predict(
    model: Any,
    inv_labels: Dict[int, str],
    target_t: int,
    feature_window: np.ndarray,
) -> Tuple[str, float, np.ndarray]:
    """Run prediction with a SimpleTCN model.

    Resamples the input window to the model's expected time dimension,
    runs inference, and returns the predicted label with confidence.

    Args:
        model: Loaded SimpleTCN model.
        inv_labels: Mapping from class index to label string.
        target_t: Target time dimension for resampling.
        feature_window: Input feature window of shape (T_raw, D).

    Returns:
        Tuple of (predicted_label, confidence, probability_array).
    """
    resampled_feature_window = resample_time(
        feature_window.astype(np.float32),
        int(target_t),
    )
    feature_window_tensor = torch.from_numpy(resampled_feature_window).unsqueeze(0)
    with torch.no_grad():
        logits = model(feature_window_tensor)
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]
        predicted_class_index = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_class_index])
        predicted_label = inv_labels.get(
            predicted_class_index,
            str(predicted_class_index),
        )
    return predicted_label, confidence, probabilities
