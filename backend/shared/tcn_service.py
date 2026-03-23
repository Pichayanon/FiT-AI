from __future__ import annotations

import re
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from .tcn_models import SimpleTCN, PhaseTCN
from .video_utils import resample_time


def _get_simple_tcn_config(
    checkpoint: Dict[str, Any],
) -> Tuple[Tuple[int, ...], float, bool]:
    metadata = checkpoint.get("meta", {})
    state_dict = checkpoint.get("model_state") or checkpoint.get("state_dict") or {}

    channels_raw = metadata.get("channels")
    if isinstance(channels_raw, (list, tuple)) and channels_raw:
        channels = tuple(int(ch) for ch in channels_raw)
    else:
        inferred_channels = []
        indexed_channels = []
        for key, value in state_dict.items():
            match = re.fullmatch(r"tcn\.(\d+)\.conv1\.weight", key)
            if match:
                indexed_channels.append((int(match.group(1)), int(value.shape[0])))
        if indexed_channels:
            channels = tuple(ch for _, ch in sorted(indexed_channels))
        elif "fc.weight" in state_dict and hasattr(state_dict["fc.weight"], "shape"):
            inferred_channels = [int(state_dict["fc.weight"].shape[1])]
            channels = tuple(inferred_channels)
        else:
            channels = (64, 64, 64)

    dropout = float(metadata.get("dropout", 0.1))
    use_attention_meta = metadata.get("use_attention")
    if isinstance(use_attention_meta, bool):
        use_attention = use_attention_meta
    else:
        use_attention = (
            "attention.weight" in state_dict or "attention.bias" in state_dict
        )

    return channels, dropout, use_attention


def load_sequence_tcn(
    path: str,
) -> Tuple[
    Optional[SimpleTCN],
    Optional[int],
    Optional[Dict[int, str]],
    Optional[int],
]:
    try:
        checkpoint = torch.load(path, map_location="cpu")
        input_dim = int(checkpoint["in_dim"])
        target_window_size = int(checkpoint["T"])
        label_map = checkpoint["label_map"]
        inverse_label_map = {value: key for key, value in label_map.items()}
        channels, dropout, use_attention = _get_simple_tcn_config(checkpoint)
        model = SimpleTCN(
            in_dim=input_dim,
            num_classes=len(inverse_label_map),
            channels=channels,
            dropout=dropout,
            use_attention=use_attention,
        )
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        print(
            f"[MODEL] Loaded: {path} in_dim={input_dim} T={target_window_size} "
            f"classes={inverse_label_map} channels={channels} "
            f"use_attention={use_attention}"
        )
        return model, target_window_size, inverse_label_map, input_dim
    except Exception as e:
        print(f"[MODEL] Cannot load: {path} err={e}")
        return None, None, None, None


def load_phase_tcn(
    path: str,
) -> Tuple[Optional[PhaseTCN], Optional[int], Optional[int]]:
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
    except Exception as e:
        print(f"[MODEL] Cannot load phase TCN: {path} err={e}")
        return None, None, None


def predict_sequence_tcn(
    model: Any,
    inv_labels: Dict[int, str],
    target_t: int,
    feature_window: np.ndarray,
) -> Tuple[str, float, np.ndarray]:
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


load_tcn = load_sequence_tcn
tcn_predict = predict_sequence_tcn
