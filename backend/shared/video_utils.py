"""
Video recording utilities for streaming sessions.

Provides video writer creation with codec fallback and
time-series resampling for TCN model input preparation.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np


def create_video_writer(
    path_no_ext: str, w: int, h: int, fps: float
) -> Tuple[Optional[cv2.VideoWriter], str]:
    """Create a video writer with MP4 codec, falling back to AVI/MJPG.

    Args:
        path_no_ext: Output file path without extension.
        w: Frame width in pixels.
        h: Frame height in pixels.
        fps: Target frames per second.

    Returns:
        Tuple of (writer_or_None, actual_file_path).
    """
    # Try MP4 first
    mp4_path = f"{path_no_ext}.mp4"
    try:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(mp4_path, fourcc, fps, (w, h))
        if writer.isOpened():
            return writer, mp4_path
    except Exception:  # pylint: disable=broad-except
        pass

    # Fallback to AVI/MJPG
    avi_path = f"{path_no_ext}.avi"
    try:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(avi_path, fourcc, fps, (w, h))
        if writer.isOpened():
            return writer, avi_path
    except Exception:  # pylint: disable=broad-except
        pass

    return None, ""


def resample_time(x: np.ndarray, target_t: int) -> np.ndarray:
    """Linear interpolation over the time axis to a target length.

    Args:
        x: Input array of shape (T, D).
        target_t: Desired number of time steps.

    Returns:
        Resampled array of shape (target_t, D), dtype float32.
    """
    t, d = x.shape
    if t == target_t:
        return x.astype(np.float32)
    if t < 2:
        return np.repeat(x, target_t, axis=0)[:target_t].astype(np.float32)

    src = np.linspace(0, 1, t)
    dst = np.linspace(0, 1, target_t)
    out = np.zeros((target_t, d), dtype=np.float32)
    for j in range(d):
        out[:, j] = np.interp(dst, src, x[:, j])
    return out


def normalize_per_sample(x: np.ndarray) -> np.ndarray:
    """Z-normalize per sample (per feature dimension) across time.

    Args:
        x: Input array of shape (T, D).

    Returns:
        Normalized array of same shape, dtype float32.
    """
    mu = x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True) + 1e-6
    return ((x - mu) / sd).astype(np.float32)
