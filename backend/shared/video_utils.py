from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np


def create_video_writer(
    path_no_ext: str, w: int, h: int, fps: float
) -> Tuple[Optional[cv2.VideoWriter], str]:
    mp4_path = f"{path_no_ext}.mp4"
    try:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(mp4_path, fourcc, fps, (w, h))
        if writer.isOpened():
            return writer, mp4_path
    except Exception:
        pass

    avi_path = f"{path_no_ext}.avi"
    try:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(avi_path, fourcc, fps, (w, h))
        if writer.isOpened():
            return writer, avi_path
    except Exception:
        pass

    return None, ""


def resample_time(x: np.ndarray, target_t: int) -> np.ndarray:
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
    mu = x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True) + 1e-6
    return ((x - mu) / sd).astype(np.float32)
