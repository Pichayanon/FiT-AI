"""Shared wall-sit feature extraction helpers.

Used by both training and streaming so wall-sit feature computation stays
consistent across offline and realtime paths.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from shared.math_utils import angle_3pts, dist, get_xyz, position_normalize


L_SHO, R_SHO = 11, 12
L_HIP, R_HIP = 23, 24
L_KNE, R_KNE = 25, 26
L_ANK, R_ANK = 27, 28


def extract_frame_features(lm: list, side: str) -> Tuple[float, float, float]:
    """Extract normalized wall-sit features from one landmark frame.

    All selected joints are first normalized in xyz space relative to the
    chosen-side hip using torso length as the primary scale. The resulting
    features are:
        - foot_wall_norm: horizontal ankle offset from hip
        - knee_angle_norm: hip-knee-ankle angle divided by 180
        - torso_alignment: horizontal shoulder offset from hip
    """
    xyz = get_xyz(lm)

    if side == "right":
        hip_idx, knee_idx, ankle_idx, shoulder_idx = R_HIP, R_KNE, R_ANK, R_SHO
    else:
        hip_idx, knee_idx, ankle_idx, shoulder_idx = L_HIP, L_KNE, L_ANK, L_SHO

    hip = xyz[hip_idx]
    knee = xyz[knee_idx]
    ankle = xyz[ankle_idx]
    shoulder = xyz[shoulder_idx]

    torso_len = dist(hip, shoulder)
    shoulder_width = dist(xyz[L_SHO], xyz[R_SHO])
    hip_width = dist(xyz[L_HIP], xyz[R_HIP])
    scale   = hip_width if hip_width > 1e-4 else (shoulder_width if shoulder_width > 1e-4 else 1e-6)

    hip_norm = position_normalize(hip, center=hip, scale=scale)
    knee_norm = position_normalize(knee, center=hip, scale=scale)
    ankle_norm = position_normalize(ankle, center=hip, scale=scale)
    shoulder_norm = position_normalize(shoulder, center=hip, scale=scale)

    print(f"XXX {hip_norm}")
    foot_wall = abs(ankle_norm[0] - hip_norm[0])
    knee_angle = angle_3pts(hip_norm, knee_norm, ankle_norm) / 180.0
    torso_alignment = abs(shoulder_norm[0] - hip_norm[0])

    return float(foot_wall), float(knee_angle), float(torso_alignment)


def aggregate_window(vals: List[Tuple[float, float, float]]) -> np.ndarray:
    """Aggregate per-frame tuples into the model's 5-D wall-sit feature vector."""
    fw = [v[0] for v in vals]
    knee = [v[1] for v in vals]
    torso = [v[2] for v in vals]

    return np.array([
        np.mean(fw),
        np.std(fw),
        np.mean(knee),
        np.min(knee),
        np.mean(torso),
    ], dtype=np.float32)
