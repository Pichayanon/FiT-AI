"""
features.py — Lunge feature extraction functions.

Single source of truth used by both training scripts and lunges_streaming.py.

Import in training:
    from lunges.features import extract_bottom_features, extract_phase_features

Import in streaming:
    from lunges.features import extract_bottom_features, extract_phase_features, LandmarkSmoother
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from shared.math_utils import angle_3pts, safe_norm, dist


# ---------------------------------------------------------------
# Constants (shared between training and streaming)
# ---------------------------------------------------------------

BOTTOM_FEATURE_DIM = 42

L_EAR, R_EAR = 7,  8
L_SHO, R_SHO = 11, 12
L_HIP, R_HIP = 23, 24
L_KNE, R_KNE = 25, 26
L_ANK, R_ANK = 27, 28
L_HEEL, R_HEEL = 29, 30
L_FOOT, R_FOOT = 31, 32


# ---------------------------------------------------------------
# Bottom features (42-D) — matches lunges/extract_bottom_lunges.py
# ---------------------------------------------------------------

def extract_bottom_features(kp: np.ndarray) -> np.ndarray:
    """Extract 42-dim features for the lunge bottom TCN.

    Input:  (33, 4) single-frame landmarks  OR  (T, 33, 4) batch
    Output: (42,)                           OR  (T, 42)

    Dimensions:
        [0-29]  10 joints × 3 xyz (body-centric, torso-length normalized,
                front/back sorted by ankle X position)
        [30-33] knee/hip angles front+back / 180
        [34]    torso tilt / 180
        [35]    stride length ratio
        [36-37] knee-over-toe (signed X diff) front+back
        [38-39] knee height (depth) front+back
        [40]    spine angle (ear-sho-hip) / 180
        [41]    hip drop / scale

    Matches the feature vector produced by lunges/extract_bottom_lunges.py.
    """
    if kp.ndim == 2:
        kp = kp[np.newaxis, ...]
        squeeze = True
    else:
        squeeze = False

    T   = kp.shape[0]
    xyz = kp[..., :3].astype(np.float32)
    out = np.zeros((T, BOTTOM_FEATURE_DIM), dtype=np.float32)

    for t in range(T):
        p = xyz[t]

        # Detect facing direction & normalize X
        avg_dir    = (p[L_FOOT][0] - p[L_HEEL][0]) + (p[R_FOOT][0] - p[R_HEEL][0])
        facing_right = avg_dir >= 0
        p_norm = p.copy()
        if not facing_right:
            p_norm[:, 0] = -p_norm[:, 0]

        # Identify front vs back leg
        is_l_front = p_norm[L_ANK][0] > p_norm[R_ANK][0]
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

        f_ear, b_ear = p_norm[IDX_F_EAR], p_norm[IDX_B_EAR]
        f_sho, b_sho = p_norm[IDX_F_SHO], p_norm[IDX_B_SHO]
        f_hip, b_hip = p_norm[IDX_F_HIP], p_norm[IDX_B_HIP]
        f_kne, b_kne = p_norm[IDX_F_KNE], p_norm[IDX_B_KNE]
        f_ank, b_ank = p_norm[IDX_F_ANK], p_norm[IDX_B_ANK]

        mid_hip = 0.5 * (f_hip + b_hip)
        mid_sho = 0.5 * (f_sho + b_sho)
        mid_ear = 0.5 * (f_ear + b_ear)

        torso_len = dist(mid_hip, mid_sho)
        scale     = torso_len if torso_len > 1e-4 else 1.0

        # [0-29] Body-centric XYZ (front/back sorted)
        sorted_joints = [
            IDX_F_EAR, IDX_B_EAR,
            IDX_F_SHO, IDX_B_SHO,
            IDX_F_HIP, IDX_B_HIP,
            IDX_F_KNE, IDX_B_KNE,
            IDX_F_ANK, IDX_B_ANK,
        ]
        for i, j_idx in enumerate(sorted_joints):
            out[t, i * 3:(i + 1) * 3] = (p_norm[j_idx] - mid_hip) / scale

        # [30-33] Angles
        out[t, 30] = angle_3pts(f_hip, f_kne, f_ank) / 180.0
        out[t, 31] = angle_3pts(b_hip, b_kne, b_ank) / 180.0
        out[t, 32] = angle_3pts(f_sho, f_hip, f_kne) / 180.0
        out[t, 33] = angle_3pts(b_sho, b_hip, b_kne) / 180.0

        # [34] Torso tilt
        spine_vec = mid_sho - mid_hip
        vertical  = np.array([0.0, -1.0, 0.0], dtype=np.float32)
        denom     = (safe_norm(spine_vec) * safe_norm(vertical)) + 1e-6
        cosang    = float(np.clip(np.dot(spine_vec, vertical) / denom, -1.0, 1.0))
        out[t, 34] = float(np.degrees(np.arccos(cosang))) / 180.0

        # [35] Stride length ratio
        out[t, 35] = dist(f_ank, b_ank) / scale

        # [36-37] Knee over toe (signed X diff)
        out[t, 36] = f_kne[0] - f_ank[0]
        out[t, 37] = b_kne[0] - b_ank[0]

        # [38-39] Knee height (depth)
        ground_y   = max(f_ank[1], b_ank[1])
        out[t, 38] = ground_y - f_kne[1]
        out[t, 39] = ground_y - b_kne[1]

        # [40] Spine angle (ear-sho-hip)
        out[t, 40] = angle_3pts(mid_ear, mid_sho, mid_hip) / 180.0

        # [41] Hip drop
        out[t, 41] = (ground_y - mid_hip[1]) / scale

    return out[0] if squeeze else out


# ---------------------------------------------------------------
# Phase features (6-D) — matches lunges/extract_phase.py
# ---------------------------------------------------------------

def extract_phase_features(
    lm: object,
    prev_vals: Optional[Tuple[float, float, float]],
) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    """Extract 6-dim features for the lunge phase TCN.

    Features: hip_h, shoulder_h, knee_h, hip_v, shoulder_v, knee_v
    All heights normalized by torso length; velocities relative to prev frame.

    Returns:
        feats:     (6,) float32 array
        curr_vals: (hip_h, shoulder_h, knee_h) — pass as prev_vals next frame

    Matches the feature vector produced by lunges/extract_phase.py.
    """
    if isinstance(lm, np.ndarray):
        def get_pt(i: int) -> np.ndarray:
            return lm[i, :2]
    else:
        def get_pt(i: int) -> np.ndarray:
            return np.array([lm[i].x, lm[i].y], dtype=np.float32)

    mid_hip      = (get_pt(L_HIP) + get_pt(R_HIP)) * 0.5
    mid_shoulder = (get_pt(L_SHO) + get_pt(R_SHO)) * 0.5
    mid_knee     = (get_pt(L_KNE) + get_pt(R_KNE)) * 0.5

    torso_len = float(abs(mid_shoulder[1] - mid_hip[1]) + 1e-6)

    def ny(p: np.ndarray) -> float:
        return float((p[1] - mid_hip[1]) / torso_len)

    hip_h      = ny(mid_hip)
    shoulder_h = ny(mid_shoulder)
    knee_h     = ny(mid_knee)

    if prev_vals is None:
        hip_v = shoulder_v = knee_v = 0.0
    else:
        hip_v      = hip_h      - prev_vals[0]
        shoulder_v = shoulder_h - prev_vals[1]
        knee_v     = knee_h     - prev_vals[2]

    feats = np.array([hip_h, shoulder_h, knee_h, hip_v, shoulder_v, knee_v], dtype=np.float32)
    return feats, (hip_h, shoulder_h, knee_h)


# ---------------------------------------------------------------
# Landmark smoother (used by streaming, not training)
# ---------------------------------------------------------------

class LandmarkSmoother:
    """Exponential moving average smoother for MediaPipe landmarks."""

    def __init__(self, alpha: float = 0.6) -> None:
        self.alpha = alpha
        self.prev: Optional[np.ndarray] = None

    def reset(self) -> None:
        self.prev = None

    def update(self, curr: np.ndarray) -> np.ndarray:
        if self.prev is None:
            self.prev = curr
            return curr
        self.prev = self.alpha * curr + (1.0 - self.alpha) * self.prev
        return self.prev
