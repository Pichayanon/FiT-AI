"""
features.py — Squat feature extraction functions.

Single source of truth used by both training scripts and squat_streaming.py.
All three models (phase, stand, bottom) extract features here.

Import in training:
    from squat.features import extract_phase_features, extract_stand_features, extract_bottom_features

Import in streaming:
    from squat.features import extract_phase_features, extract_stand_features, extract_bottom_features
"""

from __future__ import annotations

from typing import List

import numpy as np

from shared.math_utils import angle_3pts, safe_norm, dist, get_xyz


# ---------------------------------------------------------------
# Constants (shared between training and streaming)
# ---------------------------------------------------------------

BOTTOM_FEATURE_DIM = 41
STAND_FEATURE_DIM = 16

KEY_JOINTS = [7, 8, 11, 12, 23, 24, 25, 26, 27, 28]  # L/R: ear, sho, hip, knee, ankle

L_SHO, R_SHO = 11, 12
L_HIP, R_HIP = 23, 24
L_KNE, R_KNE = 25, 26
L_ANK, R_ANK = 27, 28


# ---------------------------------------------------------------
# Phase features (10-D) — matches squat/extract_phase.py
# ---------------------------------------------------------------

def extract_phase_features(lm: list) -> np.ndarray:
    """Extract 10-dim features for phase TCN.

    Features: normalized Y positions of 8 joints + 2 knee angles.

    Matches the feature vector produced by squat/extract_phase.py.
    """
    idx_map = {
        "l_shoulder": L_SHO, "r_shoulder": R_SHO,
        "l_hip": L_HIP, "r_hip": R_HIP,
        "l_knee": L_KNE, "r_knee": R_KNE,
        "l_ankle": L_ANK, "r_ankle": R_ANK,
    }
    pts = {}
    for name, idx in idx_map.items():
        pts[name] = np.array([lm[idx].x, lm[idx].y], dtype=np.float32)

    mid_hip_y = (pts["l_hip"][1] + pts["r_hip"][1]) / 2
    mid_shoulder_y = (pts["l_shoulder"][1] + pts["r_shoulder"][1]) / 2
    torso_len = abs(mid_shoulder_y - mid_hip_y) + 1e-6

    def ny(p: np.ndarray) -> float:
        return (p[1] - mid_hip_y) / torso_len

    l_knee_angle = angle_3pts(
        tuple(pts["l_hip"]), tuple(pts["l_knee"]), tuple(pts["l_ankle"])
    )
    r_knee_angle = angle_3pts(
        tuple(pts["r_hip"]), tuple(pts["r_knee"]), tuple(pts["r_ankle"])
    )
    return np.array([
        ny(pts["l_shoulder"]), ny(pts["r_shoulder"]),
        ny(pts["l_hip"]),      ny(pts["r_hip"]),
        ny(pts["l_knee"]),     ny(pts["r_knee"]),
        ny(pts["l_ankle"]),    ny(pts["r_ankle"]),
        l_knee_angle / 180.0,  r_knee_angle / 180.0,
    ], dtype=np.float32)


# ---------------------------------------------------------------
# Stand features (16-D) — matches squat/extract_standing_squat.py
# ---------------------------------------------------------------

def extract_stand_features(lm: list) -> np.ndarray:
    """Extract focused stand features (16-D).

    Dimensions:
        [0-3]   width ratios: ankle/hip, ankle/sho, knee/hip, knee/sho
        [4-9]   x positions (norm by hip width): L/R ankle, knee, hip
        [10-12] angles/180: knee_L, knee_R, torso_tilt
        [13]    feet distance (ankle_w / scale)
        [14]    shoulder distance (sho_w / scale)
        [15]    feet/shoulder ratio (ankle_w / sho_w)

    Matches the feature vector produced by squat/extract_standing_squat.py.
    """
    xyz = get_xyz(lm)
    lhip, rhip = xyz[L_HIP], xyz[R_HIP]
    lsho, rsho = xyz[L_SHO], xyz[R_SHO]
    lkne, rkne = xyz[L_KNE], xyz[R_KNE]
    lank, rank = xyz[L_ANK], xyz[R_ANK]

    mid_hip = 0.5 * (lhip + rhip)
    hip_w   = dist(lhip, rhip)
    sho_w   = dist(lsho, rsho)
    ankle_w = dist(lank, rank)
    knee_w  = dist(lkne, rkne)
    scale   = hip_w if hip_w > 1e-4 else (sho_w if sho_w > 1e-4 else 1.0)

    out = np.zeros(STAND_FEATURE_DIM, dtype=np.float32)
    out[0]  = ankle_w / (hip_w + 1e-6)
    out[1]  = ankle_w / (sho_w + 1e-6)
    out[2]  = knee_w  / (hip_w + 1e-6)
    out[3]  = knee_w  / (sho_w + 1e-6)
    out[4]  = (lank[0] - mid_hip[0]) / (scale + 1e-6)
    out[5]  = (rank[0] - mid_hip[0]) / (scale + 1e-6)
    out[6]  = (lkne[0] - mid_hip[0]) / (scale + 1e-6)
    out[7]  = (rkne[0] - mid_hip[0]) / (scale + 1e-6)
    out[8]  = (lhip[0] - mid_hip[0]) / (scale + 1e-6)
    out[9]  = (rhip[0] - mid_hip[0]) / (scale + 1e-6)
    out[10] = angle_3pts(lhip, lkne, lank) / 180.0
    out[11] = angle_3pts(rhip, rkne, rank) / 180.0

    mid_sho = 0.5 * (lsho + rsho)
    v   = (mid_sho - mid_hip).astype(np.float32)
    up  = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    denom   = (safe_norm(v) * safe_norm(up)) + 1e-6
    cosang  = float(np.clip(np.dot(v, up) / denom, -1.0, 1.0))
    out[12] = float(np.degrees(np.arccos(cosang))) / 180.0
    out[13] = ankle_w / (scale + 1e-6)
    out[14] = sho_w   / (scale + 1e-6)
    out[15] = ankle_w / (sho_w + 1e-6)

    return out


# ---------------------------------------------------------------
# Bottom features (41-D) — matches squat/extract_bottom_squat.py
# ---------------------------------------------------------------

def extract_bottom_features(lm: list) -> np.ndarray:
    """Extract focused bottom features (41-D).

    Dimensions:
        [0-29]  10 key joints × 3 xyz (body-centric, hip-width normalized)
        [30-36] angles/180: knee_L, knee_R, hip_L, hip_R, torso_tilt, neck_tilt, spine
        [37-40] width ratios: knee/hip, knee/ankle, ankle/hip, sho/hip

    Matches the feature vector produced by squat/extract_bottom_squat.py.
    """
    xyz   = get_xyz(lm)
    lear, rear = xyz[7],  xyz[8]
    lhip, rhip = xyz[23], xyz[24]
    lsho, rsho = xyz[11], xyz[12]
    lkne, rkne = xyz[25], xyz[26]
    lank, rank = xyz[27], xyz[28]

    mid_hip = 0.5 * (lhip + rhip)
    hip_w   = dist(lhip, rhip)
    sho_w   = dist(lsho, rsho)
    scale   = hip_w if hip_w > 1e-4 else (sho_w if sho_w > 1e-4 else 1.0)

    out = np.zeros(BOTTOM_FEATURE_DIM, dtype=np.float32)

    for i, j_idx in enumerate(KEY_JOINTS):
        normed = (xyz[j_idx] - mid_hip) / (scale + 1e-6)
        out[i * 3:(i + 1) * 3] = normed

    out[30] = angle_3pts(lhip, lkne, lank) / 180.0
    out[31] = angle_3pts(rhip, rkne, rank) / 180.0
    out[32] = angle_3pts(lsho, lhip, lkne) / 180.0
    out[33] = angle_3pts(rsho, rhip, rkne) / 180.0

    mid_sho = 0.5 * (lsho + rsho)
    mid_ear = 0.5 * (lear + rear)
    up      = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    v      = (mid_sho - mid_hip).astype(np.float32)
    denom  = (safe_norm(v) * safe_norm(up)) + 1e-6
    cosang = float(np.clip(np.dot(v, up) / denom, -1.0, 1.0))
    out[34] = float(np.degrees(np.arccos(cosang))) / 180.0

    v_neck      = (mid_ear - mid_sho).astype(np.float32)
    denom_neck  = (safe_norm(v_neck) * safe_norm(up)) + 1e-6
    cosang_neck = float(np.clip(np.dot(v_neck, up) / denom_neck, -1.0, 1.0))
    out[35] = float(np.degrees(np.arccos(cosang_neck))) / 180.0

    out[36] = angle_3pts(mid_ear, mid_sho, mid_hip) / 180.0

    knee_w  = dist(lkne, rkne)
    ankle_w = dist(lank, rank)
    out[37] = knee_w  / (hip_w + 1e-6)
    out[38] = knee_w  / (ankle_w + 1e-6)
    out[39] = ankle_w / (hip_w + 1e-6)
    out[40] = sho_w   / (hip_w + 1e-6)

    return out
