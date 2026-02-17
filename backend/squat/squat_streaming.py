"""
squat_streaming.py — OOP version (same pattern as wall_sit_streaming).

- FRONT-VIEW gate (FrontViewGate) → READY after READY_STREAK_N frames
- Phase from phase TCN only (eccentric / concentric); knee EMA + minima for bottom event
- Bottom TCN: trigger at knee-EMA minima → window → predict (rep counting)
- Stand TCN: CHECK ONCE before first rep; then only bottom results

WS: start | frame | stop
Server: status, phase, result (mode=bottom|stand), info
"""

from __future__ import annotations

import os
import time
import json
import base64
import argparse
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple, List
from collections import deque

import numpy as np
import cv2
import mediapipe as mp
import torch
import torch.nn as nn

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware


# -----------------------------
# Config
# -----------------------------
TCN_MODEL_PATH   = "squat/models/squat_knees_in_tcn.pt"
STAND_MODEL_PATH = "squat/models/feet_too_close_tcn.pt"
# Phase TCN from train_squat_phase (10-dim, window 30, 2 classes: eccentric / concentric)
PHASE_MODEL_PATH = "squat/models/squat_phase_tcn.pt"

FEATURE_MODE = "A"   # "A" (111 dims) or "RAW" (132 dims)

PRE_FRAMES  = 5
POST_FRAMES = 5

MIN_GAP = 18  # min frames between bottom events (eccentric→concentric)

READY_STREAK_N = 3

FRONT_VIS_TH = 0.6
FRONT_MIN_SHOULDER_X_GAP = 0.08
FRONT_MIN_HIP_X_GAP      = 0.06

MP_MIN_DET_CONF = 0.80
MP_MIN_TRACK_CONF = 0.80

STATUS_SEND_EVERY_N_FRAMES = 3
PHASE_SEND_EVERY_N_FRAMES  = 2

SAVE_VIDEO = True
RECORD_DIR = "recordings_squat"
RECORD_FPS = 10.0
RECORD_ONLY_WHEN_READY = False
PRINT_EVERY_SAVED_FRAMES = 30
os.makedirs(RECORD_DIR, exist_ok=True)

STAND_MIN_STREAK     = 6
STAND_PRED_COOLDOWN  = 12
STAND_WIN_FRAMES     = PRE_FRAMES + POST_FRAMES + 1

STAND_OK_LABELS = {"stand_ok"}
GOAL_GOOD_REPS = 5

DEBUG = True


# -----------------------------
# App
# -----------------------------
app = FastAPI(title="FiT-AI Squat Streaming Backend (OOP)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)


# -----------------------------
# FrameDecoder (same role as wall-sit)
# -----------------------------
class FrameDecoder:
    @staticmethod
    def decode_jpeg_base64(jpeg_b64: str) -> Optional[np.ndarray]:
        try:
            raw = base64.b64decode(jpeg_b64)
            arr = np.frombuffer(raw, np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except Exception:
            return None


# -----------------------------
# MediaPipe
# -----------------------------
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

L_SHO, R_SHO = 11, 12
L_HIP, R_HIP = 23, 24
L_KNE, R_KNE = 25, 26
L_ANK, R_ANK = 27, 28

FRONT_LM = [
    mp_pose.PoseLandmark.LEFT_SHOULDER,
    mp_pose.PoseLandmark.RIGHT_SHOULDER,
    mp_pose.PoseLandmark.LEFT_HIP,
    mp_pose.PoseLandmark.RIGHT_HIP,
    mp_pose.PoseLandmark.LEFT_KNEE,
    mp_pose.PoseLandmark.RIGHT_KNEE,
    mp_pose.PoseLandmark.LEFT_ANKLE,
    mp_pose.PoseLandmark.RIGHT_ANKLE,
]

FRONT_LM_LABELS = {
    mp_pose.PoseLandmark.LEFT_SHOULDER: "L_SHO",
    mp_pose.PoseLandmark.RIGHT_SHOULDER: "R_SHO",
    mp_pose.PoseLandmark.LEFT_HIP: "L_HIP",
    mp_pose.PoseLandmark.RIGHT_HIP: "R_HIP",
    mp_pose.PoseLandmark.LEFT_KNEE: "L_KNE",
    mp_pose.PoseLandmark.RIGHT_KNEE: "R_KNE",
    mp_pose.PoseLandmark.LEFT_ANKLE: "L_ANK",
    mp_pose.PoseLandmark.RIGHT_ANKLE: "R_ANK",
}


# -----------------------------
# TCN model
# -----------------------------
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
    def __init__(self, in_dim, num_classes=2, channels=(128, 128, 128), k=3, dropout=0.1):
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
        x = x.transpose(1, 2)  # (B,T,D) -> (B,D,T)
        y = self.tcn(x)
        y = self.pool(y).squeeze(-1)
        return self.fc(y)


def load_tcn(path: str):
    """
    Load TCN checkpoint dict with keys: in_dim, T, label_map, model_state.
    """
    try:
        ckpt = torch.load(path, map_location="cpu")
        in_dim = int(ckpt["in_dim"])
        T = int(ckpt["T"])
        label_map = ckpt["label_map"]
        inv = {v: k for k, v in label_map.items()}
        model = SimpleTCN(in_dim=in_dim, num_classes=len(inv), channels=(128, 128, 128), dropout=0.1)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        print(f"[MODEL] Loaded: {path} in_dim={in_dim} T={T} classes={inv}")
        return model, T, inv, in_dim
    except Exception as e:
        print(f"[MODEL] Cannot load: {path} err={e}")
        return None, None, None, None


# -----------------------------
# Phase TCN (from train_squat_phase: 10-dim, window 30, 2 classes)
# -----------------------------
class PhaseTemporalBlock(nn.Module):
    """Same as train_squat_phase TemporalBlock (k=3, d=1, no dropout)."""
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, d: int = 1):
        super().__init__()
        pad = (k - 1) * d
        self.conv1 = nn.Conv1d(in_ch, out_ch, k, padding=pad, dilation=d)
        self.conv2 = nn.Conv1d(out_ch, out_ch, k, padding=pad, dilation=d)
        self.relu = nn.ReLU()
        self.down = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x):
        y = self.relu(self.conv1(x))
        y = self.relu(self.conv2(y))
        if self.down is not None:
            x = self.down(x)
        return y[..., : x.size(-1)] + x


class PhaseTCN(nn.Module):
    """TCN from train_squat_phase: in_dim=10, num_classes=2, output (B, W, C)."""
    def __init__(self, in_dim: int = 10, num_classes: int = 2):
        super().__init__()
        self.tcn = nn.Sequential(
            PhaseTemporalBlock(in_dim, 64, d=1),
            PhaseTemporalBlock(64, 64, d=2),
            PhaseTemporalBlock(64, 64, d=4),
        )
        self.fc = nn.Conv1d(64, num_classes, 1)

    def forward(self, x):
        x = x.transpose(1, 2)   # (B, W, F) -> (B, F, W)
        x = self.tcn(x)
        x = self.fc(x)
        return x.transpose(1, 2)  # (B, W, C)


PHASE_LABELS = {0: "eccentric", 1: "concentric"}


def load_phase_tcn(path: str):
    """Load phase model from train_squat_phase checkpoint (state_dict, in_dim, num_classes, window)."""
    try:
        ckpt = torch.load(path, map_location="cpu")
        in_dim = int(ckpt.get("in_dim", 10))
        num_classes = int(ckpt.get("num_classes", 2))
        window = int(ckpt.get("window", 30))
        model = PhaseTCN(in_dim=in_dim, num_classes=num_classes)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        print(f"[MODEL] Phase TCN loaded: {path} in_dim={in_dim} window={window} classes={list(PHASE_LABELS.values())}")
        return model, window, in_dim
    except Exception as e:
        print(f"[MODEL] Cannot load phase TCN: {path} err={e}")
        return None, None, None


def extract_phase_features(lm) -> np.ndarray:
    """10-dim features matching extract_phase.py (for phase TCN from train_squat_phase)."""
    pts = {}
    idx_map = {
        "l_shoulder": L_SHO, "r_shoulder": R_SHO,
        "l_hip": L_HIP, "r_hip": R_HIP,
        "l_knee": L_KNE, "r_knee": R_KNE,
        "l_ankle": L_ANK, "r_ankle": R_ANK,
    }
    for name, idx in idx_map.items():
        pts[name] = np.array([lm[idx].x, lm[idx].y], dtype=np.float32)
    mid_hip_y = (pts["l_hip"][1] + pts["r_hip"][1]) / 2
    mid_shoulder_y = (pts["l_shoulder"][1] + pts["r_shoulder"][1]) / 2
    torso_len = abs(mid_shoulder_y - mid_hip_y) + 1e-6

    def ny(p):
        return (p[1] - mid_hip_y) / torso_len

    l_knee_angle = angle_3pts(
        tuple(pts["l_hip"]), tuple(pts["l_knee"]), tuple(pts["l_ankle"])
    )
    r_knee_angle = angle_3pts(
        tuple(pts["r_hip"]), tuple(pts["r_knee"]), tuple(pts["r_ankle"])
    )
    return np.array([
        ny(pts["l_shoulder"]), ny(pts["r_shoulder"]),
        ny(pts["l_hip"]), ny(pts["r_hip"]),
        ny(pts["l_knee"]), ny(pts["r_knee"]),
        ny(pts["l_ankle"]), ny(pts["r_ankle"]),
        l_knee_angle / 180.0, r_knee_angle / 180.0,
    ], dtype=np.float32)


# -----------------------------
# SquatModelService (same role as ModelService in wall-sit: load + predict)
# -----------------------------
class SquatModelService:
    """Holds bottom TCN + stand TCN + optional phase TCN (from train_squat_phase)."""

    def __init__(self, bottom_path: str, stand_path: str, phase_path: Optional[str] = None) -> None:
        self.bottom_model, self.bottom_T, self.inv_labels_bottom, self.bottom_in_dim = load_tcn(bottom_path)
        self.stand_model, self.stand_T, self.inv_labels_stand, self.stand_in_dim = load_tcn(stand_path)
        self.phase_model = None
        self.phase_window = None
        self.phase_in_dim = None
        if phase_path and os.path.isfile(phase_path):
            self.phase_model, self.phase_window, self.phase_in_dim = load_phase_tcn(phase_path)

    @property
    def bottom_loaded(self) -> bool:
        return self.bottom_model is not None

    @property
    def stand_loaded(self) -> bool:
        return self.stand_model is not None

    @property
    def phase_loaded(self) -> bool:
        return self.phase_model is not None and self.phase_window is not None

    def predict_bottom(self, X_win: np.ndarray) -> Tuple[str, float, np.ndarray]:
        """(pred_label, conf, probs)."""
        if self.bottom_model is None or self.bottom_T is None:
            return "unknown", 0.0, np.array([])
        return tcn_predict(
            self.bottom_model, self.inv_labels_bottom, int(self.bottom_T), X_win.astype(np.float32)
        )

    def predict_stand(self, X_win: np.ndarray) -> Tuple[str, float, np.ndarray]:
        if self.stand_model is None or self.stand_T is None:
            return "unknown", 0.0, np.array([])
        return tcn_predict(
            self.stand_model, self.inv_labels_stand, int(self.stand_T), X_win.astype(np.float32)
        )

    def predict_phase(self, X_win: np.ndarray) -> str:
        """X_win: (window, 10). Returns 'eccentric' or 'concentric' (from train_squat_phase)."""
        if self.phase_model is None or self.phase_window is None or X_win.shape[0] < self.phase_window:
            return "unknown"
        W = int(self.phase_window)
        x = X_win[-W:].astype(np.float32)
        xt = torch.from_numpy(x).unsqueeze(0)  # (1, W, 10)
        with torch.no_grad():
            logits = self.phase_model(xt)  # (1, W, 2)
            last_logits = logits[0, -1, :]
            pred_id = int(torch.argmax(last_logits).item())
        return PHASE_LABELS.get(pred_id, "unknown")


# -----------------------------
# Utils (angle, resample, normalize, tcn_predict)
# -----------------------------
def angle_3pts(a, b, c) -> float:
    """
    Compute angle ABC (degrees) from 2D points.
    """
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    c = np.array(c, dtype=np.float32)
    ba = a - b
    bc = c - b
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    cosang = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))


def landmarks_to_flat(lm) -> np.ndarray:
    """
    Flatten 33 pose landmarks into (132,) [x,y,z,vis] * 33.
    """
    arr = np.zeros((33, 4), dtype=np.float32)
    for i in range(33):
        arr[i, 0] = lm[i].x
        arr[i, 1] = lm[i].y
        arr[i, 2] = lm[i].z
        arr[i, 3] = lm[i].visibility
    return arr.reshape(-1).astype(np.float32)  # (132,)


def resample_time(x: np.ndarray, target_T: int) -> np.ndarray:
    """
    Linear interpolate over time axis to target_T.
    x: (T, D)
    """
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


def normalize_per_sample(x: np.ndarray) -> np.ndarray:
    """
    Z-normalize per sample (per feature dim) across time.
    """
    mu = x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True) + 1e-6
    return ((x - mu) / sd).astype(np.float32)


def create_video_writer(path_no_ext: str, w: int, h: int, fps: float) -> Tuple[Optional[cv2.VideoWriter], str]:
    """
    Create mp4 writer; return (writer, actual_path).
    """
    mp4_path = f"{path_no_ext}.mp4"
    try:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(mp4_path, fourcc, fps, (w, h))
        if writer.isOpened():
            return writer, mp4_path
    except Exception:
        pass
    return None, ""


# -----------------------------
# FrontViewGate (same role as SideGate in wall-sit: gate for READY)
# -----------------------------
class FrontViewGate:
    """Front-view gate: visibility + left-right separation."""

    def __init__(self, mp_pose: Any, vis_th: float, min_sho_gap: float, min_hip_gap: float) -> None:
        self.mp_pose = mp_pose
        self.vis_th = vis_th
        self.min_sho_gap = min_sho_gap
        self.min_hip_gap = min_hip_gap

    def check(self, lm) -> Tuple[bool, Dict[str, Any]]:
        vis_map: Dict[str, float] = {}
        ok_vis = True
        for idx in FRONT_LM:
            v = float(lm[idx].visibility)
            vis_map[FRONT_LM_LABELS.get(idx, str(idx))] = round(v, 3)
            if v < self.vis_th:
                ok_vis = False

        lsho_x = float(lm[self.mp_pose.PoseLandmark.LEFT_SHOULDER].x)
        rsho_x = float(lm[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].x)
        lhip_x = float(lm[self.mp_pose.PoseLandmark.LEFT_HIP].x)
        rhip_x = float(lm[self.mp_pose.PoseLandmark.RIGHT_HIP].x)
        sho_gap = abs(lsho_x - rsho_x)
        hip_gap = abs(lhip_x - rhip_x)
        ok_gap = (sho_gap >= self.min_sho_gap) and (hip_gap >= self.min_hip_gap)

        dbg = {
            "vis_ok": ok_vis,
            "gap_ok": ok_gap,
            "sho_gap": round(sho_gap, 3),
            "hip_gap": round(hip_gap, 3),
            "vis_th": float(self.vis_th),
            "vis": vis_map,
        }
        return (ok_vis and ok_gap), dbg


def draw_overlay(
    frame_bgr: np.ndarray,
    res,
    state: str,
    phase: str,
    knee_raw: Optional[float],
    knee_ema: Optional[float],
    pred_text: Optional[str] = None,
    extra_text: Optional[str] = None,
    stand_pred_text: Optional[str] = None,
    rep_text: Optional[str] = None,
    feat_dim: Optional[int] = None,
    gate_text: Optional[str] = None,
) -> np.ndarray:
    """
    Draw pose + debug texts onto frame.
    """
    out = frame_bgr.copy()
    if res.pose_landmarks:
        mp_drawing.draw_landmarks(
            out,
            res.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style()
        )

    y = 26

    def put(t: str, s: float = 0.7):
        nonlocal y
        cv2.putText(out, t, (12, y), cv2.FONT_HERSHEY_SIMPLEX, s, (255, 255, 255), 2, cv2.LINE_AA)
        y += 26

    put(f"State: {state}")
    if gate_text:
        put(gate_text, 0.60)
    put(f"Phase: {phase}")
    put(f"Knee: {knee_raw:.1f}" if knee_raw is not None else "Knee: NA")
    put(
        f"feat={FEATURE_MODE} dim={feat_dim} | bottom=ecc→conc stand=phase pre/post={PRE_FRAMES}/{POST_FRAMES}",
        0.55
    )

    if extra_text:
        put(extra_text, 0.6)

    if rep_text:
        put(rep_text, 0.65)

    if pred_text:
        cv2.putText(out, pred_text, (12, y + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (255, 255, 255), 2, cv2.LINE_AA)
        y += 30

    if stand_pred_text:
        cv2.putText(out, stand_pred_text, (12, y + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)

    return out


def tcn_predict(model, inv_labels: Dict[int, str], target_T: int, X_win: np.ndarray):
    """
    Predict with TCN.
    Args:
        model: torch model
        inv_labels: {class_id: label}
        target_T: T to resample to
        X_win: (Tw, D)
    Returns:
        (pred_label, conf, probs)
    """
    X = resample_time(X_win.astype(np.float32), int(target_T))
    X = normalize_per_sample(X)
    xt = torch.from_numpy(X).unsqueeze(0)  # (1,T,D)
    with torch.no_grad():
        logits = model(xt)
        prob = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred = int(np.argmax(prob))
        conf = float(prob[pred])
        pred_label = inv_labels.get(pred, str(pred))
    return pred_label, conf, prob


# -----------------------------
# Feature Set A (111 dims)
# -----------------------------
def _safe_norm(v: np.ndarray, eps: float = 1e-6) -> float:
    return float(np.sqrt(np.sum(v * v)) + eps)


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.sum((a - b) ** 2)))


def _angle_3pts_np(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ba = a - b
    bc = c - b
    denom = (_safe_norm(ba) * _safe_norm(bc)) + 1e-6
    cosang = float(np.clip(np.dot(ba, bc) / denom, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosang)))


def extract_feature_A_from_landmarks(lm) -> np.ndarray:
    """
    lm: MediaPipe landmark list length 33
    returns per-frame feature: (111,)
      A1) body-centric xyz normalized (origin=mid-hip, scale=hip_width) -> 99
      A2) dist/ratio (hip_w, sho_w, ankle_w, knee_w, ankle/hip, knee/hip, sho/hip) -> 7
      A3) angles (knee_L, knee_R, hip_L, hip_R, torso_tilt) -> 5
    """
    xyz = np.zeros((33, 3), dtype=np.float32)
    for i in range(33):
        xyz[i, 0] = lm[i].x
        xyz[i, 1] = lm[i].y
        xyz[i, 2] = lm[i].z

    lhip, rhip = xyz[L_HIP], xyz[R_HIP]
    lsho, rsho = xyz[L_SHO], xyz[R_SHO]
    lkne, rkne = xyz[L_KNE], xyz[R_KNE]
    lank, rank = xyz[L_ANK], xyz[R_ANK]

    mid_hip = 0.5 * (lhip + rhip)
    hip_w = _dist(lhip, rhip)
    sho_w = _dist(lsho, rsho)
    scale = hip_w if hip_w > 1e-4 else (sho_w if sho_w > 1e-4 else 1.0)

    # A1: body-centric xyz
    p_norm = (xyz - mid_hip) / (scale + 1e-6)        # (33,3)
    feat_xyz = p_norm.reshape(-1).astype(np.float32) # (99,)

    # A2: distances / ratios
    ankle_w = _dist(lank, rank)
    knee_w  = _dist(lkne, rkne)
    ankle_hip = ankle_w / (scale + 1e-6)
    knee_hip  = knee_w / (scale + 1e-6)
    sho_hip   = sho_w / (scale + 1e-6)

    feat_dist = np.array(
        [hip_w, sho_w, ankle_w, knee_w, ankle_hip, knee_hip, sho_hip],
        dtype=np.float32
    )

    # A3: angles
    knee_L = _angle_3pts_np(lhip, lkne, lank)
    knee_R = _angle_3pts_np(rhip, rkne, rank)
    hip_L  = _angle_3pts_np(lsho, lhip, lkne)
    hip_R  = _angle_3pts_np(rsho, rhip, rkne)

    mid_sho = 0.5 * (lsho + rsho)
    v = (mid_sho - mid_hip).astype(np.float32)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    denom = (_safe_norm(v) * _safe_norm(up)) + 1e-6
    cosang = float(np.clip(np.dot(v, up) / denom, -1.0, 1.0))
    torso_tilt = float(np.degrees(np.arccos(cosang)))

    feat_ang = np.array([knee_L, knee_R, hip_L, hip_R, torso_tilt], dtype=np.float32)

    return np.concatenate([feat_xyz, feat_dist, feat_ang], axis=0).astype(np.float32)  # (111,)


def extract_stream_feature(lm) -> np.ndarray:
    """
    Return per-frame feature vector consistent with FEATURE_MODE.
    """
    if FEATURE_MODE == "RAW":
        return landmarks_to_flat(lm)  # (132,)
    if FEATURE_MODE == "A":
        return extract_feature_A_from_landmarks(lm)  # (111,)
    raise ValueError(f"Invalid FEATURE_MODE={FEATURE_MODE}")


def _expected_dim() -> Optional[int]:
    """
    Expected feature dim for current FEATURE_MODE.
    """
    if FEATURE_MODE == "RAW":
        return 132
    if FEATURE_MODE == "A":
        return 111
    return None


# -----------------------------
# FeatureExtractor (same role as wall-sit: per-frame features for model)
# -----------------------------
class FeatureExtractor:
    """Per-frame feature vector (Feature A or RAW) for squat TCNs."""

    def __init__(self, feature_mode: str) -> None:
        self.feature_mode = feature_mode

    def extract(self, lm) -> np.ndarray:
        if self.feature_mode == "RAW":
            return landmarks_to_flat(lm)
        if self.feature_mode == "A":
            return extract_feature_A_from_landmarks(lm)
        raise ValueError(f"Invalid feature_mode={self.feature_mode}")


# -----------------------------
# StatusSender (same role as wall-sit: send status/info; + send_phase for squat)
# -----------------------------
class StatusSender:
    def __init__(self, every_n_frames: int, phase_every_n: int = 2) -> None:
        self.every_n_frames = max(1, int(every_n_frames))
        self.phase_every_n = max(1, int(phase_every_n))

    async def send_info(self, websocket: WebSocket, msg: str, extra: Optional[Dict[str, Any]] = None) -> None:
        payload: Dict[str, Any] = {"type": "info", "message": msg}
        if extra:
            payload.update(extra)
        await websocket.send_text(json.dumps(payload))

    async def send_status(
        self,
        websocket: WebSocket,
        st: "StreamState",
        state: str,
        extra: Optional[Dict[str, Any]] = None,
        force: bool = False,
    ) -> None:
        st.status_tick += 1
        if not force:
            if (st.status_tick % self.every_n_frames != 0) and (state == st.last_status):
                return
        payload: Dict[str, Any] = {"type": "status", "state": state, "session_id": st.session_id}
        if extra:
            payload.update(extra)
        st.last_status = state
        await websocket.send_text(json.dumps(payload))

    async def send_phase(
        self,
        websocket: WebSocket,
        st: "StreamState",
        phase: str,
        force: bool = False,
    ) -> None:
        st.phase_tick += 1
        if not force and (st.phase_tick % self.phase_every_n != 0) and (phase == st.last_phase):
            return
        payload: Dict[str, Any] = {
            "type": "phase",
            "phase": phase,
            "session_id": st.session_id,
        }
        st.last_phase = phase
        await websocket.send_text(json.dumps(payload))


# -----------------------------
# StreamState (same role as wall-sit)
# -----------------------------
@dataclass
class StreamState:
    started: bool = False
    session_id: str = ""
    out_path_no_ext: str = ""

    # gate
    ready: bool = False
    ready_streak: int = 0
    last_gate_debug: Dict[str, Any] = field(default_factory=dict)

    # recording
    writer: Optional[cv2.VideoWriter] = None
    writer_size: Optional[Tuple[int, int]] = None
    actual_video_path: str = ""
    saved_frames: int = 0

    # status/phase throttles
    status_tick: int = 0
    phase_tick: int = 0
    last_status: str = ""
    last_phase: str = "stand"

    # last predictions for overlay
    last_pred_label: str = ""
    last_pred_conf: Optional[float] = None

    # standing predict state (CHECK ONCE ONLY)
    stand_streak: int = 0
    last_stand_pred_i: int = -10**9
    last_stand_pred_label: str = ""
    last_stand_pred_conf: Optional[float] = None

    # NEW: check stand once, then never send stand again
    stand_checked_once: bool = False
    stand_ok: bool = False

    # rep counting (bottom-event based)
    total_reps: int = 0
    good_reps: int = 0
    bad_reps: int = 0
    last_counted_event_i: int = -10**9

    # de-dup sending
    last_sent_bottom_event_i: int = -10**9
    last_sent_stand_label: str = ""

    # history for window cutting: (i, feat(D), frame_for_overlay, knee_raw, knee_ema)
    hist: deque = field(default_factory=lambda: deque(maxlen=PRE_FRAMES + POST_FRAMES + 240))

    # phase TCN buffer (10-dim from extract_phase_features)
    phase_feat_buffer: deque = field(default_factory=lambda: deque(maxlen=35))
    prev_phase: str = ""  # previous frame phase for eccentric→concentric transition
    last_phase_bottom_i: int = -10**9  # cooldown: last frame we triggered bottom from transition

    # pending capture after bottom event until post frames collected
    pending: Optional[Dict[str, Any]] = None


def update_rep_counter(st: StreamState, event_i: int, pred_label: str) -> None:
    """Count once per event. Good rep: pred_label != 'knees_in'; Bad: pred_label == 'knees_in'."""
    if event_i == st.last_counted_event_i:
        return
    st.last_counted_event_i = event_i
    st.total_reps += 1
    if pred_label == "knees_in":
        st.bad_reps += 1
    else:
        st.good_reps += 1


# -----------------------------
# Shared instances (same pattern as wall-sit)
# -----------------------------
model_service = SquatModelService(TCN_MODEL_PATH, STAND_MODEL_PATH, PHASE_MODEL_PATH)
front_view_gate = FrontViewGate(
    mp_pose, FRONT_VIS_TH, FRONT_MIN_SHOULDER_X_GAP, FRONT_MIN_HIP_X_GAP
)
feature_extractor = FeatureExtractor(FEATURE_MODE)
status_sender = StatusSender(STATUS_SEND_EVERY_N_FRAMES, PHASE_SEND_EVERY_N_FRAMES)


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "feature_mode": FEATURE_MODE,
        "expected_dim": _expected_dim(),
        "bottom_loaded": model_service.bottom_loaded,
        "stand_loaded": model_service.stand_loaded,
        "phase_loaded": model_service.phase_loaded,
        "bottom_in_dim": model_service.bottom_in_dim,
        "stand_in_dim": model_service.stand_in_dim,
        "stand_ok_labels": list(STAND_OK_LABELS),
        "front_vis_th": FRONT_VIS_TH,
        "front_min_sho_gap": FRONT_MIN_SHOULDER_X_GAP,
        "front_min_hip_gap": FRONT_MIN_HIP_X_GAP,
        "record_dir": os.path.abspath(RECORD_DIR),
        "timestamp": int(time.time()),
    }


# -----------------------------
# SquatWebSocketSession (same role as WallSitWebSocketSession)
# -----------------------------
class SquatWebSocketSession:
    def __init__(
        self,
        websocket: WebSocket,
        model_svc: SquatModelService,
        gate: FrontViewGate,
        feat: FeatureExtractor,
        status: StatusSender,
        ready_streak_n: int,
        debug: bool,
    ) -> None:
        self.ws = websocket
        self.model_svc = model_svc
        self.gate = gate
        self.feat = feat
        self.status = status
        self.ready_streak_n = ready_streak_n
        self.debug = debug

        self.st = StreamState()
        self.frame_i = 0

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=MP_MIN_DET_CONF,
            min_tracking_confidence=MP_MIN_TRACK_CONF,
        )

    async def run(self) -> None:
        await self.ws.accept()

        exp_dim = _expected_dim()
        if self.model_svc.bottom_loaded and exp_dim != self.model_svc.bottom_in_dim:
            await self.status.send_info(
                self.ws,
                f"WARNING: Bottom model in_dim={self.model_svc.bottom_in_dim} but FEATURE_MODE={FEATURE_MODE} gives dim={exp_dim}",
            )
        if self.model_svc.stand_loaded and exp_dim != self.model_svc.stand_in_dim:
            await self.status.send_info(
                self.ws,
                f"WARNING: Stand model in_dim={self.model_svc.stand_in_dim} but FEATURE_MODE={FEATURE_MODE} gives dim={exp_dim}",
            )

        await self.status.send_info(
            self.ws,
            "WebSocket connected",
            {
                "record_dir": os.path.abspath(RECORD_DIR),
                "feature_mode": FEATURE_MODE,
                "front_gate": {
                    "vis_th": FRONT_VIS_TH,
                    "min_sho_gap": FRONT_MIN_SHOULDER_X_GAP,
                    "min_hip_gap": FRONT_MIN_HIP_X_GAP,
                    "needed_streak": self.ready_streak_n,
                },
                "stand_once_only": True,
                "stand_ok_labels": list(STAND_OK_LABELS),
            },
        )

        try:
            while True:
                msg = await self.ws.receive_text()
                data = self._parse_json(msg)
                if data is None:
                    await self.status.send_info(self.ws, "Invalid JSON")
                    continue
                mtype = data.get("type")
                if mtype == "start":
                    await self._handle_start()
                    continue
                if mtype == "stop":
                    await self._handle_stop()
                    continue
                if mtype != "frame" or not self.st.started:
                    continue
                await self._handle_frame(data)
        except WebSocketDisconnect:
            print(f"[WS] disconnect session_id={self.st.session_id}")
            await self._cleanup_recording()
            return
        except Exception as e:
            print(f"[WS] error: {e}")
            await self._cleanup_recording()
            try:
                await self.status.send_info(self.ws, f"Server error: {e}")
            except Exception:
                pass

    def _parse_json(self, msg: str) -> Optional[Dict[str, Any]]:
        try:
            obj = json.loads(msg)
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None

    async def _cleanup_recording(self) -> None:
        if self.st.writer is not None:
            try:
                self.st.writer.release()
                print(f"[RECORD] STOP path={self.st.actual_video_path} frames={self.st.saved_frames}")
            except Exception as e:
                print(f"[RECORD] release error: {e}")
        self.st.writer = None
        self.st.writer_size = None

    async def _start_recording_for_frame(self, frame_bgr: np.ndarray) -> None:
        if not SAVE_VIDEO:
            return
        h, w = frame_bgr.shape[:2]
        if self.st.writer is None:
            writer, actual_path = create_video_writer(self.st.out_path_no_ext, w, h, RECORD_FPS)
            if writer is None:
                await self.status.send_info(self.ws, "Recording disabled: cannot create VideoWriter")
                return
            self.st.writer = writer
            self.st.writer_size = (w, h)
            self.st.actual_video_path = actual_path
            self.st.saved_frames = 0
            print(f"[RECORD] START path={actual_path} size={w}x{h}@{RECORD_FPS}")
            await self.status.send_info(self.ws, "Recording started", {"video_path": actual_path})

    async def _handle_start(self) -> None:
        await self._cleanup_recording()
        self.st = StreamState(started=True)
        self.st.session_id = str(int(time.time() * 1000))
        self.st.out_path_no_ext = os.path.join(RECORD_DIR, f"session_{self.st.session_id}")
        self.frame_i = 0
        self.st.total_reps = 0
        self.st.good_reps = 0
        self.st.bad_reps = 0
        self.st.last_counted_event_i = -10**9
        self.st.last_sent_bottom_event_i = -10**9
        self.st.last_sent_stand_label = ""
        self.st.hist.clear()
        self.st.pending = None
        self.st.stand_ok = False
        self.st.stand_streak = 0
        self.st.last_stand_pred_i = -10**9
        self.st.last_stand_pred_label = ""
        self.st.last_stand_pred_conf = None
        self.st.phase_feat_buffer.clear()
        self.st.prev_phase = ""
        self.st.last_phase_bottom_i = -10**9
        self.frame_i = 0
        print(f"[SESSION] START session_id={self.st.session_id}")
        await self.status.send_info(self.ws, "Start streaming", {"session_id": self.st.session_id})
        await self.status.send_status(self.ws, self.st, "waiting", {"reason": "session_started"}, force=True)

    async def _handle_stop(self) -> None:
        print(f"[SESSION] STOP session_id={self.st.session_id}")
        self.st.started = False
        await self._cleanup_recording()
        await self.status.send_info(
            self.ws,
            "Stop streaming",
            {
                "session_id": self.st.session_id,
                "video_path": self.st.actual_video_path,
                "saved_frames": self.st.saved_frames,
                "reps": {
                    "total": int(self.st.total_reps),
                    "correct": int(self.st.good_reps),
                    "incorrect": int(self.st.bad_reps),
                    "goal_correct": int(GOAL_GOOD_REPS),
                },
                "stand_ok": bool(self.st.stand_ok),
            },
        )
        await self.status.send_status(self.ws, self.st, "waiting", {"reason": "session_stopped"}, force=True)

    async def _handle_frame(self, data: Dict[str, Any]) -> None:
        frame = FrameDecoder.decode_jpeg_base64(data.get("jpeg_b64", ""))
        if frame is None:
            await self.status.send_info(self.ws, "Decode failed")
            return

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.pose.process(img_rgb)

        if not res.pose_landmarks:
            self.st.ready = False
            self.st.ready_streak = 0
            self.st.stand_streak = 0
            self.st.last_gate_debug = {}
            self.st.hist.clear()
            self.st.phase_feat_buffer.clear()
            self.st.prev_phase = ""
            self.st.pending = None
            self.st.last_sent_bottom_event_i = -10**9
            self.st.last_sent_stand_label = ""
            await self.status.send_status(self.ws, self.st, "waiting", {"ready_streak": 0, "needed_streak": self.ready_streak_n})
            self.frame_i += 1
            return

        lm = res.pose_landmarks.landmark
        ok_front, gate_dbg = self.gate.check(lm)
        self.st.last_gate_debug = gate_dbg

        if not ok_front:
            self.st.ready = False
            self.st.ready_streak = 0
            self.st.stand_streak = 0
            self.st.hist.clear()
            self.st.phase_feat_buffer.clear()
            self.st.prev_phase = ""
            self.st.pending = None
            self.st.last_sent_bottom_event_i = -10**9
            self.st.last_sent_stand_label = ""
            await self.status.send_status(
                self.ws,
                self.st,
                "waiting",
                {
                    "ready_streak": 0,
                    "needed_streak": self.ready_streak_n,
                    "gate": gate_dbg if self.debug else None,
                    "reason": "front_gate_not_ok",
                },
            )
            overlay = draw_overlay(
                frame_bgr=frame,
                res=res,
                state="waiting",
                phase="stand",
                knee_raw=None,
                knee_ema=None,
                gate_text=f"FRONT GATE: NO (vis_ok={gate_dbg.get('vis_ok')} gap_ok={gate_dbg.get('gap_ok')})",
                feat_dim=None,
            )
            if SAVE_VIDEO and ((not RECORD_ONLY_WHEN_READY) or self.st.ready):
                await self._start_recording_for_frame(overlay)
                if self.st.writer is not None:
                    tw, th = self.st.writer_size if self.st.writer_size else (overlay.shape[1], overlay.shape[0])
                    if (overlay.shape[1], overlay.shape[0]) != (tw, th):
                        overlay = cv2.resize(overlay, (tw, th))
                    self.st.writer.write(overlay)
                    self.st.saved_frames += 1
                    if self.st.saved_frames % PRINT_EVERY_SAVED_FRAMES == 0:
                        print(f"[RECORD] saved_frames={self.st.saved_frames} path={self.st.actual_video_path}")
            self.frame_i += 1
            return

        self.st.ready_streak += 1
        if (not self.st.ready) and (self.st.ready_streak >= self.ready_streak_n):
            self.st.ready = True
            self.st.hist.clear()
            self.st.phase_feat_buffer.clear()
            self.st.prev_phase = ""
            self.st.last_phase_bottom_i = -10**9
            self.st.pending = None
            self.st.last_sent_bottom_event_i = -10**9
            self.st.last_sent_stand_label = ""
            await self.status.send_info(self.ws, "Front view ready", {"session_id": self.st.session_id, "gate": gate_dbg if self.debug else None})
            await self.status.send_status(self.ws, self.st, "ready", {"ready_streak": self.st.ready_streak}, force=True)
        elif not self.st.ready:
            await self.status.send_status(
                self.ws,
                self.st,
                "warming_up",
                {
                    "ready_streak": self.st.ready_streak,
                    "needed_streak": self.ready_streak_n,
                    "gate": gate_dbg if self.debug else None,
                },
            )

        if not self.st.ready:
            overlay = draw_overlay(
                frame_bgr=frame,
                res=res,
                state="warming_up",
                phase="stand",
                knee_raw=None,
                knee_ema=None,
                gate_text=f"FRONT GATE: OK streak {self.st.ready_streak}/{self.ready_streak_n}",
                feat_dim=None,
            )
            if SAVE_VIDEO and ((not RECORD_ONLY_WHEN_READY) or self.st.ready):
                await self._start_recording_for_frame(overlay)
                if self.st.writer is not None:
                    tw, th = self.st.writer_size if self.st.writer_size else (overlay.shape[1], overlay.shape[0])
                    if (overlay.shape[1], overlay.shape[0]) != (tw, th):
                        overlay = cv2.resize(overlay, (tw, th))
                    self.st.writer.write(overlay)
                    self.st.saved_frames += 1
                    if self.st.saved_frames % PRINT_EVERY_SAVED_FRAMES == 0:
                        print(f"[RECORD] saved_frames={self.st.saved_frames} path={self.st.actual_video_path}")
            self.frame_i += 1
            return

        lhip = (lm[L_HIP].x, lm[L_HIP].y)
        rhip = (lm[R_HIP].x, lm[R_HIP].y)
        lknee = (lm[L_KNE].x, lm[L_KNE].y)
        rknee = (lm[R_KNE].x, lm[R_KNE].y)
        lank = (lm[L_ANK].x, lm[L_ANK].y)
        rank = (lm[R_ANK].x, lm[R_ANK].y)
        knee_l = angle_3pts(lhip, lknee, lank)
        knee_r = angle_3pts(rhip, rknee, rank)
        knee_raw = float((knee_l + knee_r) * 0.5)

        feat = self.feat.extract(lm)

        self.st.phase_feat_buffer.append(extract_phase_features(lm))
        if self.model_svc.phase_loaded and len(self.st.phase_feat_buffer) >= self.model_svc.phase_window:
            phase = self.model_svc.predict_phase(np.array(self.st.phase_feat_buffer))
        else:
            phase = "unknown"
        await self.status.send_phase(self.ws, self.st, phase)

        # Bottom = ช่วงเปลี่ยนผ่าน eccentric → concentric
        event_i = None
        if phase == "concentric" and self.st.prev_phase == "eccentric":
            if self.frame_i - self.st.last_phase_bottom_i >= MIN_GAP:
                event_i = self.frame_i
                self.st.last_phase_bottom_i = self.frame_i
        self.st.prev_phase = phase

        self.st.hist.append((self.frame_i, feat, frame.copy(), knee_raw, None))

        is_stand = (phase == "eccentric")
        if is_stand:
            self.st.stand_streak += 1
        else:
            self.st.stand_streak = 0

        pred_text = ""
        if self.st.last_pred_label:
            pred_text = (
                f"BottomPred: {self.st.last_pred_label} ({self.st.last_pred_conf:.3f})"
                if self.st.last_pred_conf is not None
                else f"BottomPred: {self.st.last_pred_label}"
            )
        stand_pred_text = ""
        if self.st.last_stand_pred_label:
            stand_pred_text = (
                f"StandPred(once): {self.st.last_stand_pred_label} ({self.st.last_stand_pred_conf:.3f})"
                if self.st.last_stand_pred_conf is not None
                else f"StandPred(once): {self.st.last_stand_pred_label}"
            )
        rep_text = f"Reps correct/incorrect/total: {self.st.good_reps}/{self.st.bad_reps}/{self.st.total_reps} (goal correct={GOAL_GOOD_REPS})"

        overlay = draw_overlay(
            frame_bgr=frame,
            res=res,
            state="ready",
            phase=phase,
            knee_raw=knee_raw,
            knee_ema=None,
            pred_text=pred_text if pred_text else None,
            extra_text=(f"EVENT bottom @ {event_i}" if event_i is not None else None),
            stand_pred_text=stand_pred_text if stand_pred_text else None,
            rep_text=rep_text,
            feat_dim=int(feat.shape[0]),
            gate_text=f"FRONT GATE: OK (sho_gap={gate_dbg.get('sho_gap')} hip_gap={gate_dbg.get('hip_gap')})",
        )

        if SAVE_VIDEO and ((not RECORD_ONLY_WHEN_READY) or self.st.ready):
            await self._start_recording_for_frame(overlay)
            if self.st.writer is not None:
                tw, th = self.st.writer_size if self.st.writer_size else (overlay.shape[1], overlay.shape[0])
                if (overlay.shape[1], overlay.shape[0]) != (tw, th):
                    overlay = cv2.resize(overlay, (tw, th))
                self.st.writer.write(overlay)
                self.st.saved_frames += 1
                if self.st.saved_frames % PRINT_EVERY_SAVED_FRAMES == 0:
                    print(f"[RECORD] saved_frames={self.st.saved_frames} path={self.st.actual_video_path}")

        if event_i is not None and self.model_svc.bottom_loaded and self.model_svc.bottom_T is not None:
            start = event_i - PRE_FRAMES
            end   = event_i + POST_FRAMES
            need = PRE_FRAMES + POST_FRAMES + 1
            win = [r for r in self.st.hist if start <= r[0] <= end]
            if len(win) < need:
                self.st.pending = {"event": event_i, "start": start, "end": end}
            else:
                self.st.pending = None
                await self.status.send_status(
                    self.ws,
                    self.st,
                    "predicting",
                    {
                        "mode": "bottom",
                        "phase": phase,
                        "event_i": int(event_i),
                        "window_frames": int(len(win)),
                        "T": int(self.model_svc.bottom_T),
                        "D": int(win[0][1].shape[0]),
                    },
                )
                X_win = np.stack([r[1] for r in win], axis=0).astype(np.float32)
                pred_label, conf, _ = self.model_svc.predict_bottom(X_win)
                self.st.last_pred_label = pred_label
                self.st.last_pred_conf = conf
                update_rep_counter(self.st, int(event_i), pred_label)
                payload = {
                    "type": "result",
                    "mode": "bottom",
                    "prediction": pred_label,
                    "confidence": round(conf, 3),
                    "session_id": self.st.session_id,
                    "event_i": int(event_i),
                    "window": {"pre": PRE_FRAMES, "post": POST_FRAMES},
                    "T": int(self.model_svc.bottom_T),
                    "feature_mode": FEATURE_MODE,
                    "reps": {
                        "total": int(self.st.total_reps),
                        "correct": int(self.st.good_reps),
                        "incorrect": int(self.st.bad_reps),
                        "goal_correct": int(GOAL_GOOD_REPS),
                        "good": int(self.st.good_reps),
                        "bad": int(self.st.bad_reps),
                        "goal_good": int(GOAL_GOOD_REPS),
                        "is_correct_rep": bool(pred_label != "knees_in"),
                        "is_good_rep": bool(pred_label != "knees_in"),
                    },
                }
                if self.debug:
                    print("[PRED-BOTTOM]", payload)
                if int(payload.get("event_i", -1)) != self.st.last_sent_bottom_event_i:
                    self.st.last_sent_bottom_event_i = int(payload.get("event_i", -1))
                    await self.ws.send_text(json.dumps(payload))

        if self.st.pending is not None and self.model_svc.bottom_loaded and self.model_svc.bottom_T is not None:
            start = self.st.pending["start"]
            end   = self.st.pending["end"]
            need = PRE_FRAMES + POST_FRAMES + 1
            win = [r for r in self.st.hist if start <= r[0] <= end]
            if len(win) >= need:
                event_i2 = int(self.st.pending["event"])
                self.st.pending = None
                await self.status.send_status(
                    self.ws,
                    self.st,
                    "predicting",
                    {
                        "mode": "bottom",
                        "phase": phase,
                        "event_i": int(event_i2),
                        "window_frames": int(len(win)),
                        "T": int(self.model_svc.bottom_T),
                        "D": int(win[0][1].shape[0]),
                    },
                )
                X_win = np.stack([r[1] for r in win], axis=0).astype(np.float32)
                pred_label, conf, _ = self.model_svc.predict_bottom(X_win)
                self.st.last_pred_label = pred_label
                self.st.last_pred_conf = conf
                update_rep_counter(self.st, int(event_i2), pred_label)
                payload = {
                    "type": "result",
                    "mode": "bottom",
                    "prediction": pred_label,
                    "confidence": round(conf, 3),
                    "session_id": self.st.session_id,
                    "event_i": int(event_i2),
                    "window": {"pre": PRE_FRAMES, "post": POST_FRAMES},
                    "T": int(self.model_svc.bottom_T),
                    "feature_mode": FEATURE_MODE,
                    "reps": {
                        "total": int(self.st.total_reps),
                        "correct": int(self.st.good_reps),
                        "incorrect": int(self.st.bad_reps),
                        "goal_correct": int(GOAL_GOOD_REPS),
                        "good": int(self.st.good_reps),
                        "bad": int(self.st.bad_reps),
                        "goal_good": int(GOAL_GOOD_REPS),
                        "is_correct_rep": bool(pred_label != "knees_in"),
                        "is_good_rep": bool(pred_label != "knees_in"),
                    },
                }
                if self.debug:
                    print("[PRED-BOTTOM]", payload)
                if int(payload.get("event_i", -1)) != self.st.last_sent_bottom_event_i:
                    self.st.last_sent_bottom_event_i = int(payload.get("event_i", -1))
                    await self.ws.send_text(json.dumps(payload))

        if (
            (not self.st.stand_ok)
            and (self.st.total_reps == 0)
            and self.model_svc.stand_loaded
            and self.model_svc.stand_T is not None
            and is_stand
            and (self.st.stand_streak >= STAND_MIN_STREAK)
            and (len(self.st.hist) >= STAND_WIN_FRAMES)
            and (self.frame_i - self.st.last_stand_pred_i >= STAND_PRED_COOLDOWN)
        ):
            recent = list(self.st.hist)[-STAND_WIN_FRAMES:]
            X_win = np.stack([r[1] for r in recent], axis=0).astype(np.float32)
            pred_label, conf, _ = self.model_svc.predict_stand(X_win)
            self.st.last_stand_pred_i = self.frame_i
            self.st.last_stand_pred_label = pred_label
            self.st.last_stand_pred_conf = conf
            is_ok = pred_label in STAND_OK_LABELS
            if is_ok:
                self.st.stand_ok = True
            payload = {
                "type": "result",
                "mode": "stand",
                "prediction": pred_label,
                "confidence": round(conf, 3),
                "session_id": self.st.session_id,
                "frame_i": int(self.frame_i),
                "T": int(self.model_svc.stand_T),
                "feature_mode": FEATURE_MODE,
                "stand_ok": bool(is_ok),
            }
            if self.debug:
                print("[PRED-STAND]", payload)
            await self.ws.send_text(json.dumps(payload))

        self.frame_i += 1


@app.websocket("/ws/video")
async def ws_video(websocket: WebSocket) -> None:
    session = SquatWebSocketSession(
        websocket=websocket,
        model_svc=model_service,
        gate=front_view_gate,
        feat=feature_extractor,
        status=status_sender,
        ready_streak_n=READY_STREAK_N,
        debug=DEBUG,
    )
    await session.run()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--serve", action="store_true")
    args = parser.parse_args()

    if not args.serve:
        args.serve = True

    if args.serve:
        import uvicorn
        uvicorn.run(
            "squat_streaming:app",
            host="0.0.0.0",
            port=5051,
            reload=False,
            log_level="info",
        )


if __name__ == "__main__":
    main()
