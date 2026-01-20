"""
squat_streaming.py (MINIMA-BASED, SAME AS TRAINING) + STAND PREDICTION (CHECK ONCE ONLY)
+ FEATURE SET A (same style as your enhanced training)
+ REP COUNTING (goal = 5 good reps)
+ FRONT-VIEW GATE (require both left+right landmarks visible before READY)

CHANGE (as you requested):
- Stand model is checked ONLY ONCE (before rep 1).
- If stand == OK, server will NEVER send any stand-related result again.
- After that, server sends ONLY bottom (mode="bottom") results (rep-based).

Rules:
- FRONT gate: must pass front-view landmarks visibility + left-right separation for READY_STREAK_N frames
- Bottom model: trigger at knee_ema local minima -> cut [event-pre, event+post]
- Stand model : CHECK ONCE ONLY before first rep (phase == stand for STAND_MIN_STREAK frames -> use recent window)
- Rep counting (SQUAT):
    - Count ONLY when bottom-event prediction happens (mode="bottom")
    - Good rep = prediction != "knees_in"
    - Bad rep  = prediction == "knees_in"
    - frontend decides when to finish

Feature modes (must match both training checkpoints):
- FEATURE_MODE = "RAW" -> 132 dims (x,y,z,vis)
- FEATURE_MODE = "A"   -> 111 dims (body-centric xyz + dist/ratio + angles)

IMPORTANT:
- stand_tcn.pt and squat_knees_in_tcn.pt MUST be trained with same feature dim as streaming uses.
"""

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
TCN_MODEL_PATH   = "models/squat_knees_in_tcn.pt"   # bottom-event model
STAND_MODEL_PATH = "models/feet_too_close_tcn.pt"   # standing model

# Feature mode (must match both training checkpoints)
FEATURE_MODE = "A"   # "A" (111 dims) or "RAW" (132 dims)

PRE_FRAMES  = 5
POST_FRAMES = 5

# Minima detector
EMA_ALPHA      = 0.30
MAX_BOTTOM_DEG = 140.0
MIN_GAP        = 18

# UX/phase
STAND_DEG = 165.0
BOTTOM_HOLD_SHOW = 8

# gate
READY_STREAK_N = 3

# FRONT-VIEW gate: require BOTH left+right landmarks visible
FRONT_VIS_TH = 0.80
# optional: ensure left-right separation (reject side view / too rotated)
FRONT_MIN_SHOULDER_X_GAP = 0.08   # normalized [0..1]
FRONT_MIN_HIP_X_GAP      = 0.06

# MediaPipe
MP_MIN_DET_CONF = 0.80
MP_MIN_TRACK_CONF = 0.80

# status throttle
STATUS_SEND_EVERY_N_FRAMES = 3
PHASE_SEND_EVERY_N_FRAMES  = 2

# record
SAVE_VIDEO = True
RECORD_DIR = "recordings_squat"
RECORD_FPS = 10.0
RECORD_ONLY_WHEN_READY = False
PRINT_EVERY_SAVED_FRAMES = 30
os.makedirs(RECORD_DIR, exist_ok=True)

# STAND prediction control
STAND_MIN_STREAK     = 6
STAND_PRED_COOLDOWN  = 12
STAND_WIN_FRAMES     = PRE_FRAMES + POST_FRAMES + 1

# Stand "OK" labels (must match your stand model label_map)
# If your stand model outputs "stand_ok" when good, keep this.
# If it outputs "dataset_correct", change to {"dataset_correct"}.
STAND_OK_LABELS = {"stand_ok"}

# REP goal (frontend can decide; server just sends counts)
GOAL_GOOD_REPS = 5

DEBUG = True


# -----------------------------
# FastAPI
# -----------------------------
app = FastAPI(
    title="FiT-AI Squat Streaming Backend (Front Gate + Minima + Bottom TCN + Stand TCN (Once) + FeatureA + RepCounting)"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)


# -----------------------------
# MediaPipe
# -----------------------------
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

# indices (MediaPipe Pose)
L_SHO, R_SHO = 11, 12
L_HIP, R_HIP = 23, 24
L_KNE, R_KNE = 25, 26
L_ANK, R_ANK = 27, 28

# FRONT landmarks: require BOTH sides
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


TCN_MODEL, TCN_T, INV_LABELS, TCN_IN_DIM = load_tcn(TCN_MODEL_PATH)
STAND_MODEL, STAND_T, STAND_INV_LABELS, STAND_IN_DIM = load_tcn(STAND_MODEL_PATH)


# -----------------------------
# Utils
# -----------------------------
def decode_jpeg_base64(jpeg_b64: str) -> Optional[np.ndarray]:
    """
    Decode base64 JPEG into BGR image.
    """
    try:
        raw = base64.b64decode(jpeg_b64)
        arr = np.frombuffer(raw, np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return None


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


def front_view_ok(lm, vis_th: float) -> Tuple[bool, Dict[str, Any]]:
    """
    FRONT view gate:
      - all FRONT_LM visibility >= vis_th
      - left-right separation (shoulders/hips not collapsed) to avoid side view
    returns (ok, debug)
    """
    vis_map: Dict[str, float] = {}
    ok_vis = True

    for idx in FRONT_LM:
        v = float(lm[idx].visibility)
        vis_map[FRONT_LM_LABELS.get(idx, str(idx))] = round(v, 3)
        if v < vis_th:
            ok_vis = False

    lsho_x = float(lm[mp_pose.PoseLandmark.LEFT_SHOULDER].x)
    rsho_x = float(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].x)
    lhip_x = float(lm[mp_pose.PoseLandmark.LEFT_HIP].x)
    rhip_x = float(lm[mp_pose.PoseLandmark.RIGHT_HIP].x)

    sho_gap = abs(lsho_x - rsho_x)
    hip_gap = abs(lhip_x - rhip_x)

    ok_gap = (sho_gap >= FRONT_MIN_SHOULDER_X_GAP) and (hip_gap >= FRONT_MIN_HIP_X_GAP)

    dbg = {
        "vis_ok": ok_vis,
        "gap_ok": ok_gap,
        "sho_gap": round(sho_gap, 3),
        "hip_gap": round(hip_gap, 3),
        "vis_th": float(vis_th),
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
    put(f"Knee raw: {knee_raw:.1f}" if knee_raw is not None else "Knee raw: NA")
    put(f"Knee ema: {knee_ema:.1f}" if knee_ema is not None else "Knee ema: NA")
    put(
        f"feat={FEATURE_MODE} dim={feat_dim} | front_vis>={FRONT_VIS_TH:.2f} "
        f"minima<= {MAX_BOTTOM_DEG:.0f} stand>= {STAND_DEG:.0f} pre/post={PRE_FRAMES}/{POST_FRAMES}",
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
# Minima detector
# -----------------------------
class KneeMinimaDetector:
    def __init__(self, ema_alpha=0.3, max_bottom_deg=140.0, min_gap=18):
        self.ema_alpha = float(ema_alpha)
        self.max_bottom_deg = float(max_bottom_deg)
        self.min_gap = int(min_gap)

        self.knee_ema = None
        self.k_hist = deque(maxlen=3)
        self.i_hist = deque(maxlen=3)

        self.last_event_i = -10**9
        self.last_event_countdown = 0

    def update(self, knee_deg: Optional[float], frame_i: int):
        if knee_deg is None:
            return None, self.knee_ema

        k_raw = float(knee_deg)
        if self.knee_ema is None:
            self.knee_ema = k_raw
        else:
            a = self.ema_alpha
            self.knee_ema = a * k_raw + (1.0 - a) * self.knee_ema

        self.k_hist.append(self.knee_ema)
        self.i_hist.append(frame_i)

        if len(self.k_hist) < 3:
            return None, self.knee_ema

        k0, k1, k2 = self.k_hist[0], self.k_hist[1], self.k_hist[2]
        i1 = self.i_hist[1]

        is_min = (k1 < k0) and (k1 < k2)
        if is_min and (k1 <= self.max_bottom_deg) and (i1 - self.last_event_i >= self.min_gap):
            self.last_event_i = i1
            self.last_event_countdown = BOTTOM_HOLD_SHOW
            return i1, self.knee_ema

        return None, self.knee_ema

    def phase_from_trend(self):
        if self.knee_ema is None:
            return "stand"
        if self.knee_ema >= STAND_DEG:
            return "stand"
        if self.last_event_countdown > 0:
            self.last_event_countdown -= 1
            return "bottom"
        if len(self.k_hist) >= 2:
            if self.k_hist[-1] < self.k_hist[-2] - 0.5:
                return "lowering"
            if self.k_hist[-1] > self.k_hist[-2] + 0.5:
                return "rising"
        return "hold"


# -----------------------------
# Stream State
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

    # pending capture after bottom event until post frames collected
    pending: Optional[Dict[str, Any]] = None


def update_rep_counter(st: StreamState, event_i: int, pred_label: str):
    """
    Count ONLY once per event.
    Good rep: pred_label != "knees_in"
    Bad rep : pred_label == "knees_in"
    """
    if event_i == st.last_counted_event_i:
        return
    st.last_counted_event_i = event_i

    st.total_reps += 1
    if pred_label == "knees_in":
        st.bad_reps += 1
    else:
        st.good_reps += 1


@app.get("/health")
def health():
    """
    Healthcheck endpoint.
    """
    return {
        "status": "ok",
        "feature_mode": FEATURE_MODE,
        "expected_dim": _expected_dim(),
        "bottom_loaded": TCN_MODEL is not None,
        "stand_loaded": STAND_MODEL is not None,
        "bottom_in_dim": TCN_IN_DIM,
        "stand_in_dim": STAND_IN_DIM,
        "stand_ok_labels": list(STAND_OK_LABELS),
        "front_vis_th": FRONT_VIS_TH,
        "front_min_sho_gap": FRONT_MIN_SHOULDER_X_GAP,
        "front_min_hip_gap": FRONT_MIN_HIP_X_GAP,
        "record_dir": os.path.abspath(RECORD_DIR),
        "timestamp": int(time.time()),
    }


@app.websocket("/ws/video")
async def ws_video(websocket: WebSocket):
    await websocket.accept()

    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=MP_MIN_DET_CONF,
        min_tracking_confidence=MP_MIN_TRACK_CONF,
    )

    st = StreamState()
    det = KneeMinimaDetector(ema_alpha=EMA_ALPHA, max_bottom_deg=MAX_BOTTOM_DEG, min_gap=MIN_GAP)
    frame_i = 0

    async def send_info(msg: str, extra: Optional[Dict[str, Any]] = None):
        payload = {"type": "info", "message": msg}
        if extra:
            payload.update(extra)
        await websocket.send_text(json.dumps(payload))

    async def send_status(state: str, extra: Optional[Dict[str, Any]] = None, force: bool = False):
        st.status_tick += 1
        if not force:
            if (st.status_tick % STATUS_SEND_EVERY_N_FRAMES != 0) and (state == st.last_status):
                return
        payload = {"type": "status", "state": state, "session_id": st.session_id}
        if extra:
            payload.update(extra)
        st.last_status = state
        await websocket.send_text(json.dumps(payload))

    async def send_phase(phase: str, knee_ema: Optional[float], force: bool = False):
        st.phase_tick += 1
        if not force and (st.phase_tick % PHASE_SEND_EVERY_N_FRAMES != 0) and (phase == st.last_phase):
            return
        payload = {
            "type": "phase",
            "phase": phase,
            "session_id": st.session_id,
            "knee_ema": round(float(knee_ema), 1) if knee_ema is not None else None,
        }
        st.last_phase = phase
        await websocket.send_text(json.dumps(payload))

    async def cleanup_recording():
        if st.writer is not None:
            try:
                st.writer.release()
                print(f"[RECORD] STOP path={st.actual_video_path} frames={st.saved_frames}")
            except Exception as e:
                print(f"[RECORD] release error: {e}")
        st.writer = None
        st.writer_size = None

    async def start_recording_for_frame(frame_bgr: np.ndarray):
        if not SAVE_VIDEO:
            return
        h, w = frame_bgr.shape[:2]
        if st.writer is None:
            writer, actual_path = create_video_writer(st.out_path_no_ext, w, h, RECORD_FPS)
            if writer is None:
                await send_info("Recording disabled: cannot create VideoWriter")
                return
            st.writer = writer
            st.writer_size = (w, h)
            st.actual_video_path = actual_path
            st.saved_frames = 0
            print(f"[RECORD] START path={actual_path} size={w}x{h}@{RECORD_FPS}")
            await send_info("Recording started", {"video_path": actual_path})

    # sanity check: model dims vs feature mode
    exp_dim = _expected_dim()
    if TCN_MODEL is not None and TCN_IN_DIM is not None and exp_dim != TCN_IN_DIM:
        await send_info(f"WARNING: Bottom model in_dim={TCN_IN_DIM} but FEATURE_MODE={FEATURE_MODE} gives dim={exp_dim}")
    if STAND_MODEL is not None and STAND_IN_DIM is not None and exp_dim != STAND_IN_DIM:
        await send_info(f"WARNING: Stand model in_dim={STAND_IN_DIM} but FEATURE_MODE={FEATURE_MODE} gives dim={exp_dim}")

    await send_info(
        "WebSocket connected",
        {
            "record_dir": os.path.abspath(RECORD_DIR),
            "feature_mode": FEATURE_MODE,
            "front_gate": {
                "vis_th": FRONT_VIS_TH,
                "min_sho_gap": FRONT_MIN_SHOULDER_X_GAP,
                "min_hip_gap": FRONT_MIN_HIP_X_GAP,
                "needed_streak": READY_STREAK_N,
            },
            "stand_once_only": True,
            "stand_ok_labels": list(STAND_OK_LABELS),
        },
    )

    try:
        while True:
            msg = await websocket.receive_text()
            try:
                data = json.loads(msg)
            except json.JSONDecodeError:
                await send_info("Invalid JSON")
                continue

            mtype = data.get("type")

            if mtype == "start":
                await cleanup_recording()
                st = StreamState(started=True)
                st.session_id = str(int(time.time() * 1000))
                st.out_path_no_ext = os.path.join(RECORD_DIR, f"session_{st.session_id}")
                frame_i = 0
                det = KneeMinimaDetector(ema_alpha=EMA_ALPHA, max_bottom_deg=MAX_BOTTOM_DEG, min_gap=MIN_GAP)

                # reset counters
                st.total_reps = 0
                st.good_reps = 0
                st.bad_reps = 0
                st.last_counted_event_i = -10**9
                st.last_sent_bottom_event_i = -10**9
                st.last_sent_stand_label = ""
                st.hist.clear()
                st.pending = None

                # reset stand-once flags
                st.stand_checked_once = False
                st.stand_ok = False
                st.stand_streak = 0
                st.last_stand_pred_i = -10**9
                st.last_stand_pred_label = ""
                st.last_stand_pred_conf = None

                print(f"[SESSION] START session_id={st.session_id}")
                await send_info("Start streaming", {"session_id": st.session_id})
                await send_status("waiting", {"reason": "session_started"}, force=True)
                continue

            if mtype == "stop":
                print(f"[SESSION] STOP session_id={st.session_id}")
                st.started = False
                await cleanup_recording()
                await send_info(
                    "Stop streaming",
                    {
                        "session_id": st.session_id,
                        "video_path": st.actual_video_path,
                        "saved_frames": st.saved_frames,
                        "reps": {
                            "total": int(st.total_reps),
                            "dataset_correct": int(st.good_reps),
                            "incorrect": int(st.bad_reps),
                            "goal_correct": int(GOAL_GOOD_REPS),
                        },
                        "stand_checked_once": bool(st.stand_checked_once),
                        "stand_ok": bool(st.stand_ok),
                    },
                )
                await send_status("waiting", {"reason": "session_stopped"}, force=True)
                continue

            if mtype != "frame" or not st.started:
                continue

            frame = decode_jpeg_base64(data.get("jpeg_b64", ""))
            if frame is None:
                await send_info("Decode failed")
                continue

            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(img_rgb)

            # -----------------------------
            # FRONT VIEW GATE
            # -----------------------------
            if not res.pose_landmarks:
                # lost pose -> reset
                st.ready = False
                st.ready_streak = 0
                st.stand_streak = 0
                st.last_gate_debug = {}

                # IMPORTANT: clear history/pending to avoid wrong event windows after reacquire
                st.hist.clear()
                st.pending = None

                # reset dedup (so next reacquire can send results again)
                st.last_sent_bottom_event_i = -10**9
                st.last_sent_stand_label = ""

                await send_status("waiting", {"ready_streak": 0, "needed_streak": READY_STREAK_N})
                continue

            lm = res.pose_landmarks.landmark
            ok_front, gate_dbg = front_view_ok(lm, FRONT_VIS_TH)
            st.last_gate_debug = gate_dbg

            if not ok_front:
                # not front-ready -> reset streak + buffers
                st.ready = False
                st.ready_streak = 0
                st.stand_streak = 0

                st.hist.clear()
                st.pending = None

                st.last_sent_bottom_event_i = -10**9
                st.last_sent_stand_label = ""

                await send_status(
                    "waiting",
                    {
                        "ready_streak": 0,
                        "needed_streak": READY_STREAK_N,
                        "gate": gate_dbg if DEBUG else None,
                        "reason": "front_gate_not_ok",
                    },
                )

                # still record overlay if you want, but no prediction / no minima
                overlay = draw_overlay(
                    frame_bgr=frame,
                    res=res,
                    state="waiting",
                    phase="stand",
                    knee_raw=None,
                    knee_ema=None,
                    gate_text=f"FRONT GATE: NO (vis_ok={gate_dbg.get('vis_ok')} gap_ok={gate_dbg.get('gap_ok')} "
                              f"sho_gap={gate_dbg.get('sho_gap')} hip_gap={gate_dbg.get('hip_gap')})",
                    feat_dim=None,
                )

                should_record = SAVE_VIDEO and ((not RECORD_ONLY_WHEN_READY) or st.ready)
                if should_record:
                    await start_recording_for_frame(overlay)
                    if st.writer is not None:
                        tw, th = st.writer_size if st.writer_size else (overlay.shape[1], overlay.shape[0])
                        if (overlay.shape[1], overlay.shape[0]) != (tw, th):
                            overlay = cv2.resize(overlay, (tw, th))
                        st.writer.write(overlay)
                        st.saved_frames += 1
                        if st.saved_frames % PRINT_EVERY_SAVED_FRAMES == 0:
                            print(f"[RECORD] saved_frames={st.saved_frames} path={st.actual_video_path}")

                frame_i += 1
                continue

            # front ok this frame
            st.ready_streak += 1
            if (not st.ready) and (st.ready_streak >= READY_STREAK_N):
                st.ready = True
                st.hist.clear()
                st.pending = None
                det = KneeMinimaDetector(ema_alpha=EMA_ALPHA, max_bottom_deg=MAX_BOTTOM_DEG, min_gap=MIN_GAP)

                # reset dedup on first READY
                st.last_sent_bottom_event_i = -10**9
                st.last_sent_stand_label = ""

                await send_info("Front view ready", {"session_id": st.session_id, "gate": gate_dbg if DEBUG else None})
                await send_status("ready", {"ready_streak": st.ready_streak}, force=True)
            elif not st.ready:
                await send_status(
                    "warming_up",
                    {
                        "ready_streak": st.ready_streak,
                        "needed_streak": READY_STREAK_N,
                        "gate": gate_dbg if DEBUG else None,
                    },
                )

            # If not READY yet, we still show overlay but do not run minima/predict
            if not st.ready:
                overlay = draw_overlay(
                    frame_bgr=frame,
                    res=res,
                    state="warming_up",
                    phase="stand",
                    knee_raw=None,
                    knee_ema=None,
                    gate_text=f"FRONT GATE: OK streak {st.ready_streak}/{READY_STREAK_N} "
                              f"(sho_gap={gate_dbg.get('sho_gap')} hip_gap={gate_dbg.get('hip_gap')})",
                    feat_dim=None,
                )
                should_record = SAVE_VIDEO and ((not RECORD_ONLY_WHEN_READY) or st.ready)
                if should_record:
                    await start_recording_for_frame(overlay)
                    if st.writer is not None:
                        tw, th = st.writer_size if st.writer_size else (overlay.shape[1], overlay.shape[0])
                        if (overlay.shape[1], overlay.shape[0]) != (tw, th):
                            overlay = cv2.resize(overlay, (tw, th))
                        st.writer.write(overlay)
                        st.saved_frames += 1
                        if st.saved_frames % PRINT_EVERY_SAVED_FRAMES == 0:
                            print(f"[RECORD] saved_frames={st.saved_frames} path={st.actual_video_path}")
                frame_i += 1
                continue

            # -----------------------------
            # READY: compute knee + feature + minima + predict
            # -----------------------------
            # knee angle uses 2D x,y points
            lhip = (lm[L_HIP].x, lm[L_HIP].y)
            rhip = (lm[R_HIP].x, lm[R_HIP].y)
            lknee = (lm[L_KNE].x, lm[L_KNE].y)
            rknee = (lm[R_KNE].x, lm[R_KNE].y)
            lank = (lm[L_ANK].x, lm[L_ANK].y)
            rank = (lm[R_ANK].x, lm[R_ANK].y)

            knee_l = angle_3pts(lhip, lknee, lank)
            knee_r = angle_3pts(rhip, rknee, rank)
            knee_raw = float((knee_l + knee_r) * 0.5)

            feat = extract_stream_feature(lm)  # (D,)

            # minima update + phase
            event_i, knee_ema = det.update(knee_raw, frame_i)
            phase = det.phase_from_trend()
            await send_phase(phase, knee_ema)

            # store history for window cutting
            st.hist.append((frame_i, feat, frame.copy(), knee_raw, knee_ema))

            # stand streak
            if phase == "stand":
                st.stand_streak += 1
            else:
                st.stand_streak = 0

            # overlay texts
            pred_text = ""
            if st.last_pred_label:
                pred_text = (
                    f"BottomPred: {st.last_pred_label} ({st.last_pred_conf:.3f})"
                    if st.last_pred_conf is not None
                    else f"BottomPred: {st.last_pred_label}"
                )

            stand_pred_text = ""
            if st.last_stand_pred_label:
                stand_pred_text = (
                    f"StandPred(once): {st.last_stand_pred_label} ({st.last_stand_pred_conf:.3f})"
                    if st.last_stand_pred_conf is not None
                    else f"StandPred(once): {st.last_stand_pred_label}"
                )

            rep_text = f"Reps dataset_correct/incorrect/total: {st.good_reps}/{st.bad_reps}/{st.total_reps} (goal dataset_correct={GOAL_GOOD_REPS})"

            overlay = draw_overlay(
                frame_bgr=frame,
                res=res,
                state="ready",
                phase=phase,
                knee_raw=knee_raw,
                knee_ema=knee_ema,
                pred_text=pred_text if pred_text else None,
                extra_text=(f"EVENT bottom @ {event_i}" if event_i is not None else None),
                stand_pred_text=stand_pred_text if stand_pred_text else None,
                rep_text=rep_text,
                feat_dim=int(feat.shape[0]),
                gate_text=f"FRONT GATE: OK (sho_gap={gate_dbg.get('sho_gap')} hip_gap={gate_dbg.get('hip_gap')})",
            )

            # record
            should_record = SAVE_VIDEO and ((not RECORD_ONLY_WHEN_READY) or st.ready)
            if should_record:
                await start_recording_for_frame(overlay)
                if st.writer is not None:
                    tw, th = st.writer_size if st.writer_size else (overlay.shape[1], overlay.shape[0])
                    if (overlay.shape[1], overlay.shape[0]) != (tw, th):
                        overlay = cv2.resize(overlay, (tw, th))
                    st.writer.write(overlay)
                    st.saved_frames += 1
                    if st.saved_frames % PRINT_EVERY_SAVED_FRAMES == 0:
                        print(f"[RECORD] saved_frames={st.saved_frames} path={st.actual_video_path}")

            # ---------------------------------
            # (A) Bottom event -> cut window -> bottom TCN
            # ---------------------------------
            if event_i is not None and TCN_MODEL is not None and TCN_T is not None:
                start = event_i - PRE_FRAMES
                end   = event_i + POST_FRAMES
                need = PRE_FRAMES + POST_FRAMES + 1

                win = [r for r in st.hist if start <= r[0] <= end]
                if len(win) < need:
                    st.pending = {"event": event_i, "start": start, "end": end}
                else:
                    st.pending = None

                    await send_status(
                        "predicting",
                        {
                            "mode": "bottom",
                            "phase": phase,
                            "event_i": int(event_i),
                            "window_frames": int(len(win)),
                            "T": int(TCN_T),
                            "D": int(win[0][1].shape[0]),
                        },
                    )

                    X_win = np.stack([r[1] for r in win], axis=0).astype(np.float32)  # (Tw,D)
                    pred_label, conf, _ = tcn_predict(TCN_MODEL, INV_LABELS, int(TCN_T), X_win)

                    st.last_pred_label = pred_label
                    st.last_pred_conf = conf

                    # rep counting (count ONLY when bottom prediction is produced)
                    update_rep_counter(st, int(event_i), pred_label)

                    payload = {
                        "type": "result",
                        "mode": "bottom",
                        "prediction": pred_label,
                        "confidence": round(conf, 3),
                        "session_id": st.session_id,
                        "event_i": int(event_i),
                        "window": {"pre": PRE_FRAMES, "post": POST_FRAMES},
                        "T": int(TCN_T),
                        "feature_mode": FEATURE_MODE,
                        "reps": {
                            "total": int(st.total_reps),
                            "dataset_correct": int(st.good_reps),
                            "incorrect": int(st.bad_reps),
                            "goal_correct": int(GOAL_GOOD_REPS),

                            # backward-compatible keys (optional)
                            "good": int(st.good_reps),
                            "bad": int(st.bad_reps),
                            "goal_good": int(GOAL_GOOD_REPS),

                            "is_correct_rep": bool(pred_label != "knees_in"),
                            "is_good_rep": bool(pred_label != "knees_in"),
                        },
                    }
                    if DEBUG:
                        print("[PRED-BOTTOM]", payload)

                    # de-dup: send bottom result only once per event
                    if int(payload.get("event_i", -1)) != st.last_sent_bottom_event_i:
                        st.last_sent_bottom_event_i = int(payload.get("event_i", -1))
                        await websocket.send_text(json.dumps(payload))
                    else:
                        if DEBUG:
                            print("[PRED-BOTTOM] (dedup skip)", payload)

            # pending bottom completion
            if st.pending is not None and TCN_MODEL is not None and TCN_T is not None:
                start = st.pending["start"]
                end   = st.pending["end"]
                need = PRE_FRAMES + POST_FRAMES + 1
                win = [r for r in st.hist if start <= r[0] <= end]
                if len(win) >= need:
                    event_i2 = int(st.pending["event"])
                    st.pending = None

                    await send_status(
                        "predicting",
                        {
                            "mode": "bottom",
                            "phase": phase,
                            "event_i": int(event_i2),
                            "window_frames": int(len(win)),
                            "T": int(TCN_T),
                            "D": int(win[0][1].shape[0]),
                        },
                    )

                    X_win = np.stack([r[1] for r in win], axis=0).astype(np.float32)
                    pred_label, conf, _ = tcn_predict(TCN_MODEL, INV_LABELS, int(TCN_T), X_win)

                    st.last_pred_label = pred_label
                    st.last_pred_conf = conf

                    # rep counting
                    update_rep_counter(st, int(event_i2), pred_label)

                    payload = {
                        "type": "result",
                        "mode": "bottom",
                        "prediction": pred_label,
                        "confidence": round(conf, 3),
                        "session_id": st.session_id,
                        "event_i": int(event_i2),
                        "window": {"pre": PRE_FRAMES, "post": POST_FRAMES},
                        "T": int(TCN_T),
                        "feature_mode": FEATURE_MODE,
                        "reps": {
                            "total": int(st.total_reps),
                            "dataset_correct": int(st.good_reps),
                            "incorrect": int(st.bad_reps),
                            "goal_correct": int(GOAL_GOOD_REPS),

                            # backward-compatible keys (optional)
                            "good": int(st.good_reps),
                            "bad": int(st.bad_reps),
                            "goal_good": int(GOAL_GOOD_REPS),

                            "is_correct_rep": bool(pred_label != "knees_in"),
                            "is_good_rep": bool(pred_label != "knees_in"),
                        },
                    }
                    if DEBUG:
                        print("[PRED-BOTTOM]", payload)

                    # de-dup: send bottom result only once per event
                    if int(payload.get("event_i", -1)) != st.last_sent_bottom_event_i:
                        st.last_sent_bottom_event_i = int(payload.get("event_i", -1))
                        await websocket.send_text(json.dumps(payload))
                    else:
                        if DEBUG:
                            print("[PRED-BOTTOM] (dedup skip)", payload)

            # ---------------------------------
            # (B) Stand phase -> stand TCN
            # Keep checking until stand_ok == True
            # Once stand_ok True -> stop sending stand forever
            # ---------------------------------
            if (
                    (not st.stand_ok)  # keep checking until OK
                    and (st.total_reps == 0)  # before first rep only
                    and (STAND_MODEL is not None)
                    and (STAND_T is not None)
                    and (phase == "stand")
                    and (st.stand_streak >= STAND_MIN_STREAK)
                    and (len(st.hist) >= STAND_WIN_FRAMES)
                    and (frame_i - st.last_stand_pred_i >= STAND_PRED_COOLDOWN)  # cooldown
            ):
                recent = list(st.hist)[-STAND_WIN_FRAMES:]
                X_win = np.stack([r[1] for r in recent], axis=0).astype(np.float32)  # (Tw,D)

                pred_label, conf, _ = tcn_predict(STAND_MODEL, STAND_INV_LABELS, int(STAND_T), X_win)

                st.last_stand_pred_i = frame_i
                st.last_stand_pred_label = pred_label
                st.last_stand_pred_conf = conf

                is_ok = (pred_label in STAND_OK_LABELS)

                # lock forever only when OK
                if is_ok:
                    st.stand_ok = True

                payload = {
                    "type": "result",
                    "mode": "stand",
                    "prediction": pred_label,
                    "confidence": round(conf, 3),
                    "session_id": st.session_id,
                    "frame_i": int(frame_i),
                    "T": int(STAND_T),
                    "feature_mode": FEATURE_MODE,
                    "stand_ok": bool(is_ok),
                }

                if DEBUG:
                    print("[PRED-STAND]", payload)

                # keep sending until OK, then will stop next time because st.stand_ok=True
                await websocket.send_text(json.dumps(payload))

            frame_i += 1

    except WebSocketDisconnect:
        print(f"[WS] disconnect session_id={st.session_id}")
        await cleanup_recording()
        return
    except Exception as e:
        print(f"[WS] error: {e}")
        await cleanup_recording()
        try:
            await send_info(f"Server error: {e}")
        except Exception:
            pass


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
