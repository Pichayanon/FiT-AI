"""
lunges_streaming.py — Lunge form analysis streaming backend.

Real-time lunge analysis with side-view visibility gate, phase detection
(eccentric/concentric), bottom-event TCN classification, and knee minima detection.

Key features:
    - Side-view visibility gate: validates body landmark visibility
    - Phase TCN: detects eccentric/concentric phases
    - Bottom TCN: classifies form at eccentric-to-concentric transition
    - Depth gate: ignores triggers when knee angle is too high

WS protocol (from iOS):
    {"type":"start"}
    {"type":"frame","jpeg_b64":"..."}
    {"type":"stop"}

Server -> iOS:
    {"type":"status","state":"STAND|NO_POSE|ADJUST", ...}
    {"type":"phase","phase":"eccentric|concentric|unknown", ...}
    {"type":"result","mode":"bottom","prediction":"...", ...}
    {"type":"info","message":"..."}
"""

from __future__ import annotations

import json
import os
import time
import traceback
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
import cv2
import mediapipe as mp
import torch
import torch.nn as nn

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from shared.frame_decoder import FrameDecoder
from shared.json_utils import parse_json
from shared.math_utils import dist as _dist, safe_norm as _safe_norm, angle_3pts as _angle_3pts
from shared.video_utils import resample_time


# ---------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------

MODEL_PATH = "lunges/models/lunges_bottom_tcn.pt"
PHASE_MODEL_PATH = "lunges/models/lunge_phase_tcn.pt"
FEATURE_DIM = 42

# MediaPipe Indices
L_EAR, R_EAR = 7, 8
L_SHO, R_SHO = 11, 12
L_HIP, R_HIP = 23, 24
L_KNE, R_KNE = 25, 26
L_ANK, R_ANK = 27, 28
KEY_JOINTS = [L_EAR, R_EAR, L_SHO, R_SHO, L_HIP, R_HIP, L_KNE, R_KNE, L_ANK, R_ANK]

PRE_FRAMES = 15
POST_FRAMES = 15
WIN_SIZE = PRE_FRAMES + POST_FRAMES + 1

# Min frames between bottom events (match extract_bottom_lunges.py default)
MIN_GAP = 18

# Max knee angle (avg) to consider as a potential bottom candidate
MAX_BOTTOM_DEG = 130.0

DEBUG = True
SAVE_RECORDING = True
RECORDING_DIR = "lunges/recordings"

# Send status periodically
STATUS_SEND_EVERY_N_FRAMES = 3

# Depth Gate Threshold (Degrees)
# If min_knee_angle > this, we consider it "Standing" or "Too High" -> Ignore triggers
GATE_KNEE_ANGLE = 135.0

# Visibility thresholds
VIS_LEGS = [23, 24, 25, 26, 27, 28]  # Hips, Knees, Ankles
VIS_SHOULDERS = [11, 12]
VIS_TH = 0.65


# ---------------------------------------------------------------
# FastAPI Application
# ---------------------------------------------------------------

app = FastAPI(title="FiT-AI Lunges Streaming Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------
# Lunge Feature Extraction (exercise-specific, matches training)
# ---------------------------------------------------------------

def _angle_2d(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """2D angle ABC in degrees."""
    ba = a - b
    bc = c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6
    cosang = float(np.clip(np.dot(ba, bc) / denom, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosang)))


def extract_lunge_features(kp: np.ndarray) -> np.ndarray:
    """Extract 42-dim features relevant to lunges.

    Input: (T, 33, 4) or (33, 4)
    Output: (T, 42) or (42,)
    """
    if kp.ndim == 2:
        kp = kp[np.newaxis, ...]
        squeeze = True
    else:
        squeeze = False

    T = kp.shape[0]
    xyz = kp[..., :3].astype(np.float32)
    out = np.zeros((T, FEATURE_DIM), dtype=np.float32)

    L_HEEL, R_HEEL = 29, 30
    L_FOOT, R_FOOT = 31, 32

    for t in range(T):
        p = xyz[t]

        # 1. Detect Facing & Normalize X
        l_dir = p[L_FOOT][0] - p[L_HEEL][0]
        r_dir = p[R_FOOT][0] - p[R_HEEL][0]
        avg_dir = l_dir + r_dir

        facing_right = avg_dir >= 0

        p_norm = p.copy()
        if not facing_right:
            p_norm[:, 0] = -p_norm[:, 0]

        # 2. Identify Front vs Back Leg
        l_ank_x = p_norm[L_ANK][0]
        r_ank_x = p_norm[R_ANK][0]
        is_l_front = l_ank_x > r_ank_x

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

        torso_len = _dist(mid_hip, mid_sho)
        scale = torso_len if torso_len > 1e-4 else 1.0

        # 3. Features Construction
        SORTED_JOINTS = [
            IDX_F_EAR, IDX_B_EAR,
            IDX_F_SHO, IDX_B_SHO,
            IDX_F_HIP, IDX_B_HIP,
            IDX_F_KNE, IDX_B_KNE,
            IDX_F_ANK, IDX_B_ANK,
        ]

        # [0-29] Body-centric XYZ (Front/Back sorted)
        for i, j_idx in enumerate(SORTED_JOINTS):
            normed = (p_norm[j_idx] - mid_hip) / scale
            out[t, i * 3:(i + 1) * 3] = normed

        # [30-33] Angles (F/B)
        out[t, 30] = _angle_3pts(f_hip, f_kne, f_ank) / 180.0  # Front Knee
        out[t, 31] = _angle_3pts(b_hip, b_kne, b_ank) / 180.0  # Back Knee
        out[t, 32] = _angle_3pts(f_sho, f_hip, f_kne) / 180.0  # Front Hip
        out[t, 33] = _angle_3pts(b_sho, b_hip, b_kne) / 180.0  # Back Hip

        # [34] Torso tilt (vertical alignment)
        spine_vec = mid_sho - mid_hip
        vertical = np.array([0.0, -1.0, 0.0], dtype=np.float32)
        denom = (_safe_norm(spine_vec) * _safe_norm(vertical)) + 1e-6
        cosang = float(np.clip(np.dot(spine_vec, vertical) / denom, -1.0, 1.0))
        out[t, 34] = float(np.degrees(np.arccos(cosang))) / 180.0

        # [35] Stride Length Ratio
        stride_dist = _dist(f_ank, b_ank)
        out[t, 35] = stride_dist / scale

        # [36-37] Knee Over Toe (Signed X difference)
        out[t, 36] = f_kne[0] - f_ank[0]  # Front
        out[t, 37] = b_kne[0] - b_ank[0]  # Back

        # [38-39] Knee Height (Depth)
        ground_y = max(f_ank[1], b_ank[1])
        out[t, 38] = ground_y - f_kne[1]  # Front knee height
        out[t, 39] = ground_y - b_kne[1]  # Back knee height

        # [40] Spine Angle (Ear-Sho-Hip)
        out[t, 40] = _angle_3pts(mid_ear, mid_sho, mid_hip) / 180.0

        # [41] Hip Drop
        out[t, 41] = (ground_y - mid_hip[1]) / scale

    if squeeze:
        return out[0]
    return out


def extract_phase_features_from_lm(
    lm: Any, prev_hip_y: Optional[float]
) -> Tuple[np.ndarray, float]:
    """Extract 9-dim features matching backend/lunges/extract_phase.py."""

    if isinstance(lm, np.ndarray):
        def get_pt(i: int) -> np.ndarray:
            return lm[i, :2]
    else:
        def get_pt(i: int) -> np.ndarray:
            return np.array([lm[i].x, lm[i].y], dtype=np.float32)

    pts: Dict[str, np.ndarray] = {}
    pts["l_shoulder"] = get_pt(11)
    pts["r_shoulder"] = get_pt(12)
    pts["l_hip"] = get_pt(23)
    pts["r_hip"] = get_pt(24)
    pts["l_knee"] = get_pt(25)
    pts["r_knee"] = get_pt(26)
    pts["l_ankle"] = get_pt(27)
    pts["r_ankle"] = get_pt(28)

    mid_hip = (pts["l_hip"] + pts["r_hip"]) * 0.5
    mid_sho = (pts["l_shoulder"] + pts["r_shoulder"]) * 0.5
    torso_len = float(abs(mid_sho[1] - mid_hip[1]) + 1e-6)

    def ny(p: np.ndarray) -> float:
        return float((p[1] - mid_hip[1]) / torso_len)

    l_knee_angle = _angle_2d(pts["l_hip"], pts["l_knee"], pts["l_ankle"])
    r_knee_angle = _angle_2d(pts["r_hip"], pts["r_knee"], pts["r_ankle"])

    # Detect front leg (side view) by ankle x
    if float(pts["l_ankle"][0]) > float(pts["r_ankle"][0]):
        front_knee_angle = l_knee_angle
        back_knee_angle = r_knee_angle
        front_knee = pts["l_knee"]
        front_ankle = pts["l_ankle"]
    else:
        front_knee_angle = r_knee_angle
        back_knee_angle = l_knee_angle
        front_knee = pts["r_knee"]
        front_ankle = pts["r_ankle"]

    step_length = float(abs(pts["l_ankle"][0] - pts["r_ankle"][0]) / torso_len)
    knee_forward = float((front_knee[0] - front_ankle[0]) / torso_len)

    hip_y = float(mid_hip[1])

    if prev_hip_y is None:
        hip_velocity = 0.0
    else:
        hip_velocity = float((hip_y - float(prev_hip_y)) / torso_len)

    torso_angle = _angle_2d(mid_sho, mid_hip, front_knee)
    shoulder_height = ny(mid_sho)
    knee_angle_diff = float(abs(l_knee_angle - r_knee_angle))

    feats = np.array([
        ny(mid_hip),        # hip_height_norm
        shoulder_height,
        float(front_knee_angle) / 180.0,
        float(back_knee_angle) / 180.0,
        knee_angle_diff / 180.0,
        step_length,
        knee_forward,
        float(torso_angle) / 180.0,
        hip_velocity,
    ], dtype=np.float32)
    return feats, hip_y


# ---------------------------------------------------------------
# Landmark Smoother
# ---------------------------------------------------------------

class LandmarkSmoother:
    """Exponential moving average smoother for MediaPipe landmarks."""

    def __init__(self, alpha: float = 0.6) -> None:
        self.alpha = alpha
        self.prev: Optional[np.ndarray] = None

    def update(self, curr: np.ndarray) -> np.ndarray:
        if self.prev is None:
            self.prev = curr
            return curr
        self.prev = self.alpha * curr + (1.0 - self.alpha) * self.prev
        return self.prev


# ---------------------------------------------------------------
# Knee Minima Detector
# ---------------------------------------------------------------

class KneeMinimaDetector:
    """Detect knee angle minima for bottom-event triggering (fallback)."""

    def __init__(
        self, ema_alpha: float = 0.3, max_bottom_deg: float = 140.0, min_gap: int = 20
    ) -> None:
        self.ema_alpha = ema_alpha
        self.max_bottom_deg = max_bottom_deg
        self.min_gap = min_gap
        self.knee_ema: Optional[float] = None
        self.last_event_frame = -9999
        self.k_hist: deque = deque(maxlen=3)
        self.f_hist: deque = deque(maxlen=3)

    def update(
        self, knee_deg: Optional[float], frame_idx: int
    ) -> Tuple[Optional[int], Optional[float]]:
        if knee_deg is None:
            return None, self.knee_ema

        if self.knee_ema is None:
            self.knee_ema = knee_deg
        else:
            self.knee_ema = self.ema_alpha * knee_deg + (1.0 - self.ema_alpha) * self.knee_ema

        self.k_hist.append(self.knee_ema)
        self.f_hist.append(frame_idx)

        if len(self.k_hist) < 3:
            return None, self.knee_ema

        k0, k1, k2 = self.k_hist[0], self.k_hist[1], self.k_hist[2]
        f1 = self.f_hist[1]

        is_min = (k1 < k0) and (k1 < k2)
        if is_min and (k1 <= self.max_bottom_deg) and (f1 - self.last_event_frame >= self.min_gap):
            self.last_event_frame = f1
            return f1, self.knee_ema

        return None, self.knee_ema


# ---------------------------------------------------------------
# Session State
# ---------------------------------------------------------------

@dataclass
class SessionState:
    """Session-level state for a lunge streaming WebSocket connection."""

    session_id: str
    total_reps: int = 0
    correct_reps: int = 0
    incorrect_reps: int = 0

    # History for TCN
    hist: Deque[Tuple[int, np.ndarray]] = field(
        default_factory=lambda: deque(maxlen=WIN_SIZE + 20)
    )

    # Rep Logic
    pending_event_i: int = -1
    last_triggered_event_i: int = -(10**9)

    # Phase buffer
    phase_feat_buffer: Deque[np.ndarray] = field(
        default_factory=lambda: deque(maxlen=120)
    )
    last_phase: str = "unknown"
    prev_hip_y: Optional[float] = None

    # Status de-dup
    last_status: str = ""


# ---------------------------------------------------------------
# Lunge Model Services
# ---------------------------------------------------------------

class LungeModelService:
    """TCN model for lunge bottom classification."""

    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.model = None
        self.label_map: Dict[str, int] = {}
        self.idx_to_label: Dict[int, str] = {}
        self.loaded = False
        self.T = 30

    def load(self) -> None:
        if not os.path.exists(MODEL_PATH):
            print(f"[WARN] Model not found: {MODEL_PATH}")
            return
        try:
            from shared.tcn_models import SimpleTCN

            ckpt = torch.load(MODEL_PATH, map_location=self.device)
            self.label_map = ckpt["label_map"]
            self.idx_to_label = {v: k for k, v in self.label_map.items()}
            self.T = ckpt.get("T", 30)
            in_dim = ckpt.get("in_dim", FEATURE_DIM)

            self.model = SimpleTCN(
                in_dim=in_dim, num_classes=len(self.label_map)
            ).to(self.device)
            self.model.load_state_dict(ckpt["model_state"])
            self.model.eval()
            self.loaded = True
            print(f"[LUNGE] Loaded TCN model from {MODEL_PATH}")
        except Exception as e:
            print(f"[ERR] Failed to load model: {e}")
            traceback.print_exc()

    def predict(self, x_win: np.ndarray) -> Tuple[str, float]:
        """Predict form from a feature window. Returns (label, confidence)."""
        if not self.loaded or self.model is None:
            return "...", 0.0

        x = resample_time(x_win.astype(np.float32), self.T)
        t_in = torch.from_numpy(x).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(t_in)
            probs = torch.softmax(logits, dim=1)
            conf, pred_idx = torch.max(probs, dim=1)

        lbl = self.idx_to_label.get(pred_idx.item(), "unknown")
        return lbl, float(conf.item())


class LungePhaseTCN(nn.Module):
    """Lunge-specific phase TCN architecture (matches train_lunge_phase.py).

    Uses an inner _Block class with a different forward pass than the
    shared PhaseTCN. Kept separate to guarantee checkpoint compatibility.
    """

    def __init__(self, in_dim: int = 9, num_classes: int = 2) -> None:
        super().__init__()

        class _Block(nn.Module):
            def __init__(self, in_ch: int, out_ch: int, k: int = 3, d: int = 1):
                super().__init__()
                pad = (k - 1) * d
                self.conv1 = nn.Conv1d(in_ch, out_ch, k, padding=pad, dilation=d)
                self.conv2 = nn.Conv1d(out_ch, out_ch, k, padding=pad, dilation=d)
                self.relu = nn.ReLU()
                self.down = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                y = self.relu(self.conv1(x))
                y = self.relu(self.conv2(y))
                if self.down:
                    x = self.down(x)
                return y[..., : x.size(-1)] + x

        self.tcn = nn.Sequential(
            _Block(in_dim, 64, d=1),
            _Block(64, 64, d=2),
            _Block(64, 64, d=4),
        )
        self.fc = nn.Conv1d(64, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)  # (B, W, F) -> (B, F, W)
        x = self.tcn(x)
        x = self.fc(x)
        return x.transpose(1, 2)  # (B, W, C)


class LungePhaseService:
    """Phase TCN service for lunge eccentric/concentric detection."""

    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.model: Optional[LungePhaseTCN] = None
        self.window: int = 30
        self.loaded: bool = False

    def load(self) -> None:
        if not os.path.exists(PHASE_MODEL_PATH):
            print(f"[WARN] Phase model not found: {PHASE_MODEL_PATH}")
            return
        try:
            ckpt = torch.load(PHASE_MODEL_PATH, map_location=self.device)
            in_dim = int(ckpt.get("in_dim", 9))
            num_classes = int(ckpt.get("num_classes", 2))
            self.window = int(ckpt.get("window", 30))
            self.model = LungePhaseTCN(
                in_dim=in_dim, num_classes=num_classes
            ).to(self.device)
            self.model.load_state_dict(ckpt["state_dict"])
            self.model.eval()
            self.loaded = True
            print(
                f"[LUNGE] Loaded phase model from {PHASE_MODEL_PATH} window={self.window}"
            )
        except Exception as e:
            print(f"[ERR] Failed to load phase model: {e}")
            traceback.print_exc()

    def predict_phase(self, feats: np.ndarray) -> str:
        if not self.loaded or self.model is None:
            return "unknown"
        if feats.shape[0] < self.window:
            return "unknown"

        x = feats[-self.window:].astype(np.float32)
        xt = torch.from_numpy(x).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(xt)
            pred = int(torch.argmax(logits[0, -1], dim=-1).item())
        return "eccentric" if pred == 0 else "concentric"


# ---------------------------------------------------------------
# Shared service instances
# ---------------------------------------------------------------

model_service = LungeModelService()
model_service.load()

phase_service = LungePhaseService()
phase_service.load()


# ---------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------

def _ts() -> str:
    return time.strftime("%H:%M:%S")


def _log(tag: str, frame_i: int, elapsed: float, msg: str) -> None:
    print(f"[{_ts()}] [{tag} f={frame_i} t={elapsed:.1f}s] {msg}")


def _log_summary(st: SessionState, elapsed: float) -> None:
    print("\n" + "=" * 50)
    print(f"[{_ts()}] SESSION SUMMARY  id={st.session_id}")
    print(f"  Duration     : {elapsed:.1f}s")
    print(f"  Total reps   : {st.total_reps}")
    print(f"  Correct      : {st.correct_reps}")
    print(f"  Incorrect    : {st.incorrect_reps}")
    pct = (st.correct_reps / max(1, st.total_reps)) * 100
    print(f"  Accuracy     : {pct:.1f}%")
    print("=" * 50 + "\n")


# ---------------------------------------------------------------
# Visibility Check
# ---------------------------------------------------------------

def check_full_body(lm_arr: np.ndarray) -> Tuple[bool, str]:
    """Check if legs and at least one shoulder are visible."""
    for i in VIS_LEGS:
        if lm_arr[i, 3] < VIS_TH:
            return False, "Legs/Feet not visible"
    s1 = lm_arr[11, 3] >= VIS_TH
    s2 = lm_arr[12, 3] >= VIS_TH
    if not s1 and not s2:
        return False, "Upper body not visible"
    return True, "OK"


# ---------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------

@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "bottom_loaded": model_service.loaded,
        "bottom_labels": model_service.label_map,
        "phase_loaded": phase_service.loaded,
        "phase_window": phase_service.window,
        "feature_dim": FEATURE_DIM,
        "win_size": WIN_SIZE,
        "min_gap": MIN_GAP,
    }


# ---------------------------------------------------------------
# WebSocket Handler (procedural — preserved from original)
# ---------------------------------------------------------------

@app.websocket("/ws/video")
async def ws_video(websocket: WebSocket) -> None:
    await websocket.accept()
    print(f"[{_ts()}] [WS] Client connected")

    session_t0 = time.time()

    def elapsed() -> float:
        return time.time() - session_t0

    st = SessionState(session_id=str(int(time.time())))
    detector = KneeMinimaDetector(max_bottom_deg=MAX_BOTTOM_DEG, min_gap=MIN_GAP)
    smoother = LandmarkSmoother(alpha=0.6)

    mp_pose = mp.solutions.pose
    mp_draw = mp.solutions.drawing_utils
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    frame_i = 0
    last_debug_frame = -999
    last_pred_text = ""

    # Recording
    video_writer: Optional[cv2.VideoWriter] = None
    rec_path = ""
    if SAVE_RECORDING:
        os.makedirs(RECORDING_DIR, exist_ok=True)
        rec_path = os.path.join(RECORDING_DIR, f"session_{st.session_id}.mp4")
        _log("REC", 0, 0.0, f"Recording will be saved to {rec_path}")

    try:
        while True:
            msg = await websocket.receive_text()

            data = parse_json(msg)
            if data is None:
                if "," in msg:
                    _, b64_data = msg.split(",", 1)
                else:
                    b64_data = msg
                img = FrameDecoder.decode_jpeg_base64(b64_data)
            else:
                mtype = str(data.get("type", "frame"))
                if mtype == "start":
                    session_t0 = time.time()
                    st = SessionState(session_id=str(int(time.time())))
                    smoother = LandmarkSmoother(alpha=0.6)
                    _log("CTRL", 0, 0.0, "Session started")
                    await websocket.send_text(json.dumps({
                        "type": "status",
                        "state": "STAND",
                        "message": "Session started — stand sideways",
                    }))
                    st.last_status = "STAND"
                    continue
                if mtype == "stop":
                    _log("CTRL", frame_i, elapsed(), "Session stopped by client")
                    await websocket.send_text(json.dumps({
                        "type": "status",
                        "state": "WAITING",
                        "message": "Session stopped",
                    }))
                    st.last_status = "WAITING"
                    break
                if mtype != "frame":
                    continue
                img = FrameDecoder.decode_jpeg_base64(str(data.get("jpeg_b64", "")))

            if img is None:
                await websocket.send_text(json.dumps({
                    "type": "info",
                    "message": "Decode failed",
                }))
                continue

            # --- Pose ---
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)

            # Init video writer on first frame
            if SAVE_RECORDING and video_writer is None:
                h, w = img.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_writer = cv2.VideoWriter(rec_path, fourcc, 15.0, (w, h))

            if not res.pose_landmarks:
                smoother.prev = None

                if st.last_status != "NO_POSE" or (frame_i % STATUS_SEND_EVERY_N_FRAMES == 0):
                    await websocket.send_text(json.dumps({
                        "type": "status",
                        "state": "NO_POSE",
                        "message": "No body detected",
                    }))
                if st.last_status != "NO_POSE":
                    _log("POSE", frame_i, elapsed(), "Lost pose → NO_POSE")
                st.last_status = "NO_POSE"

                if SAVE_RECORDING and video_writer is not None:
                    cv2.putText(
                        img, f"f={frame_i} NO_POSE", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
                    )
                    video_writer.write(img)
                frame_i += 1
                continue

            # --- Landmarks Processing & Smoothing ---
            lm_raw = res.pose_landmarks.landmark
            curr_arr = np.zeros((33, 4), dtype=np.float32)
            for i in range(33):
                curr_arr[i] = [lm_raw[i].x, lm_raw[i].y, lm_raw[i].z, lm_raw[i].visibility]

            lm_smooth = smoother.update(curr_arr)

            # Visibility check (using smoothed)
            is_visible, vis_msg = check_full_body(lm_smooth)

            # Draw pose on frame for recording
            if SAVE_RECORDING:
                color = (0, 255, 0) if is_visible else (0, 0, 255)
                mp_draw.draw_landmarks(
                    img, res.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_draw.DrawingSpec(
                        color=color, thickness=2, circle_radius=2
                    ),
                    connection_drawing_spec=mp_draw.DrawingSpec(
                        color=color, thickness=1
                    ),
                )

            if not is_visible:
                if st.last_status != "ADJUST" or (frame_i % STATUS_SEND_EVERY_N_FRAMES == 0):
                    await websocket.send_text(json.dumps({
                        "type": "status",
                        "state": "ADJUST",
                        "message": vis_msg,
                    }))
                    if st.last_status != "ADJUST":
                        _log("POSE", frame_i, elapsed(), f"Visibility fail: {vis_msg} → Resetting buffers")
                    st.last_status = "ADJUST"

                if SAVE_RECORDING and video_writer is not None:
                    cv2.putText(
                        img, f"VIS FAIL: {vis_msg}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
                    )
                    video_writer.write(img)

                # Reset state to prevent splicing bad data
                st.prev_hip_y = None
                st.phase_feat_buffer.clear()
                st.hist.clear()
                st.last_triggered_event_i = -1000
                st.pending_event_i = -1

                frame_i += 1
                continue

            # --- Visibility OK: send STAND status ---
            if st.last_status != "STAND":
                _log("POSE", frame_i, elapsed(), "Full body visible → STAND")
            if st.last_status != "STAND" or (frame_i % STATUS_SEND_EVERY_N_FRAMES == 0):
                await websocket.send_text(json.dumps({
                    "type": "status",
                    "state": "STAND",
                    "message": "Side View OK",
                }))
                st.last_status = "STAND"

            # --- Feature Extraction (using smoothed landmarks) ---
            feats = extract_lunge_features(lm_smooth[np.newaxis, ...])[0]
            st.hist.append((frame_i, feats))

            phase_feats, st.prev_hip_y = extract_phase_features_from_lm(
                lm_smooth, st.prev_hip_y
            )
            st.phase_feat_buffer.append(phase_feats)

            knee_avg = (feats[30] + feats[31]) * 0.5 * 180.0

            phase = phase_service.predict_phase(np.asarray(st.phase_feat_buffer))
            if phase != "unknown" and (frame_i % 2 == 0):
                await websocket.send_text(json.dumps({
                    "type": "phase",
                    "phase": phase,
                    "knee_avg": round(float(knee_avg), 1),
                    "session_id": st.session_id,
                }))

            # --- Debug log every 30 frames ---
            if DEBUG and (frame_i - last_debug_frame) >= 30:
                _log(
                    "DBG", frame_i, elapsed(),
                    f"phase={phase} knee={knee_avg:.1f}° "
                    f"buf={len(st.phase_feat_buffer)} hist={len(st.hist)} "
                    f"reps={st.total_reps}",
                )
                last_debug_frame = frame_i

            # --- Bottom trigger ---
            # PRIMARY: Phase Transition (eccentric -> concentric)
            event_frame: Optional[int] = None

            # Update detector for logging/debugging
            ef, _ = detector.update(knee_avg, frame_i)

            if phase != "unknown":
                if st.last_phase == "eccentric" and phase == "concentric":
                    # DEPTH GATE: Only trigger if we are somewhat deep
                    if knee_avg <= GATE_KNEE_ANGLE:
                        event_frame = frame_i
                        _log(
                            "PHASE", frame_i, elapsed(),
                            f"Transition eccentric→concentric | knee={knee_avg:.1f}° → BOTTOM EVENT",
                        )
                    else:
                        _log(
                            "GATE", frame_i, elapsed(),
                            f"Ignored trigger (knee={knee_avg:.1f}° > {GATE_KNEE_ANGLE}°)",
                        )

                st.last_phase = phase

            if event_frame is not None:
                if (
                    st.pending_event_i == -1
                    and (int(event_frame) - st.last_triggered_event_i) >= MIN_GAP
                ):
                    st.pending_event_i = int(event_frame)
                    _log(
                        "EVT", frame_i, elapsed(),
                        f"Pending bottom @ f={st.pending_event_i}, "
                        f"will predict after f={st.pending_event_i + POST_FRAMES}",
                    )

            if st.pending_event_i != -1:
                if frame_i >= (st.pending_event_i + POST_FRAMES):
                    start_f = st.pending_event_i - PRE_FRAMES
                    end_f = st.pending_event_i + POST_FRAMES

                    hist_map: Dict[int, np.ndarray] = {i: v for i, v in st.hist}
                    if all((f in hist_map) for f in range(start_f, end_f + 1)):
                        valid_win = [hist_map[f] for f in range(start_f, end_f + 1)]
                        X_win = np.stack(valid_win, axis=0)
                        pred_label, conf = model_service.predict(X_win)

                        is_correct = pred_label == "correct"
                        st.total_reps += 1
                        if is_correct:
                            st.correct_reps += 1
                        else:
                            st.incorrect_reps += 1

                        tag = "✓" if is_correct else "✗"
                        last_pred_text = f"{tag} {pred_label} ({conf:.2f})"
                        _log(
                            "PRED", frame_i, elapsed(),
                            f"{tag} {pred_label} ({conf:.2f}) | "
                            f"Reps: {st.correct_reps}/{st.incorrect_reps}/"
                            f"{st.total_reps} (good/bad/total)",
                        )

                        resp = {
                            "type": "result",
                            "mode": "bottom",
                            "prediction": pred_label,
                            "confidence": conf,
                            "reps": {
                                "total": st.total_reps,
                                "correct": st.correct_reps,
                                "incorrect": st.incorrect_reps,
                            },
                        }
                        await websocket.send_text(json.dumps(resp))
                    else:
                        _log(
                            "EVT", frame_i, elapsed(),
                            f"Window incomplete for pending @ f={st.pending_event_i}, skipping",
                        )

                    st.last_triggered_event_i = int(st.pending_event_i)
                    st.pending_event_i = -1

            # --- Write frame to recording with debug overlay ---
            if SAVE_RECORDING and video_writer is not None:
                y0 = 30
                lines = [
                    f"f={frame_i} t={elapsed():.1f}s",
                    f"phase={phase} knee={knee_avg:.1f}",
                    f"reps={st.correct_reps}/{st.incorrect_reps}/{st.total_reps}",
                ]
                if last_pred_text:
                    lines.append(f"PRED: {last_pred_text}")
                if st.pending_event_i != -1:
                    lines.append(f"PENDING @ f={st.pending_event_i}")
                for li, txt in enumerate(lines):
                    cv2.putText(
                        img, txt, (10, y0 + li * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2,
                    )
                video_writer.write(img)

            frame_i += 1

    except WebSocketDisconnect:
        _log("WS", frame_i, elapsed(), "Client disconnected")
    except Exception as e:
        _log("ERR", frame_i, elapsed(), f"{e}")
        traceback.print_exc()
    finally:
        if video_writer is not None:
            video_writer.release()
            _log("REC", frame_i, elapsed(), f"Recording saved: {rec_path}")
        _log_summary(st, elapsed())


if __name__ == "__main__":
    import uvicorn

    print(f"[{_ts()}] Starting Lunges backend on port 5053")
    print(f"  Bottom model : {MODEL_PATH} (loaded={model_service.loaded})")
    print(f"  Phase model  : {PHASE_MODEL_PATH} (loaded={phase_service.loaded})")
    print(f"  WIN_SIZE={WIN_SIZE} PRE={PRE_FRAMES} POST={POST_FRAMES} MIN_GAP={MIN_GAP}")
    uvicorn.run(app, host="0.0.0.0", port=5053)
