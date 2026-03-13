"""
lunges_streaming.py — Lunge form analysis streaming backend.

Real-time lunge analysis with side-view visibility gate, phase detection
(eccentric/concentric), bottom-event TCN classification, and depth gating.

Key features:
    - Side-view gate (SideViewGateDynamic): validates body landmark visibility
    - Phase TCN: detects eccentric/concentric phases
    - Bottom TCN: classifies form at eccentric-to-concentric transition
    - Depth gate: ignores triggers when knee angle is too high

WS protocol (from iOS):
    {"type":"start"}
    {"type":"frame","jpeg_b64":"..."}
    {"type":"stop"}

Server -> iOS:
    {"type":"status","state":"waiting|warming_up|ready|predicting", ...}
    {"type":"phase","phase":"eccentric|concentric|unknown", ...}
    {"type":"result","mode":"bottom","prediction":"...", ...}
    {"type":"info","message":"..."}
"""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import cv2
import mediapipe as mp
import torch
import torch.nn as nn

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from shared.frame_decoder import FrameDecoder
from shared.json_utils import parse_json
from shared.math_utils import dist, safe_norm, angle_3pts
from shared.side_view_gate_dynamic import SideViewGateDynamic
from shared.status_sender import StatusSender
from shared.tcn_service import load_tcn, tcn_predict
from shared.video_utils import resample_time


# ---------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------

BOTTOM_MODEL_PATH = "lunges/models/lunges_bottom_tcn.pt"
PHASE_MODEL_PATH = "lunges/models/lunge_phase_tcn.pt"

BOTTOM_FEATURE_DIM = 42

# MediaPipe Indices
L_EAR, R_EAR = 7, 8
L_SHO, R_SHO = 11, 12
L_HIP, R_HIP = 23, 24
L_KNE, R_KNE = 25, 26
L_ANK, R_ANK = 27, 28
KEY_JOINTS = [L_EAR, R_EAR, L_SHO, R_SHO, L_HIP, R_HIP, L_KNE, R_KNE, L_ANK, R_ANK]

PRE_FRAMES = 15
POST_FRAMES = 15
MIN_GAP = 18  # min frames between bottom events

# Depth Gate: knee angle threshold (degrees)
GATE_KNEE_ANGLE = 130.0

# Visibility
VIS_TH = 0.65

READY_STREAK_N = 3

MP_MIN_DET_CONF = 0.50
MP_MIN_TRACK_CONF = 0.50

STATUS_SEND_EVERY_N_FRAMES = 3
PHASE_SEND_EVERY_N_FRAMES = 2



GOAL_GOOD_REPS = 5

DEBUG = True

PHASE_LABELS = {0: "eccentric", 1: "concentric"}

mp_pose = mp.solutions.pose


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
# Lunge Feature Extractors (exercise-specific, match training scripts)
# ---------------------------------------------------------------



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
    out = np.zeros((T, BOTTOM_FEATURE_DIM), dtype=np.float32)

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

        torso_len = dist(mid_hip, mid_sho)
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
        out[t, 30] = angle_3pts(f_hip, f_kne, f_ank) / 180.0  # Front Knee
        out[t, 31] = angle_3pts(b_hip, b_kne, b_ank) / 180.0  # Back Knee
        out[t, 32] = angle_3pts(f_sho, f_hip, f_kne) / 180.0  # Front Hip
        out[t, 33] = angle_3pts(b_sho, b_hip, b_kne) / 180.0  # Back Hip

        # [34] Torso tilt (vertical alignment)
        spine_vec = mid_sho - mid_hip
        vertical = np.array([0.0, -1.0, 0.0], dtype=np.float32)
        denom = (safe_norm(spine_vec) * safe_norm(vertical)) + 1e-6
        cosang = float(np.clip(np.dot(spine_vec, vertical) / denom, -1.0, 1.0))
        out[t, 34] = float(np.degrees(np.arccos(cosang))) / 180.0

        # [35] Stride Length Ratio
        stride_dist = dist(f_ank, b_ank)
        out[t, 35] = stride_dist / scale

        # [36-37] Knee Over Toe (Signed X difference)
        out[t, 36] = f_kne[0] - f_ank[0]  # Front
        out[t, 37] = b_kne[0] - b_ank[0]  # Back

        # [38-39] Knee Height (Depth)
        ground_y = max(f_ank[1], b_ank[1])
        out[t, 38] = ground_y - f_kne[1]  # Front knee height
        out[t, 39] = ground_y - b_kne[1]  # Back knee height

        # [40] Spine Angle (Ear-Sho-Hip)
        out[t, 40] = angle_3pts(mid_ear, mid_sho, mid_hip) / 180.0

        # [41] Hip Drop
        out[t, 41] = (ground_y - mid_hip[1]) / scale

    if squeeze:
        return out[0]
    return out


def extract_phase_features_from_lm(
    lm: Any, prev_vals: Optional[Tuple[float, float, float]]
) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    """Extract 6-dim features matching lunges/extract_phase.py.

    Features: hip_h, shoulder_h, knee_h, hip_v, shoulder_v, knee_v
    (all height-normalized by torso length, velocities relative to prev frame)
    """

    if isinstance(lm, np.ndarray):
        def get_pt(i: int) -> np.ndarray:
            return lm[i, :2]
    else:
        def get_pt(i: int) -> np.ndarray:
            return np.array([lm[i].x, lm[i].y], dtype=np.float32)

    l_shoulder = get_pt(11)
    r_shoulder = get_pt(12)
    l_hip = get_pt(23)
    r_hip = get_pt(24)
    l_knee = get_pt(25)
    r_knee = get_pt(26)

    mid_hip = (l_hip + r_hip) * 0.5
    mid_shoulder = (l_shoulder + r_shoulder) * 0.5
    mid_knee = (l_knee + r_knee) * 0.5

    torso_len = float(abs(mid_shoulder[1] - mid_hip[1]) + 1e-6)

    def ny(p: np.ndarray) -> float:
        return float((p[1] - mid_hip[1]) / torso_len)

    hip_h = ny(mid_hip)
    shoulder_h = ny(mid_shoulder)
    knee_h = ny(mid_knee)

    if prev_vals is None:
        hip_v = 0.0
        shoulder_v = 0.0
        knee_v = 0.0
    else:
        hip_v = hip_h - prev_vals[0]
        shoulder_v = shoulder_h - prev_vals[1]
        knee_v = knee_h - prev_vals[2]

    feats = np.array([
        hip_h,
        shoulder_h,
        knee_h,
        hip_v,
        shoulder_v,
        knee_v,
    ], dtype=np.float32)
    return feats, (hip_h, shoulder_h, knee_h)


# ---------------------------------------------------------------
# Landmark Smoother
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


# ---------------------------------------------------------------
# LungePhaseTCN (kept separate for checkpoint compatibility)
# ---------------------------------------------------------------

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


# ---------------------------------------------------------------
# Lunge Model Service (holds bottom + phase models)
# ---------------------------------------------------------------

class LungeModelService:
    """Manages TCN models for lunge analysis."""

    def __init__(
        self, bottom_path: str, phase_path: Optional[str] = None
    ) -> None:
        # Load bottom model via shared service
        self.bottom_model, self.bottom_T, self.inv_labels_bottom, self.bottom_in_dim = load_tcn(bottom_path)

        # Load phase model (lunge-specific architecture)
        self.phase_model: Optional[LungePhaseTCN] = None
        self.phase_window: Optional[int] = None
        self.phase_in_dim: Optional[int] = None
        if phase_path and os.path.isfile(phase_path):
            self._load_phase(phase_path)

    def _load_phase(self, path: str) -> None:
        """Load phase TCN with lunge-specific architecture."""
        try:
            ckpt = torch.load(path, map_location="cpu")
            in_dim = int(ckpt.get("in_dim", 9))
            num_classes = int(ckpt.get("num_classes", 2))
            self.phase_window = int(ckpt.get("window", 30))
            self.phase_in_dim = in_dim
            self.phase_model = LungePhaseTCN(
                in_dim=in_dim, num_classes=num_classes
            )
            self.phase_model.load_state_dict(ckpt["state_dict"])
            self.phase_model.eval()
            print(
                f"[MODEL] Lunge phase TCN loaded: {path} "
                f"in_dim={in_dim} window={self.phase_window} num_classes={num_classes}"
            )
        except Exception as e:  # pylint: disable=broad-except
            print(f"[MODEL] Cannot load lunge phase TCN: {path} err={e}")

    @property
    def bottom_loaded(self) -> bool:
        return self.bottom_model is not None

    @property
    def phase_loaded(self) -> bool:
        return self.phase_model is not None and self.phase_window is not None

    def predict_bottom(self, x_win: np.ndarray) -> Tuple[str, float, np.ndarray]:
        """Run bottom TCN prediction. Returns (label, confidence, probs)."""
        if self.bottom_model is None or self.bottom_T is None:
            return "unknown", 0.0, np.array([])
        return tcn_predict(
            self.bottom_model, self.inv_labels_bottom, int(self.bottom_T),
            x_win.astype(np.float32),
        )

    def predict_phase(self, x_win: np.ndarray) -> str:
        """Run phase TCN prediction. Returns 'eccentric', 'concentric', or 'unknown'."""
        if self.phase_model is None or self.phase_window is None or x_win.shape[0] < self.phase_window:
            return "unknown"
        w = int(self.phase_window)
        x = x_win[-w:].astype(np.float32)
        xt = torch.from_numpy(x).unsqueeze(0)  # (1, W, 9)
        with torch.no_grad():
            logits = self.phase_model(xt)  # (1, W, 2)
            last_logits = logits[0, -1, :]
            pred_id = int(torch.argmax(last_logits).item())
        return PHASE_LABELS.get(pred_id, "unknown")



# ---------------------------------------------------------------
# Rep counter
# ---------------------------------------------------------------

def update_rep_counter(st: "StreamState", event_i: int, pred_label: str) -> None:
    """Count a rep once per event. Good rep: label starts with 'good' or == 'correct'."""
    if event_i == st.last_counted_event_i:
        return
    st.last_counted_event_i = event_i
    st.total_reps += 1
    if pred_label.startswith("good") or pred_label == "correct":
        st.good_reps += 1
    else:
        st.bad_reps += 1


# ---------------------------------------------------------------
# Stream State
# ---------------------------------------------------------------

@dataclass
class StreamState:
    """Session-level state for a lunge streaming WebSocket connection."""

    started: bool = False
    session_id: str = ""

    # Gate
    ready: bool = False
    ready_streak: int = 0
    last_gate_debug: Dict[str, Any] = field(default_factory=dict)

    # Status/phase throttles
    status_tick: int = 0
    phase_tick: int = 0
    last_status: str = ""
    last_phase: str = "unknown"


    # Rep counting
    total_reps: int = 0
    good_reps: int = 0
    bad_reps: int = 0
    last_counted_event_i: int = -10**9

    # De-dup sending
    last_sent_bottom_event_i: int = -10**9

    # History: (i, bottom_feat)
    hist: deque = field(default_factory=lambda: deque(maxlen=PRE_FRAMES + POST_FRAMES + 240))

    # Phase TCN buffer
    phase_feat_buffer: deque = field(default_factory=lambda: deque(maxlen=120))
    prev_phase: str = ""
    last_phase_bottom_i: int = -10**9
    prev_phase_vals: Optional[Tuple[float, float, float]] = None

    # Pending bottom event
    pending: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------
# Shared service instances
# ---------------------------------------------------------------

model_service = LungeModelService(BOTTOM_MODEL_PATH, PHASE_MODEL_PATH)
side_view_gate = SideViewGateDynamic(mp_pose, VIS_TH)
status_sender = StatusSender(STATUS_SEND_EVERY_N_FRAMES, PHASE_SEND_EVERY_N_FRAMES)


# ---------------------------------------------------------------
# Routes
# ---------------------------------------------------------------

@app.get("/health")
def health() -> Dict[str, Any]:
    """Health check endpoint with model and configuration details."""
    return {
        "status": "ok",
        "bottom_feature_dim": BOTTOM_FEATURE_DIM,
        "bottom_loaded": model_service.bottom_loaded,
        "phase_loaded": model_service.phase_loaded,
        "bottom_in_dim": model_service.bottom_in_dim,
        "vis_th": VIS_TH,
        "timestamp": int(time.time()),
    }


# ---------------------------------------------------------------
# Lunge WebSocket Session
# ---------------------------------------------------------------

class LungeWebSocketSession:
    """Handle a single lunge WebSocket streaming session."""

    def __init__(
        self,
        websocket: WebSocket,
        model_svc: LungeModelService,
        gate: SideViewGateDynamic,
        status: StatusSender,
        ready_streak_n: int,
        debug: bool,
    ) -> None:
        self.ws = websocket
        self.model_svc = model_svc
        self.gate = gate
        self.status = status
        self.ready_streak_n = ready_streak_n
        self.debug = debug

        self.st = StreamState()
        self.frame_i = 0
        self.smoother = LandmarkSmoother(alpha=0.6)

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=MP_MIN_DET_CONF,
            min_tracking_confidence=MP_MIN_TRACK_CONF,
        )

    # ---------------------------------------------------------------
    # Main loop
    # ---------------------------------------------------------------

    async def run(self) -> None:
        """Main receive loop."""
        await self.ws.accept()

        # Dimension mismatch warnings
        if self.model_svc.bottom_loaded and BOTTOM_FEATURE_DIM != self.model_svc.bottom_in_dim:
            await self.status.send_info(
                self.ws,
                f"WARNING: Bottom model in_dim={self.model_svc.bottom_in_dim} "
                f"but extractor gives dim={BOTTOM_FEATURE_DIM}",
            )

        await self.status.send_info(
            self.ws,
            "WebSocket connected",
            {
                "bottom_feature_dim": BOTTOM_FEATURE_DIM,
                "side_gate": {
                    "vis_th": VIS_TH,
                    "needed_streak": self.ready_streak_n,
                },
            },
        )

        try:
            while True:
                msg = await self.ws.receive_text()
                data = parse_json(msg)
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
            return
        except Exception as e:  # pylint: disable=broad-except
            print(f"[WS] error: {e}")
            print(traceback.format_exc())
            try:
                await self.status.send_info(self.ws, f"Server error: {e}")
            except Exception:  # pylint: disable=broad-except
                pass

    # ---------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------


    def _reset_buffers(self) -> None:
        """Reset all tracking buffers and streaks (used on gate failure or ready transition)."""
        self.st.ready = False
        self.st.ready_streak = 0
        self.st.last_gate_debug = {}
        self.st.hist.clear()
        self.st.phase_feat_buffer.clear()
        self.st.prev_phase = ""
        self.st.pending = None
        self.st.last_sent_bottom_event_i = -10**9
        self.st.prev_phase_vals = None
        self.smoother.reset()

    # ---------------------------------------------------------------
    # Start / Stop handlers
    # ---------------------------------------------------------------

    async def _handle_start(self) -> None:
        """Handle session start. Initialize all state."""
        self.st = StreamState(started=True)
        self.st.session_id = str(int(time.time() * 1000))
        self.frame_i = 0
        self.smoother = LandmarkSmoother(alpha=0.6)
        print(f"[SESSION] START session_id={self.st.session_id}")
        await self.status.send_info(self.ws, "Start streaming", {"session_id": self.st.session_id})
        await self.status.send_status(self.ws, self.st, "waiting", {"reason": "session_started"}, force=True)

    async def _handle_stop(self) -> None:
        """Handle session stop. Clean up and report summary."""
        print(f"[SESSION] STOP session_id={self.st.session_id}")
        self.st.started = False
        await self.status.send_info(
            self.ws,
            "Stop streaming",
            {
                "session_id": self.st.session_id,
                "reps": {
                    "total": int(self.st.total_reps),
                    "correct": int(self.st.good_reps),
                    "incorrect": int(self.st.bad_reps),
                    "goal_correct": int(GOAL_GOOD_REPS),
                },
            },
        )
        await self.status.send_status(self.ws, self.st, "waiting", {"reason": "session_stopped"}, force=True)

    # ---------------------------------------------------------------
    # Bottom prediction
    # ---------------------------------------------------------------

    async def _predict_and_send_bottom(
        self,
        event_i: int,
        phase: str,
    ) -> None:
        """Run bottom TCN prediction for a given event and send result.

        Extracts the feature window from history, runs prediction,
        updates rep counter, and sends the result payload.
        """
        start = event_i - PRE_FRAMES
        end = event_i + POST_FRAMES
        need = PRE_FRAMES + POST_FRAMES + 1
        win = [r for r in self.st.hist if start <= r[0] <= end]

        if len(win) < need:
            # Not enough frames yet — set as pending
            self.st.pending = {"event": event_i, "start": start, "end": end}
            return

        self.st.pending = None
        await self.status.send_status(
            self.ws, self.st, "predicting",
            {
                "mode": "bottom",
                "phase": phase,
                "event_i": int(event_i),
                "window_frames": int(len(win)),
                "T": int(self.model_svc.bottom_T),
                "D": BOTTOM_FEATURE_DIM,
            },
        )

        x_win = np.stack([r[1] for r in win], axis=0).astype(np.float32)
        pred_label, conf, _ = self.model_svc.predict_bottom(x_win)
        is_good = pred_label.startswith("good") or pred_label == "correct"
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
            "feature_dim": BOTTOM_FEATURE_DIM,
            "reps": {
                "total": int(self.st.total_reps),
                "correct": int(self.st.good_reps),
                "incorrect": int(self.st.bad_reps),
                "goal_correct": int(GOAL_GOOD_REPS),
                "good": int(self.st.good_reps),
                "bad": int(self.st.bad_reps),
                "goal_good": int(GOAL_GOOD_REPS),
                "is_correct_rep": bool(is_good),
                "is_good_rep": bool(is_good),
            },
        }
        if self.debug:
            print("[PRED-BOTTOM]", payload)
        if int(payload.get("event_i", -1)) != self.st.last_sent_bottom_event_i:
            self.st.last_sent_bottom_event_i = int(payload.get("event_i", -1))
            await self.ws.send_text(json.dumps(payload))



    # ---------------------------------------------------------------
    # Frame handler
    # ---------------------------------------------------------------

    async def _handle_frame(self, data: Dict[str, Any]) -> None:
        """Process a single video frame through the full lunge pipeline."""
        # Step 1: Decode frame
        frame = FrameDecoder.decode_jpeg_base64(data.get("jpeg_b64", ""))
        if frame is None:
            await self.status.send_info(self.ws, "Decode failed")
            return

        # Step 2: Run pose detection
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.pose.process(img_rgb)

        # Step 3: Handle no pose
        if not res.pose_landmarks:
            self._reset_buffers()
            self.smoother.reset()
            await self.status.send_status(
                self.ws, self.st, "waiting",
                {"ready_streak": 0, "needed_streak": self.ready_streak_n},
            )
            self.frame_i += 1
            return

        # Step 4: Side-view gate check
        lm = res.pose_landmarks.landmark

        # Build numpy array and apply smoothing
        lm_arr = np.zeros((33, 4), dtype=np.float32)
        for idx in range(33):
            lm_arr[idx] = [lm[idx].x, lm[idx].y, lm[idx].z, lm[idx].visibility]

        lm_arr = self.smoother.update(lm_arr)

        ok_side, gate_dbg = self.gate.check(lm_arr)
        self.st.last_gate_debug = gate_dbg

        if not ok_side:
            self._reset_buffers()
            await self.status.send_status(
                self.ws, self.st, "waiting",
                {
                    "ready_streak": 0,
                    "needed_streak": self.ready_streak_n,
                    "gate": gate_dbg if self.debug else None,
                    "reason": gate_dbg.get("reason", "side_gate_not_ok"),
                },
            )
            self.frame_i += 1
            return

        # Step 5: Track ready streak
        self.st.ready_streak += 1
        if (not self.st.ready) and (self.st.ready_streak >= self.ready_streak_n):
            self.st.ready = True
            self.st.hist.clear()
            self.st.phase_feat_buffer.clear()
            self.st.prev_phase = ""
            self.st.last_phase_bottom_i = -10**9
            self.st.pending = None
            self.st.prev_phase_vals = None
            self.st.last_sent_bottom_event_i = -10**9
            await self.status.send_info(
                self.ws, "Side View OK",
                {"session_id": self.st.session_id, "gate": gate_dbg if self.debug else None},
            )
            await self.status.send_status(
                self.ws, self.st, "ready",
                {"ready_streak": self.st.ready_streak}, force=True,
            )
        elif not self.st.ready:
            await self.status.send_status(
                self.ws, self.st, "warming_up",
                {
                    "ready_streak": self.st.ready_streak,
                    "needed_streak": self.ready_streak_n,
                    "gate": gate_dbg if self.debug else None,
                },
            )

        if not self.st.ready:
            self.frame_i += 1
            return

        # Step 6: Extract features
        feats = extract_lunge_features(lm_arr[np.newaxis, ...])[0]

        # Compute average knee angle from features (indices 30, 31 are knee angles / 180)
        knee_avg = float((feats[30] + feats[31]) * 0.5 * 180.0)

        # Step 7: Phase detection
        phase_feats, self.st.prev_phase_vals = extract_phase_features_from_lm(
            lm_arr, self.st.prev_phase_vals
        )
        self.st.phase_feat_buffer.append(phase_feats)
        if self.model_svc.phase_loaded and len(self.st.phase_feat_buffer) >= self.model_svc.phase_window:
            phase = self.model_svc.predict_phase(np.array(self.st.phase_feat_buffer))
        else:
            phase = "unknown"
        await self.status.send_phase(self.ws, self.st, phase)

        # Step 8: Detect bottom event (eccentric -> concentric transition + depth gate)
        event_i = None
        if phase == "concentric" and self.st.prev_phase == "eccentric":
            if self.frame_i - self.st.last_phase_bottom_i >= MIN_GAP:
                # Depth gate: only trigger if knee angle is low enough
                if knee_avg <= GATE_KNEE_ANGLE:
                    event_i = self.frame_i
                    self.st.last_phase_bottom_i = self.frame_i
                    if self.debug:
                        print(f"[PHASE] Transition eccentric→concentric | knee={knee_avg:.1f}° → BOTTOM EVENT")
                else:
                    if self.debug:
                        print(f"[GATE] Ignored trigger (knee={knee_avg:.1f}° > {GATE_KNEE_ANGLE}°)")
        self.st.prev_phase = phase

        # Add to history
        self.st.hist.append((self.frame_i, feats))



        # Step 9: Bottom prediction (immediate)
        if event_i is not None and self.model_svc.bottom_loaded and self.model_svc.bottom_T is not None:
            await self._predict_and_send_bottom(event_i, phase)

        # Step 10: Bottom prediction (pending — waiting for post-frames)
        if self.st.pending is not None and self.model_svc.bottom_loaded and self.model_svc.bottom_T is not None:
            pending_event = int(self.st.pending["event"])
            start = self.st.pending["start"]
            end = self.st.pending["end"]
            need = PRE_FRAMES + POST_FRAMES + 1
            win = [r for r in self.st.hist if start <= r[0] <= end]
            if len(win) >= need:
                self.st.pending = None
                await self._predict_and_send_bottom(pending_event, phase)

        self.frame_i += 1


# ---------------------------------------------------------------
# WebSocket route
# ---------------------------------------------------------------

@app.websocket("/ws/video")
async def ws_video(websocket: WebSocket) -> None:
    """WebSocket endpoint for lunge streaming sessions."""
    session = LungeWebSocketSession(
        websocket=websocket,
        model_svc=model_service,
        gate=side_view_gate,
        status=status_sender,
        ready_streak_n=READY_STREAK_N,
        debug=DEBUG,
    )
    await session.run()


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def main() -> None:
    """Run server via uvicorn."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--serve", action="store_true")
    args = parser.parse_args()

    if not args.serve:
        args.serve = True

    if args.serve:
        import uvicorn
        uvicorn.run(
            "lunges.lunges_streaming:app",
            host="0.0.0.0",
            port=5053,
            reload=False,
            log_level="info",
        )


if __name__ == "__main__":
    main()
