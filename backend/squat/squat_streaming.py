"""
squat_streaming.py — Squat form analysis streaming backend.

Real-time squat analysis with front-view gate, phase detection (eccentric/concentric),
bottom-event TCN classification, and standing posture TCN assessment.

Key features:
    - Front-view gate (FrontViewGate): validates body visibility and separation
    - Phase TCN: detects eccentric/concentric phases from train_squat_phase
    - Bottom TCN: classifies form at eccentric-to-concentric transition
    - Stand TCN: evaluates standing posture before first rep

WS protocol (from iOS):
    {"type":"start"}
    {"type":"frame","jpeg_b64":"..."}
    {"type":"stop"}

Server -> iOS:
    {"type":"status","state":"waiting|warming_up|ready|predicting", ...}
    {"type":"phase","phase":"eccentric|concentric|unknown", ...}
    {"type":"result","mode":"bottom|stand","prediction":"...", ...}
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

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from shared.frame_decoder import FrameDecoder
from shared.front_gate import FrontViewGate
from shared.json_utils import parse_json
from shared.math_utils import angle_3pts, safe_norm, dist, get_xyz
from shared.status_sender import StatusSender
from shared.tcn_models import PhaseTCN
from shared.tcn_service import load_tcn, load_phase_tcn, tcn_predict
from shared.video_utils import create_video_writer


# ---------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------

TCN_MODEL_PATH = "squat/models/squat_bottom_tcn.pt"
STAND_MODEL_PATH = "squat/models/squat_stand_tcn.pt"
PHASE_MODEL_PATH = "squat/models/squat_phase_tcn.pt"

STAND_FEATURE_DIM = 16   # focused stand features
BOTTOM_FEATURE_DIM = 41  # focused bottom features

KEY_JOINTS = [7, 8, 11, 12, 23, 24, 25, 26, 27, 28]  # L/R: ear, sho, hip, knee, ankle

PRE_FRAMES = 5
POST_FRAMES = 5
MIN_GAP = 18  # min frames between bottom events

# Stand gate thresholds
STAND_KNEE_ANGLE_DEG_TH = 155.0
STAND_KNEE_DELTA_MAX_DEG = 5.0

READY_STREAK_N = 3

FRONT_VIS_TH = 0.6
FRONT_MIN_SHOULDER_X_GAP = 0.08
FRONT_MIN_HIP_X_GAP = 0.06

MP_MIN_DET_CONF = 0.80
MP_MIN_TRACK_CONF = 0.80

STATUS_SEND_EVERY_N_FRAMES = 3
PHASE_SEND_EVERY_N_FRAMES = 2

SAVE_VIDEO = True
RECORD_DIR = "recordings_squat"
RECORD_FPS = 10.0
RECORD_ONLY_WHEN_READY = False
PRINT_EVERY_SAVED_FRAMES = 30
os.makedirs(RECORD_DIR, exist_ok=True)

STAND_MIN_STREAK = 6
STAND_PRED_COOLDOWN = 12
STAND_WIN_FRAMES = PRE_FRAMES + POST_FRAMES + 1

STAND_OK_LABELS = {"good_stand"}
GOAL_GOOD_REPS = 5

DEBUG = True

# MediaPipe landmark indices
L_SHO, R_SHO = 11, 12
L_HIP, R_HIP = 23, 24
L_KNE, R_KNE = 25, 26
L_ANK, R_ANK = 27, 28

PHASE_LABELS = {0: "eccentric", 1: "concentric"}

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles


# ---------------------------------------------------------------
# FastAPI Application
# ---------------------------------------------------------------

app = FastAPI(title="FiT-AI Squat Streaming Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------
# Squat Feature Extractors (exercise-specific, match training scripts)
# ---------------------------------------------------------------

def extract_phase_features(lm: list) -> np.ndarray:
    """Extract 10-dim features for phase TCN (matches extract_phase.py).

    Features: normalized Y positions of 8 joints + 2 knee angles.
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
        ny(pts["l_hip"]), ny(pts["r_hip"]),
        ny(pts["l_knee"]), ny(pts["r_knee"]),
        ny(pts["l_ankle"]), ny(pts["r_ankle"]),
        l_knee_angle / 180.0, r_knee_angle / 180.0,
    ], dtype=np.float32)


def extract_stand_features_from_lm(lm: list) -> np.ndarray:
    """Extract focused stand features (16 dims) — matches train_squat_stand.py.

    Dimensions:
        [0-3]   width ratios: ankle/hip, ankle/sho, knee/hip, knee/sho
        [4-9]   x positions (norm by hip width): L/R ankle, knee, hip
        [10-12] angles/180: knee_L, knee_R, torso_tilt
        [13]    feet distance (ankle_w / scale)
        [14]    shoulder distance (sho_w / scale)
        [15]    feet/shoulder ratio (ankle_w / sho_w)
    """
    xyz = get_xyz(lm)
    lhip, rhip = xyz[L_HIP], xyz[R_HIP]
    lsho, rsho = xyz[L_SHO], xyz[R_SHO]
    lkne, rkne = xyz[L_KNE], xyz[R_KNE]
    lank, rank = xyz[L_ANK], xyz[R_ANK]

    mid_hip = 0.5 * (lhip + rhip)
    hip_w = dist(lhip, rhip)
    sho_w = dist(lsho, rsho)
    ankle_w = dist(lank, rank)
    knee_w = dist(lkne, rkne)
    scale = hip_w if hip_w > 1e-4 else (sho_w if sho_w > 1e-4 else 1.0)

    out = np.zeros(16, dtype=np.float32)
    out[0] = ankle_w / (hip_w + 1e-6)
    out[1] = ankle_w / (sho_w + 1e-6)
    out[2] = knee_w / (hip_w + 1e-6)
    out[3] = knee_w / (sho_w + 1e-6)
    out[4] = (lank[0] - mid_hip[0]) / (scale + 1e-6)
    out[5] = (rank[0] - mid_hip[0]) / (scale + 1e-6)
    out[6] = (lkne[0] - mid_hip[0]) / (scale + 1e-6)
    out[7] = (rkne[0] - mid_hip[0]) / (scale + 1e-6)
    out[8] = (lhip[0] - mid_hip[0]) / (scale + 1e-6)
    out[9] = (rhip[0] - mid_hip[0]) / (scale + 1e-6)
    out[10] = angle_3pts(lhip, lkne, lank) / 180.0
    out[11] = angle_3pts(rhip, rkne, rank) / 180.0
    mid_sho = 0.5 * (lsho + rsho)
    v = (mid_sho - mid_hip).astype(np.float32)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    denom = (safe_norm(v) * safe_norm(up)) + 1e-6
    cosang = float(np.clip(np.dot(v, up) / denom, -1.0, 1.0))
    out[12] = float(np.degrees(np.arccos(cosang))) / 180.0
    out[13] = ankle_w / (scale + 1e-6)
    out[14] = sho_w / (scale + 1e-6)
    out[15] = ankle_w / (sho_w + 1e-6)

    return out


def extract_bottom_features_from_lm(lm: list) -> np.ndarray:
    """Extract focused bottom features (41 dims) — matches train_squat_bottom.py.

    Dimensions:
        [0-29]  10 key joints x 3 xyz (body-centric, hip-width normalized)
        [30-36] angles/180: knee_L, knee_R, hip_L, hip_R, torso_tilt, neck_tilt, spine
        [37-40] width ratios: knee/hip, knee/ankle, ankle/hip, sho/hip
    """
    xyz = get_xyz(lm)
    lear, rear = xyz[7], xyz[8]
    lhip, rhip = xyz[23], xyz[24]
    lsho, rsho = xyz[11], xyz[12]
    lkne, rkne = xyz[25], xyz[26]
    lank, rank = xyz[27], xyz[28]

    mid_hip = 0.5 * (lhip + rhip)
    hip_w = dist(lhip, rhip)
    sho_w = dist(lsho, rsho)
    scale = hip_w if hip_w > 1e-4 else (sho_w if sho_w > 1e-4 else 1.0)

    out = np.zeros(BOTTOM_FEATURE_DIM, dtype=np.float32)

    # Body-centric xyz for 10 key joints (30 dims)
    for i, j_idx in enumerate(KEY_JOINTS):
        normed = (xyz[j_idx] - mid_hip) / (scale + 1e-6)
        out[i * 3:(i + 1) * 3] = normed

    # Angles
    out[30] = angle_3pts(lhip, lkne, lank) / 180.0
    out[31] = angle_3pts(rhip, rkne, rank) / 180.0
    out[32] = angle_3pts(lsho, lhip, lkne) / 180.0
    out[33] = angle_3pts(rsho, rhip, rkne) / 180.0

    mid_sho = 0.5 * (lsho + rsho)
    mid_ear = 0.5 * (lear + rear)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    v = (mid_sho - mid_hip).astype(np.float32)
    denom = (safe_norm(v) * safe_norm(up)) + 1e-6
    cosang = float(np.clip(np.dot(v, up) / denom, -1.0, 1.0))
    out[34] = float(np.degrees(np.arccos(cosang))) / 180.0

    v_neck = (mid_ear - mid_sho).astype(np.float32)
    denom_neck = (safe_norm(v_neck) * safe_norm(up)) + 1e-6
    cosang_neck = float(np.clip(np.dot(v_neck, up) / denom_neck, -1.0, 1.0))
    out[35] = float(np.degrees(np.arccos(cosang_neck))) / 180.0

    out[36] = angle_3pts(mid_ear, mid_sho, mid_hip) / 180.0

    # Width ratios
    knee_w = dist(lkne, rkne)
    ankle_w = dist(lank, rank)
    out[37] = knee_w / (hip_w + 1e-6)
    out[38] = knee_w / (ankle_w + 1e-6)
    out[39] = ankle_w / (hip_w + 1e-6)
    out[40] = sho_w / (hip_w + 1e-6)

    return out


# ---------------------------------------------------------------
# Squat Model Service (holds bottom + stand + phase models)
# ---------------------------------------------------------------

class SquatModelService:
    """Manages multiple TCN models for squat analysis."""

    def __init__(
        self, bottom_path: str, stand_path: str, phase_path: Optional[str] = None
    ) -> None:
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

    def predict_bottom(self, x_win: np.ndarray) -> Tuple[str, float, np.ndarray]:
        """Run bottom TCN prediction. Returns (label, confidence, probs)."""
        if self.bottom_model is None or self.bottom_T is None:
            return "unknown", 0.0, np.array([])
        return tcn_predict(
            self.bottom_model, self.inv_labels_bottom, int(self.bottom_T),
            x_win.astype(np.float32),
        )

    def predict_stand(self, x_win: np.ndarray) -> Tuple[str, float, np.ndarray]:
        """Run stand TCN prediction. Returns (label, confidence, probs)."""
        if self.stand_model is None or self.stand_T is None:
            return "unknown", 0.0, np.array([])
        return tcn_predict(
            self.stand_model, self.inv_labels_stand, int(self.stand_T),
            x_win.astype(np.float32),
        )

    def predict_phase(self, x_win: np.ndarray) -> str:
        """Run phase TCN prediction. Returns 'eccentric', 'concentric', or 'unknown'."""
        if self.phase_model is None or self.phase_window is None or x_win.shape[0] < self.phase_window:
            return "unknown"
        w = int(self.phase_window)
        x = x_win[-w:].astype(np.float32)
        xt = torch.from_numpy(x).unsqueeze(0)  # (1, W, 10)
        with torch.no_grad():
            logits = self.phase_model(xt)  # (1, W, 2)
            last_logits = logits[0, -1, :]
            pred_id = int(torch.argmax(last_logits).item())
        return PHASE_LABELS.get(pred_id, "unknown")


# ---------------------------------------------------------------
# Overlay drawing (debug visualization)
# ---------------------------------------------------------------

def draw_overlay(
    frame_bgr: np.ndarray,
    res: Any,
    state: str,
    phase: str,
    knee_raw: Optional[float],
    knee_ema: Optional[float],
    pred_text: Optional[str] = None,
    extra_text: Optional[str] = None,
    stand_gate_text: Optional[str] = None,
    stand_pred_text: Optional[str] = None,
    rep_text: Optional[str] = None,
    feat_dim: Optional[int] = None,
    gate_text: Optional[str] = None,
) -> np.ndarray:
    """Draw pose landmarks and debug info onto a frame for recording."""
    out = frame_bgr.copy()
    if res.pose_landmarks:
        mp_drawing.draw_landmarks(
            out,
            res.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style(),
        )

    y = 26

    def put(t: str, s: float = 0.7) -> None:
        nonlocal y
        cv2.putText(out, t, (12, y), cv2.FONT_HERSHEY_SIMPLEX, s,
                     (255, 255, 255), 2, cv2.LINE_AA)
        y += 26

    put(f"State: {state}")
    if gate_text:
        put(gate_text, 0.60)
    put(f"Phase: {phase}")
    put(f"Knee: {knee_raw:.1f}" if knee_raw is not None else "Knee: NA")
    put(
        f"stand={STAND_FEATURE_DIM}d bottom={BOTTOM_FEATURE_DIM}d | "
        f"ecc->conc knee>={STAND_KNEE_ANGLE_DEG_TH:.0f} "
        f"d<={STAND_KNEE_DELTA_MAX_DEG:.1f}",
        0.55,
    )
    if extra_text:
        put(extra_text, 0.6)
    if stand_gate_text:
        put(stand_gate_text, 0.6)
    if rep_text:
        put(rep_text, 0.65)
    if pred_text:
        cv2.putText(out, pred_text, (12, y + 6), cv2.FONT_HERSHEY_SIMPLEX,
                     0.82, (255, 255, 255), 2, cv2.LINE_AA)
        y += 30
    if stand_pred_text:
        cv2.putText(out, stand_pred_text, (12, y + 6), cv2.FONT_HERSHEY_SIMPLEX,
                     0.72, (255, 255, 255), 2, cv2.LINE_AA)

    return out


# ---------------------------------------------------------------
# Rep counter
# ---------------------------------------------------------------

def update_rep_counter(st: "StreamState", event_i: int, pred_label: str) -> None:
    """Count a rep once per event. Good rep: label starts with 'good'."""
    if event_i == st.last_counted_event_i:
        return
    st.last_counted_event_i = event_i
    st.total_reps += 1
    if pred_label.startswith("good"):
        st.good_reps += 1
    else:
        st.bad_reps += 1


# ---------------------------------------------------------------
# Stream State
# ---------------------------------------------------------------

@dataclass
class StreamState:
    """Session-level state for a squat streaming WebSocket connection."""

    started: bool = False
    session_id: str = ""
    out_path_no_ext: str = ""

    # Gate
    ready: bool = False
    ready_streak: int = 0
    last_gate_debug: Dict[str, Any] = field(default_factory=dict)

    # Recording
    writer: Optional[cv2.VideoWriter] = None
    writer_size: Optional[Tuple[int, int]] = None
    actual_video_path: str = ""
    saved_frames: int = 0

    # Status/phase throttles
    status_tick: int = 0
    phase_tick: int = 0
    last_status: str = ""
    last_phase: str = "stand"

    # Last predictions for overlay
    last_pred_label: str = ""
    last_pred_conf: Optional[float] = None

    # Standing predict state
    stand_streak: int = 0
    prev_knee_raw: Optional[float] = None
    last_stand_pred_i: int = -10**9
    last_stand_pred_label: str = ""
    last_stand_pred_conf: Optional[float] = None
    stand_checked_once: bool = False
    stand_ok: bool = False

    # Rep counting
    total_reps: int = 0
    good_reps: int = 0
    bad_reps: int = 0
    last_counted_event_i: int = -10**9

    # De-dup sending
    last_sent_bottom_event_i: int = -10**9
    last_sent_stand_label: str = ""

    # History: (i, stand_feat, bottom_feat, frame_bgr, knee_raw, knee_ema)
    hist: deque = field(default_factory=lambda: deque(maxlen=PRE_FRAMES + POST_FRAMES + 240))

    # Phase TCN buffer
    phase_feat_buffer: deque = field(default_factory=lambda: deque(maxlen=35))
    prev_phase: str = ""
    last_phase_bottom_i: int = -10**9

    # Pending bottom event
    pending: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------
# Shared service instances
# ---------------------------------------------------------------

model_service = SquatModelService(TCN_MODEL_PATH, STAND_MODEL_PATH, PHASE_MODEL_PATH)
front_view_gate = FrontViewGate(
    mp_pose, FRONT_VIS_TH, FRONT_MIN_SHOULDER_X_GAP, FRONT_MIN_HIP_X_GAP
)
status_sender = StatusSender(STATUS_SEND_EVERY_N_FRAMES, PHASE_SEND_EVERY_N_FRAMES)


# ---------------------------------------------------------------
# Routes
# ---------------------------------------------------------------

@app.get("/health")
def health() -> Dict[str, Any]:
    """Health check endpoint with model and configuration details."""
    return {
        "status": "ok",
        "stand_feature_dim": STAND_FEATURE_DIM,
        "bottom_feature_dim": BOTTOM_FEATURE_DIM,
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


# ---------------------------------------------------------------
# Squat WebSocket Session
# ---------------------------------------------------------------

class SquatWebSocketSession:
    """Handle a single squat WebSocket streaming session."""

    def __init__(
        self,
        websocket: WebSocket,
        model_svc: SquatModelService,
        gate: FrontViewGate,
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
        if self.model_svc.stand_loaded and STAND_FEATURE_DIM != self.model_svc.stand_in_dim:
            await self.status.send_info(
                self.ws,
                f"WARNING: Stand model in_dim={self.model_svc.stand_in_dim} "
                f"but extractor gives dim={STAND_FEATURE_DIM}",
            )

        await self.status.send_info(
            self.ws,
            "WebSocket connected",
            {
                "record_dir": os.path.abspath(RECORD_DIR),
                "stand_feature_dim": STAND_FEATURE_DIM,
                "bottom_feature_dim": BOTTOM_FEATURE_DIM,
                "front_gate": {
                    "vis_th": FRONT_VIS_TH,
                    "min_sho_gap": FRONT_MIN_SHOULDER_X_GAP,
                    "min_hip_gap": FRONT_MIN_HIP_X_GAP,
                    "needed_streak": self.ready_streak_n,
                },
                "stand_once_only": False,
                "stand_ok_labels": list(STAND_OK_LABELS),
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
            await self._cleanup_recording()
            return
        except Exception as e:  # pylint: disable=broad-except
            print(f"[WS] error: {e}")
            print(traceback.format_exc())
            await self._cleanup_recording()
            try:
                await self.status.send_info(self.ws, f"Server error: {e}")
            except Exception:  # pylint: disable=broad-except
                pass

    # ---------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------


    async def _cleanup_recording(self) -> None:
        """Release video writer if active."""
        if self.st.writer is not None:
            try:
                self.st.writer.release()
                print(f"[RECORD] STOP path={self.st.actual_video_path} frames={self.st.saved_frames}")
            except Exception as e:  # pylint: disable=broad-except
                print(f"[RECORD] release error: {e}")
        self.st.writer = None
        self.st.writer_size = None

    async def _start_recording_for_frame(self, frame_bgr: np.ndarray) -> None:
        """Initialize video writer on first frame if recording is enabled."""
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

    def _write_frame_to_recording(self, overlay: np.ndarray) -> None:
        """Write a frame to the video recording, handling resize if needed."""
        if self.st.writer is None:
            return
        tw, th = self.st.writer_size if self.st.writer_size else (overlay.shape[1], overlay.shape[0])
        if (overlay.shape[1], overlay.shape[0]) != (tw, th):
            overlay = cv2.resize(overlay, (tw, th))
        self.st.writer.write(overlay)
        self.st.saved_frames += 1
        if self.st.saved_frames % PRINT_EVERY_SAVED_FRAMES == 0:
            print(f"[RECORD] saved_frames={self.st.saved_frames} path={self.st.actual_video_path}")

    def _reset_buffers(self) -> None:
        """Reset all tracking buffers and streaks (used on gate failure or ready transition)."""
        self.st.ready = False
        self.st.ready_streak = 0
        self.st.stand_streak = 0
        self.st.prev_knee_raw = None
        self.st.last_gate_debug = {}
        self.st.hist.clear()
        self.st.phase_feat_buffer.clear()
        self.st.prev_phase = ""
        self.st.pending = None
        self.st.last_sent_bottom_event_i = -10**9
        self.st.last_sent_stand_label = ""

    # ---------------------------------------------------------------
    # Start / Stop handlers
    # ---------------------------------------------------------------

    async def _handle_start(self) -> None:
        """Handle session start. Initialize all state."""
        await self._cleanup_recording()
        self.st = StreamState(started=True)
        self.st.session_id = str(int(time.time() * 1000))
        self.st.out_path_no_ext = os.path.join(RECORD_DIR, f"session_{self.st.session_id}")
        self.frame_i = 0
        print(f"[SESSION] START session_id={self.st.session_id}")
        await self.status.send_info(self.ws, "Start streaming", {"session_id": self.st.session_id})
        await self.status.send_status(self.ws, self.st, "waiting", {"reason": "session_started"}, force=True)

    async def _handle_stop(self) -> None:
        """Handle session stop. Clean up and report summary."""
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

    # ---------------------------------------------------------------
    # Bottom prediction (consolidated from duplicated code)
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

        x_win = np.stack([r[2] for r in win], axis=0).astype(np.float32)
        pred_label, conf, _ = self.model_svc.predict_bottom(x_win)
        self.st.last_pred_label = pred_label
        self.st.last_pred_conf = conf
        is_good = pred_label.startswith("good")
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
    # Recording helper
    # ---------------------------------------------------------------

    async def _record_overlay(self, overlay: np.ndarray) -> None:
        """Write overlay frame to recording if enabled."""
        if SAVE_VIDEO and ((not RECORD_ONLY_WHEN_READY) or self.st.ready):
            await self._start_recording_for_frame(overlay)
            self._write_frame_to_recording(overlay)

    # ---------------------------------------------------------------
    # Frame handler
    # ---------------------------------------------------------------

    async def _handle_frame(self, data: Dict[str, Any]) -> None:
        """Process a single video frame through the full squat pipeline."""
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
            await self.status.send_status(
                self.ws, self.st, "waiting",
                {"ready_streak": 0, "needed_streak": self.ready_streak_n},
            )
            self.frame_i += 1
            return

        # Step 4: Front-view gate check
        lm = res.pose_landmarks.landmark
        ok_front, gate_dbg = self.gate.check(lm)
        self.st.last_gate_debug = gate_dbg

        if not ok_front:
            self._reset_buffers()
            await self.status.send_status(
                self.ws, self.st, "waiting",
                {
                    "ready_streak": 0,
                    "needed_streak": self.ready_streak_n,
                    "gate": gate_dbg if self.debug else None,
                    "reason": "front_gate_not_ok",
                },
            )
            overlay = draw_overlay(
                frame_bgr=frame, res=res, state="waiting", phase="stand",
                knee_raw=None, knee_ema=None,
                gate_text=f"FRONT GATE: NO (vis_ok={gate_dbg.get('vis_ok')} gap_ok={gate_dbg.get('gap_ok')})",
            )
            await self._record_overlay(overlay)
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
            self.st.prev_knee_raw = None
            self.st.last_sent_bottom_event_i = -10**9
            self.st.last_sent_stand_label = ""
            await self.status.send_info(
                self.ws, "Front View OK",
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
            overlay = draw_overlay(
                frame_bgr=frame, res=res, state="warming_up", phase="stand",
                knee_raw=None, knee_ema=None,
                gate_text=f"FRONT GATE: OK streak {self.st.ready_streak}/{self.ready_streak_n}",
            )
            await self._record_overlay(overlay)
            self.frame_i += 1
            return

        # Step 6: Extract features and compute knee angle
        lhip = (lm[L_HIP].x, lm[L_HIP].y)
        rhip = (lm[R_HIP].x, lm[R_HIP].y)
        lknee = (lm[L_KNE].x, lm[L_KNE].y)
        rknee = (lm[R_KNE].x, lm[R_KNE].y)
        lank = (lm[L_ANK].x, lm[L_ANK].y)
        rank = (lm[R_ANK].x, lm[R_ANK].y)
        knee_l = angle_3pts(lhip, lknee, lank)
        knee_r = angle_3pts(rhip, rknee, rank)
        knee_raw = float((knee_l + knee_r) * 0.5)

        # Stand gate
        knee_delta = abs(knee_raw - self.st.prev_knee_raw) if self.st.prev_knee_raw is not None else 0.0
        is_stand = knee_raw >= STAND_KNEE_ANGLE_DEG_TH
        self.st.prev_knee_raw = knee_raw
        if is_stand:
            self.st.stand_streak += 1
        else:
            self.st.stand_streak = 0
        stand_gate_text = (
            f"STAND_GATE: {'YES' if is_stand else 'no'} "
            f"streak {self.st.stand_streak}/{STAND_MIN_STREAK} "
            f"knee={knee_raw:.1f} d={knee_delta:.1f}"
        )

        stand_feat = extract_stand_features_from_lm(lm)
        bottom_feat = extract_bottom_features_from_lm(lm)

        # Step 7: Phase detection
        self.st.phase_feat_buffer.append(extract_phase_features(lm))
        if self.model_svc.phase_loaded and len(self.st.phase_feat_buffer) >= self.model_svc.phase_window:
            phase = self.model_svc.predict_phase(np.array(self.st.phase_feat_buffer))
        else:
            phase = "unknown"
        await self.status.send_phase(self.ws, self.st, phase)

        # Step 8: Detect bottom event (eccentric -> concentric transition)
        event_i = None
        if phase == "concentric" and self.st.prev_phase == "eccentric":
            if self.frame_i - self.st.last_phase_bottom_i >= MIN_GAP:
                event_i = self.frame_i
                self.st.last_phase_bottom_i = self.frame_i
        self.st.prev_phase = phase

        # Add to history
        self.st.hist.append((self.frame_i, stand_feat, bottom_feat, frame.copy(), knee_raw, None))

        # Build overlay text
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
        rep_text = (
            f"Reps correct/incorrect/total: {self.st.good_reps}/{self.st.bad_reps}/"
            f"{self.st.total_reps} (goal correct={GOAL_GOOD_REPS})"
        )

        overlay = draw_overlay(
            frame_bgr=frame, res=res, state="ready", phase=phase,
            knee_raw=knee_raw, knee_ema=None,
            pred_text=pred_text if pred_text else None,
            extra_text=(f"EVENT bottom @ {event_i}" if event_i is not None else None),
            stand_gate_text=stand_gate_text,
            stand_pred_text=stand_pred_text if stand_pred_text else None,
            rep_text=rep_text,
            feat_dim=BOTTOM_FEATURE_DIM,
            gate_text=f"FRONT GATE: OK (sho_gap={gate_dbg.get('sho_gap')} hip_gap={gate_dbg.get('hip_gap')})",
        )
        await self._record_overlay(overlay)

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

        # Step 11: Standing posture prediction (before first rep)
        if (
            (self.st.total_reps == 0)
            and self.model_svc.stand_loaded
            and self.model_svc.stand_T is not None
            and is_stand
            and (self.st.stand_streak >= STAND_MIN_STREAK)
            and (len(self.st.hist) >= STAND_WIN_FRAMES)
            and (self.frame_i - self.st.last_stand_pred_i >= STAND_PRED_COOLDOWN)
        ):
            recent = list(self.st.hist)[-STAND_WIN_FRAMES:]
            x_win = np.stack([r[1] for r in recent], axis=0).astype(np.float32)
            pred_label, conf, _ = self.model_svc.predict_stand(x_win)
            self.st.last_stand_pred_i = self.frame_i
            self.st.last_stand_pred_label = pred_label
            self.st.last_stand_pred_conf = conf
            is_ok = pred_label in STAND_OK_LABELS
            self.st.stand_ok = bool(is_ok)

            payload = {
                "type": "result",
                "mode": "stand",
                "prediction": pred_label,
                "confidence": round(conf, 3),
                "session_id": self.st.session_id,
                "frame_i": int(self.frame_i),
                "T": int(self.model_svc.stand_T),
                "feature_dim": STAND_FEATURE_DIM,
                "stand_ok": bool(is_ok),
            }
            if self.debug:
                print("[PRED-STAND]", payload)
            await self.ws.send_text(json.dumps(payload))

        self.frame_i += 1


# ---------------------------------------------------------------
# WebSocket route
# ---------------------------------------------------------------

@app.websocket("/ws/video")
async def ws_video(websocket: WebSocket) -> None:
    """WebSocket endpoint for squat streaming sessions."""
    session = SquatWebSocketSession(
        websocket=websocket,
        model_svc=model_service,
        gate=front_view_gate,
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
            "squat_streaming:app",
            host="0.0.0.0",
            port=5051,
            reload=False,
            log_level="info",
        )


if __name__ == "__main__":
    main()
