"""
wall_sit_streaming.py — Wall Sit posture analysis streaming backend.

WebSocket streaming server for real-time wall sit form assessment using
side-view pose detection and sklearn-based classification.

Phases sent to iOS:
    NO_POSE, HAVE_POSE, BUFFERING, INFERENCING

WS protocol (from iOS):
    {"type":"start"}
    {"type":"frame","jpeg_b64":"..."}
    {"type":"stop"}

Server -> iOS:
    {"type":"status","state":"NO_POSE|HAVE_POSE|BUFFERING|INFERENCING", ...}
    {"type":"result","prediction":"...", "confidence":..., ...}
    {"type":"info","message":"..."}
"""

from __future__ import annotations

import json
import os
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

warnings.filterwarnings(
    "ignore",
    message=r".*SymbolDatabase\.GetPrototype\(\) is deprecated.*",
    category=UserWarning,
    module=r"google\.protobuf\.symbol_database",
)

import mediapipe as mp
from fastapi import FastAPI, WebSocket

from shared.app_factory import make_app
from shared.server_utils import serve
from shared.sklearn_model_service import SklearnModelService
from shared.side_gate import SideGate
from shared.label_mapper import LabelMapper
from shared.status_sender import StatusSender
from wall_sit.session import (
    StreamState,
    WallSitFeatureExtractor,
    WallSitWebSocketSession,
)


# ---------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------

_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(_DIR, "models", "wall_sit_model.pkl")

LABELS: Dict[int, str] = {
    0: "correct",
    1: "feet_too_close",
    2: "feet_too_far",
    3: "back_of_wall",
    4: "not_deep_enough",
}

WINDOW_FRAMES = 15          # frames per inference window
READY_STREAK_N = 3          # consecutive side-view frames required
VIS_TH = 0.80               # side landmark visibility threshold

DEBUG = False

# MediaPipe confidence thresholds
MP_MIN_DET_CONF = 0.80
MP_MIN_TRACK_CONF = 0.80

# Status message throttle
STATUS_SEND_EVERY_N_FRAMES = 3

# Side selection: "auto" | "left" | "right"
SIDE_MODE = "auto"

# Status phase constants
PHASE_NO_POSE = "NO_POSE"
PHASE_HAVE_POSE = "HAVE_POSE"
PHASE_BUFFERING = "BUFFERING"
PHASE_INFERENCING = "INFERENCING"

# NO_POSE watchdog: alert after this many seconds of no pose
NO_POSE_ADJUST_SECONDS = 5.0

# DARK watchdog: alert after this many seconds of dark frames
DARK_ADJUST_SECONDS = 5.0
# Mean grayscale brightness threshold (0..255)
DARK_BRIGHTNESS_TH = 55.0

# Standing gate: avoid predicting while user stands upright
# knee_angle ~ 165-180 degrees = standing (not in wall-sit yet)
STAND_KNEE_ANGLE_DEG_TH = 165.0
STAND_STREAK_N = 3


# ---------------------------------------------------------------
# FastAPI Application
# ---------------------------------------------------------------

app = make_app("FiT-AI WallSit Streaming Backend")






# ---------------------------------------------------------------
# Shared service instances
# ---------------------------------------------------------------

model_service = SklearnModelService(MODEL_PATH)
label_mapper = LabelMapper(LABELS)

mp_pose = mp.solutions.pose
side_gate = SideGate(mp_pose=mp_pose, side_mode=SIDE_MODE, vis_th=VIS_TH)
feature_extractor = WallSitFeatureExtractor(mp_pose=mp_pose)

status_sender = StatusSender(every_n_frames=STATUS_SEND_EVERY_N_FRAMES)


# ---------------------------------------------------------------
# Routes
# ---------------------------------------------------------------

@app.get("/health")
def health() -> Dict[str, Any]:
    """Health check endpoint with server configuration details."""
    return {
        "status": "ok",
        "model_loaded": model_service.loaded,
        "timestamp": int(time.time()),
        "window_frames": WINDOW_FRAMES,
        "ready_streak_n": READY_STREAK_N,
        "vis_th": VIS_TH,
        "side_mode": SIDE_MODE,
        "mp_det_conf": MP_MIN_DET_CONF,
        "mp_track_conf": MP_MIN_TRACK_CONF,
        "cwd": os.path.abspath("."),
        "no_pose_adjust_seconds": NO_POSE_ADJUST_SECONDS,
        "dark_adjust_seconds": DARK_ADJUST_SECONDS,
        "dark_brightness_th": DARK_BRIGHTNESS_TH,
    }


@app.websocket("/ws/video")
async def ws_video(websocket: WebSocket) -> None:
    """WebSocket endpoint for wall sit streaming sessions."""
    session = WallSitWebSocketSession(
        websocket=websocket,
        model_svc=model_service,
        gate=side_gate,
        feat=feature_extractor,
        labels=label_mapper,
        status=status_sender,
        window_frames=WINDOW_FRAMES,
        ready_streak_n=READY_STREAK_N,
        debug=DEBUG,
        side_mode=SIDE_MODE,
        vis_th=VIS_TH,
        mp_min_det_conf=MP_MIN_DET_CONF,
        mp_min_track_conf=MP_MIN_TRACK_CONF,
        no_pose_adjust_seconds=NO_POSE_ADJUST_SECONDS,
        dark_adjust_seconds=DARK_ADJUST_SECONDS,
        dark_brightness_th=DARK_BRIGHTNESS_TH,
        stand_knee_angle_deg_th=STAND_KNEE_ANGLE_DEG_TH,
        stand_streak_n=STAND_STREAK_N,
        phase_no_pose=PHASE_NO_POSE,
        phase_have_pose=PHASE_HAVE_POSE,
        phase_buffering=PHASE_BUFFERING,
        phase_inferencing=PHASE_INFERENCING,
        status_send_every_n_frames=STATUS_SEND_EVERY_N_FRAMES,
    )
    await session.run()


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

if __name__ == "__main__":
    serve("wall_sit.wall_sit_streaming:app", port=5050)
