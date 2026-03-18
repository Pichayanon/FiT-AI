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

if __package__ in {None, ""}:
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import json
import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import cv2
import mediapipe as mp
from fastapi import WebSocket

from shared.app_factory import make_app
from shared.server_utils import serve
from shared.side_view_gate_dynamic import SideViewGateDynamic
from shared.status_sender import StatusSender
from lunges.features import BOTTOM_FEATURE_DIM
from lunges.session import (
    LungeModelService,
    LungeWebSocketSession,
)


# ---------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------

_DIR = os.path.dirname(__file__)
BOTTOM_MODEL_PATH = os.path.join(_DIR, "models", "lunges_bottom_tcn.pt")
PHASE_MODEL_PATH = os.path.join(_DIR, "models", "lunge_phase_tcn.pt")

# BOTTOM_FEATURE_DIM, joint indices imported from lunges.features

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

DEBUG = False

mp_pose = mp.solutions.pose

# ---------------------------------------------------------------
# FastAPI Application
# ---------------------------------------------------------------

app = make_app("FiT-AI Lunges Streaming Backend")


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
        bottom_feature_dim=BOTTOM_FEATURE_DIM,
        pre_frames=PRE_FRAMES,
        post_frames=POST_FRAMES,
        min_gap=MIN_GAP,
        gate_knee_angle=GATE_KNEE_ANGLE,
        goal_good_reps=GOAL_GOOD_REPS,
        mp_min_det_conf=MP_MIN_DET_CONF,
        mp_min_track_conf=MP_MIN_TRACK_CONF,
    )
    await session.run()


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

if __name__ == "__main__":
    serve("lunges.lunges_streaming:app", port=5053)
