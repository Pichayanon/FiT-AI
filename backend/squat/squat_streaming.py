"""
squat_streaming.py — Squat form analysis streaming backend.

Real-time squat analysis with front-view gate, phase detection (eccentric/concentric),
bottom-event TCN classification, and standing posture TCN assessment.

Key features:
    - Front-view gate (FrontViewGateDynamic): validates body visibility and separation
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
import torch

from fastapi import FastAPI, WebSocket

from shared.app_factory import make_app
from shared.server_utils import serve
from shared.front_view_gate_dynamic import FrontViewGateDynamic
from shared.status_sender import StatusSender
from squat.features import (
    BOTTOM_FEATURE_DIM,
    STAND_FEATURE_DIM,
)
from squat.session import (
    SquatModelService,
    SquatWebSocketSession,
    StreamState,
    update_rep_counter,
)


# ---------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------

_DIR = os.path.dirname(__file__)
TCN_MODEL_PATH = os.path.join(_DIR, "models", "squat_bottom_tcn.pt")
STAND_MODEL_PATH = os.path.join(_DIR, "models", "squat_stand_tcn.pt")
PHASE_MODEL_PATH = os.path.join(_DIR, "models", "squat_phase_tcn.pt")

PRE_FRAMES = 5
POST_FRAMES = 5
MIN_GAP = 18  # min frames between bottom events

# Stand gate thresholds
STAND_KNEE_ANGLE_DEG_TH = 160.0
STAND_KNEE_DELTA_MAX_DEG = 5.0

READY_STREAK_N = 3

DEBUG = False

FRONT_VIS_TH = 0.6
FRONT_MIN_SHOULDER_X_GAP = 0.08
FRONT_MIN_HIP_X_GAP = 0.06

MP_MIN_DET_CONF = 0.80
MP_MIN_TRACK_CONF = 0.80

STATUS_SEND_EVERY_N_FRAMES = 3
PHASE_SEND_EVERY_N_FRAMES = 2


STAND_MIN_STREAK = 6
STAND_PRED_COOLDOWN = 12
STAND_WIN_FRAMES = PRE_FRAMES + POST_FRAMES + 1

STAND_OK_LABELS = {"good_stand"}
GOAL_GOOD_REPS = 5


PHASE_LABELS = {0: "eccentric", 1: "concentric"}

mp_pose = mp.solutions.pose

# ---------------------------------------------------------------
# FastAPI Application
# ---------------------------------------------------------------

app = make_app("FiT-AI Squat Streaming Backend")


# ---------------------------------------------------------------
# Shared service instances
# ---------------------------------------------------------------

model_service = SquatModelService(TCN_MODEL_PATH, STAND_MODEL_PATH, PHASE_MODEL_PATH)
front_view_gate = FrontViewGateDynamic(
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
        "timestamp": int(time.time()),
    }




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
        bottom_feature_dim=BOTTOM_FEATURE_DIM,
        stand_feature_dim=STAND_FEATURE_DIM,
        stand_ok_labels=STAND_OK_LABELS,
        pre_frames=PRE_FRAMES,
        post_frames=POST_FRAMES,
        min_gap=MIN_GAP,
        stand_knee_angle_deg_th=STAND_KNEE_ANGLE_DEG_TH,
        stand_knee_delta_max_deg=STAND_KNEE_DELTA_MAX_DEG,
        stand_min_streak=STAND_MIN_STREAK,
        stand_pred_cooldown=STAND_PRED_COOLDOWN,
        goal_good_reps=GOAL_GOOD_REPS,
        mp_min_det_conf=MP_MIN_DET_CONF,
        mp_min_track_conf=MP_MIN_TRACK_CONF,
    )
    await session.run()


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

if __name__ == "__main__":
    serve("squat.squat_streaming:app", port=5051)
