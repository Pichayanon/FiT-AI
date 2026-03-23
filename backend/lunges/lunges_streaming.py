from __future__ import annotations

if __package__ in {None, ""}:
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import os
import time
from typing import Any, Dict

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

_DIR = os.path.dirname(__file__)
BOTTOM_MODEL_PATH = os.path.join(_DIR, "models", "lunges_bottom_tcn.pt")
PHASE_MODEL_PATH = os.path.join(_DIR, "models", "lunge_phase_tcn.pt")

PRE_FRAMES = 15
POST_FRAMES = 15
MIN_GAP = 18

GATE_KNEE_ANGLE = 130.0

VIS_TH = 0.65

READY_STREAK_N = 3

MP_MIN_DET_CONF = 0.50
MP_MIN_TRACK_CONF = 0.50

STATUS_SEND_EVERY_N_FRAMES = 3
PHASE_SEND_EVERY_N_FRAMES = 2

DARK_ADJUST_SECONDS = 3.0
DARK_BRIGHTNESS_TH = 55.0

GOAL_GOOD_REPS = 5

DEBUG = False

pose_module = mp.solutions.pose

app = make_app("FiT-AI Lunges Streaming Backend")

model_service = LungeModelService(BOTTOM_MODEL_PATH, PHASE_MODEL_PATH)
side_view_gate = SideViewGateDynamic(pose_module, VIS_TH)
status_sender = StatusSender(STATUS_SEND_EVERY_N_FRAMES, PHASE_SEND_EVERY_N_FRAMES)


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "bottom_feature_dim": BOTTOM_FEATURE_DIM,
        "bottom_loaded": model_service.bottom_loaded,
        "phase_loaded": model_service.phase_loaded,
        "bottom_in_dim": model_service.bottom_in_dim,
        "vis_th": VIS_TH,
        "dark_adjust_seconds": DARK_ADJUST_SECONDS,
        "dark_brightness_th": DARK_BRIGHTNESS_TH,
        "timestamp": int(time.time()),
    }


@app.websocket("/ws/video")
async def ws_video(websocket: WebSocket) -> None:
    session = LungeWebSocketSession(
        websocket=websocket,
        model_service=model_service,
        gate=side_view_gate,
        status_sender=status_sender,
        ready_streak_n=READY_STREAK_N,
        debug=DEBUG,
        bottom_feature_dim=BOTTOM_FEATURE_DIM,
        pre_frames=PRE_FRAMES,
        post_frames=POST_FRAMES,
        min_gap=MIN_GAP,
        gate_knee_angle=GATE_KNEE_ANGLE,
        dark_adjust_seconds=DARK_ADJUST_SECONDS,
        dark_brightness_th=DARK_BRIGHTNESS_TH,
        goal_good_reps=GOAL_GOOD_REPS,
        mp_min_det_conf=MP_MIN_DET_CONF,
        mp_min_track_conf=MP_MIN_TRACK_CONF,
    )
    await session.run()


if __name__ == "__main__":
    serve(app, port=5053)
