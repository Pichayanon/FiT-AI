from __future__ import annotations

if __package__ in {None, ""}:
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import os
import time
import warnings
from typing import Any, Dict

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
from shared.label_mapper import LabelMapper
from shared.side_gate import SideGate
from shared.status_sender import StatusSender
from plank.features import PlankFeatureExtractor
from plank.session import PlankWebSocketSession

PLANK_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(PLANK_DIR, "models", "plank_model.pkl")

LABELS: Dict[int, str] = {
    0: "correct",
    1: "hips_too_high",
    2: "hips_too_low",
}

WINDOW_FRAMES = 45
READY_STREAK_N = 3
VIS_TH = 0.80

DEBUG = False

MP_MIN_DET_CONF = 0.80
MP_MIN_TRACK_CONF = 0.80

STATUS_SEND_EVERY_N_FRAMES = 3

SIDE_MODE = "auto"

PHASE_NO_POSE = "NO_POSE"
PHASE_HAVE_POSE = "HAVE_POSE"
PHASE_BUFFERING = "BUFFERING"
PHASE_INFERENCING = "INFERENCING"

NO_POSE_ADJUST_SECONDS = 5.0

DARK_ADJUST_SECONDS = 3.0

DARK_BRIGHTNESS_TH = 55.0

PLANK_READY_MAX_BODY_AXIS_ANGLE_DEG = 35.0
PLANK_READY_MAX_TORSO_ANGLE_DEG = 45.0
PLANK_READY_MAX_LEG_ANGLE_DEG = 45.0

app = make_app("FiT-AI Plank Streaming Backend")

model_service = SklearnModelService(MODEL_PATH)
label_mapper = LabelMapper(LABELS)

pose_module = mp.solutions.pose
side_gate = SideGate(mp_pose=pose_module, side_mode=SIDE_MODE, vis_th=VIS_TH)
feature_extractor = PlankFeatureExtractor(mp_pose=pose_module)

status_sender = StatusSender(every_n_frames=STATUS_SEND_EVERY_N_FRAMES)


@app.get("/health")
def health() -> Dict[str, Any]:
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
        "plank_ready_max_body_axis_angle_deg": PLANK_READY_MAX_BODY_AXIS_ANGLE_DEG,
        "plank_ready_max_torso_angle_deg": PLANK_READY_MAX_TORSO_ANGLE_DEG,
        "plank_ready_max_leg_angle_deg": PLANK_READY_MAX_LEG_ANGLE_DEG,
    }


@app.websocket("/ws/video")
async def ws_video(websocket: WebSocket) -> None:
    session = PlankWebSocketSession(
        websocket=websocket,
        model_service=model_service,
        gate=side_gate,
        feature_extractor=feature_extractor,
        labels=label_mapper,
        status_sender=status_sender,
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
        plank_ready_max_body_axis_angle_deg=PLANK_READY_MAX_BODY_AXIS_ANGLE_DEG,
        plank_ready_max_torso_angle_deg=PLANK_READY_MAX_TORSO_ANGLE_DEG,
        plank_ready_max_leg_angle_deg=PLANK_READY_MAX_LEG_ANGLE_DEG,
        phase_no_pose=PHASE_NO_POSE,
        phase_have_pose=PHASE_HAVE_POSE,
        phase_buffering=PHASE_BUFFERING,
        phase_inferencing=PHASE_INFERENCING,
        status_send_every_n_frames=STATUS_SEND_EVERY_N_FRAMES,
    )
    await session.run()


if __name__ == "__main__":
    serve(app, port=5052)
