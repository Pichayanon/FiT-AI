"""
main.py - Unified FiT-AI streaming backend.

All exercise services on a single port with path-based routing.

Routes:
    /squat/ws/video      - Squat WebSocket streaming
    /plank/ws/video      - Plank WebSocket streaming
    /lunges/ws/video     - Lunges WebSocket streaming
    /wall_sit/ws/video   - Wall Sit WebSocket streaming
    /health              - Global health check
    /{exercise}/health   - Per-exercise health check
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=r".*InconsistentVersionWarning.*")

import time
from typing import Any, Dict

from fastapi import WebSocket

from shared.app_factory import make_app

# --- Squat imports ---
from squat.squat_streaming import (
    model_service as squat_model_svc,
    front_view_gate as squat_gate,
    status_sender as squat_status,
    READY_STREAK_N as SQUAT_READY_STREAK_N,
    DEBUG as SQUAT_DEBUG,
    PRE_FRAMES as SQUAT_PRE_FRAMES,
    POST_FRAMES as SQUAT_POST_FRAMES,
    MIN_GAP as SQUAT_MIN_GAP,
    STAND_KNEE_ANGLE_DEG_TH as SQUAT_STAND_KNEE_ANGLE_DEG_TH,
    STAND_KNEE_DELTA_MAX_DEG as SQUAT_STAND_KNEE_DELTA_MAX_DEG,
    STAND_MIN_STREAK as SQUAT_STAND_MIN_STREAK,
    STAND_PRED_COOLDOWN as SQUAT_STAND_PRED_COOLDOWN,
    GOAL_GOOD_REPS as SQUAT_GOAL_GOOD_REPS,
    MP_MIN_DET_CONF as SQUAT_MP_MIN_DET_CONF,
    MP_MIN_TRACK_CONF as SQUAT_MP_MIN_TRACK_CONF,
    STAND_OK_LABELS as SQUAT_STAND_OK_LABELS,
    PHASE_DECISION_MODE as SQUAT_PHASE_DECISION_MODE,
    DARK_ADJUST_SECONDS as SQUAT_DARK_ADJUST_SECONDS,
    DARK_BRIGHTNESS_TH as SQUAT_DARK_BRIGHTNESS_TH,
)
from squat.features import BOTTOM_FEATURE_DIM as SQUAT_BOTTOM_FEATURE_DIM, STAND_FEATURE_DIM as SQUAT_STAND_FEATURE_DIM
from squat.session import SquatWebSocketSession

# --- Plank imports ---
from plank.plank_streaming import (
    model_service as plank_model_svc,
    side_gate as plank_gate,
    feature_extractor as plank_feature_extractor,
    label_mapper as plank_labels,
    status_sender as plank_status,
    WINDOW_FRAMES as PLANK_WINDOW_FRAMES,
    READY_STREAK_N as PLANK_READY_STREAK_N,
    DEBUG as PLANK_DEBUG,
    SIDE_MODE as PLANK_SIDE_MODE,
    VIS_TH as PLANK_VIS_TH,
    MP_MIN_DET_CONF as PLANK_MP_MIN_DET_CONF,
    MP_MIN_TRACK_CONF as PLANK_MP_MIN_TRACK_CONF,
    NO_POSE_ADJUST_SECONDS as PLANK_NO_POSE_ADJUST_SECONDS,
    DARK_ADJUST_SECONDS as PLANK_DARK_ADJUST_SECONDS,
    DARK_BRIGHTNESS_TH as PLANK_DARK_BRIGHTNESS_TH,
    PHASE_NO_POSE as PLANK_PHASE_NO_POSE,
    PHASE_HAVE_POSE as PLANK_PHASE_HAVE_POSE,
    PHASE_BUFFERING as PLANK_PHASE_BUFFERING,
    PHASE_INFERENCING as PLANK_PHASE_INFERENCING,
    STATUS_SEND_EVERY_N_FRAMES as PLANK_STATUS_SEND_EVERY_N_FRAMES,
    PLANK_READY_MAX_BODY_AXIS_ANGLE_DEG,
    PLANK_READY_MAX_TORSO_ANGLE_DEG,
    PLANK_READY_MAX_LEG_ANGLE_DEG,
)
from plank.session import PlankWebSocketSession

# --- Lunges imports ---
from lunges.lunges_streaming import (
    model_service as lunges_model_svc,
    side_view_gate as lunges_gate,
    status_sender as lunges_status,
    READY_STREAK_N as LUNGES_READY_STREAK_N,
    DEBUG as LUNGES_DEBUG,
    PRE_FRAMES as LUNGES_PRE_FRAMES,
    POST_FRAMES as LUNGES_POST_FRAMES,
    MIN_GAP as LUNGES_MIN_GAP,
    GATE_KNEE_ANGLE as LUNGES_GATE_KNEE_ANGLE,
    GOAL_GOOD_REPS as LUNGES_GOAL_GOOD_REPS,
    MP_MIN_DET_CONF as LUNGES_MP_MIN_DET_CONF,
    MP_MIN_TRACK_CONF as LUNGES_MP_MIN_TRACK_CONF,
    DARK_ADJUST_SECONDS as LUNGES_DARK_ADJUST_SECONDS,
    DARK_BRIGHTNESS_TH as LUNGES_DARK_BRIGHTNESS_TH,
)
from lunges.features import BOTTOM_FEATURE_DIM as LUNGES_BOTTOM_FEATURE_DIM
from lunges.session import LungeWebSocketSession

# --- Wall Sit imports ---
from wall_sit.wall_sit_streaming import (
    model_service as wall_sit_model_svc,
    side_gate as wall_sit_gate,
    feature_extractor as wall_sit_feature_extractor,
    label_mapper as wall_sit_labels,
    status_sender as wall_sit_status,
    WINDOW_FRAMES as WALL_SIT_WINDOW_FRAMES,
    READY_STREAK_N as WALL_SIT_READY_STREAK_N,
    DEBUG as WALL_SIT_DEBUG,
    SIDE_MODE as WALL_SIT_SIDE_MODE,
    VIS_TH as WALL_SIT_VIS_TH,
    MP_MIN_DET_CONF as WALL_SIT_MP_MIN_DET_CONF,
    MP_MIN_TRACK_CONF as WALL_SIT_MP_MIN_TRACK_CONF,
    NO_POSE_ADJUST_SECONDS as WALL_SIT_NO_POSE_ADJUST_SECONDS,
    DARK_ADJUST_SECONDS as WALL_SIT_DARK_ADJUST_SECONDS,
    DARK_BRIGHTNESS_TH as WALL_SIT_DARK_BRIGHTNESS_TH,
    STAND_KNEE_ANGLE_DEG_TH as WALL_SIT_STAND_KNEE_ANGLE_DEG_TH,
    STAND_STREAK_N as WALL_SIT_STAND_STREAK_N,
    PHASE_NO_POSE as WALL_SIT_PHASE_NO_POSE,
    PHASE_HAVE_POSE as WALL_SIT_PHASE_HAVE_POSE,
    PHASE_BUFFERING as WALL_SIT_PHASE_BUFFERING,
    PHASE_INFERENCING as WALL_SIT_PHASE_INFERENCING,
    STATUS_SEND_EVERY_N_FRAMES as WALL_SIT_STATUS_SEND_EVERY_N_FRAMES,
)
from wall_sit.session import WallSitWebSocketSession

# ---------------------------------------------------------------
# Unified FastAPI Application
# ---------------------------------------------------------------

app = make_app("FiT-AI Unified Streaming Backend")


# ---------------------------------------------------------------
# Health
# ---------------------------------------------------------------

@app.get("/")
@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "services": ["squat", "plank", "lunges", "wall_sit"],
        "timestamp": int(time.time()),
    }


@app.get("/squat/health")
def squat_health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "bottom_loaded": squat_model_svc.bottom_loaded,
        "stand_loaded": squat_model_svc.stand_loaded,
        "phase_loaded": squat_model_svc.phase_loaded,
    }


@app.get("/plank/health")
def plank_health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "model_loaded": plank_model_svc.loaded,
    }


@app.get("/lunges/health")
def lunges_health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "bottom_loaded": lunges_model_svc.bottom_loaded,
        "phase_loaded": lunges_model_svc.phase_loaded,
    }


@app.get("/wall_sit/health")
def wall_sit_health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "model_loaded": wall_sit_model_svc.loaded,
    }


# ---------------------------------------------------------------
# Squat
# ---------------------------------------------------------------

@app.websocket("/squat/ws/video")
async def squat_ws(websocket: WebSocket) -> None:
    session = SquatWebSocketSession(
        websocket=websocket,
        model_service=squat_model_svc,
        gate=squat_gate,
        status_sender=squat_status,
        ready_streak_n=SQUAT_READY_STREAK_N,
        debug=SQUAT_DEBUG,
        bottom_feature_dim=SQUAT_BOTTOM_FEATURE_DIM,
        stand_feature_dim=SQUAT_STAND_FEATURE_DIM,
        stand_ok_labels=SQUAT_STAND_OK_LABELS,
        pre_frames=SQUAT_PRE_FRAMES,
        post_frames=SQUAT_POST_FRAMES,
        min_gap=SQUAT_MIN_GAP,
        stand_knee_angle_deg_th=SQUAT_STAND_KNEE_ANGLE_DEG_TH,
        stand_knee_delta_max_deg=SQUAT_STAND_KNEE_DELTA_MAX_DEG,
        stand_min_streak=SQUAT_STAND_MIN_STREAK,
        stand_pred_cooldown=SQUAT_STAND_PRED_COOLDOWN,
        goal_good_reps=SQUAT_GOAL_GOOD_REPS,
        phase_decision_mode=SQUAT_PHASE_DECISION_MODE,
        dark_adjust_seconds=SQUAT_DARK_ADJUST_SECONDS,
        dark_brightness_th=SQUAT_DARK_BRIGHTNESS_TH,
        mp_min_det_conf=SQUAT_MP_MIN_DET_CONF,
        mp_min_track_conf=SQUAT_MP_MIN_TRACK_CONF,
    )
    await session.run()


# ---------------------------------------------------------------
# Plank
# ---------------------------------------------------------------

@app.websocket("/plank/ws/video")
async def plank_ws(websocket: WebSocket) -> None:
    session = PlankWebSocketSession(
        websocket=websocket,
        model_service=plank_model_svc,
        gate=plank_gate,
        feature_extractor=plank_feature_extractor,
        labels=plank_labels,
        status_sender=plank_status,
        window_frames=PLANK_WINDOW_FRAMES,
        ready_streak_n=PLANK_READY_STREAK_N,
        debug=PLANK_DEBUG,
        side_mode=PLANK_SIDE_MODE,
        vis_th=PLANK_VIS_TH,
        mp_min_det_conf=PLANK_MP_MIN_DET_CONF,
        mp_min_track_conf=PLANK_MP_MIN_TRACK_CONF,
        no_pose_adjust_seconds=PLANK_NO_POSE_ADJUST_SECONDS,
        dark_adjust_seconds=PLANK_DARK_ADJUST_SECONDS,
        dark_brightness_th=PLANK_DARK_BRIGHTNESS_TH,
        plank_ready_max_body_axis_angle_deg=PLANK_READY_MAX_BODY_AXIS_ANGLE_DEG,
        plank_ready_max_torso_angle_deg=PLANK_READY_MAX_TORSO_ANGLE_DEG,
        plank_ready_max_leg_angle_deg=PLANK_READY_MAX_LEG_ANGLE_DEG,
        phase_no_pose=PLANK_PHASE_NO_POSE,
        phase_have_pose=PLANK_PHASE_HAVE_POSE,
        phase_buffering=PLANK_PHASE_BUFFERING,
        phase_inferencing=PLANK_PHASE_INFERENCING,
        status_send_every_n_frames=PLANK_STATUS_SEND_EVERY_N_FRAMES,
    )
    await session.run()


# ---------------------------------------------------------------
# Lunges
# ---------------------------------------------------------------

@app.websocket("/lunges/ws/video")
async def lunges_ws(websocket: WebSocket) -> None:
    session = LungeWebSocketSession(
        websocket=websocket,
        model_service=lunges_model_svc,
        gate=lunges_gate,
        status_sender=lunges_status,
        ready_streak_n=LUNGES_READY_STREAK_N,
        debug=LUNGES_DEBUG,
        bottom_feature_dim=LUNGES_BOTTOM_FEATURE_DIM,
        pre_frames=LUNGES_PRE_FRAMES,
        post_frames=LUNGES_POST_FRAMES,
        min_gap=LUNGES_MIN_GAP,
        gate_knee_angle=LUNGES_GATE_KNEE_ANGLE,
        dark_adjust_seconds=LUNGES_DARK_ADJUST_SECONDS,
        dark_brightness_th=LUNGES_DARK_BRIGHTNESS_TH,
        goal_good_reps=LUNGES_GOAL_GOOD_REPS,
        mp_min_det_conf=LUNGES_MP_MIN_DET_CONF,
        mp_min_track_conf=LUNGES_MP_MIN_TRACK_CONF,
    )
    await session.run()


# ---------------------------------------------------------------
# Wall Sit
# ---------------------------------------------------------------

@app.websocket("/wall_sit/ws/video")
async def wall_sit_ws(websocket: WebSocket) -> None:
    session = WallSitWebSocketSession(
        websocket=websocket,
        model_service=wall_sit_model_svc,
        gate=wall_sit_gate,
        feature_extractor=wall_sit_feature_extractor,
        labels=wall_sit_labels,
        status_sender=wall_sit_status,
        window_frames=WALL_SIT_WINDOW_FRAMES,
        ready_streak_n=WALL_SIT_READY_STREAK_N,
        debug=WALL_SIT_DEBUG,
        side_mode=WALL_SIT_SIDE_MODE,
        vis_th=WALL_SIT_VIS_TH,
        mp_min_det_conf=WALL_SIT_MP_MIN_DET_CONF,
        mp_min_track_conf=WALL_SIT_MP_MIN_TRACK_CONF,
        no_pose_adjust_seconds=WALL_SIT_NO_POSE_ADJUST_SECONDS,
        dark_adjust_seconds=WALL_SIT_DARK_ADJUST_SECONDS,
        dark_brightness_th=WALL_SIT_DARK_BRIGHTNESS_TH,
        stand_knee_angle_deg_th=WALL_SIT_STAND_KNEE_ANGLE_DEG_TH,
        stand_streak_n=WALL_SIT_STAND_STREAK_N,
        phase_no_pose=WALL_SIT_PHASE_NO_POSE,
        phase_have_pose=WALL_SIT_PHASE_HAVE_POSE,
        phase_buffering=WALL_SIT_PHASE_BUFFERING,
        phase_inferencing=WALL_SIT_PHASE_INFERENCING,
        status_send_every_n_frames=WALL_SIT_STATUS_SEND_EVERY_N_FRAMES,
    )
    await session.run()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=7860, reload=False, log_level="info")
