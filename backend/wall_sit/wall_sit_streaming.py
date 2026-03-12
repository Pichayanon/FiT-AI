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

import argparse
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

import cv2
import mediapipe as mp
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from shared.frame_decoder import FrameDecoder
from shared.frame_quality import FrameQuality
from shared.json_utils import parse_json
from shared.math_utils import angle_3pts
from shared.sklearn_model_service import SklearnModelService
from shared.side_gate import SideGate
from shared.label_mapper import LabelMapper
from shared.status_sender import StatusSender


# ---------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------

MODEL_PATH = "wall_sit/models/wall_sit_model.pkl"

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

DEBUG = True

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

app = FastAPI(title="FiT-AI WallSit Streaming Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------
# Wall Sit Stream State
# ---------------------------------------------------------------

@dataclass
class StreamState:
    """Session-level state for a wall sit streaming WebSocket connection."""

    started: bool = False
    foot_wall_vals: List[Tuple[float, float, float]] = field(default_factory=list)
    stand_streak: int = 0

    # Gate
    ready: bool = False
    ready_streak: int = 0
    chosen_side: Optional[str] = None

    # Session metadata
    session_id: str = ""

    # Status throttle
    last_status: str = ""
    status_tick: int = 0

    # Last prediction (for logs/debugging)
    last_pred_label: str = ""
    last_pred_conf: Optional[float] = None

    # Last sent (for optional deduplication)
    last_sent_label: str = ""
    last_sent_conf: Optional[float] = None

    # Frame counter
    frame_count: int = 0

    # NO_POSE watchdog
    no_pose_since: Optional[float] = None
    no_pose_alerted: bool = False

    # DARK watchdog
    dark_since: Optional[float] = None
    dark_alerted: bool = False


# ---------------------------------------------------------------
# Wall Sit Feature Extractor
# ---------------------------------------------------------------

class WallSitFeatureExtractor:
    """Extract per-frame and window-level features for the wall sit model.

    Per-frame feature tuple:
        (foot_wall_norm, knee_angle, torso_alignment)
    Window-level aggregate feature (5-D):
        mean_fw, std_fw, mean_knee, min_knee, mean_torso
    """

    def __init__(self, mp_pose: Any) -> None:
        self.mp_pose = mp_pose

    @staticmethod
    def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        """Compute angle ABC (degrees) from 2D points."""
        return angle_3pts(a, b, c)

    def extract_features(
        self, res: Any, side: str
    ) -> Optional[Tuple[float, float, float]]:
        """Extract per-frame feature tuple for wall sit.

        Args:
            res: MediaPipe pose result.
            side: "left" or "right".

        Returns:
            Tuple of (foot_wall_norm, knee_angle, torso_alignment),
            or None if landmarks are missing.
        """
        if not res.pose_landmarks:
            return None

        lm = res.pose_landmarks.landmark

        if side == "right":
            hip = self.mp_pose.PoseLandmark.RIGHT_HIP
            knee = self.mp_pose.PoseLandmark.RIGHT_KNEE
            ankle = self.mp_pose.PoseLandmark.RIGHT_ANKLE
            shoulder = self.mp_pose.PoseLandmark.RIGHT_SHOULDER
        else:
            hip = self.mp_pose.PoseLandmark.LEFT_HIP
            knee = self.mp_pose.PoseLandmark.LEFT_KNEE
            ankle = self.mp_pose.PoseLandmark.LEFT_ANKLE
            shoulder = self.mp_pose.PoseLandmark.LEFT_SHOULDER

        # Foot-wall distance normalized by shoulder width
        shoulder_width = abs(
            lm[self.mp_pose.PoseLandmark.LEFT_SHOULDER].x
            - lm[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].x
        ) + 1e-6
        foot_wall = abs(lm[ankle].x - lm[hip].x) / shoulder_width

        # Knee angle (hip-knee-ankle)
        knee_angle = self._angle(
            [lm[hip].x, lm[hip].y],
            [lm[knee].x, lm[knee].y],
            [lm[ankle].x, lm[ankle].y],
        )

        # Torso alignment (shoulder-hip horizontal distance)
        torso_alignment = abs(lm[shoulder].x - lm[hip].x)

        return float(foot_wall), float(knee_angle), float(torso_alignment)

    @staticmethod
    def aggregate_window(vals: List[Tuple[float, float, float]]) -> np.ndarray:
        """Aggregate a window of per-frame tuples into a 5-D feature vector.

        Returns:
            Array of [mean_fw, std_fw, mean_knee, min_knee, mean_torso].
        """
        fw = [v[0] for v in vals]
        knee = [v[1] for v in vals]
        torso = [v[2] for v in vals]

        return np.array([
            np.mean(fw),
            np.std(fw),
            np.mean(knee),
            np.min(knee),
            np.mean(torso),
        ], dtype=np.float32)


# ---------------------------------------------------------------
# Wall Sit WebSocket Session
# ---------------------------------------------------------------

class WallSitWebSocketSession:
    """Handle a single wall sit WebSocket streaming session.

    Orchestrates: frame decoding, quality checks, pose detection,
    side-view gating, standing gate, feature extraction, buffering,
    inference, and result sending.
    """

    def __init__(
        self,
        websocket: WebSocket,
        model_svc: SklearnModelService,
        gate: SideGate,
        feat: WallSitFeatureExtractor,
        labels: LabelMapper,
        status: StatusSender,
        window_frames: int,
        ready_streak_n: int,
        debug: bool,
    ) -> None:
        self.ws = websocket
        self.model_svc = model_svc
        self.gate = gate
        self.feat = feat
        self.labels = labels
        self.status = status
        self.window_frames = int(window_frames)
        self.ready_streak_n = int(ready_streak_n)
        self.debug = debug

        self.st = StreamState()

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
        """Main receive loop for the WebSocket session."""
        await self.ws.accept()
        await self.status.send_info(self.ws, "WebSocket connected")

        print(
            f"[BOOT] side_mode={SIDE_MODE} VIS_TH={VIS_TH} "
            f"det={MP_MIN_DET_CONF} track={MP_MIN_TRACK_CONF}"
        )
        print(
            f"[BOOT] WINDOW_FRAMES={self.window_frames} "
            f"READY_STREAK_N={self.ready_streak_n}"
        )
        if not self.model_svc.loaded:
            await self.status.send_info(
                self.ws,
                "Model not loaded (check MODEL_PATH)",
                {"model_path": MODEL_PATH},
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
                elif mtype == "stop":
                    await self._handle_stop()
                elif mtype == "frame":
                    await self._handle_frame(data)
                else:
                    continue

        except WebSocketDisconnect:
            print(f"[WS] disconnect session_id={self.st.session_id}")
            return
        except Exception as e:  # pylint: disable=broad-except
            print(f"[WS] error: {e}")
            try:
                await self.status.send_info(self.ws, f"Server error: {e}")
            except Exception:  # pylint: disable=broad-except
                pass
            return


    # ---------------------------------------------------------------
    # Start / Stop handlers
    # ---------------------------------------------------------------

    async def _handle_start(self) -> None:
        """Handle session start message. Reset all state."""
        self.st = StreamState(started=True)
        self.st.session_id = str(int(time.time() * 1000))

        self.st.no_pose_since = None
        self.st.no_pose_alerted = False
        self.st.dark_since = None
        self.st.dark_alerted = False

        print(f"[SESSION] START session_id={self.st.session_id}")

        await self.status.send_info(
            self.ws, "Start streaming", {"session_id": self.st.session_id}
        )
        await self.status.send_status(
            self.ws, self.st, PHASE_NO_POSE,
            {"reason": "session_started"}, force=True,
        )

    async def _handle_stop(self) -> None:
        """Handle session stop message. Reset gate and buffers."""
        print(f"[SESSION] STOP session_id={self.st.session_id}")
        self.st.started = False

        await self.status.send_info(
            self.ws, "Stop streaming", {"session_id": self.st.session_id}
        )
        await self.status.send_status(
            self.ws, self.st, PHASE_NO_POSE,
            {"reason": "session_stopped"}, force=True,
        )

        self._reset_gate_and_buffers(reset_watchdog=True)

    # ---------------------------------------------------------------
    # State reset
    # ---------------------------------------------------------------

    def _reset_gate_and_buffers(self, reset_watchdog: bool) -> None:
        """Reset gate state, feature buffers, and optionally watchdogs."""
        self.st.foot_wall_vals.clear()
        self.st.stand_streak = 0
        self.st.ready = False
        self.st.ready_streak = 0
        self.st.chosen_side = None

        self.st.last_sent_label = ""
        self.st.last_sent_conf = None
        self.st.last_pred_label = ""
        self.st.last_pred_conf = None
        self.st.frame_count = 0

        if reset_watchdog:
            self.st.no_pose_since = None
            self.st.no_pose_alerted = False
            self.st.dark_since = None
            self.st.dark_alerted = False

    # ---------------------------------------------------------------
    # Frame handler
    # ---------------------------------------------------------------

    async def _handle_frame(self, data: Dict[str, Any]) -> None:
        """Process a single video frame through the full pipeline."""
        if not self.st.started:
            return

        self.st.frame_count += 1

        # Step 1: Decode frame
        frame = FrameDecoder.decode_jpeg_base64(data.get("jpeg_b64", ""))
        if frame is None:
            await self.status.send_info(self.ws, "Decode failed")
            return

        # Step 2: Check brightness
        too_dark, brightness_mean = FrameQuality.is_too_dark(frame, DARK_BRIGHTNESS_TH)

        # Step 3: Run pose detection
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.pose.process(img_rgb)

        # Step 4: Side-view gate
        side_debug: Dict[str, Any] = {}
        chosen_side: Optional[str] = None

        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark
            if self.st.ready and self.st.chosen_side is not None:
                chosen_side = self.st.chosen_side
            else:
                chosen_side, side_debug = self.gate.choose_best_side(lm)

        # Gate failure: NO_POSE watchdog + DARK watchdog
        if (not res.pose_landmarks) or (chosen_side is None):
            await self._handle_no_pose(too_dark, brightness_mean, side_debug)
            return

        # Pose regained: reset watchdogs
        self.st.no_pose_since = None
        self.st.no_pose_alerted = False
        self.st.dark_since = None
        self.st.dark_alerted = False

        # Step 5: Track ready streak
        self.st.ready_streak += 1
        self.st.chosen_side = chosen_side

        # Not ready yet: HAVE_POSE phase
        if (not self.st.ready) and (self.st.ready_streak < self.ready_streak_n):
            await self.status.send_status(
                self.ws, self.st, PHASE_HAVE_POSE,
                {
                    "chosen_side": chosen_side,
                    "ready_streak": self.st.ready_streak,
                    "needed_streak": self.ready_streak_n,
                    "window_fill": 0,
                    "window_size": self.window_frames,
                },
            )
            return

        # First time ready: enter BUFFERING
        if (not self.st.ready) and (self.st.ready_streak >= self.ready_streak_n):
            self.st.ready = True
            self.st.foot_wall_vals.clear()
            self.st.last_sent_label = ""
            self.st.last_sent_conf = None

            print(f"[GATE] READY session_id={self.st.session_id} side={chosen_side}")
            await self.status.send_info(
                self.ws, "Side View OK",
                {"session_id": self.st.session_id, "side": chosen_side},
            )
            await self.status.send_status(
                self.ws, self.st, PHASE_BUFFERING,
                {"chosen_side": chosen_side, "window_fill": 0, "window_size": self.window_frames},
                force=True,
            )

        # Model not loaded or not ready: stay in BUFFERING
        if (not self.model_svc.loaded) or (not self.st.ready) or (self.st.chosen_side is None):
            await self.status.send_status(
                self.ws, self.st, PHASE_BUFFERING,
                {
                    "chosen_side": self.st.chosen_side,
                    "window_fill": len(self.st.foot_wall_vals),
                    "window_size": self.window_frames,
                },
            )
            return

        # Step 6: Extract features
        feat_tuple = self.feat.extract_features(res, self.st.chosen_side)

        if feat_tuple is None:
            self.st.foot_wall_vals.clear()
            self.st.stand_streak = 0
            await self.status.send_status(
                self.ws, self.st, PHASE_BUFFERING,
                {
                    "chosen_side": self.st.chosen_side,
                    "window_fill": 0,
                    "window_size": self.window_frames,
                },
            )
            return

        # Step 7: Standing gate — avoid predicting while standing upright
        knee_angle = float(feat_tuple[1])
        if knee_angle >= STAND_KNEE_ANGLE_DEG_TH:
            self.st.stand_streak += 1
        else:
            self.st.stand_streak = 0

        if self.st.stand_streak >= STAND_STREAK_N:
            if self.st.stand_streak == STAND_STREAK_N:
                print(
                    f"[WALLSIT] STANDING gate: session_id={self.st.session_id} "
                    f"side={self.st.chosen_side} knee_angle={knee_angle:.1f} th={STAND_KNEE_ANGLE_DEG_TH}"
                )
            self.st.foot_wall_vals.clear()
            await self.status.send_status(
                self.ws, self.st, PHASE_HAVE_POSE,
                {
                    "chosen_side": self.st.chosen_side,
                    "ready_streak": self.st.ready_streak,
                    "needed_streak": self.ready_streak_n,
                    "window_fill": 0,
                    "window_size": self.window_frames,
                    "standing": True,
                    "knee_angle": round(knee_angle, 1),
                    "knee_th": STAND_KNEE_ANGLE_DEG_TH,
                },
            )
            return

        self.st.foot_wall_vals.append(feat_tuple)

        # Step 8: Not enough frames yet: BUFFERING
        if len(self.st.foot_wall_vals) < self.window_frames:
            await self.status.send_status(
                self.ws, self.st, PHASE_BUFFERING,
                {
                    "chosen_side": self.st.chosen_side,
                    "window_fill": len(self.st.foot_wall_vals),
                    "window_size": self.window_frames,
                },
            )
            return

        # Step 9: Enough frames — run inference
        await self.status.send_status(
            self.ws, self.st, PHASE_INFERENCING,
            {
                "chosen_side": self.st.chosen_side,
                "window_fill": len(self.st.foot_wall_vals),
                "window_size": self.window_frames,
            },
        )

        vals = self.st.foot_wall_vals[-self.window_frames:]
        agg_feat = self.feat.aggregate_window(vals)

        pred_id, conf = self.model_svc.predict(agg_feat)
        pred_label = self.labels.label_of(pred_id)

        self.st.last_pred_label = pred_label
        self.st.last_pred_conf = conf

        # Step 10: Send result
        payload: Dict[str, Any] = {
            "type": "result",
            "prediction": pred_label,
            "confidence": round(conf, 3) if conf is not None else None,
            "window": self.window_frames,
            "session_id": self.st.session_id,
            "side": self.st.chosen_side,
        }
        if self.debug:
            print(f"[PRED] {payload}")
        await self.ws.send_text(json.dumps(payload))

        # Bound feature buffer to prevent unbounded growth
        if len(self.st.foot_wall_vals) > (self.window_frames + 60):
            self.st.foot_wall_vals = self.st.foot_wall_vals[-(self.window_frames + 60):]

    # ---------------------------------------------------------------
    # Watchdog handlers
    # ---------------------------------------------------------------

    async def _handle_no_pose(
        self,
        too_dark: bool,
        brightness_mean: float,
        side_debug: Dict[str, Any],
    ) -> None:
        """Handle frames where pose or side gate fails.

        Manages the NO_POSE and DARK watchdog timers, sending
        user-facing messages after configured timeout periods.
        """
        now = time.time()

        # Start NO_POSE watchdog
        if self.st.no_pose_since is None:
            self.st.no_pose_since = now
            self.st.no_pose_alerted = False

        # Track DARK watchdog
        if too_dark:
            if self.st.dark_since is None:
                self.st.dark_since = now
                self.st.dark_alerted = False
        else:
            self.st.dark_since = None
            self.st.dark_alerted = False

        # DARK alert (takes priority over NO_POSE)
        if (
            too_dark
            and (self.st.dark_since is not None)
            and (not self.st.dark_alerted)
            and (now - self.st.dark_since >= DARK_ADJUST_SECONDS)
        ):
            await self.status.send_info(
                self.ws,
                "Please adjust your lights.",
                {"brightness_mean": round(brightness_mean, 1), "brightness_th": DARK_BRIGHTNESS_TH},
            )
            self.st.dark_alerted = True

        # NO_POSE alert (only if not explained by darkness)
        if (
            (not self.st.no_pose_alerted)
            and (now - self.st.no_pose_since >= NO_POSE_ADJUST_SECONDS)
            and (not too_dark)
        ):
            await self.status.send_info(
                self.ws,
                "Adjust your camera to see your full body",
                {"brightness_mean": round(brightness_mean, 1), "brightness_th": DARK_BRIGHTNESS_TH},
            )
            self.st.no_pose_alerted = True

        # Reset gate/buffers but keep watchdog timers running
        self._reset_gate_and_buffers(reset_watchdog=False)

        await self.status.send_status(
            self.ws, self.st, PHASE_NO_POSE,
            {
                "chosen_side": None,
                "ready_streak": 0,
                "needed_streak": self.ready_streak_n,
                "window_fill": 0,
                "window_size": self.window_frames,
                "too_dark": too_dark,
                "brightness_mean": round(brightness_mean, 1),
                "brightness_th": DARK_BRIGHTNESS_TH,
                "debug": side_debug if self.debug else None,
            },
        )


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
    )
    await session.run()


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def main() -> None:
    """Run server via uvicorn."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--serve", action="store_true", help="Run WebSocket server")
    args = parser.parse_args()

    if not args.serve:
        args.serve = True

    if args.serve:
        import uvicorn

        uvicorn.run(
            "wall_sit_streaming:app",
            host="0.0.0.0",
            port=5050,
            reload=False,
            log_level="info",
        )


if __name__ == "__main__":
    main()
