"""
Wall Sit WebSocket session handler.

Manages a single wall sit streaming session including pose detection,
side-view gating, standing gate, feature extraction, buffering,
inference, and result sending.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from fastapi import WebSocket

from shared.base_session import BaseWebSocketSession
from shared.frame_decoder import FrameDecoder
from shared.frame_quality import FrameQuality
from shared.sklearn_model_service import SklearnModelService
from shared.side_gate import SideGate
from shared.label_mapper import LabelMapper
from shared.status_sender import StatusSender
from wall_sit.features import aggregate_window, extract_frame_features


# ---------------------------------------------------------------
# Wall Sit Stream State
# ---------------------------------------------------------------

@dataclass
class StreamState:
    """Session-level state for a wall sit streaming WebSocket connection."""

    started: bool = False
    frame_feature_values: List[Tuple[float, float, float]] = field(default_factory=list)
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

    def extract_features(
        self,
        pose_result: Any,
        side: str,
    ) -> Optional[Tuple[float, float, float]]:
        """Extract per-frame feature tuple for wall sit.

        Args:
            pose_result: MediaPipe pose result.
            side: "left" or "right".

        Returns:
            Tuple of (foot_wall_norm, knee_angle, torso_alignment),
            or None if landmarks are missing.
        """
        if not pose_result.pose_landmarks:
            return None

        return extract_frame_features(pose_result.pose_landmarks.landmark, side)

    @staticmethod
    def aggregate_window(
        frame_feature_values: List[Tuple[float, float, float]]
    ) -> np.ndarray:
        """Aggregate a window of per-frame tuples into a 5-D feature vector.

        Returns:
            Array of [mean_fw, std_fw, mean_knee, min_knee, mean_torso].
        """
        return aggregate_window(frame_feature_values)


# ---------------------------------------------------------------
# Wall Sit WebSocket Session
# ---------------------------------------------------------------

class WallSitWebSocketSession(BaseWebSocketSession):
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
        feature_extractor: WallSitFeatureExtractor,
        labels: LabelMapper,
        status: StatusSender,
        window_frames: int,
        ready_streak_n: int,
        debug: bool,
        # Configuration constants
        side_mode: str,
        vis_th: float,
        mp_min_det_conf: float,
        mp_min_track_conf: float,
        no_pose_adjust_seconds: float,
        dark_adjust_seconds: float,
        dark_brightness_th: float,
        stand_knee_angle_deg_th: float,
        stand_streak_n: int,
        phase_no_pose: str,
        phase_have_pose: str,
        phase_buffering: str,
        phase_inferencing: str,
        status_send_every_n_frames: int,
    ) -> None:
        self.ws = websocket
        self.model_svc = model_svc
        self.gate = gate
        self.feature_extractor = feature_extractor
        self.labels = labels
        self.status = status
        self.window_frames = int(window_frames)
        self.ready_streak_n = int(ready_streak_n)
        self.debug = debug

        # Store configuration
        self.side_mode = side_mode
        self.vis_th = vis_th
        self.mp_min_det_conf = mp_min_det_conf
        self.mp_min_track_conf = mp_min_track_conf
        self.no_pose_adjust_seconds = no_pose_adjust_seconds
        self.dark_adjust_seconds = dark_adjust_seconds
        self.dark_brightness_th = dark_brightness_th
        self.stand_knee_angle_deg_th = stand_knee_angle_deg_th
        self.stand_streak_n = stand_streak_n
        self.phase_no_pose = phase_no_pose
        self.phase_have_pose = phase_have_pose
        self.phase_buffering = phase_buffering
        self.phase_inferencing = phase_inferencing
        self.status_send_every_n_frames = status_send_every_n_frames

        self.st = StreamState()

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=mp_min_det_conf,
            min_tracking_confidence=mp_min_track_conf,
        )

    # ---------------------------------------------------------------
    # Connection hook (replaces duplicated run() boilerplate)
    # ---------------------------------------------------------------

    async def _on_connected(self) -> None:
        """Send boot info and welcome message after ws.accept()."""
        await self.status.send_info(self.ws, "WebSocket connected")
        print(
            f"[BOOT] side_mode={self.side_mode} VIS_TH={self.vis_th} "
            f"det={self.mp_min_det_conf} track={self.mp_min_track_conf}"
        )
        print(
            f"[BOOT] WINDOW_FRAMES={self.window_frames} "
            f"READY_STREAK_N={self.ready_streak_n}"
        )
        if not self.model_svc.loaded:
            await self.status.send_info(
                self.ws,
                "Model not loaded (check MODEL_PATH)",
            )

    # ---------------------------------------------------------------
    # State factory and hooks (implement BaseWebSocketSession contract)
    # ---------------------------------------------------------------

    def _create_state(self) -> StreamState:
        """Return a fresh StreamState; base will set started=True and session_id."""
        return StreamState()

    @property
    def _initial_status(self) -> str:
        """Wall-sit uses NO_POSE as its initial/idle phase."""
        return self.phase_no_pose

    async def _on_stop(self) -> None:
        """Reset gate and buffers including watchdog timers after session ends."""
        self._reset_gate_and_buffers(reset_watchdog=True)

    # ---------------------------------------------------------------
    # State reset
    # ---------------------------------------------------------------

    def _reset_gate_and_buffers(self, reset_watchdog: bool) -> None:
        """Reset gate state, feature buffers, and optionally watchdogs."""
        self.st.frame_feature_values.clear()
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

    async def _update_dark_watchdog(
        self,
        too_dark: bool,
        brightness_mean: float,
    ) -> None:
        """Send a light warning once darkness persists beyond the threshold."""
        now = time.time()

        if not too_dark:
            self.st.dark_since = None
            self.st.dark_alerted = False
            return

        if self.st.dark_since is None:
            self.st.dark_since = now
            self.st.dark_alerted = False
            return

        if self.st.dark_alerted:
            return

        if now - self.st.dark_since >= self.dark_adjust_seconds:
            await self.status.send_info(
                self.ws,
                "Please adjust your lights.",
                {
                    "brightness_mean": round(brightness_mean, 1),
                    "brightness_th": self.dark_brightness_th,
                },
            )
            self.st.dark_alerted = True

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
        too_dark, brightness_mean = FrameQuality.is_too_dark(frame, self.dark_brightness_th)
        await self._update_dark_watchdog(too_dark, brightness_mean)

        # Step 3: Run pose detection
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_result = self.pose.process(img_rgb)

        # Step 4: Side-view gate
        side_debug: Dict[str, Any] = {}
        chosen_side: Optional[str] = None

        if pose_result.pose_landmarks:
            landmarks = pose_result.pose_landmarks.landmark
            if self.st.ready and self.st.chosen_side is not None:
                chosen_side = self.st.chosen_side
            else:
                chosen_side, side_debug = self.gate.choose_best_side(landmarks)

        # Gate failure: NO_POSE watchdog + DARK watchdog
        if (not pose_result.pose_landmarks) or (chosen_side is None):
            await self._handle_no_pose(too_dark, brightness_mean, side_debug)
            return

        # Pose regained: reset watchdogs
        self.st.no_pose_since = None
        self.st.no_pose_alerted = False

        # Step 5: Track ready streak
        self.st.ready_streak += 1
        self.st.chosen_side = chosen_side

        # Not ready yet: HAVE_POSE phase
        if (not self.st.ready) and (self.st.ready_streak < self.ready_streak_n):
            await self.status.send_status(
                self.ws, self.st, self.phase_have_pose,
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
            self.st.frame_feature_values.clear()
            self.st.last_sent_label = ""
            self.st.last_sent_conf = None

            print(f"[GATE] READY session_id={self.st.session_id} side={chosen_side}")
            await self.status.send_info(
                self.ws, "Side View OK",
                {"session_id": self.st.session_id, "side": chosen_side},
            )
            await self.status.send_status(
                self.ws, self.st, self.phase_buffering,
                {"chosen_side": chosen_side, "window_fill": 0, "window_size": self.window_frames},
                force=True,
            )

        # Model not loaded or not ready: stay in BUFFERING
        if (not self.model_svc.loaded) or (not self.st.ready) or (self.st.chosen_side is None):
            await self.status.send_status(
                self.ws, self.st, self.phase_buffering,
                {
                    "chosen_side": self.st.chosen_side,
                    "window_fill": len(self.st.frame_feature_values),
                    "window_size": self.window_frames,
                },
            )
            return

        # Step 6: Extract features
        frame_feature_tuple = self.feature_extractor.extract_features(
            pose_result,
            self.st.chosen_side,
        )

        if frame_feature_tuple is None:
            self.st.frame_feature_values.clear()
            self.st.stand_streak = 0
            await self.status.send_status(
                self.ws, self.st, self.phase_buffering,
                {
                    "chosen_side": self.st.chosen_side,
                    "window_fill": 0,
                    "window_size": self.window_frames,
                },
            )
            return

        # Step 7: Standing gate — avoid predicting while standing upright
        knee_angle = float(frame_feature_tuple[1])
        if knee_angle >= self.stand_knee_angle_deg_th:
            self.st.stand_streak += 1
        else:
            self.st.stand_streak = 0

        if self.st.stand_streak >= self.stand_streak_n:
            if self.st.stand_streak == self.stand_streak_n:
                print(
                    f"[WALLSIT] STANDING gate: session_id={self.st.session_id} "
                    f"side={self.st.chosen_side} knee_angle={knee_angle:.1f} th={self.stand_knee_angle_deg_th}"
                )
            self.st.frame_feature_values.clear()
            await self.status.send_status(
                self.ws, self.st, self.phase_have_pose,
                {
                    "chosen_side": self.st.chosen_side,
                    "ready_streak": self.st.ready_streak,
                    "needed_streak": self.ready_streak_n,
                    "window_fill": 0,
                    "window_size": self.window_frames,
                    "standing": True,
                    "knee_angle": round(knee_angle, 1),
                    "knee_th": self.stand_knee_angle_deg_th,
                },
            )
            return

        self.st.frame_feature_values.append(frame_feature_tuple)

        # Step 8: Not enough frames yet: BUFFERING
        if len(self.st.frame_feature_values) < self.window_frames:
            await self.status.send_status(
                self.ws, self.st, self.phase_buffering,
                {
                    "chosen_side": self.st.chosen_side,
                    "window_fill": len(self.st.frame_feature_values),
                    "window_size": self.window_frames,
                },
            )
            return

        # Step 9: Enough frames — run inference
        await self.status.send_status(
            self.ws, self.st, self.phase_inferencing,
            {
                "chosen_side": self.st.chosen_side,
                "window_fill": len(self.st.frame_feature_values),
                "window_size": self.window_frames,
            },
        )

        feature_window_values = self.st.frame_feature_values[-self.window_frames:]
        aggregated_feature_vector = self.feature_extractor.aggregate_window(
            feature_window_values
        )

        pred_id, conf = self.model_svc.predict(aggregated_feature_vector)
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
        if len(self.st.frame_feature_values) > (self.window_frames + 60):
            self.st.frame_feature_values = self.st.frame_feature_values[
                -(self.window_frames + 60):
            ]

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

        # NO_POSE alert (only if not explained by darkness)
        if (
            (not self.st.no_pose_alerted)
            and (now - self.st.no_pose_since >= self.no_pose_adjust_seconds)
            and (not too_dark)
        ):
            await self.status.send_info(
                self.ws,
                "Adjust your camera to see your full body",
                {"brightness_mean": round(brightness_mean, 1), "brightness_th": self.dark_brightness_th},
            )
            self.st.no_pose_alerted = True

        # Reset gate/buffers but keep watchdog timers running
        self._reset_gate_and_buffers(reset_watchdog=False)

        await self.status.send_status(
            self.ws, self.st, self.phase_no_pose,
            {
                "chosen_side": None,
                "ready_streak": 0,
                "needed_streak": self.ready_streak_n,
                "window_fill": 0,
                "window_size": self.window_frames,
                "too_dark": too_dark,
                "brightness_mean": round(brightness_mean, 1),
                "brightness_th": self.dark_brightness_th,
                "debug": side_debug if self.debug else None,
            },
        )
