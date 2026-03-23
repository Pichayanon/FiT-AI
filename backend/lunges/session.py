"""Lunge WebSocket session handler."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np
import cv2
import mediapipe as mp
from fastapi import WebSocket

from shared.base_session import BaseWebSocketSession
from shared.frame_decoder import FrameDecoder
from shared.frame_quality import FrameQuality
from shared.phase_bottom_session import PhaseBottomSessionMixin
from shared.phase_bottom_state import PhaseBottomStreamState
from shared.side_view_gate_dynamic import SideViewGateDynamic
from shared.status_sender import StatusSender
from shared.tcn_model_service import PhaseAwareTCNModelService

from lunges.features import (
    BOTTOM_FEATURE_DIM,
    extract_bottom_features,
    extract_phase_features as extract_phase_features_from_lm,
    LandmarkSmoother,
)


class LungeModelService(PhaseAwareTCNModelService):
    """Load and serve lunge bottom and phase TCN models."""

    def __init__(
        self,
        bottom_path: str,
        phase_path: Optional[str] = None,
    ) -> None:
        super().__init__(bottom_path, phase_path=phase_path)


# ---------------------------------------------------------------
# Stream State
# ---------------------------------------------------------------

@dataclass
class StreamState(PhaseBottomStreamState):
    """Session state for lunge streaming."""

    last_phase: str = "unknown"

    history: deque = field(default_factory=lambda: deque(maxlen=15 + 15 + 240))
    phase_features: deque = field(default_factory=lambda: deque(maxlen=120))
    previous_phase_values: Optional[Tuple[float, float, float]] = None


# ---------------------------------------------------------------
# Lunge WebSocket Session
# ---------------------------------------------------------------

class LungeWebSocketSession(PhaseBottomSessionMixin, BaseWebSocketSession):
    """Handle a single lunge WebSocket streaming session."""

    def __init__(
        self,
        websocket: WebSocket,
        model_service: LungeModelService,
        gate: SideViewGateDynamic,
        status_sender: StatusSender,
        ready_streak_n: int,
        debug: bool,
        # Configuration constants
        bottom_feature_dim: int,
        pre_frames: int,
        post_frames: int,
        min_gap: int,
        gate_knee_angle: float,
        dark_adjust_seconds: float,
        dark_brightness_th: float,
        goal_good_reps: int,
        mp_min_det_conf: float,
        mp_min_track_conf: float,
    ) -> None:
        self.websocket = websocket
        self.model_service = model_service
        self.gate = gate
        self.status_sender = status_sender
        self.ready_streak_n = ready_streak_n
        self.debug = debug

        # Store configuration
        self.bottom_feature_dim = bottom_feature_dim
        self.pre_frames = pre_frames
        self.post_frames = post_frames
        self.min_gap = min_gap
        self.gate_knee_angle = gate_knee_angle
        self.dark_adjust_seconds = dark_adjust_seconds
        self.dark_brightness_th = dark_brightness_th
        self.goal_good_reps = goal_good_reps

        self.state = StreamState()
        self.frame_index = 0
        self.smoother = LandmarkSmoother(alpha=0.6)

        self.pose_module = mp.solutions.pose
        self.pose = self.pose_module.Pose(
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
        """Send dimension warnings and welcome message after ws.accept()."""
        if self.model_service.bottom_loaded and self.bottom_feature_dim != self.model_service.bottom_in_dim:
            await self.status_sender.send_info(
                self.websocket,
                f"WARNING: Bottom model in_dim={self.model_service.bottom_in_dim} "
                f"but extractor gives dim={self.bottom_feature_dim}",
            )
        await self.status_sender.send_info(
            self.websocket,
            "WebSocket connected",
            {
                "bottom_feature_dim": self.bottom_feature_dim,
            },
        )

    # ---------------------------------------------------------------
    # State factory and hooks (implement BaseWebSocketSession contract)
    # ---------------------------------------------------------------

    def _create_state(self) -> StreamState:
        """Return a fresh StreamState; base will set started=True and session_id."""
        return StreamState(
            history=deque(maxlen=self.pre_frames + self.post_frames + 240)
        )

    async def _on_start(self) -> None:
        """Reset frame counter and landmark smoother for the new session."""
        self.frame_index = 0
        self.smoother = LandmarkSmoother(alpha=0.6)

    def _stop_extra(self) -> Dict[str, Any]:
        """Include rep counts in the stop payload."""
        return {
            "reps": {
                "total": int(self.state.total_reps),
                "correct": int(self.state.good_reps),
                "incorrect": int(self.state.bad_reps),
                "goal_correct": int(self.goal_good_reps),
            },
        }

    # ---------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------

    def _reset_buffers(self) -> None:
        """Reset all tracking buffers and streaks (used on gate failure or ready transition)."""
        self._reset_phase_bottom_state()

    def _after_full_buffer_reset(self) -> None:
        """Reset lunge-specific tracking fields on gate failure."""
        self.state.previous_phase_values = None
        self.smoother.reset()

    def _after_ready_transition(self) -> None:
        """Reset only the stateful phase features when the gate becomes ready."""
        self.state.previous_phase_values = None

    def _is_good_rep_label(self, pred_label: str) -> bool:
        """Treat both 'good*' and 'correct' lunge labels as valid reps."""
        return pred_label.startswith("good") or pred_label == "correct"

    def _bottom_feature_from_history_record(self, record: Any) -> np.ndarray:
        """Return the bottom-model feature vector from a lunge history tuple."""
        return record[1]

    # ---------------------------------------------------------------
    # Frame handler
    # ---------------------------------------------------------------

    async def _handle_frame(self, data: Dict[str, Any]) -> None:
        """Process a single video frame through the full lunge pipeline."""
        # Step 1: Decode frame
        frame = FrameDecoder.decode_jpeg_base64(data.get("jpeg_b64", ""))
        if frame is None:
            await self.status_sender.send_info(self.websocket, "Decode failed")
            return

        # Step 2: Check brightness and run pose detection
        too_dark, brightness_mean = FrameQuality.is_too_dark(
            frame,
            self.dark_brightness_th,
        )
        await self._update_dark_watchdog(too_dark, brightness_mean)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_result = self.pose.process(image_rgb)

        # Step 3: Handle no pose
        if not pose_result.pose_landmarks:
            self._reset_buffers()
            await self._send_waiting_status(
                too_dark=too_dark,
                brightness_mean=brightness_mean,
            )
            self.frame_index += 1
            return

        # Step 4: Side-view gate check
        landmarks = pose_result.pose_landmarks.landmark

        # Build numpy array and apply smoothing
        landmark_array = np.zeros((33, 4), dtype=np.float32)
        for idx in range(33):
            landmark_array[idx] = [
                landmarks[idx].x,
                landmarks[idx].y,
                landmarks[idx].z,
                landmarks[idx].visibility,
            ]

        landmark_array = self.smoother.update(landmark_array)

        side_gate_ok, gate_debug = self.gate.evaluate(landmark_array)
        self.state.last_gate_debug = gate_debug
        side_view_ok = side_gate_ok or bool(gate_debug.get("single_side_profile_ok"))

        if not side_view_ok:
            self._reset_buffers()
            await self._send_waiting_status(
                too_dark=too_dark,
                brightness_mean=brightness_mean,
                gate_debug=gate_debug,
                reason=gate_debug.get("reason", "side_gate_not_ok"),
            )
            self.frame_index += 1
            return

        # Step 5: Track ready streak
        if not await self._advance_ready_streak(
            gate_debug=gate_debug,
            ok_message="Side View OK",
        ):
            self.frame_index += 1
            return

        # Step 6: Extract features
        bottom_features = extract_bottom_features(landmark_array)

        # Compute average knee angle from features (indices 30, 31 are knee angles / 180)
        average_knee_angle = float(
            (bottom_features[30] + bottom_features[31]) * 0.5 * 180.0
        )

        # Step 7: Phase detection
        current_phase_features, self.state.previous_phase_values = extract_phase_features_from_lm(
            landmark_array,
            self.state.previous_phase_values,
        )
        self.state.phase_features.append(current_phase_features)
        if self.model_service.phase_loaded and len(self.state.phase_features) >= self.model_service.phase_window:
            phase = self.model_service.predict_phase(np.array(self.state.phase_features))
        else:
            phase = "unknown"
        await self.status_sender.send_phase(self.websocket, self.state, phase)

        # Step 8: Detect bottom event (eccentric -> concentric transition + depth gate)
        event_frame = None
        if phase == "concentric" and self.state.prev_phase == "eccentric":
            if self.frame_index - self.state.last_bottom_event_frame >= self.min_gap:
                if average_knee_angle <= self.gate_knee_angle:
                    event_frame = self.frame_index
                    self.state.last_bottom_event_frame = self.frame_index
                    if self.debug:
                        print(
                            "[PHASE] Transition eccentric→concentric | "
                            f"knee={average_knee_angle:.1f}° → BOTTOM EVENT"
                        )
                else:
                    if self.debug:
                        print(
                            "[GATE] Ignored trigger "
                            f"(knee={average_knee_angle:.1f}° > {self.gate_knee_angle}°)"
                        )
        self.state.prev_phase = phase

        # Add to history
        self.state.history.append((self.frame_index, bottom_features))

        # Step 9: Bottom prediction (immediate)
        if (
            event_frame is not None
            and self.model_service.bottom_loaded
            and self.model_service.bottom_T is not None
        ):
            await self._predict_and_send_bottom(event_frame, phase)

        # Step 10: Bottom prediction (pending — waiting for post-frames)
        await self._resolve_pending_bottom_prediction(phase)

        self.frame_index += 1
