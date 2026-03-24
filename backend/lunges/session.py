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
    def __init__(
        self,
        bottom_path: str,
        phase_path: Optional[str] = None,
    ) -> None:
        super().__init__(bottom_path, phase_path=phase_path)


@dataclass
class StreamState(PhaseBottomStreamState):
    last_phase: str = "unknown"

    history: deque = field(default_factory=lambda: deque(maxlen=15 + 15 + 240))
    phase_features: deque = field(default_factory=lambda: deque(maxlen=120))
    previous_phase_values: Optional[Tuple[float, float, float]] = None


class LungeWebSocketSession(PhaseBottomSessionMixin, BaseWebSocketSession):
    def __init__(
        self,
        websocket: WebSocket,
        model_service: LungeModelService,
        gate: SideViewGateDynamic,
        status_sender: StatusSender,
        ready_streak_n: int,
        debug: bool,
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

        self.mp_min_det_conf = mp_min_det_conf
        self.mp_min_track_conf = mp_min_track_conf
        self.pose_module = mp.solutions.pose
        self.pose = None  # created lazily in _on_start()

    async def _on_connected(self) -> None:
        if (
            self.model_service.bottom_loaded
            and self.bottom_feature_dim != self.model_service.bottom_in_dim
        ):
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

    def _create_state(self) -> StreamState:
        return StreamState(
            history=deque(maxlen=self.pre_frames + self.post_frames + 240)
        )

    async def _on_start(self) -> None:
        self.frame_index = 0
        self.smoother = LandmarkSmoother(alpha=0.6)
        try:
            self.pose.close()
        except Exception:
            pass
        self.pose = self.pose_module.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=self.mp_min_det_conf,
            min_tracking_confidence=self.mp_min_track_conf,
        )

    async def _on_stop(self) -> None:
        try:
            self.pose.close()
        except Exception:
            pass

    def _stop_extra(self) -> Dict[str, Any]:
        return {
            "reps": {
                "total": int(self.state.total_reps),
                "correct": int(self.state.good_reps),
                "incorrect": int(self.state.bad_reps),
                "goal_correct": int(self.goal_good_reps),
            },
        }

    def _reset_buffers(self) -> None:
        self._reset_phase_bottom_state()

    def _after_full_buffer_reset(self) -> None:
        self.state.previous_phase_values = None
        self.smoother.reset()

    def _after_ready_transition(self) -> None:
        self.state.previous_phase_values = None

    def _is_good_rep_label(self, pred_label: str) -> bool:
        return pred_label.startswith("good") or pred_label == "correct"

    def _bottom_feature_from_history_record(self, record: Any) -> np.ndarray:
        return record[1]

    async def _handle_frame(self, data: Dict[str, Any]) -> None:
        frame = FrameDecoder.decode_jpeg_base64(data.get("jpeg_b64", ""))
        if frame is None:
            await self.status_sender.send_info(self.websocket, "Decode failed")
            return

        too_dark, brightness_mean = FrameQuality.is_too_dark(
            frame,
            self.dark_brightness_th,
        )
        await self._update_dark_watchdog(too_dark, brightness_mean)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_result = self.pose.process(image_rgb)

        if not pose_result.pose_landmarks:
            self._reset_buffers()
            await self._send_waiting_status(
                too_dark=too_dark,
                brightness_mean=brightness_mean,
            )
            self.frame_index += 1
            return

        landmarks = pose_result.pose_landmarks.landmark

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

        if not await self._advance_ready_streak(
            gate_debug=gate_debug,
            ok_message="Side View OK",
        ):
            self.frame_index += 1
            return

        bottom_features = extract_bottom_features(landmark_array)

        average_knee_angle = float(
            (bottom_features[30] + bottom_features[31]) * 0.5 * 180.0
        )

        (
            current_phase_features,
            self.state.previous_phase_values,
        ) = extract_phase_features_from_lm(
            landmark_array,
            self.state.previous_phase_values,
        )
        self.state.phase_features.append(current_phase_features)
        if (
            self.model_service.phase_loaded
            and len(self.state.phase_features) >= self.model_service.phase_window
        ):
            phase = self.model_service.predict_phase(
                np.array(self.state.phase_features)
            )
        else:
            phase = "unknown"
        await self.status_sender.send_phase(self.websocket, self.state, phase)

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

        self.state.history.append((self.frame_index, bottom_features))

        if (
            event_frame is not None
            and self.model_service.bottom_loaded
            and self.model_service.bottom_T is not None
        ):
            await self._predict_and_send_bottom(event_frame, phase)

        await self._resolve_pending_bottom_prediction(phase)

        self.frame_index += 1
