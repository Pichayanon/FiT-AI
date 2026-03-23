from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import mediapipe as mp
from fastapi import WebSocket

from shared.base_session import BaseWebSocketSession
from shared.frame_decoder import FrameDecoder
from shared.frame_quality import FrameQuality
from shared.math_utils import angle_from_points
from shared.phase_bottom_session import PhaseBottomSessionMixin
from shared.phase_bottom_state import PhaseBottomStreamState
from shared.status_sender import StatusSender
from shared.tcn_model_service import PhaseAwareTCNModelService
from shared.front_view_gate_dynamic import FrontViewGateDynamic

from squat.features import (
    BOTTOM_FEATURE_DIM,
    STAND_FEATURE_DIM,
    L_HIP,
    R_HIP,
    L_KNE,
    R_KNE,
    L_ANK,
    R_ANK,
    extract_phase_features,
    extract_stand_features,
    extract_bottom_features,
)


class SquatModelService(PhaseAwareTCNModelService):
    def __init__(
        self,
        bottom_path: str,
        stand_path: str,
        phase_path: Optional[str] = None,
    ) -> None:
        super().__init__(
            bottom_path,
            stand_path=stand_path,
            phase_path=phase_path,
        )


@dataclass
class StreamState(PhaseBottomStreamState):
    last_phase: str = "stand"

    stand_streak: int = 0
    previous_knee_angle: Optional[float] = None
    last_stand_prediction_frame: int = -(10**9)
    stand_checked_once: bool = False
    stand_ok: bool = False

    last_sent_stand_label: str = ""

    history: deque = field(default_factory=deque)
    phase_features: deque = field(default_factory=lambda: deque(maxlen=35))


class SquatWebSocketSession(PhaseBottomSessionMixin, BaseWebSocketSession):
    def __init__(
        self,
        websocket: WebSocket,
        model_service: SquatModelService,
        gate: FrontViewGateDynamic,
        status_sender: StatusSender,
        ready_streak_n: int,
        debug: bool,
        bottom_feature_dim: int,
        stand_feature_dim: int,
        stand_ok_labels: set,
        pre_frames: int,
        post_frames: int,
        min_gap: int,
        phase_decision_mode: str,
        stand_knee_angle_deg_th: float,
        stand_knee_delta_max_deg: float,
        stand_min_streak: int,
        stand_pred_cooldown: int,
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
        self.stand_feature_dim = stand_feature_dim
        self.stand_ok_labels = stand_ok_labels
        self.pre_frames = pre_frames
        self.post_frames = post_frames
        self.min_gap = min_gap
        self.phase_decision_mode = phase_decision_mode
        self.stand_knee_angle_deg_th = stand_knee_angle_deg_th
        self.stand_knee_delta_max_deg = stand_knee_delta_max_deg
        self.stand_min_streak = stand_min_streak
        self.stand_pred_cooldown = stand_pred_cooldown
        self.dark_adjust_seconds = dark_adjust_seconds
        self.dark_brightness_th = dark_brightness_th
        self.goal_good_reps = goal_good_reps

        self.state = StreamState()
        self.frame_index = 0

        self.pose_module = mp.solutions.pose
        self.pose = self.pose_module.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=mp_min_det_conf,
            min_tracking_confidence=mp_min_track_conf,
        )

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
        if (
            self.model_service.stand_loaded
            and self.stand_feature_dim != self.model_service.stand_in_dim
        ):
            await self.status_sender.send_info(
                self.websocket,
                f"WARNING: Stand model in_dim={self.model_service.stand_in_dim} "
                f"but extractor gives dim={self.stand_feature_dim}",
            )
        await self.status_sender.send_info(
            self.websocket,
            "WebSocket connected",
            {
                "stand_feature_dim": self.stand_feature_dim,
                "bottom_feature_dim": self.bottom_feature_dim,
                "front_gate": {
                    "needed_streak": self.ready_streak_n,
                },
                "phase_decision_mode": self.phase_decision_mode,
                "stand_once_only": False,
                "stand_ok_labels": list(self.stand_ok_labels),
            },
        )

    def _create_state(self) -> StreamState:
        return StreamState(
            history=deque(maxlen=self.pre_frames + self.post_frames + 240)
        )

    async def _on_start(self) -> None:
        self.frame_index = 0

    def _stop_extra(self) -> Dict[str, Any]:
        return {
            "reps": {
                "total": int(self.state.total_reps),
                "correct": int(self.state.good_reps),
                "incorrect": int(self.state.bad_reps),
                "goal_correct": int(self.goal_good_reps),
            },
            "stand_ok": bool(self.state.stand_ok),
        }

    def _reset_buffers(self) -> None:
        self._reset_phase_bottom_state()
        self.state.last_sent_stand_label = ""

    def _after_full_buffer_reset(self) -> None:
        self.state.stand_streak = 0
        self.state.previous_knee_angle = None

    def _after_ready_transition(self) -> None:
        self.state.previous_knee_angle = None

    def _bottom_feature_from_history_record(self, record: Any) -> np.ndarray:
        return record[2]

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
        front_view_ok, gate_debug = self.gate.evaluate(landmarks)
        self.state.last_gate_debug = gate_debug

        if not front_view_ok:
            self._reset_buffers()
            await self._send_waiting_status(
                too_dark=too_dark,
                brightness_mean=brightness_mean,
                gate_debug=gate_debug,
                reason="front_gate_not_ok",
            )
            self.frame_index += 1
            return

        if not await self._advance_ready_streak(
            gate_debug=gate_debug,
            ok_message="Front View OK",
        ):
            self.frame_index += 1
            return

        left_hip_xy = (landmarks[L_HIP].x, landmarks[L_HIP].y)
        right_hip_xy = (landmarks[R_HIP].x, landmarks[R_HIP].y)
        left_knee_xy = (landmarks[L_KNE].x, landmarks[L_KNE].y)
        right_knee_xy = (landmarks[R_KNE].x, landmarks[R_KNE].y)
        left_ankle_xy = (landmarks[L_ANK].x, landmarks[L_ANK].y)
        right_ankle_xy = (landmarks[R_ANK].x, landmarks[R_ANK].y)
        left_knee_angle = angle_from_points(left_hip_xy, left_knee_xy, left_ankle_xy)
        right_knee_angle = angle_from_points(
            right_hip_xy,
            right_knee_xy,
            right_ankle_xy,
        )
        knee_angle = float((left_knee_angle + right_knee_angle) * 0.5)

        is_standing = knee_angle >= self.stand_knee_angle_deg_th
        self.state.previous_knee_angle = knee_angle
        if is_standing:
            self.state.stand_streak += 1
        else:
            self.state.stand_streak = 0

        stand_features = extract_stand_features(landmarks)
        bottom_features = extract_bottom_features(landmarks)

        self.state.phase_features.append(extract_phase_features(landmarks))
        if (
            self.model_service.phase_loaded
            and len(self.state.phase_features) >= self.model_service.phase_window
        ):
            phase = self.model_service.predict_phase(
                np.array(self.state.phase_features),
                decision_mode=self.phase_decision_mode,
            )
        else:
            phase = "unknown"
        await self.status_sender.send_phase(self.websocket, self.state, phase)

        event_frame = None
        if phase == "concentric" and self.state.prev_phase == "eccentric":
            if self.frame_index - self.state.last_bottom_event_frame >= self.min_gap:
                event_frame = self.frame_index
                self.state.last_bottom_event_frame = self.frame_index
        self.state.prev_phase = phase

        self.state.history.append(
            (
                self.frame_index,
                stand_features,
                bottom_features,
            )
        )

        if (
            event_frame is not None
            and self.model_service.bottom_loaded
            and self.model_service.bottom_T is not None
        ):
            await self._predict_and_send_bottom(event_frame, phase)

        await self._resolve_pending_bottom_prediction(phase)

        stand_win_frames = self.pre_frames + self.post_frames + 1
        if (
            (self.state.total_reps == 0)
            and self.model_service.stand_loaded
            and self.model_service.stand_T is not None
            and is_standing
            and (self.state.stand_streak >= self.stand_min_streak)
            and (len(self.state.history) >= stand_win_frames)
            and (
                self.frame_index - self.state.last_stand_prediction_frame
                >= self.stand_pred_cooldown
            )
        ):
            recent_history = list(self.state.history)[-stand_win_frames:]
            stand_window_features = np.stack(
                [record[1] for record in recent_history],
                axis=0,
            ).astype(np.float32)
            pred_label, prediction_confidence, _ = self.model_service.predict_stand(
                stand_window_features
            )
            self.state.last_stand_prediction_frame = self.frame_index
            is_ok = pred_label in self.stand_ok_labels
            self.state.stand_ok = bool(is_ok)

            payload = {
                "type": "result",
                "mode": "stand",
                "prediction": pred_label,
                "confidence": round(prediction_confidence, 3),
                "session_id": self.state.session_id,
                "frame_i": int(self.frame_index),
                "T": int(self.model_service.stand_T),
                "feature_dim": self.stand_feature_dim,
                "stand_ok": bool(is_ok),
            }
            if self.debug:
                print("[PRED-STAND]", payload)
            await self.websocket.send_text(json.dumps(payload))

        self.frame_index += 1
