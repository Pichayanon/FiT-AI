from __future__ import annotations

import json
from typing import Any, Dict, Optional

import cv2
import mediapipe as mp
from fastapi import WebSocket

from shared.base_session import BaseWebSocketSession
from shared.frame_decoder import FrameDecoder
from shared.frame_quality import FrameQuality
from shared.label_mapper import LabelMapper
from shared.side_gate import SideGate
from shared.sklearn_model_service import SklearnModelService
from shared.status_sender import StatusSender


class SideWindowSession(BaseWebSocketSession):
    def __init__(
        self,
        websocket: WebSocket,
        model_service: SklearnModelService,
        gate: SideGate,
        feature_extractor: Any,
        labels: LabelMapper,
        status_sender: StatusSender,
        window_frames: int,
        ready_streak_n: int,
        debug: bool,
        side_mode: str,
        vis_th: float,
        mp_min_det_conf: float,
        mp_min_track_conf: float,
        no_pose_adjust_seconds: float,
        dark_adjust_seconds: float,
        dark_brightness_th: float,
        phase_no_pose: str,
        phase_have_pose: str,
        phase_buffering: str,
        phase_inferencing: str,
        status_send_every_n_frames: int,
    ) -> None:
        self.websocket = websocket
        self.model_service = model_service
        self.gate = gate
        self.feature_extractor = feature_extractor
        self.labels = labels
        self.status_sender = status_sender
        self.window_frames = int(window_frames)
        self.ready_streak_n = int(ready_streak_n)
        self.debug = debug

        self.side_mode = side_mode
        self.vis_th = vis_th
        self.mp_min_det_conf = mp_min_det_conf
        self.mp_min_track_conf = mp_min_track_conf
        self.no_pose_adjust_seconds = no_pose_adjust_seconds
        self.dark_adjust_seconds = dark_adjust_seconds
        self.dark_brightness_th = dark_brightness_th
        self.phase_no_pose = phase_no_pose
        self.phase_have_pose = phase_have_pose
        self.phase_buffering = phase_buffering
        self.phase_inferencing = phase_inferencing
        self.status_send_every_n_frames = status_send_every_n_frames

        self.state = self._create_state()

        self.pose_module = mp.solutions.pose
        self.pose = self.pose_module.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=mp_min_det_conf,
            min_tracking_confidence=mp_min_track_conf,
        )

    async def _on_connected(self) -> None:
        await self.status_sender.send_info(self.websocket, "WebSocket connected")
        print(
            f"[BOOT] side_mode={self.side_mode} VIS_TH={self.vis_th} "
            f"det={self.mp_min_det_conf} track={self.mp_min_track_conf}"
        )
        print(
            f"[BOOT] WINDOW_FRAMES={self.window_frames} "
            f"READY_STREAK_N={self.ready_streak_n}"
        )
        if not self.model_service.loaded:
            await self.status_sender.send_info(
                self.websocket,
                "Model not loaded (check MODEL_PATH)",
            )

    @property
    def _initial_status(self) -> str:
        return self.phase_no_pose

    async def _on_stop(self) -> None:
        self._reset_gate_and_buffers(reset_watchdog=True)

    def _reset_gate_specific_fields(self) -> None:
        pass

    def _reset_gate_and_buffers(self, reset_watchdog: bool) -> None:
        self.state.frame_features.clear()
        self.state.ready = False
        self.state.ready_streak = 0
        self.state.chosen_side = None

        self.state.last_sent_label = ""
        self.state.last_sent_confidence = None
        self.state.last_prediction_label = ""
        self.state.last_prediction_confidence = None
        self.state.frame_count = 0
        self._reset_gate_specific_fields()

        if reset_watchdog:
            self.state.no_pose_since = None
            self.state.no_pose_alerted = False
            self.state.dark_since = None
            self.state.dark_alerted = False

    def _window_payload(
        self,
        chosen_side: Optional[str],
        window_fill: int,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "chosen_side": chosen_side,
            "window_fill": window_fill,
            "window_size": self.window_frames,
        }
        if extra:
            payload.update(extra)
        return payload

    def _pre_ready_pose_payload(
        self,
        landmarks: Any,
        chosen_side: str,
    ) -> Optional[Dict[str, Any]]:
        return None

    def _on_missing_features(self) -> None:
        pass

    def _pre_buffer_feature_payload(
        self,
        frame_features: Any,
    ) -> Optional[Dict[str, Any]]:
        return None

    async def _handle_frame(self, data: Dict[str, Any]) -> None:
        if not self.state.started:
            return

        self.state.frame_count += 1

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

        gate_debug: Dict[str, Any] = {}
        chosen_side: Optional[str] = None

        if pose_result.pose_landmarks:
            landmarks = pose_result.pose_landmarks.landmark
            if self.state.ready and self.state.chosen_side is not None:
                chosen_side = self.state.chosen_side
            else:
                chosen_side, gate_debug = self.gate.choose_best_side(landmarks)

            if chosen_side is not None:
                pose_gate_payload = self._pre_ready_pose_payload(
                    landmarks,
                    chosen_side,
                )
                if pose_gate_payload is not None:
                    self._reset_gate_and_buffers(reset_watchdog=False)
                    await self.status_sender.send_status(
                        self.websocket,
                        self.state,
                        self.phase_have_pose,
                        pose_gate_payload,
                    )
                    return

        if (not pose_result.pose_landmarks) or (chosen_side is None):
            await self._handle_no_pose(too_dark, brightness_mean, gate_debug)
            return

        self.state.no_pose_since = None
        self.state.no_pose_alerted = False

        self.state.ready_streak += 1
        self.state.chosen_side = chosen_side

        if (not self.state.ready) and (self.state.ready_streak < self.ready_streak_n):
            await self.status_sender.send_status(
                self.websocket,
                self.state,
                self.phase_have_pose,
                self._window_payload(
                    chosen_side,
                    0,
                    {
                        "ready_streak": self.state.ready_streak,
                        "needed_streak": self.ready_streak_n,
                    },
                ),
            )
            return

        if (not self.state.ready) and (self.state.ready_streak >= self.ready_streak_n):
            self.state.ready = True
            self.state.frame_features.clear()
            self.state.last_sent_label = ""
            self.state.last_sent_confidence = None

            print(
                f"[GATE] READY session_id={self.state.session_id} "
                f"side={chosen_side}"
            )
            await self.status_sender.send_info(
                self.websocket,
                "Side View OK",
                {"session_id": self.state.session_id, "side": chosen_side},
            )
            await self.status_sender.send_status(
                self.websocket,
                self.state,
                self.phase_buffering,
                self._window_payload(chosen_side, 0),
                force=True,
            )

        if (
            (not self.model_service.loaded)
            or (not self.state.ready)
            or (self.state.chosen_side is None)
        ):
            await self.status_sender.send_status(
                self.websocket,
                self.state,
                self.phase_buffering,
                self._window_payload(
                    self.state.chosen_side,
                    len(self.state.frame_features),
                ),
            )
            return

        frame_features = self.feature_extractor.extract_features(
            pose_result,
            self.state.chosen_side,
        )

        if frame_features is None:
            self.state.frame_features.clear()
            self._on_missing_features()
            await self.status_sender.send_status(
                self.websocket,
                self.state,
                self.phase_buffering,
                self._window_payload(self.state.chosen_side, 0),
            )
            return

        feature_gate_payload = self._pre_buffer_feature_payload(frame_features)
        if feature_gate_payload is not None:
            await self.status_sender.send_status(
                self.websocket,
                self.state,
                self.phase_have_pose,
                feature_gate_payload,
            )
            return

        self.state.frame_features.append(frame_features)

        if len(self.state.frame_features) < self.window_frames:
            await self.status_sender.send_status(
                self.websocket,
                self.state,
                self.phase_buffering,
                self._window_payload(
                    self.state.chosen_side,
                    len(self.state.frame_features),
                ),
            )
            return

        await self.status_sender.send_status(
            self.websocket,
            self.state,
            self.phase_inferencing,
            self._window_payload(
                self.state.chosen_side,
                len(self.state.frame_features),
            ),
        )

        window_features = self.state.frame_features[-self.window_frames :]
        aggregated_features = self.feature_extractor.aggregate_window(window_features)

        predicted_id, prediction_confidence = self.model_service.predict(
            aggregated_features
        )
        pred_label = self.labels.label_of(predicted_id)

        self.state.last_prediction_label = pred_label
        self.state.last_prediction_confidence = prediction_confidence

        payload: Dict[str, Any] = {
            "type": "result",
            "prediction": pred_label,
            "confidence": (
                round(prediction_confidence, 3)
                if prediction_confidence is not None
                else None
            ),
            "window": self.window_frames,
            "session_id": self.state.session_id,
            "side": self.state.chosen_side,
        }
        if self.debug:
            print(f"[PRED] {payload}")
        await self.websocket.send_text(json.dumps(payload))

        if len(self.state.frame_features) > (self.window_frames + 60):
            self.state.frame_features = self.state.frame_features[
                -(self.window_frames + 60) :
            ]
