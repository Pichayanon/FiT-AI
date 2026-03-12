"""
Plank WebSocket session handler.

Manages a single plank streaming session including pose detection,
side-view gating, feature extraction, model inference, and result
sending over WebSocket.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, Optional

import cv2
import mediapipe as mp
import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

from shared.frame_decoder import FrameDecoder
from shared.frame_quality import FrameQuality
from shared.json_utils import parse_json
from shared.label_mapper import LabelMapper
from shared.sklearn_model_service import SklearnModelService
from shared.side_gate import SideGate
from shared.status_sender import StatusSender

from .feature_extractor import FeatureExtractor
from .stream_state import StreamState


class PlankWebSocketSession:
    """Handle a single plank WebSocket streaming session.

    Orchestrates the full pipeline: frame decoding, quality checks,
    pose detection, side-view gating, feature extraction, buffering,
    inference, and result sending.
    """

    def __init__(
        self,
        websocket: WebSocket,
        model_svc: SklearnModelService,
        gate: SideGate,
        feat: FeatureExtractor,
        labels: LabelMapper,
        status: StatusSender,
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
        self.ws = websocket
        self.model_svc = model_svc
        self.gate = gate
        self.feat = feat
        self.labels = labels
        self.status = status
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

        self.st = StreamState()

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=self.mp_min_det_conf,
            min_tracking_confidence=self.mp_min_track_conf,
        )

    # ---------------------------------------------------------------
    # Main loop
    # ---------------------------------------------------------------

    async def run(self) -> None:
        """Main receive loop for the WebSocket session."""
        await self.ws.accept()
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

        try:
            while True:
                msg = await self.ws.receive_text()
                data = parse_json(msg)
                if data is None:
                    await self.status.send_info(self.ws, "Invalid JSON")
                    continue

                msg_type = data.get("type")
                if msg_type == "start":
                    await self._handle_start()
                elif msg_type == "stop":
                    await self._handle_stop()
                elif msg_type == "frame":
                    await self._handle_frame(data)
        except WebSocketDisconnect:
            print(f"[WS] disconnect session_id={self.st.session_id}")
        except Exception as exc:  # pylint: disable=broad-except
            print(f"[WS] error: {exc}")
            try:
                await self.status.send_info(self.ws, f"Server error: {exc}")
            except Exception:  # pylint: disable=broad-except
                pass


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
            self.ws,
            "Start streaming",
            {"session_id": self.st.session_id},
        )
        await self.status.send_status(
            self.ws,
            self.st,
            self.phase_no_pose,
            {"reason": "session_started"},
            force=True,
        )

    async def _handle_stop(self) -> None:
        """Handle session stop message. Reset gate and buffers."""
        print(f"[SESSION] STOP session_id={self.st.session_id}")
        self.st.started = False

        await self.status.send_info(
            self.ws,
            "Stop streaming",
            {"session_id": self.st.session_id},
        )
        await self.status.send_status(
            self.ws,
            self.st,
            self.phase_no_pose,
            {"reason": "session_stopped"},
            force=True,
        )

        self._reset_gate_and_buffers(reset_watchdog=True)

    # ---------------------------------------------------------------
    # State reset
    # ---------------------------------------------------------------

    def _reset_gate_and_buffers(self, reset_watchdog: bool) -> None:
        """Reset gate state, feature buffers, and optionally watchdogs."""
        self.st.feats.clear()
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
        too_dark, brightness_mean = FrameQuality.is_too_dark(
            frame,
            self.dark_brightness_th,
        )

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
                self.ws,
                self.st,
                self.phase_have_pose,
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
            self.st.feats.clear()
            self.st.last_sent_label = ""
            self.st.last_sent_conf = None

            print(
                f"[GATE] READY session_id={self.st.session_id} "
                f"side={chosen_side}"
            )
            await self.status.send_info(
                self.ws,
                "Side View OK",
                {"session_id": self.st.session_id, "side": chosen_side},
            )
            await self.status.send_status(
                self.ws,
                self.st,
                self.phase_buffering,
                {
                    "chosen_side": chosen_side,
                    "window_fill": 0,
                    "window_size": self.window_frames,
                },
                force=True,
            )

        # Model not loaded or not ready: stay in BUFFERING
        if (not self.model_svc.loaded) or (not self.st.ready) or (
            self.st.chosen_side is None
        ):
            await self.status.send_status(
                self.ws,
                self.st,
                self.phase_buffering,
                {
                    "chosen_side": self.st.chosen_side,
                    "window_fill": len(self.st.feats),
                    "window_size": self.window_frames,
                },
            )
            return

        # Step 6: Extract features
        feat_tuple = self.feat.extract_features(res, self.st.chosen_side)

        if feat_tuple is None:
            self.st.feats.clear()
            await self.status.send_status(
                self.ws,
                self.st,
                self.phase_buffering,
                {
                    "chosen_side": self.st.chosen_side,
                    "window_fill": 0,
                    "window_size": self.window_frames,
                },
            )
            return

        self.st.feats.append(feat_tuple)

        # Step 7: Check if enough frames for inference
        if len(self.st.feats) < self.window_frames:
            await self.status.send_status(
                self.ws,
                self.st,
                self.phase_buffering,
                {
                    "chosen_side": self.st.chosen_side,
                    "window_fill": len(self.st.feats),
                    "window_size": self.window_frames,
                },
            )
            return

        # Step 8: Run inference
        await self.status.send_status(
            self.ws,
            self.st,
            self.phase_inferencing,
            {
                "chosen_side": self.st.chosen_side,
                "window_fill": len(self.st.feats),
                "window_size": self.window_frames,
            },
        )

        values = self.st.feats[-self.window_frames:]
        agg_feat = self.feat.aggregate_window(values)

        pred_id, conf = self.model_svc.predict(agg_feat)
        pred_label = self.labels.label_of(pred_id)

        self.st.last_pred_label = pred_label
        self.st.last_pred_conf = conf

        # Step 9: Send result
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

        # Step 10: Bound feature buffer to prevent unbounded growth
        if len(self.st.feats) > (self.window_frames + 60):
            self.st.feats = self.st.feats[-(self.window_frames + 60):]

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
            and (now - self.st.dark_since >= self.dark_adjust_seconds)
        ):
            await self.status.send_info(
                self.ws,
                "Please adjust your lights.",
                {
                    "brightness_mean": round(brightness_mean, 1),
                    "brightness_th": self.dark_brightness_th,
                },
            )
            self.st.dark_alerted = True

        # NO_POSE alert (only if not explained by darkness)
        if (
            (not self.st.no_pose_alerted)
            and (now - self.st.no_pose_since >= self.no_pose_adjust_seconds)
            and (not too_dark)
        ):
            await self.status.send_info(
                self.ws,
                "Adjust your camera to see your full body",
                {
                    "brightness_mean": round(brightness_mean, 1),
                    "brightness_th": self.dark_brightness_th,
                },
            )
            self.st.no_pose_alerted = True

        # Reset gate/buffers but keep watchdog timers running
        self._reset_gate_and_buffers(reset_watchdog=False)

        await self.status.send_status(
            self.ws,
            self.st,
            self.phase_no_pose,
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
