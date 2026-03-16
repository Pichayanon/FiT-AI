"""
Squat WebSocket session handler.

Manages a single squat streaming session including pose detection,
front-view gating, phase detection, bottom-event TCN classification,
and standing posture TCN assessment.
"""

from __future__ import annotations

import json
import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import mediapipe as mp
from fastapi import WebSocket

from shared.base_session import BaseWebSocketSession
from shared.frame_decoder import FrameDecoder
from shared.json_utils import parse_json
from shared.math_utils import angle_3pts
from shared.status_sender import StatusSender
from shared.tcn_service import load_tcn, load_phase_tcn, tcn_predict
from shared.front_view_gate_dynamic import FrontViewGateDynamic

from squat.features import (
    BOTTOM_FEATURE_DIM,
    STAND_FEATURE_DIM,
    KEY_JOINTS,
    L_SHO, R_SHO, L_HIP, R_HIP, L_KNE, R_KNE, L_ANK, R_ANK,
    extract_phase_features,
    extract_stand_features,
    extract_bottom_features,
)

_PHASE_LABELS: Dict[int, str] = {0: "eccentric", 1: "concentric"}


# ---------------------------------------------------------------
# Squat Model Service (holds bottom + stand + phase models)
# ---------------------------------------------------------------

class SquatModelService:
    """Manages multiple TCN models for squat analysis."""

    def __init__(
        self, bottom_path: str, stand_path: str, phase_path: Optional[str] = None
    ) -> None:
        self.bottom_model, self.bottom_T, self.inv_labels_bottom, self.bottom_in_dim = load_tcn(bottom_path)
        self.stand_model, self.stand_T, self.inv_labels_stand, self.stand_in_dim = load_tcn(stand_path)
        self.phase_model = None
        self.phase_window = None
        self.phase_in_dim = None
        if phase_path and os.path.isfile(phase_path):
            self.phase_model, self.phase_window, self.phase_in_dim = load_phase_tcn(phase_path)

    @property
    def bottom_loaded(self) -> bool:
        return self.bottom_model is not None

    @property
    def stand_loaded(self) -> bool:
        return self.stand_model is not None

    @property
    def phase_loaded(self) -> bool:
        return self.phase_model is not None and self.phase_window is not None

    def predict_bottom(self, x_win: np.ndarray) -> Tuple[str, float, np.ndarray]:
        """Run bottom TCN prediction. Returns (label, confidence, probs)."""
        if self.bottom_model is None or self.bottom_T is None:
            return "unknown", 0.0, np.array([])
        return tcn_predict(
            self.bottom_model, self.inv_labels_bottom, int(self.bottom_T),
            x_win.astype(np.float32),
        )

    def predict_stand(self, x_win: np.ndarray) -> Tuple[str, float, np.ndarray]:
        """Run stand TCN prediction. Returns (label, confidence, probs)."""
        if self.stand_model is None or self.stand_T is None:
            return "unknown", 0.0, np.array([])
        return tcn_predict(
            self.stand_model, self.inv_labels_stand, int(self.stand_T),
            x_win.astype(np.float32),
        )

    def predict_phase(self, x_win: np.ndarray) -> str:
        """Run phase TCN prediction. Returns 'eccentric', 'concentric', or 'unknown'."""
        if self.phase_model is None or self.phase_window is None or x_win.shape[0] < self.phase_window:
            return "unknown"
        w = int(self.phase_window)
        x = x_win[-w:].astype(np.float32)
        xt = torch.from_numpy(x).unsqueeze(0)  # (1, W, 10)
        with torch.no_grad():
            logits = self.phase_model(xt)  # (1, W, 2)
            last_logits = logits[0, -1, :]
            pred_id = int(torch.argmax(last_logits).item())
        return _PHASE_LABELS.get(pred_id, "unknown")


# ---------------------------------------------------------------
# Rep counter
# ---------------------------------------------------------------

def update_rep_counter(st: StreamState, event_i: int, pred_label: str) -> None:
    """Count a rep once per event. Good rep: label starts with 'good'."""
    if event_i == st.last_counted_event_i:
        return
    st.last_counted_event_i = event_i
    st.total_reps += 1
    if pred_label.startswith("good"):
        st.good_reps += 1
    else:
        st.bad_reps += 1


# ---------------------------------------------------------------
# Stream State
# ---------------------------------------------------------------

@dataclass
class StreamState:
    """Session-level state for a squat streaming WebSocket connection."""

    started: bool = False
    session_id: str = ""

    # Gate
    ready: bool = False
    ready_streak: int = 0
    last_gate_debug: Dict[str, Any] = field(default_factory=dict)

    # Status/phase throttles
    status_tick: int = 0
    phase_tick: int = 0
    last_status: str = ""
    last_phase: str = "stand"

    # Standing predict state
    stand_streak: int = 0
    prev_knee_raw: Optional[float] = None
    last_stand_pred_i: int = -10**9
    last_stand_pred_label: str = ""
    last_stand_pred_conf: Optional[float] = None
    stand_checked_once: bool = False
    stand_ok: bool = False

    # Rep counting
    total_reps: int = 0
    good_reps: int = 0
    bad_reps: int = 0
    last_counted_event_i: int = -10**9

    # De-dup sending
    last_sent_bottom_event_i: int = -10**9
    last_sent_stand_label: str = ""

    # History: (i, stand_feat, bottom_feat, frame_bgr, knee_raw, knee_ema)
    # Note: PRE_FRAMES and POST_FRAMES are defined in squat_streaming.py
    hist: deque = field(default_factory=lambda: deque(maxlen=5 + 5 + 240))

    # Phase TCN buffer
    phase_feat_buffer: deque = field(default_factory=lambda: deque(maxlen=35))
    prev_phase: str = ""
    last_phase_bottom_i: int = -10**9

    # Pending bottom event
    pending: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------
# Squat WebSocket Session
# ---------------------------------------------------------------

class SquatWebSocketSession(BaseWebSocketSession):
    """Handle a single squat WebSocket streaming session."""

    def __init__(
        self,
        websocket: WebSocket,
        model_svc: SquatModelService,
        gate: FrontViewGateDynamic,
        status: StatusSender,
        ready_streak_n: int,
        debug: bool,
        # Configuration constants from squat_streaming.py
        bottom_feature_dim: int,
        stand_feature_dim: int,
        stand_ok_labels: set,
        pre_frames: int,
        post_frames: int,
        min_gap: int,
        stand_knee_angle_deg_th: float,
        stand_knee_delta_max_deg: float,
        stand_min_streak: int,
        stand_pred_cooldown: int,
        goal_good_reps: int,
        mp_min_det_conf: float,
        mp_min_track_conf: float,
    ) -> None:
        self.ws = websocket
        self.model_svc = model_svc
        self.gate = gate
        self.status = status
        self.ready_streak_n = ready_streak_n
        self.debug = debug

        # Store configuration
        self.bottom_feature_dim = bottom_feature_dim
        self.stand_feature_dim = stand_feature_dim
        self.stand_ok_labels = stand_ok_labels
        self.pre_frames = pre_frames
        self.post_frames = post_frames
        self.min_gap = min_gap
        self.stand_knee_angle_deg_th = stand_knee_angle_deg_th
        self.stand_knee_delta_max_deg = stand_knee_delta_max_deg
        self.stand_min_streak = stand_min_streak
        self.stand_pred_cooldown = stand_pred_cooldown
        self.goal_good_reps = goal_good_reps

        self.st = StreamState()
        self.frame_i = 0

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
        """Send dimension warnings and welcome message after ws.accept()."""
        if self.model_svc.bottom_loaded and self.bottom_feature_dim != self.model_svc.bottom_in_dim:
            await self.status.send_info(
                self.ws,
                f"WARNING: Bottom model in_dim={self.model_svc.bottom_in_dim} "
                f"but extractor gives dim={self.bottom_feature_dim}",
            )
        if self.model_svc.stand_loaded and self.stand_feature_dim != self.model_svc.stand_in_dim:
            await self.status.send_info(
                self.ws,
                f"WARNING: Stand model in_dim={self.model_svc.stand_in_dim} "
                f"but extractor gives dim={self.stand_feature_dim}",
            )
        await self.status.send_info(
            self.ws,
            "WebSocket connected",
            {
                "stand_feature_dim": self.stand_feature_dim,
                "bottom_feature_dim": self.bottom_feature_dim,
                "front_gate": {
                    "needed_streak": self.ready_streak_n,
                },
                "stand_once_only": False,
                "stand_ok_labels": list(self.stand_ok_labels),
            },
        )

    # ---------------------------------------------------------------
    # State factory and hooks (implement BaseWebSocketSession contract)
    # ---------------------------------------------------------------

    def _create_state(self) -> StreamState:
        """Return a fresh StreamState; base will set started=True and session_id."""
        return StreamState()

    async def _on_start(self) -> None:
        """Reset the frame counter for the new session."""
        self.frame_i = 0

    def _stop_extra(self) -> Dict[str, Any]:
        """Include rep counts and stand result in the stop payload."""
        return {
            "reps": {
                "total": int(self.st.total_reps),
                "correct": int(self.st.good_reps),
                "incorrect": int(self.st.bad_reps),
                "goal_correct": int(self.goal_good_reps),
            },
            "stand_ok": bool(self.st.stand_ok),
        }

    # ---------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------

    def _reset_buffers(self) -> None:
        """Reset all tracking buffers and streaks (used on gate failure or ready transition)."""
        self.st.ready = False
        self.st.ready_streak = 0
        self.st.stand_streak = 0
        self.st.prev_knee_raw = None
        self.st.last_gate_debug = {}
        self.st.hist.clear()
        self.st.phase_feat_buffer.clear()
        self.st.prev_phase = ""
        self.st.pending = None
        self.st.last_sent_bottom_event_i = -10**9
        self.st.last_sent_stand_label = ""

    # ---------------------------------------------------------------
    # Bottom prediction
    # ---------------------------------------------------------------

    async def _predict_and_send_bottom(
        self,
        event_i: int,
        phase: str,
    ) -> None:
        """Run bottom TCN prediction for a given event and send result.

        Extracts the feature window from history, runs prediction,
        updates rep counter, and sends the result payload.
        """
        start = event_i - self.pre_frames
        end = event_i + self.post_frames
        need = self.pre_frames + self.post_frames + 1
        win = [r for r in self.st.hist if start <= r[0] <= end]

        if len(win) < need:
            # Not enough frames yet — set as pending
            self.st.pending = {"event": event_i, "start": start, "end": end}
            return

        self.st.pending = None
        await self.status.send_status(
            self.ws, self.st, "predicting",
            {
                "mode": "bottom",
                "phase": phase,
                "event_i": int(event_i),
                "window_frames": int(len(win)),
                "T": int(self.model_svc.bottom_T),
                "D": self.bottom_feature_dim,
            },
        )

        x_win = np.stack([r[2] for r in win], axis=0).astype(np.float32)
        pred_label, conf, _ = self.model_svc.predict_bottom(x_win)
        is_good = pred_label.startswith("good")
        update_rep_counter(self.st, int(event_i), pred_label)

        payload = {
            "type": "result",
            "mode": "bottom",
            "prediction": pred_label,
            "confidence": round(conf, 3),
            "session_id": self.st.session_id,
            "event_i": int(event_i),
            "window": {"pre": self.pre_frames, "post": self.post_frames},
            "T": int(self.model_svc.bottom_T),
            "feature_dim": self.bottom_feature_dim,
            "reps": {
                "total": int(self.st.total_reps),
                "correct": int(self.st.good_reps),
                "incorrect": int(self.st.bad_reps),
                "goal_correct": int(self.goal_good_reps),
                "is_correct_rep": bool(is_good),
            },
        }
        if self.debug:
            print("[PRED-BOTTOM]", payload)
        if int(payload.get("event_i", -1)) != self.st.last_sent_bottom_event_i:
            self.st.last_sent_bottom_event_i = int(payload.get("event_i", -1))
            await self.ws.send_text(json.dumps(payload))

    # ---------------------------------------------------------------
    # Frame handler
    # ---------------------------------------------------------------

    async def _handle_frame(self, data: Dict[str, Any]) -> None:
        """Process a single video frame through the full squat pipeline."""
        # Step 1: Decode frame
        frame = FrameDecoder.decode_jpeg_base64(data.get("jpeg_b64", ""))
        if frame is None:
            await self.status.send_info(self.ws, "Decode failed")
            return

        # Step 2: Run pose detection
        import cv2
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.pose.process(img_rgb)

        # Step 3: Handle no pose
        if not res.pose_landmarks:
            self._reset_buffers()
            await self.status.send_status(
                self.ws, self.st, "waiting",
                {"ready_streak": 0, "needed_streak": self.ready_streak_n},
            )
            self.frame_i += 1
            return

        # Step 4: Front-view gate check
        lm = res.pose_landmarks.landmark
        ok_front, gate_dbg = self.gate.check(lm)
        self.st.last_gate_debug = gate_dbg

        if not ok_front:
            self._reset_buffers()
            await self.status.send_status(
                self.ws, self.st, "waiting",
                {
                    "ready_streak": 0,
                    "needed_streak": self.ready_streak_n,
                    "gate": gate_dbg if self.debug else None,
                    "reason": "front_gate_not_ok",
                },
            )
            self.frame_i += 1
            return

        # Step 5: Track ready streak
        self.st.ready_streak += 1
        if (not self.st.ready) and (self.st.ready_streak >= self.ready_streak_n):
            self.st.ready = True
            self.st.hist.clear()
            self.st.phase_feat_buffer.clear()
            self.st.prev_phase = ""
            self.st.last_phase_bottom_i = -10**9
            self.st.pending = None
            self.st.prev_knee_raw = None
            self.st.last_sent_bottom_event_i = -10**9
            await self.status.send_info(
                self.ws, "Front View OK",
                {"session_id": self.st.session_id, "gate": gate_dbg if self.debug else None},
            )
            await self.status.send_status(
                self.ws, self.st, "ready",
                {"ready_streak": self.st.ready_streak}, force=True,
            )
        elif not self.st.ready:
            await self.status.send_status(
                self.ws, self.st, "warming_up",
                {
                    "ready_streak": self.st.ready_streak,
                    "needed_streak": self.ready_streak_n,
                    "gate": gate_dbg if self.debug else None,
                },
            )

        if not self.st.ready:
            self.frame_i += 1
            return

        # Step 6: Extract features and compute knee angle
        lhip = (lm[L_HIP].x, lm[L_HIP].y)
        rhip = (lm[R_HIP].x, lm[R_HIP].y)
        lknee = (lm[L_KNE].x, lm[L_KNE].y)
        rknee = (lm[R_KNE].x, lm[R_KNE].y)
        lank = (lm[L_ANK].x, lm[L_ANK].y)
        rank = (lm[R_ANK].x, lm[R_ANK].y)
        knee_l = angle_3pts(lhip, lknee, lank)
        knee_r = angle_3pts(rhip, rknee, rank)
        knee_raw = float((knee_l + knee_r) * 0.5)

        # Stand gate
        knee_delta = abs(knee_raw - self.st.prev_knee_raw) if self.st.prev_knee_raw is not None else 0.0
        is_stand = knee_raw >= self.stand_knee_angle_deg_th
        self.st.prev_knee_raw = knee_raw
        if is_stand:
            self.st.stand_streak += 1
        else:
            self.st.stand_streak = 0
        stand_gate_text = (
            f"STAND_GATE: {'YES' if is_stand else 'no'} "
            f"streak {self.st.stand_streak}/{self.stand_min_streak} "
            f"knee={knee_raw:.1f} d={knee_delta:.1f}"
        )

        stand_feat = extract_stand_features(lm)
        bottom_feat = extract_bottom_features(lm)

        # Step 7: Phase detection
        self.st.phase_feat_buffer.append(extract_phase_features(lm))
        if self.model_svc.phase_loaded and len(self.st.phase_feat_buffer) >= self.model_svc.phase_window:
            phase = self.model_svc.predict_phase(np.array(self.st.phase_feat_buffer))
        else:
            phase = "unknown"
        await self.status.send_phase(self.ws, self.st, phase)

        # Step 8: Detect bottom event (eccentric -> concentric transition)
        event_i = None
        if phase == "concentric" and self.st.prev_phase == "eccentric":
            if self.frame_i - self.st.last_phase_bottom_i >= self.min_gap:
                event_i = self.frame_i
                self.st.last_phase_bottom_i = self.frame_i
        self.st.prev_phase = phase

        # Add to history
        self.st.hist.append((self.frame_i, stand_feat, bottom_feat, frame.copy(), knee_raw, None))

        # Step 9: Bottom prediction (immediate)
        if event_i is not None and self.model_svc.bottom_loaded and self.model_svc.bottom_T is not None:
            await self._predict_and_send_bottom(event_i, phase)

        # Step 10: Bottom prediction (pending — waiting for post-frames)
        if self.st.pending is not None and self.model_svc.bottom_loaded and self.model_svc.bottom_T is not None:
            pending_event = int(self.st.pending["event"])
            start = self.st.pending["start"]
            end = self.st.pending["end"]
            need = self.pre_frames + self.post_frames + 1
            win = [r for r in self.st.hist if start <= r[0] <= end]
            if len(win) >= need:
                self.st.pending = None
                await self._predict_and_send_bottom(pending_event, phase)

        # Step 11: Standing posture prediction (before first rep)
        stand_win_frames = self.pre_frames + self.post_frames + 1
        if (
            (self.st.total_reps == 0)
            and self.model_svc.stand_loaded
            and self.model_svc.stand_T is not None
            and is_stand
            and (self.st.stand_streak >= self.stand_min_streak)
            and (len(self.st.hist) >= stand_win_frames)
            and (self.frame_i - self.st.last_stand_pred_i >= self.stand_pred_cooldown)
        ):
            recent = list(self.st.hist)[-stand_win_frames:]
            x_win = np.stack([r[1] for r in recent], axis=0).astype(np.float32)
            pred_label, conf, _ = self.model_svc.predict_stand(x_win)
            self.st.last_stand_pred_i = self.frame_i
            is_ok = pred_label in self.stand_ok_labels
            self.st.stand_ok = bool(is_ok)

            payload = {
                "type": "result",
                "mode": "stand",
                "prediction": pred_label,
                "confidence": round(conf, 3),
                "session_id": self.st.session_id,
                "frame_i": int(self.frame_i),
                "T": int(self.model_svc.stand_T),
                "feature_dim": self.stand_feature_dim,
                "stand_ok": bool(is_ok),
            }
            if self.debug:
                print("[PRED-STAND]", payload)
            await self.ws.send_text(json.dumps(payload))

        self.frame_i += 1
