"""
Lunge WebSocket session handler.

Manages a single lunge streaming session including pose detection,
side-view gating, phase detection, bottom-event TCN classification.
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
import torch.nn as nn
import cv2
import mediapipe as mp
from fastapi import WebSocket

from shared.base_session import BaseWebSocketSession
from shared.frame_decoder import FrameDecoder
from shared.json_utils import parse_json
from shared.math_utils import angle_3pts
from shared.side_view_gate_dynamic import SideViewGateDynamic
from shared.status_sender import StatusSender
from shared.tcn_service import load_tcn, tcn_predict

from lunges.features import (
    BOTTOM_FEATURE_DIM,
    L_EAR, R_EAR, L_SHO, R_SHO, L_HIP, R_HIP, L_KNE, R_KNE, L_ANK, R_ANK,
    extract_bottom_features,
    extract_phase_features as extract_phase_features_from_lm,
    LandmarkSmoother,
)


_PHASE_LABELS: Dict[int, str] = {0: "eccentric", 1: "concentric"}


# ---------------------------------------------------------------
# LungePhaseTCN (kept separate for checkpoint compatibility)
# ---------------------------------------------------------------

class LungePhaseTCN(nn.Module):
    """Lunge-specific phase TCN architecture (matches train_lunge_phase.py).

    Uses an inner _Block class with a different forward pass than the
    shared PhaseTCN. Kept separate to guarantee checkpoint compatibility.
    """

    def __init__(self, in_dim: int = 9, num_classes: int = 2) -> None:
        super().__init__()

        class _Block(nn.Module):
            def __init__(self, in_ch: int, out_ch: int, k: int = 3, d: int = 1):
                super().__init__()
                pad = (k - 1) * d
                self.conv1 = nn.Conv1d(in_ch, out_ch, k, padding=pad, dilation=d)
                self.conv2 = nn.Conv1d(out_ch, out_ch, k, padding=pad, dilation=d)
                self.relu = nn.ReLU()
                self.down = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                y = self.relu(self.conv1(x))
                y = self.relu(self.conv2(y))
                if self.down:
                    x = self.down(x)
                return y[..., : x.size(-1)] + x

        self.tcn = nn.Sequential(
            _Block(in_dim, 64, d=1),
            _Block(64, 64, d=2),
            _Block(64, 64, d=4),
        )
        self.fc = nn.Conv1d(64, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)  # (B, W, F) -> (B, F, W)
        x = self.tcn(x)
        x = self.fc(x)
        return x.transpose(1, 2)  # (B, W, C)


# ---------------------------------------------------------------
# Lunge Model Service (holds bottom + phase models)
# ---------------------------------------------------------------

class LungeModelService:
    """Manages TCN models for lunge analysis."""

    def __init__(
        self, bottom_path: str, phase_path: Optional[str] = None
    ) -> None:
        # Load bottom model via shared service
        self.bottom_model, self.bottom_T, self.inv_labels_bottom, self.bottom_in_dim = load_tcn(bottom_path)

        # Load phase model (lunge-specific architecture)
        self.phase_model: Optional[LungePhaseTCN] = None
        self.phase_window: Optional[int] = None
        self.phase_in_dim: Optional[int] = None
        if phase_path and os.path.isfile(phase_path):
            self._load_phase(phase_path)

    def _load_phase(self, path: str) -> None:
        """Load phase TCN with lunge-specific architecture."""
        try:
            ckpt = torch.load(path, map_location="cpu")
            in_dim = int(ckpt.get("in_dim", 9))
            num_classes = int(ckpt.get("num_classes", 2))
            self.phase_window = int(ckpt.get("window", 30))
            self.phase_in_dim = in_dim
            self.phase_model = LungePhaseTCN(
                in_dim=in_dim, num_classes=num_classes
            )
            self.phase_model.load_state_dict(ckpt["state_dict"])
            self.phase_model.eval()
            print(
                f"[MODEL] Lunge phase TCN loaded: {path} "
                f"in_dim={in_dim} window={self.phase_window} num_classes={num_classes}"
            )
        except Exception as e:  # pylint: disable=broad-except
            print(f"[MODEL] Cannot load lunge phase TCN: {path} err={e}")

    @property
    def bottom_loaded(self) -> bool:
        return self.bottom_model is not None

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

    def predict_phase(self, x_win: np.ndarray) -> str:
        """Run phase TCN prediction. Returns 'eccentric', 'concentric', or 'unknown'."""
        if self.phase_model is None or self.phase_window is None or x_win.shape[0] < self.phase_window:
            return "unknown"
        w = int(self.phase_window)
        x = x_win[-w:].astype(np.float32)
        xt = torch.from_numpy(x).unsqueeze(0)  # (1, W, 9)
        with torch.no_grad():
            logits = self.phase_model(xt)  # (1, W, 2)
            last_logits = logits[0, -1, :]
            pred_id = int(torch.argmax(last_logits).item())
        return _PHASE_LABELS.get(pred_id, "unknown")


# ---------------------------------------------------------------
# Rep counter
# ---------------------------------------------------------------

def update_rep_counter(st: StreamState, event_i: int, pred_label: str) -> None:
    """Count a rep once per event. Good rep: label starts with 'good' or == 'correct'."""
    if event_i == st.last_counted_event_i:
        return
    st.last_counted_event_i = event_i
    st.total_reps += 1
    if pred_label.startswith("good") or pred_label == "correct":
        st.good_reps += 1
    else:
        st.bad_reps += 1


# ---------------------------------------------------------------
# Stream State
# ---------------------------------------------------------------

@dataclass
class StreamState:
    """Session-level state for a lunge streaming WebSocket connection."""

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
    last_phase: str = "unknown"

    # Rep counting
    total_reps: int = 0
    good_reps: int = 0
    bad_reps: int = 0
    last_counted_event_i: int = -10**9

    # De-dup sending
    last_sent_bottom_event_i: int = -10**9

    # History: (i, bottom_feat)
    # Note: PRE_FRAMES and POST_FRAMES are defined in lunges_streaming.py
    hist: deque = field(default_factory=lambda: deque(maxlen=15 + 15 + 240))

    # Phase TCN buffer
    phase_feat_buffer: deque = field(default_factory=lambda: deque(maxlen=120))
    prev_phase: str = ""
    last_phase_bottom_i: int = -10**9
    prev_phase_vals: Optional[Tuple[float, float, float]] = None

    # Pending bottom event
    pending: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------
# Lunge WebSocket Session
# ---------------------------------------------------------------

class LungeWebSocketSession(BaseWebSocketSession):
    """Handle a single lunge WebSocket streaming session."""

    def __init__(
        self,
        websocket: WebSocket,
        model_svc: LungeModelService,
        gate: SideViewGateDynamic,
        status: StatusSender,
        ready_streak_n: int,
        debug: bool,
        # Configuration constants
        bottom_feature_dim: int,
        pre_frames: int,
        post_frames: int,
        min_gap: int,
        gate_knee_angle: float,
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
        self.pre_frames = pre_frames
        self.post_frames = post_frames
        self.min_gap = min_gap
        self.gate_knee_angle = gate_knee_angle
        self.goal_good_reps = goal_good_reps

        self.st = StreamState()
        self.frame_i = 0
        self.smoother = LandmarkSmoother(alpha=0.6)

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
        await self.status.send_info(
            self.ws,
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
        return StreamState()

    async def _on_start(self) -> None:
        """Reset frame counter and landmark smoother for the new session."""
        self.frame_i = 0
        self.smoother = LandmarkSmoother(alpha=0.6)

    def _stop_extra(self) -> Dict[str, Any]:
        """Include rep counts in the stop payload."""
        return {
            "reps": {
                "total": int(self.st.total_reps),
                "correct": int(self.st.good_reps),
                "incorrect": int(self.st.bad_reps),
                "goal_correct": int(self.goal_good_reps),
            },
        }

    # ---------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------

    def _reset_buffers(self) -> None:
        """Reset all tracking buffers and streaks (used on gate failure or ready transition)."""
        self.st.ready = False
        self.st.ready_streak = 0
        self.st.last_gate_debug = {}
        self.st.hist.clear()
        self.st.phase_feat_buffer.clear()
        self.st.prev_phase = ""
        self.st.pending = None
        self.st.last_sent_bottom_event_i = -10**9
        self.st.prev_phase_vals = None
        self.smoother.reset()

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

        x_win = np.stack([r[1] for r in win], axis=0).astype(np.float32)
        pred_label, conf, _ = self.model_svc.predict_bottom(x_win)
        is_good = pred_label.startswith("good") or pred_label == "correct"
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
        """Process a single video frame through the full lunge pipeline."""
        # Step 1: Decode frame
        frame = FrameDecoder.decode_jpeg_base64(data.get("jpeg_b64", ""))
        if frame is None:
            await self.status.send_info(self.ws, "Decode failed")
            return

        # Step 2: Run pose detection
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.pose.process(img_rgb)

        # Step 3: Handle no pose
        if not res.pose_landmarks:
            self._reset_buffers()
            self.smoother.reset()
            await self.status.send_status(
                self.ws, self.st, "waiting",
                {"ready_streak": 0, "needed_streak": self.ready_streak_n},
            )
            self.frame_i += 1
            return

        # Step 4: Side-view gate check
        lm = res.pose_landmarks.landmark

        # Build numpy array and apply smoothing
        lm_arr = np.zeros((33, 4), dtype=np.float32)
        for idx in range(33):
            lm_arr[idx] = [lm[idx].x, lm[idx].y, lm[idx].z, lm[idx].visibility]

        lm_arr = self.smoother.update(lm_arr)

        ok_side, gate_dbg = self.gate.check(lm_arr)
        self.st.last_gate_debug = gate_dbg

        if not ok_side:
            self._reset_buffers()
            await self.status.send_status(
                self.ws, self.st, "waiting",
                {
                    "ready_streak": 0,
                    "needed_streak": self.ready_streak_n,
                    "gate": gate_dbg if self.debug else None,
                    "reason": gate_dbg.get("reason", "side_gate_not_ok"),
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
            self.st.prev_phase_vals = None
            self.st.last_sent_bottom_event_i = -10**9
            await self.status.send_info(
                self.ws, "Side View OK",
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

        # Step 6: Extract features
        feats = extract_bottom_features(lm_arr[np.newaxis, ...])[0]

        # Compute average knee angle from features (indices 30, 31 are knee angles / 180)
        knee_avg = float((feats[30] + feats[31]) * 0.5 * 180.0)

        # Step 7: Phase detection
        phase_feats, self.st.prev_phase_vals = extract_phase_features_from_lm(
            lm_arr, self.st.prev_phase_vals
        )
        self.st.phase_feat_buffer.append(phase_feats)
        if self.model_svc.phase_loaded and len(self.st.phase_feat_buffer) >= self.model_svc.phase_window:
            phase = self.model_svc.predict_phase(np.array(self.st.phase_feat_buffer))
        else:
            phase = "unknown"
        await self.status.send_phase(self.ws, self.st, phase)

        # Step 8: Detect bottom event (eccentric -> concentric transition + depth gate)
        event_i = None
        if phase == "concentric" and self.st.prev_phase == "eccentric":
            if self.frame_i - self.st.last_phase_bottom_i >= self.min_gap:
                # Depth gate: only trigger if knee angle is low enough
                if knee_avg <= self.gate_knee_angle:
                    event_i = self.frame_i
                    self.st.last_phase_bottom_i = self.frame_i
                    if self.debug:
                        print(f"[PHASE] Transition eccentric→concentric | knee={knee_avg:.1f}° → BOTTOM EVENT")
                else:
                    if self.debug:
                        print(f"[GATE] Ignored trigger (knee={knee_avg:.1f}° > {self.gate_knee_angle}°)")
        self.st.prev_phase = phase

        # Add to history
        self.st.hist.append((self.frame_i, feats))

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

        self.frame_i += 1
