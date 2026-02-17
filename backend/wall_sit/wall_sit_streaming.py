"""
wall_sit_streaming_oop.py (NO OVERLAY / NO RECORDING) - OOP VERSION

Streaming (WebSocket) + SIDE-VIEW gate (one side only)
+ PRINT logs + Status to iOS
+ (Optional) DEDUP RULE: Send {"type":"result"} ONLY when prediction label changes

PHASE (status.state) to iOS:
  - NO_POSE
  - HAVE_POSE
  - BUFFERING
  - INFERENCING

Run:
  python wall_sit_streaming_oop.py --serve

WS protocol (from iOS):
  - {"type":"start"}
  - {"type":"frame","jpeg_b64":"..."}
  - {"type":"stop"}

Server -> iOS:
  - {"type":"status","state":"NO_POSE|HAVE_POSE|BUFFERING|INFERENCING", ...}
  - {"type":"result","prediction":"...", "confidence":..., ...}
  - {"type":"info","message":"..."}  (optional)
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import warnings

warnings.filterwarnings(
    "ignore",
    message=r".*SymbolDatabase\.GetPrototype\(\) is deprecated.*",
    category=UserWarning,
    module=r"google\.protobuf\.symbol_database",
)

import cv2
import joblib
import mediapipe as mp
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware


# -----------------------------
# Config
# -----------------------------
MODEL_PATH = "/Users/pichayanon/Desktop/FiT-AI/backend/wall_sit/models/wall_sit_model.pkl"

LABELS: Dict[int, str] = {
    0: "correct",
    1: "feet_too_close",
    2: "feet_too_far",
    3: "back_of_wall",
    4: "not_deep_enough",
    5: "standing",
}

WINDOW_FRAMES = 15          # frames per window
READY_STREAK_N = 3          # require side landmarks N consecutive frames
VIS_TH = 0.80               # side landmark visibility threshold

DEBUG = True

# MediaPipe confidence
MP_MIN_DET_CONF = 0.80
MP_MIN_TRACK_CONF = 0.80

# Status to iOS (throttle)
STATUS_SEND_EVERY_N_FRAMES = 3

# Choose side mode: "auto" | "left" | "right"
SIDE_MODE = "auto"

# Status phases
PHASE_NO_POSE = "NO_POSE"
PHASE_HAVE_POSE = "HAVE_POSE"
PHASE_BUFFERING = "BUFFERING"
PHASE_INFERENCING = "INFERENCING"

# NO_POSE watchdog seconds
NO_POSE_ADJUST_SECONDS = 5.0

# DARK watchdog (check before NO_POSE message)
DARK_ADJUST_SECONDS = 5.0
# Mean grayscale brightness threshold (0..255). Lower = darker.
DARK_BRIGHTNESS_TH = 55.0


# -----------------------------
# App
# -----------------------------
app = FastAPI(title="FiT-AI WallSit Streaming Backend (OOP)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------
# Stream State
# -----------------------------
@dataclass
class StreamState:
    started: bool = False
    foot_wall_vals: List[float] = field(default_factory=list)
    knee_angles: List[float] = field(default_factory=list)
    torso_alignments: List[float] = field(default_factory=list)
    hip_angles: List[float] = field(default_factory=list)

    # gate
    ready: bool = False
    ready_streak: int = 0
    chosen_side: Optional[str] = None

    # session
    session_id: str = ""

    # status throttle
    last_status: str = ""
    status_tick: int = 0

    # last prediction (for memory/logs)
    last_pred_label: str = ""
    last_pred_conf: Optional[float] = None

    # last sent (for optional dedup)
    last_sent_label: str = ""
    last_sent_conf: Optional[float] = None

    # frame count (misc)
    frame_count: int = 0

    # NO_POSE watchdog
    no_pose_since: Optional[float] = None
    no_pose_alerted: bool = False

    # DARK watchdog (only meaningful when NO_POSE)
    dark_since: Optional[float] = None
    dark_alerted: bool = False


# -----------------------------
# Helper Services
# -----------------------------
class ModelService:
    """Loads a sklearn-like model via joblib and provides predict + proba."""

    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self.model = self._load()

    def _load(self) -> Any:
        try:
            m = joblib.load(self.model_path)
            print(f"[MODEL] Loaded: {self.model_path}")
            return m
        except Exception as e:
            print(f"[MODEL] Cannot load model: {e}")
            return None

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def predict(self, feat: np.ndarray) -> Tuple[int, Optional[float]]:
        """Return (pred_id, confidence or None)."""
        if self.model is None:
            return 0, None

        pred = int(self.model.predict(feat.reshape(1, -1))[0])
        conf: Optional[float] = None
        if hasattr(self.model, "predict_proba"):
            conf = float(self.model.predict_proba(feat.reshape(1, -1))[0][pred])
        return pred, conf


class FrameDecoder:
    """Decode base64 jpeg string into BGR image."""

    @staticmethod
    def decode_jpeg_base64(jpeg_b64: str) -> Optional[np.ndarray]:
        try:
            raw = base64.b64decode(jpeg_b64)
            arr = np.frombuffer(raw, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            return img
        except Exception:
            return None


class FrameQuality:
    """Cheap frame quality checks (brightness)."""

    @staticmethod
    def compute_brightness_mean_bgr(frame_bgr: np.ndarray) -> float:
        """
        Return mean grayscale brightness in range 0..255.
        """
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        return float(np.mean(gray))

    @staticmethod
    def is_too_dark(frame_bgr: np.ndarray, th: float) -> Tuple[bool, float]:
        mean_v = FrameQuality.compute_brightness_mean_bgr(frame_bgr)
        return (mean_v < float(th)), mean_v


class LabelMapper:
    """Map class index -> label string."""

    def __init__(self, labels: Dict[int, str]) -> None:
        self.labels = labels

    def label_of(self, pred_id: int) -> str:
        return self.labels.get(int(pred_id), str(pred_id))


class SideGate:
    """Side-view gate: chooses a side and checks landmark visibility."""

    def __init__(self, mp_pose: Any, side_mode: str, vis_th: float) -> None:
        self.mp_pose = mp_pose
        self.side_mode = side_mode
        self.vis_th = vis_th

        self.SIDE_LM: Dict[str, List[int]] = {
            "left": [
                mp_pose.PoseLandmark.LEFT_SHOULDER,
                mp_pose.PoseLandmark.LEFT_HIP,
                mp_pose.PoseLandmark.LEFT_KNEE,
                mp_pose.PoseLandmark.LEFT_ANKLE,
                mp_pose.PoseLandmark.LEFT_FOOT_INDEX,
            ],
            "right": [
                mp_pose.PoseLandmark.RIGHT_SHOULDER,
                mp_pose.PoseLandmark.RIGHT_HIP,
                mp_pose.PoseLandmark.RIGHT_KNEE,
                mp_pose.PoseLandmark.RIGHT_ANKLE,
                mp_pose.PoseLandmark.RIGHT_FOOT_INDEX,
            ],
        }

        self.REQ_LM_LABELS: Dict[int, str] = {
            mp_pose.PoseLandmark.LEFT_SHOULDER: "L_SHO",
            mp_pose.PoseLandmark.LEFT_HIP: "L_HIP",
            mp_pose.PoseLandmark.LEFT_KNEE: "L_KNEE",
            mp_pose.PoseLandmark.LEFT_ANKLE: "L_ANK",
            mp_pose.PoseLandmark.LEFT_FOOT_INDEX: "L_FOOT",
            mp_pose.PoseLandmark.RIGHT_SHOULDER: "R_SHO",
            mp_pose.PoseLandmark.RIGHT_HIP: "R_HIP",
            mp_pose.PoseLandmark.RIGHT_KNEE: "R_KNEE",
            mp_pose.PoseLandmark.RIGHT_ANKLE: "R_ANK",
            mp_pose.PoseLandmark.RIGHT_FOOT_INDEX: "R_FOOT",
        }

    def side_score(self, lm: List[Any], side: str) -> Tuple[bool, float, Dict[str, float]]:
        """ok = all side landmarks visibility >= vis_th"""
        vis_map: Dict[str, float] = {}
        ok = True
        vis_sum = 0.0

        for idx in self.SIDE_LM[side]:
            v = float(lm[idx].visibility)
            vis_map[self.REQ_LM_LABELS.get(idx, str(idx))] = v
            vis_sum += v
            if v < self.vis_th:
                ok = False

        avg = vis_sum / max(1, len(self.SIDE_LM[side]))
        return ok, avg, vis_map

    def choose_best_side(self, lm: List[Any]) -> Tuple[Optional[str], Dict[str, Any]]:
        left_ok, left_avg, left_map = self.side_score(lm, "left")
        right_ok, right_avg, right_map = self.side_score(lm, "right")

        debug: Dict[str, Any] = {
            "left_ok": left_ok,
            "left_avg": round(left_avg, 3),
            "left_vis": left_map,
            "right_ok": right_ok,
            "right_avg": round(right_avg, 3),
            "right_vis": right_map,
            "mode": self.side_mode,
            "vis_th": self.vis_th,
        }

        if self.side_mode == "left":
            return ("left" if left_ok else None), debug
        if self.side_mode == "right":
            return ("right" if right_ok else None), debug

        # auto
        if left_ok and not right_ok:
            return "left", debug
        if right_ok and not left_ok:
            return "right", debug
        if left_ok and right_ok:
            return ("left" if left_avg >= right_avg else "right"), debug

        return None, debug


def _angle_deg(a: List[float], b: List[float], c: List[float]) -> float:
    """Angle at b (in degrees). a, b, c are [x, y] or [x, y, z]."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))


class FeatureExtractor:
    """Extract per-frame features used by the model (match train_wall_sit.py)."""

    def __init__(self, mp_pose: Any) -> None:
        self.mp_pose = mp_pose

    def extract_foot_wall(self, res: Any, side: str) -> Optional[float]:
        """
        foot_wall = |ankle.x - hip.x| / shoulder_width
        """
        if not res.pose_landmarks:
            return None

        lm = res.pose_landmarks.landmark

        if side == "right":
            hip = self.mp_pose.PoseLandmark.RIGHT_HIP
            ank = self.mp_pose.PoseLandmark.RIGHT_ANKLE
        else:
            hip = self.mp_pose.PoseLandmark.LEFT_HIP
            ank = self.mp_pose.PoseLandmark.LEFT_ANKLE

        shoulder_width = abs(
            lm[self.mp_pose.PoseLandmark.LEFT_SHOULDER].x
            - lm[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].x
        ) + 1e-6

        foot_wall = abs(lm[ank].x - lm[hip].x) / shoulder_width
        return float(foot_wall)

    def extract_knee_angle(self, res: Any, side: str) -> Optional[float]:
        """Knee angle (hip-knee-ankle) in degrees. ยืน ~180°, ย่อ ~90°."""
        if not res.pose_landmarks:
            return None
        lm = res.pose_landmarks.landmark
        if side == "right":
            hip = lm[self.mp_pose.PoseLandmark.RIGHT_HIP]
            knee = lm[self.mp_pose.PoseLandmark.RIGHT_KNEE]
            ankle = lm[self.mp_pose.PoseLandmark.RIGHT_ANKLE]
        else:
            hip = lm[self.mp_pose.PoseLandmark.LEFT_HIP]
            knee = lm[self.mp_pose.PoseLandmark.LEFT_KNEE]
            ankle = lm[self.mp_pose.PoseLandmark.LEFT_ANKLE]
        return _angle_deg([hip.x, hip.y], [knee.x, knee.y], [ankle.x, ankle.y])

    def extract_torso_alignment(self, res: Any, side: str) -> Optional[float]:
        """|shoulder.x - hip.x| for back-against-wall."""
        if not res.pose_landmarks:
            return None
        lm = res.pose_landmarks.landmark
        if side == "right":
            shoulder = lm[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            hip = lm[self.mp_pose.PoseLandmark.RIGHT_HIP]
        else:
            shoulder = lm[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            hip = lm[self.mp_pose.PoseLandmark.LEFT_HIP]
        return float(abs(shoulder.x - hip.x))

    def extract_hip_angle(self, res: Any, side: str) -> Optional[float]:
        """Hip angle (shoulder-hip-knee) in degrees. ยืนตรง ~180°, ย่อ/นั่ง ~90–120°."""
        if not res.pose_landmarks:
            return None
        lm = res.pose_landmarks.landmark
        if side == "right":
            shoulder = lm[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            hip = lm[self.mp_pose.PoseLandmark.RIGHT_HIP]
            knee = lm[self.mp_pose.PoseLandmark.RIGHT_KNEE]
        else:
            shoulder = lm[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            hip = lm[self.mp_pose.PoseLandmark.LEFT_HIP]
            knee = lm[self.mp_pose.PoseLandmark.LEFT_KNEE]
        return _angle_deg([shoulder.x, shoulder.y], [hip.x, hip.y], [knee.x, knee.y])

    @staticmethod
    def aggregate_window(
        foot_wall_vals: List[float],
        knee_angles: List[float],
        torso_alignments: List[float],
        hip_angles: List[float],
    ) -> np.ndarray:
        """Match train_wall_sit.py: [mean(fw), std(fw), mean(knee), min(knee), mean(torso), mean(hip), min(hip)]."""
        fw = np.array(foot_wall_vals, dtype=np.float64)
        knee = np.array(knee_angles, dtype=np.float64)
        torso = np.array(torso_alignments, dtype=np.float64)
        hip = np.array(hip_angles, dtype=np.float64)
        return np.array(
            [
                float(np.mean(fw)),
                float(np.std(fw)) if len(fw) > 1 else 0.0,
                float(np.mean(knee)),
                float(np.min(knee)),
                float(np.mean(torso)),
                float(np.mean(hip)),
                float(np.min(hip)),
            ],
            dtype=np.float32,
        )


class StatusSender:
    """Send throttled status PHASE to iOS."""

    def __init__(self, every_n_frames: int) -> None:
        self.every_n_frames = max(1, int(every_n_frames))

    async def send_info(self, websocket: WebSocket, msg: str, extra: Optional[Dict[str, Any]] = None) -> None:
        payload: Dict[str, Any] = {"type": "info", "message": msg}
        if extra:
            payload.update(extra)
        await websocket.send_text(json.dumps(payload))

    async def send_status(
        self,
        websocket: WebSocket,
        st: StreamState,
        state: str,
        extra: Optional[Dict[str, Any]] = None,
        force: bool = False,
    ) -> None:
        st.status_tick += 1

        if not force:
            if (st.status_tick % self.every_n_frames != 0) and (state == st.last_status):
                return

        payload: Dict[str, Any] = {"type": "status", "state": state, "session_id": st.session_id}
        if extra:
            payload.update(extra)

        st.last_status = state
        await websocket.send_text(json.dumps(payload))


# -----------------------------
# Session Handler (OOP)
# -----------------------------
class WallSitWebSocketSession:
    def __init__(
        self,
        websocket: WebSocket,
        model_svc: ModelService,
        gate: SideGate,
        feat: FeatureExtractor,
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
        self._latency_samples: List[Tuple[float, float]] = []  # (wall_time, frame_total_ms)
        self._last_avg_print_t: float = 0.0

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=MP_MIN_DET_CONF,
            min_tracking_confidence=MP_MIN_TRACK_CONF,
        )

    async def run(self) -> None:
        await self.ws.accept()
        await self.status.send_info(self.ws, "WebSocket connected")

        print(f"[BOOT] side_mode={SIDE_MODE} VIS_TH={VIS_TH} det={MP_MIN_DET_CONF} track={MP_MIN_TRACK_CONF}")
        print(f"[BOOT] WINDOW_FRAMES={self.window_frames} READY_STREAK_N={self.ready_streak_n}")
        if not self.model_svc.loaded:
            await self.status.send_info(self.ws, "Model not loaded (check MODEL_PATH)", {"model_path": MODEL_PATH})

        try:
            while True:
                msg = await self.ws.receive_text()
                data = self._parse_json(msg)
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
        except Exception as e:
            print(f"[WS] error: {e}")
            try:
                await self.status.send_info(self.ws, f"Server error: {e}")
            except Exception:
                pass
            return

    def _parse_json(self, msg: str) -> Optional[Dict[str, Any]]:
        try:
            obj = json.loads(msg)
            if isinstance(obj, dict):
                return obj
            return None
        except Exception:
            return None

    async def _handle_start(self) -> None:
        self.st = StreamState(started=True)
        self.st.session_id = str(int(time.time() * 1000))
        self._latency_samples = []
        self._last_avg_print_t = 0.0

        self.st.no_pose_since = None
        self.st.no_pose_alerted = False
        self.st.dark_since = None
        self.st.dark_alerted = False

        print(f"[SESSION] START session_id={self.st.session_id}")

        await self.status.send_info(self.ws, "Start streaming", {"session_id": self.st.session_id})
        await self.status.send_status(self.ws, self.st, PHASE_NO_POSE, {"reason": "session_started"}, force=True)

    async def _handle_stop(self) -> None:
        print(f"[SESSION] STOP session_id={self.st.session_id}")
        self.st.started = False

        await self.status.send_info(self.ws, "Stop streaming", {"session_id": self.st.session_id})
        await self.status.send_status(self.ws, self.st, PHASE_NO_POSE, {"reason": "session_stopped"}, force=True)

        self._reset_gate_and_buffers(reset_watchdog=True)

    def _reset_gate_and_buffers(self, reset_watchdog: bool) -> None:
        self.st.foot_wall_vals.clear()
        self.st.knee_angles.clear()
        self.st.torso_alignments.clear()
        self.st.hip_angles.clear()
        self.st.ready = False
        self.st.ready_streak = 0
        self.st.chosen_side = None

        # reset last sent/pred memories
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

    async def _handle_frame(self, data: Dict[str, Any]) -> None:
        t_frame_start = time.perf_counter()
        if not self.st.started:
            return

        self.st.frame_count += 1

        frame = FrameDecoder.decode_jpeg_base64(data.get("jpeg_b64", ""))
        if frame is None:
            await self.status.send_info(self.ws, "Decode failed")
            return

        # brightness check (used when NO_POSE)
        too_dark, brightness_mean = FrameQuality.is_too_dark(frame, DARK_BRIGHTNESS_TH)

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.pose.process(img_rgb)

        side_debug: Dict[str, Any] = {}
        chosen_side: Optional[str] = None

        # Decide side (or keep locked after READY)
        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark
            if self.st.ready and self.st.chosen_side is not None:
                chosen_side = self.st.chosen_side
            else:
                chosen_side, side_debug = self.gate.choose_best_side(lm)

        # Gate fail -> NO_POSE watchdog + DARK watchdog (dark first)
        if (not res.pose_landmarks) or (chosen_side is None):
            now = time.time()

            # start counting NO_POSE duration
            if self.st.no_pose_since is None:
                self.st.no_pose_since = now
                self.st.no_pose_alerted = False

            # start/stop counting DARK duration
            if too_dark:
                if self.st.dark_since is None:
                    self.st.dark_since = now
                    self.st.dark_alerted = False
            else:
                self.st.dark_since = None
                self.st.dark_alerted = False

            # 1) DARK message first (one-time)
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

            # 2) NO_POSE message (only if not already explained by dark)
            if (
                (not self.st.no_pose_alerted)
                and (now - self.st.no_pose_since >= NO_POSE_ADJUST_SECONDS)
                and (not too_dark)  # if dark, we prefer dark message
            ):
                await self.status.send_info(
                    self.ws,
                    "Step back so the camera can see your full body",
                    {"brightness_mean": round(brightness_mean, 1), "brightness_th": DARK_BRIGHTNESS_TH},
                )
                self.st.no_pose_alerted = True

            # reset gate/buffer, but keep watchdog running (reset_watchdog=False)
            self._reset_gate_and_buffers(reset_watchdog=False)

            await self.status.send_status(
                self.ws,
                self.st,
                PHASE_NO_POSE,
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
            return

        # regained pose -> reset watchdogs
        self.st.no_pose_since = None
        self.st.no_pose_alerted = False
        self.st.dark_since = None
        self.st.dark_alerted = False

        # Side OK this frame
        self.st.ready_streak += 1
        self.st.chosen_side = chosen_side

        # Have pose but not ready yet => HAVE_POSE
        if (not self.st.ready) and (self.st.ready_streak < self.ready_streak_n):
            await self.status.send_status(
                self.ws,
                self.st,
                PHASE_HAVE_POSE,
                {
                    "chosen_side": chosen_side,
                    "ready_streak": self.st.ready_streak,
                    "needed_streak": self.ready_streak_n,
                    "window_fill": 0,
                    "window_size": self.window_frames,
                },
            )
            return

        # First time ready => enter BUFFERING
        if (not self.st.ready) and (self.st.ready_streak >= self.ready_streak_n):
            self.st.ready = True
            self.st.foot_wall_vals.clear()
            self.st.knee_angles.clear()
            self.st.torso_alignments.clear()
            self.st.hip_angles.clear()

            # reset last sent (if you previously used dedup)
            self.st.last_sent_label = ""
            self.st.last_sent_conf = None

            print(f"[GATE] READY session_id={self.st.session_id} side={chosen_side}")
            await self.status.send_info(
                self.ws,
                "Side landmarks ready",
                {"session_id": self.st.session_id, "side": chosen_side},
            )
            await self.status.send_status(
                self.ws,
                self.st,
                PHASE_BUFFERING,
                {"chosen_side": chosen_side, "window_fill": 0, "window_size": self.window_frames},
                force=True,
            )

        # If model missing, still report phase but skip inference
        if (not self.model_svc.loaded) or (not self.st.ready) or (self.st.chosen_side is None):
            await self.status.send_status(
                self.ws,
                self.st,
                PHASE_BUFFERING,
                {
                    "chosen_side": self.st.chosen_side,
                    "window_fill": len(self.st.foot_wall_vals),
                    "window_size": self.window_frames,
                },
            )
            return

        fw = self.feat.extract_foot_wall(res, self.st.chosen_side)
        knee = self.feat.extract_knee_angle(res, self.st.chosen_side)
        torso = self.feat.extract_torso_alignment(res, self.st.chosen_side)
        hip_ang = self.feat.extract_hip_angle(res, self.st.chosen_side)
        print(f"FW {fw}")
        print(f"KNEE {knee}")
        print(f"TORSO {torso}")
        print(f"HIP ANG {hip_ang}")
        if fw is None or knee is None or torso is None or hip_ang is None:
            self.st.foot_wall_vals.clear()
            self.st.knee_angles.clear()
            self.st.torso_alignments.clear()
            self.st.hip_angles.clear()
            await self.status.send_status(
                self.ws,
                self.st,
                PHASE_BUFFERING,
                {"chosen_side": self.st.chosen_side, "window_fill": 0, "window_size": self.window_frames},
            )
            return

        self.st.foot_wall_vals.append(fw)
        self.st.knee_angles.append(knee)
        self.st.torso_alignments.append(torso)
        self.st.hip_angles.append(hip_ang)

        # Not enough frames yet => BUFFERING
        if len(self.st.foot_wall_vals) < self.window_frames:
            await self.status.send_status(
                self.ws,
                self.st,
                PHASE_BUFFERING,
                {
                    "chosen_side": self.st.chosen_side,
                    "window_fill": len(self.st.foot_wall_vals),
                    "window_size": self.window_frames,
                },
            )
            return

        # Enough frames => INFERENCING
        await self.status.send_status(
            self.ws,
            self.st,
            PHASE_INFERENCING,
            {
                "chosen_side": self.st.chosen_side,
                "window_fill": len(self.st.foot_wall_vals),
                "window_size": self.window_frames,
            },
        )

        t_infer_start = time.perf_counter()
        fw_vals = self.st.foot_wall_vals[-self.window_frames:]
        knee_vals = self.st.knee_angles[-self.window_frames:]
        torso_vals = self.st.torso_alignments[-self.window_frames:]
        hip_vals = self.st.hip_angles[-self.window_frames:]
        agg_feat = self.feat.aggregate_window(fw_vals, knee_vals, torso_vals, hip_vals)
        pred_id, conf = self.model_svc.predict(agg_feat)
        t_infer_end = time.perf_counter()
        pred_label = self.labels.label_of(pred_id)

        frame_total_ms = (t_infer_end - t_frame_start) * 1000
        inference_ms = (t_infer_end - t_infer_start) * 1000
        print(f"[WALLSIT latency] frame_total_ms={frame_total_ms:.1f} inference_ms={inference_ms:.1f} pred={pred_label}")

        now_wall = time.time()
        self._latency_samples.append((now_wall, frame_total_ms))
        cutoff = now_wall - 1.0
        self._latency_samples = [(t, ms) for t, ms in self._latency_samples if t >= cutoff]
        if now_wall - self._last_avg_print_t >= 1.0:
            if self._latency_samples:
                avg_ms = sum(ms for _, ms in self._latency_samples) / len(self._latency_samples)
                print(f"[WALLSIT] AVERAGE LATENCY OF 1 SEC: {avg_ms:.1f} ms (n={len(self._latency_samples)})")
            self._last_avg_print_t = now_wall

        self.st.last_pred_label = pred_label
        self.st.last_pred_conf = conf

        # send result every inference (dedup removed)
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

        # keep lists bounded (avoid growing forever)
        max_len = self.window_frames + 60
        if len(self.st.foot_wall_vals) > max_len:
            self.st.foot_wall_vals = self.st.foot_wall_vals[-max_len:]
            self.st.knee_angles = self.st.knee_angles[-max_len:]
            self.st.torso_alignments = self.st.torso_alignments[-max_len:]
            self.st.hip_angles = self.st.hip_angles[-max_len:]


# -----------------------------
# Routes
# -----------------------------
model_service = ModelService(MODEL_PATH)
label_mapper = LabelMapper(LABELS)

mp_pose = mp.solutions.pose
side_gate = SideGate(mp_pose=mp_pose, side_mode=SIDE_MODE, vis_th=VIS_TH)
feature_extractor = FeatureExtractor(mp_pose=mp_pose)

status_sender = StatusSender(every_n_frames=STATUS_SEND_EVERY_N_FRAMES)


@app.get("/health")
def health() -> Dict[str, Any]:
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


# -----------------------------
# Main
# -----------------------------
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