"""
plank_streaming.py (NO OVERLAY / NO RECORDING) - OOP VERSION

Streaming (WebSocket) + SIDE-VIEW gate (one side only)
Same protocol/behavior as wall_sit_streaming, but for plank posture.

PHASE (status.state) to iOS:
  - NO_POSE
  - HAVE_POSE
  - BUFFERING
  - INFERENCING

Run (from backend/):
  python plank/plank_streaming.py --serve

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
PLANK_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(PLANK_DIR, "models", "plank_model.pkl")

LABELS: Dict[int, str] = {
    0: "correct",
    1: "hips_too_high",
    2: "hips_too_low",
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
app = FastAPI(title="FiT-AI Plank Streaming Backend (OOP)")

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
    # per-frame features: (body_angle, torso_slope, hip_deviation, hip_height)
    feats: List[Tuple[float, float, float, float]] = field(default_factory=list)

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
            mp_pose.PoseLandmark.RIGHT_KNEE: "R_KNE",
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


class FeatureExtractor:
    """Extract per-frame features used by the plank model.

    Returns per-frame tuple:
        (body_angle, torso_slope, hip_deviation, hip_height)
    """

    def __init__(self, mp_pose: Any) -> None:
        self.mp_pose = mp_pose

    @staticmethod
    def angle(a, b, c):
        a, b, c = np.array(a), np.array(b), np.array(c)
        ba, bc = a - b, c - b
        cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        return np.degrees(np.arccos(np.clip(cos, -1, 1)))

    def extract_features(self, res: Any, side: str) -> Optional[Tuple[float, float, float, float]]:
        if not res.pose_landmarks:
            return None

        lm = res.pose_landmarks.landmark

        if side == "right":
            hip = self.mp_pose.PoseLandmark.RIGHT_HIP
            ankle = self.mp_pose.PoseLandmark.RIGHT_ANKLE
            shoulder = self.mp_pose.PoseLandmark.RIGHT_SHOULDER
        else:
            hip = self.mp_pose.PoseLandmark.LEFT_HIP
            ankle = self.mp_pose.PoseLandmark.LEFT_ANKLE
            shoulder = self.mp_pose.PoseLandmark.LEFT_SHOULDER

        shoulder_xy = np.array([lm[shoulder].x, lm[shoulder].y], dtype=np.float32)
        hip_xy = np.array([lm[hip].x, lm[hip].y], dtype=np.float32)
        ankle_xy = np.array([lm[ankle].x, lm[ankle].y], dtype=np.float32)

        # 1️⃣ Body angle
        body_angle = self.angle(shoulder_xy, hip_xy, ankle_xy)

        # 2️⃣ Torso slope (y over x)
        slope = (ankle_xy[1] - shoulder_xy[1]) / (ankle_xy[0] - shoulder_xy[0] + 1e-6)

        # 3️⃣ Hip deviation from shoulder-ankle line
        dev = np.abs(
            np.cross(
                ankle_xy - shoulder_xy,
                shoulder_xy - hip_xy,
            )
        ) / (np.linalg.norm(ankle_xy - shoulder_xy) + 1e-6)

        # 4️⃣ Relative hip height
        mean_ref_y = 0.5 * (shoulder_xy[1] + ankle_xy[1])
        hip_height = hip_xy[1] - mean_ref_y

        return float(body_angle), float(slope), float(dev), float(hip_height)

    @staticmethod
    def aggregate_window(vals: List[Tuple[float, float, float, float]]) -> np.ndarray:
        """
        Aggregate window into final feature vector:
        [
            mean_body_angle,
            std_body_angle,
            min_body_angle,
            max_body_angle,

            mean_torso_slope,
            std_torso_slope,

            mean_hip_deviation,
            max_hip_deviation,

            mean_hip_height,
            std_hip_height,
            min_hip_height,
            max_hip_height,
        ]
        This matches the 12-dim feature definition in train_plank.py.
        """
        body = [v[0] for v in vals]
        slope = [v[1] for v in vals]
        dev = [v[2] for v in vals]
        hip = [v[3] for v in vals]

        return np.array(
            [
                np.mean(body),
                np.std(body),
                np.min(body),
                np.max(body),

                np.mean(slope),
                np.std(slope),

                np.mean(dev),
                np.max(dev),

                np.mean(hip),
                np.std(hip),
                np.min(hip),
                np.max(hip),
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
class PlankWebSocketSession:
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
        self.st.feats.clear()
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
                    "Please Adjust Your Pose",
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
            self.st.feats.clear()

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
                    "window_fill": len(self.st.feats),
                    "window_size": self.window_frames,
                },
            )
            return

        feat_tuple = self.feat.extract_features(res, self.st.chosen_side)

        if feat_tuple is None:
            self.st.feats.clear()
            await self.status.send_status(
                self.ws,
                self.st,
                PHASE_BUFFERING,
                {
                    "chosen_side": self.st.chosen_side,
                    "window_fill": 0,
                    "window_size": self.window_frames,
                },
            )
            return

        self.st.feats.append(feat_tuple)

        # Not enough frames yet => BUFFERING
        if len(self.st.feats) < self.window_frames:
            await self.status.send_status(
                self.ws,
                self.st,
                PHASE_BUFFERING,
                {
                    "chosen_side": self.st.chosen_side,
                    "window_fill": len(self.st.feats),
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
                "window_fill": len(self.st.feats),
                "window_size": self.window_frames,
            },
        )

        vals = self.st.feats[-self.window_frames:]
        agg_feat = self.feat.aggregate_window(vals)

        pred_id, conf = self.model_svc.predict(agg_feat)
        pred_label = self.labels.label_of(pred_id)

        self.st.last_pred_label = pred_label
        self.st.last_pred_conf = conf

        # send result every inference (no dedup)
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

        # keep list bounded (avoid growing forever)
        if len(self.st.feats) > (self.window_frames + 60):
            self.st.feats = self.st.feats[-(self.window_frames + 60):]


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
    session = PlankWebSocketSession(
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
            "plank.plank_streaming:app",
            host="0.0.0.0",
            port=5052,
            reload=False,
            log_level="info",
        )


if __name__ == "__main__":
    main()

