"""
wall_sit_streaming.py

Streaming (WebSocket) + SIDE-VIEW gate (one side only) + Save video per session
+ Overlay pose + PRINT logs + Status to iOS
+ DEDUP RULE: Send {"type":"result"} ONLY when prediction label changes

Run:
  python wall_sit_streaming.py --serve

WS protocol (from iOS):
  - {"type":"start"}
  - {"type":"frame","jpeg_b64":"..."}
  - {"type":"stop"}

Server -> iOS:
  - {"type":"status","state":"waiting|warming_up|ready|predicting", ...}
  - {"type":"result","prediction":"...", "confidence":..., ...}   (ONLY when changed)
  - {"type":"info","message":"..."}  (optional)
"""

import os
import time
import json
import base64
import argparse
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple

import warnings

warnings.filterwarnings(
    "ignore",
    message=r".*SymbolDatabase\.GetPrototype\(\) is deprecated.*",
    category=UserWarning,
    module=r"google\.protobuf\.symbol_database"
)

import numpy as np
import cv2
import joblib
import mediapipe as mp
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

print("### RUNNING FILE:", __file__)

# -----------------------------
# Config
# -----------------------------
MODEL_PATH = "wall_sit_side_model.pkl"

LABELS = {
    0: "correct",
    1: "feet_too_close",
}

WINDOW_FRAMES = 15          # frames per window
READY_STREAK_N = 3          # require side landmarks N consecutive frames
VIS_TH = 0.80               # side landmark visibility threshold

DEBUG = True

# Save video per session
SAVE_VIDEO = True
RECORD_DIR = "recordings"
RECORD_FPS = 10.0
os.makedirs(RECORD_DIR, exist_ok=True)

# For debugging: record even when not READY
RECORD_ONLY_WHEN_READY = False

# MediaPipe confidence
MP_MIN_DET_CONF = 0.80
MP_MIN_TRACK_CONF = 0.80

# Print record stats every N frames
PRINT_EVERY_SAVED_FRAMES = 30

# Status to iOS (throttle)
STATUS_SEND_EVERY_N_FRAMES = 3

# Choose side mode: "auto" | "left" | "right"
SIDE_MODE = "auto"


# -----------------------------
# FastAPI
# -----------------------------
app = FastAPI(title="FiT-AI WallSit Streaming Backend (Side Gate + Dedup Result)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Load model
# -----------------------------
try:
    MODEL = joblib.load(MODEL_PATH)
    print(f"[MODEL] Loaded: {MODEL_PATH}")
except Exception as e:
    MODEL = None
    print(f"[MODEL] Cannot load model: {e}")

# -----------------------------
# MediaPipe
# -----------------------------
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

# Side landmark sets (only ONE side is required)
SIDE_LM = {
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

# Labels for overlay (optional)
REQ_LM_LABELS = {
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


# -----------------------------
# Utils
# -----------------------------
def decode_jpeg_base64(jpeg_b64: str) -> Optional[np.ndarray]:
    """
    Decode base64 JPEG to BGR image.
    """
    try:
        raw = base64.b64decode(jpeg_b64)
        arr = np.frombuffer(raw, np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return None


def label_of(pred: int) -> str:
    """
    Map numeric prediction to label.
    """
    return LABELS.get(int(pred), str(pred))


def side_score(lm, side: str, vis_th: float) -> Tuple[bool, float, Dict[str, float]]:
    """
    returns (ok, avg_visibility, vis_map)
    ok = all side landmarks visibility >= vis_th
    """
    vis_map: Dict[str, float] = {}
    ok = True
    vis_sum = 0.0
    for idx in SIDE_LM[side]:
        v = float(lm[idx].visibility)
        vis_map[REQ_LM_LABELS.get(idx, str(idx))] = v
        vis_sum += v
        if v < vis_th:
            ok = False
    avg = vis_sum / max(1, len(SIDE_LM[side]))
    return ok, avg, vis_map


def choose_best_side(lm, vis_th: float) -> Tuple[Optional[str], Dict[str, Any]]:
    """
    Choose side based on mode & visibility.
    - If SIDE_MODE = left/right: enforce that side (must pass)
    - If auto: pick side with higher avg visibility; must pass vis_th for that side.
    """
    left_ok, left_avg, left_map = side_score(lm, "left", vis_th)
    right_ok, right_avg, right_map = side_score(lm, "right", vis_th)

    debug = {
        "left_ok": left_ok, "left_avg": round(left_avg, 3), "left_vis": left_map,
        "right_ok": right_ok, "right_avg": round(right_avg, 3), "right_vis": right_map,
        "mode": SIDE_MODE,
    }

    if SIDE_MODE == "left":
        return ("left" if left_ok else None), debug
    if SIDE_MODE == "right":
        return ("right" if right_ok else None), debug

    # auto
    if left_ok and not right_ok:
        return "left", debug
    if right_ok and not left_ok:
        return "right", debug
    if left_ok and right_ok:
        return ("left" if left_avg >= right_avg else "right"), debug
    return None, debug


def extract_frame_foot_wall(res, side: str) -> Optional[float]:
    """
    A simple scalar per-frame feature:
      foot_wall = |ankle.x - hip.x| / shoulder_width
    """
    if not res.pose_landmarks:
        return None

    lm = res.pose_landmarks.landmark

    if side == "right":
        HIP = mp_pose.PoseLandmark.RIGHT_HIP
        ANK = mp_pose.PoseLandmark.RIGHT_ANKLE
    else:
        HIP = mp_pose.PoseLandmark.LEFT_HIP
        ANK = mp_pose.PoseLandmark.LEFT_ANKLE

    shoulder_width = abs(
        lm[mp_pose.PoseLandmark.LEFT_SHOULDER].x -
        lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].x
    ) + 1e-6

    foot_wall = abs(lm[ANK].x - lm[HIP].x) / shoulder_width
    return float(foot_wall)


def create_video_writer(path_no_ext: str, w: int, h: int, fps: float) -> Tuple[Optional[cv2.VideoWriter], str]:
    """
    Try mp4 then avi.
    """
    mp4_path = f"{path_no_ext}.mp4"
    try:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(mp4_path, fourcc, fps, (w, h))
        if writer.isOpened():
            return writer, mp4_path
    except Exception:
        pass

    avi_path = f"{path_no_ext}.avi"
    try:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(avi_path, fourcc, fps, (w, h))
        if writer.isOpened():
            return writer, avi_path
    except Exception:
        pass

    return None, ""


def draw_pose_overlay(
    frame_bgr: np.ndarray,
    res,
    state: str,
    side: Optional[str],
    vis_th: float,
    extra_text: Optional[str] = None,
    pred_text: Optional[str] = None,
) -> np.ndarray:
    """
    Draw pose + debug texts onto frame.
    """
    out = frame_bgr.copy()
    h, w = out.shape[:2]

    if res.pose_landmarks:
        mp_drawing.draw_landmarks(
            out,
            res.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style()
        )

        lm = res.pose_landmarks.landmark
        indices: List[int] = []
        if side in ("left", "right"):
            indices = SIDE_LM[side]
        else:
            indices = SIDE_LM["left"] + SIDE_LM["right"]

        for idx in indices:
            name = REQ_LM_LABELS.get(idx, str(idx))
            p = lm[idx]
            if p.visibility < vis_th:
                continue
            x, y = int(p.x * w), int(p.y * h)
            cv2.circle(out, (x, y), 4, (0, 255, 255), -1)
            cv2.putText(out, name, (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

    cv2.putText(out, f"State: {state}", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(out, f"Side: {side or '-'}  VIS_TH: {vis_th:.2f}  READY_STREAK: {READY_STREAK_N}",
                (12, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    if extra_text:
        cv2.putText(out, extra_text, (12, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    if pred_text:
        cv2.putText(out, pred_text, (12, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return out


# -----------------------------
# Stream State
# -----------------------------
@dataclass
class StreamState:
    started: bool = False
    foot_wall_vals: List[float] = field(default_factory=list)

    # gate
    ready: bool = False
    ready_streak: int = 0
    chosen_side: Optional[str] = None

    # session
    session_id: str = ""
    out_path_no_ext: str = ""

    # recording
    writer: Optional[cv2.VideoWriter] = None
    writer_size: Optional[Tuple[int, int]] = None
    actual_video_path: str = ""
    saved_frames: int = 0

    # status throttle
    last_status: str = ""
    status_tick: int = 0

    # last prediction (for overlay)
    last_pred_label: str = ""
    last_pred_conf: Optional[float] = None

    # de-dup send (ONLY send result when changed)
    last_sent_label: str = ""
    last_sent_conf: Optional[float] = None

    # frame count (misc)
    frame_count: int = 0


# -----------------------------
# Routes
# -----------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": MODEL is not None,
        "record_dir": os.path.abspath(RECORD_DIR),
        "timestamp": int(time.time()),
        "window_frames": WINDOW_FRAMES,
    }


@app.websocket("/ws/video")
async def ws_video(websocket: WebSocket):
    await websocket.accept()

    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=MP_MIN_DET_CONF,
        min_tracking_confidence=MP_MIN_TRACK_CONF
    )

    st = StreamState()

    async def send_info(msg: str, extra: Optional[Dict[str, Any]] = None):
        payload = {"type": "info", "message": msg}
        if extra:
            payload.update(extra)
        await websocket.send_text(json.dumps(payload))

    async def send_status(state: str, extra: Optional[Dict[str, Any]] = None, force: bool = False):
        """
        waiting | warming_up | ready | predicting
        """
        st.status_tick += 1
        if not force:
            if (st.status_tick % STATUS_SEND_EVERY_N_FRAMES != 0) and (state == st.last_status):
                return

        payload = {"type": "status", "state": state, "session_id": st.session_id}
        if extra:
            payload.update(extra)

        st.last_status = state
        await websocket.send_text(json.dumps(payload))

    async def cleanup_recording():
        if st.writer is not None:
            try:
                st.writer.release()
                print(f"[RECORD] STOP recording")
                print(f"[RECORD] path   = {st.actual_video_path}")
                print(f"[RECORD] frames = {st.saved_frames}")
            except Exception as e:
                print(f"[RECORD] release error: {e}")
        st.writer = None
        st.writer_size = None

    async def start_recording_for_frame(frame_bgr: np.ndarray):
        if not SAVE_VIDEO:
            return

        h, w = frame_bgr.shape[:2]

        if st.writer is None:
            writer, actual_path = create_video_writer(st.out_path_no_ext, w, h, RECORD_FPS)
            if writer is None:
                print("[RECORD] Failed to create VideoWriter")
                await send_info("Recording disabled: cannot create VideoWriter")
                return

            st.writer = writer
            st.writer_size = (w, h)
            st.actual_video_path = actual_path
            st.saved_frames = 0

            print(f"[RECORD] START recording")
            print(f"[RECORD] path = {actual_path}")
            print(f"[RECORD] size = {w}x{h} @ {RECORD_FPS}fps")
            print(f"[RECORD]  dir  = {os.path.abspath(RECORD_DIR)}")

            await send_info("Recording started", {"video_path": actual_path})

    await send_info("WebSocket connected", {"record_dir": os.path.abspath(RECORD_DIR)})
    print(f"[BOOT] record_dir={os.path.abspath(RECORD_DIR)}")
    print(f"[BOOT] side_mode={SIDE_MODE} VIS_TH={VIS_TH} det={MP_MIN_DET_CONF} track={MP_MIN_TRACK_CONF}")
    print(f"[BOOT] WINDOW_FRAMES={WINDOW_FRAMES}")

    if MODEL is None:
        await send_info("Model not loaded (check MODEL_PATH)")

    try:
        while True:
            msg = await websocket.receive_text()
            try:
                data = json.loads(msg)
            except json.JSONDecodeError:
                await send_info("Invalid JSON")
                continue

            mtype = data.get("type")

            if mtype == "start":
                await cleanup_recording()
                st = StreamState(started=True)

                st.session_id = str(int(time.time() * 1000))
                st.out_path_no_ext = os.path.join(RECORD_DIR, f"session_{st.session_id}")

                print(f"[SESSION] START session_id={st.session_id}")
                print(f"[SESSION] output base={st.out_path_no_ext}")

                await send_info("Start streaming", {"session_id": st.session_id})
                await send_status("waiting", {"reason": "session_started"}, force=True)
                continue

            if mtype == "stop":
                print(f"[SESSION] STOP session_id={st.session_id}")
                st.started = False
                await cleanup_recording()

                await send_info("Stop streaming", {
                    "session_id": st.session_id,
                    "video_path": st.actual_video_path,
                    "saved_frames": st.saved_frames
                })
                await send_status("waiting", {"reason": "session_stopped"}, force=True)

                # reset gate + buffers
                st.foot_wall_vals.clear()
                st.ready = False
                st.ready_streak = 0
                st.chosen_side = None
                st.last_sent_label = ""
                st.last_sent_conf = None
                st.frame_count = 0
                continue

            if mtype != "frame" or not st.started:
                continue

            st.frame_count += 1

            frame = decode_jpeg_base64(data.get("jpeg_b64", ""))
            if frame is None:
                await send_info("Decode failed")
                continue

            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(img_rgb)

            extra = ""
            side_debug = {}
            chosen_side = None

            if res.pose_landmarks:
                lm = res.pose_landmarks.landmark
                if st.ready and st.chosen_side is not None:
                    chosen_side = st.chosen_side
                else:
                    chosen_side, side_debug = choose_best_side(lm, VIS_TH)

            if (not res.pose_landmarks) or (chosen_side is None):
                # not OK -> reset gate + buffers
                st.ready = False
                st.ready_streak = 0
                st.chosen_side = None
                st.foot_wall_vals.clear()

                # IMPORTANT: reset dedup memory so next label can send again after reacquire
                st.last_sent_label = ""
                st.last_sent_conf = None

                extra = "No side landmarks yet"
                await send_status("waiting", {
                    "chosen_side": None,
                    "ready_streak": 0,
                    "needed_streak": READY_STREAK_N,
                    "window_fill": 0,
                    "window_size": WINDOW_FRAMES,
                    "debug": side_debug if DEBUG else None
                })
            else:
                # side OK this frame
                st.ready_streak += 1
                st.chosen_side = chosen_side
                extra = f"Side={chosen_side} Streak {st.ready_streak}/{READY_STREAK_N}"

                if not st.ready and st.ready_streak < READY_STREAK_N:
                    await send_status("warming_up", {
                        "chosen_side": chosen_side,
                        "ready_streak": st.ready_streak,
                        "needed_streak": READY_STREAK_N,
                        "window_fill": 0,
                        "window_size": WINDOW_FRAMES
                    })

                if not st.ready and st.ready_streak >= READY_STREAK_N:
                    st.ready = True
                    st.foot_wall_vals.clear()

                    # reset dedup when first becomes READY
                    st.last_sent_label = ""
                    st.last_sent_conf = None

                    print(f"[GATE] READY session_id={st.session_id} side={chosen_side}")
                    await send_info("Side landmarks ready", {"session_id": st.session_id, "side": chosen_side})
                    await send_status("ready", {"chosen_side": chosen_side}, force=True)

            # overlay prediction text (optional)
            pred_text = ""
            if st.last_pred_label:
                if st.last_pred_conf is None:
                    pred_text = f"Pred: {st.last_pred_label}"
                else:
                    pred_text = f"Pred: {st.last_pred_label} ({st.last_pred_conf:.3f})"

            overlay = draw_pose_overlay(
                frame_bgr=frame,
                res=res,
                state=("ready" if st.ready else ("warming_up" if st.ready_streak > 0 else "waiting")),
                side=st.chosen_side,
                vis_th=VIS_TH,
                extra_text=extra,
                pred_text=pred_text if pred_text else None,
            )

            # record
            should_record = SAVE_VIDEO and ((not RECORD_ONLY_WHEN_READY) or st.ready)
            if should_record:
                await start_recording_for_frame(overlay)
                if st.writer is not None:
                    tw, th = st.writer_size if st.writer_size else (overlay.shape[1], overlay.shape[0])
                    if (overlay.shape[1], overlay.shape[0]) != (tw, th):
                        overlay = cv2.resize(overlay, (tw, th))
                    st.writer.write(overlay)
                    st.saved_frames += 1

                    if st.saved_frames % PRINT_EVERY_SAVED_FRAMES == 0:
                        print(f"[RECORD] saved_frames={st.saved_frames} path={st.actual_video_path}")

            # Predict only when READY and model exists
            if (MODEL is None) or (not st.ready) or (st.chosen_side is None):
                continue

            fw = extract_frame_foot_wall(res, st.chosen_side)
            if fw is None:
                st.foot_wall_vals.clear()
                await send_status("predicting", {
                    "chosen_side": st.chosen_side,
                    "window_fill": 0,
                    "window_size": WINDOW_FRAMES
                })
                continue

            st.foot_wall_vals.append(fw)

            await send_status("predicting", {
                "chosen_side": st.chosen_side,
                "window_fill": len(st.foot_wall_vals),
                "window_size": WINDOW_FRAMES
            })

            # Need enough frames for a window
            if len(st.foot_wall_vals) < WINDOW_FRAMES:
                continue

            # ---- Sliding window prediction (every frame) ----
            vals = st.foot_wall_vals[-WINDOW_FRAMES:]

            agg_feat = np.array([
                np.mean(vals),
                np.std(vals),
                np.min(vals),
            ], dtype=np.float32)

            pred = int(MODEL.predict(agg_feat.reshape(1, -1))[0])
            conf = None
            if hasattr(MODEL, "predict_proba"):
                conf = float(MODEL.predict_proba(agg_feat.reshape(1, -1))[0][pred])

            pred_label = label_of(pred)
            st.last_pred_label = pred_label
            st.last_pred_conf = conf

            # ---- DEDUP SEND: send ONLY when label changes ----
            if pred_label != st.last_sent_label:
                payload = {
                    "type": "result",
                    "prediction": pred_label,
                    "confidence": round(conf, 3) if conf is not None else None,
                    "window": WINDOW_FRAMES,
                    "session_id": st.session_id,
                    "side": st.chosen_side
                }
                if DEBUG:
                    print(f"[PRED] (CHANGED) {payload}")
                await websocket.send_text(json.dumps(payload))

                st.last_sent_label = pred_label
                st.last_sent_conf = conf
            else:
                if DEBUG:
                    print(f"[PRED] (SKIP) same label={pred_label}")

            # keep list bounded (avoid growing forever)
            if len(st.foot_wall_vals) > (WINDOW_FRAMES + 60):
                st.foot_wall_vals = st.foot_wall_vals[-(WINDOW_FRAMES + 60):]

    except WebSocketDisconnect:
        print(f"[WS] disconnect session_id={st.session_id}")
        await cleanup_recording()
        return
    except Exception as e:
        print(f"[WS] error: {e}")
        await cleanup_recording()
        try:
            await send_info(f"Server error: {e}")
        except Exception:
            pass


# -----------------------------
# Main
# -----------------------------
def main():
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
            log_level="info"
        )


if __name__ == "__main__":
    main()
