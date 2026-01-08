import base64
import json
import time
import os
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

import numpy as np
import cv2
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(title="FiT-AI Video Streaming Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Video Recording Settings
# =========================
SAVE_VIDEO = True
RECORD_DIR = "recordings"
RECORD_FPS = 10.0  # ให้ใกล้กับ targetFPS ฝั่ง iOS
os.makedirs(RECORD_DIR, exist_ok=True)

# =========================
# Debug / Latency Settings
# =========================
DEBUG_STEPS = True
SEND_INFO_EVERY_FRAME = True
INFO_THROTTLE_SEC = 0.15


@dataclass
class StreamState:
    started: bool = False
    frame_count: int = 0
    total_reps: int = 0
    correct_reps: int = 0
    incorrect_reps: int = 0

    last_sent_ts: float = 0.0
    last_feedback: str = "Waiting..."

    # ---- recording ----
    session_id: str = ""
    out_path_no_ext: str = ""
    writer: Optional[cv2.VideoWriter] = None
    writer_size: Optional[Tuple[int, int]] = None  # (w, h)
    actual_video_path: str = ""

    # ---- info throttle ----
    last_info_sent_ts: float = 0.0


def decode_jpeg_base64(jpeg_b64: str) -> Optional[np.ndarray]:
    """Decode base64 JPEG -> OpenCV BGR image (H,W,3)."""
    try:
        raw = base64.b64decode(jpeg_b64)
        arr = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


def create_video_writer(path_no_ext: str, w: int, h: int, fps: float) -> Tuple[Optional[cv2.VideoWriter], str]:
    """
    Try MP4 writer first; fallback to AVI(MJPG) if MP4 not supported.
    Returns (writer, actual_path).
    """
    # 1) Try MP4
    mp4_path = f"{path_no_ext}.mp4"
    try:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(mp4_path, fourcc, fps, (w, h))
        if writer.isOpened():
            return writer, mp4_path
    except Exception:
        pass

    # 2) Fallback AVI
    avi_path = f"{path_no_ext}.avi"
    try:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(avi_path, fourcc, fps, (w, h))
        if writer.isOpened():
            return writer, avi_path
    except Exception:
        pass

    return None, ""


def dummy_inference(img_bgr: np.ndarray, st: StreamState) -> Dict[str, Any]:
    """Demo logic"""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    mean_intensity = float(np.mean(gray))

    if mean_intensity < 40:
        feedback = "Too dark / Move to brighter area"
        is_correct = False
    elif mean_intensity < 70:
        feedback = "Adjust camera distance (make sure full body is visible)"
        is_correct = True
    else:
        if st.frame_count % 50 < 10:
            feedback = "Keep your back straight"
            is_correct = False
        elif st.frame_count % 50 < 20:
            feedback = "Push knees slightly out"
            is_correct = False
        else:
            feedback = "Good form"
            is_correct = True

    # demo rep counting
    if st.frame_count % 30 == 0:
        st.total_reps += 1
        if is_correct:
            st.correct_reps += 1
        else:
            st.incorrect_reps += 1

    st.last_feedback = feedback

    return {
        "feedback": feedback,
        "totalReps": st.total_reps,
        "correctReps": st.correct_reps,
        "incorrectReps": st.incorrect_reps,
    }


@app.get("/health")
def health():
    return {"status": "ok", "ts": int(time.time())}


@app.websocket("/ws/video")
async def ws_video(websocket: WebSocket):
    """
    iOS -> backend:
      {"type":"start","ts":...}
      {"type":"frame","ts":...,"jpeg_b64":"..."}
      {"type":"stop","ts":...}

    backend -> iOS:
      {"type":"info","message":"..."}
      {"type":"result","feedback":"...","totalReps":...,"correctReps":...,"incorrectReps":...}
    """
    await websocket.accept()
    st = StreamState()

    async def send_info(message: str, force: bool = False):
        """
        ส่ง info กลับไปให้ iOS ขึ้นบนจอ (feedback)
        มี throttle กันสแปม
        """
        if not force:
            now = time.time()
            if now - st.last_info_sent_ts < INFO_THROTTLE_SEC:
                return
            st.last_info_sent_ts = now
        await websocket.send_text(json.dumps({"type": "info", "message": message}))

    async def cleanup_recording():
        if st.writer is not None:
            try:
                st.writer.release()
            except Exception:
                pass
            st.writer = None
            st.writer_size = None

    await send_info("WebSocket connected", force=True)

    try:
        while True:
            msg = await websocket.receive_text()
            try:
                data = json.loads(msg)
            except json.JSONDecodeError:
                await send_info("Invalid JSON", force=True)
                continue

            mtype = data.get("type", "")

            # -------------------
            # START
            # -------------------
            if mtype == "start":
                st.started = True
                st.frame_count = 0
                st.total_reps = 0
                st.correct_reps = 0
                st.incorrect_reps = 0
                st.last_sent_ts = 0.0
                st.last_feedback = "Session started"

                # prepare recording
                st.session_id = str(int(time.time() * 1000))
                st.out_path_no_ext = os.path.join(RECORD_DIR, f"session_{st.session_id}")
                st.actual_video_path = ""
                await cleanup_recording()

                if DEBUG_STEPS:
                    print("[1] start received")
                    await send_info("1 (start received)", force=True)

                continue

            # -------------------
            # STOP
            # -------------------
            if mtype == "stop":
                st.started = False
                await cleanup_recording()

                if DEBUG_STEPS:
                    print("[stop] session stopped")
                    saved_msg = st.actual_video_path if st.actual_video_path else "(no video writer)"
                    await send_info(f"stop (saved: {saved_msg})", force=True)

                continue

            # -------------------
            # FRAME
            # -------------------
            if mtype == "frame":
                if not st.started:
                    continue

                jpeg_b64 = data.get("jpeg_b64")
                if not jpeg_b64:
                    continue

                img = decode_jpeg_base64(jpeg_b64)
                if img is None:
                    await send_info("Decode failed", force=True)
                    continue

                st.frame_count += 1

                # ===== Latency (one-way ~)
                client_ts = data.get("ts")  # ms
                server_ms = int(time.time() * 1000)
                one_way_ms = (server_ms - client_ts) if isinstance(client_ts, int) else None

                # ===== DEBUG 2
                if DEBUG_STEPS and SEND_INFO_EVERY_FRAME:
                    print(f"[2] frame decoded | one-way~{one_way_ms} ms | frame={st.frame_count}")
                    await send_info(f"2 (decoded) one-way~{one_way_ms}ms f={st.frame_count}")
                elif DEBUG_STEPS and st.frame_count % 10 == 0:
                    # ถ้าไม่อยากสแปม ส่งทุก 10 เฟรม
                    print(f"[2] frame decoded | one-way~{one_way_ms} ms | frame={st.frame_count}")
                    await send_info(f"2 (decoded) one-way~{one_way_ms}ms f={st.frame_count}")

                # ===== Save Video
                if SAVE_VIDEO:
                    h, w = img.shape[:2]
                    if st.writer is None:
                        writer, actual_path = create_video_writer(st.out_path_no_ext, w, h, RECORD_FPS)
                        if writer is None:
                            await send_info("Recording disabled: cannot create VideoWriter (codec issue)", force=True)
                        else:
                            st.writer = writer
                            st.writer_size = (w, h)
                            st.actual_video_path = actual_path
                            await send_info(f"Recording started: {actual_path}", force=True)

                    if st.writer is not None and st.writer_size is not None:
                        tw, th = st.writer_size
                        if (w, h) != (tw, th):
                            img = cv2.resize(img, (tw, th))
                        st.writer.write(img)

                # ===== Inference
                result = dummy_inference(img, st)

                # ===== DEBUG 3
                if DEBUG_STEPS:
                    print("[3] inference done -> sending result")
                    await send_info("3 (inference done)", force=False)

                # ===== Send result (throttle ~5Hz)
                now = time.time()
                if now - st.last_sent_ts >= 0.2:
                    st.last_sent_ts = now
                    await websocket.send_text(json.dumps({
                        "type": "result",
                        "feedback": result["feedback"],
                        "totalReps": result["totalReps"],
                        "correctReps": result["correctReps"],
                        "incorrectReps": result["incorrectReps"],
                    }))

                continue

            await send_info(f"Unknown type: {mtype}", force=True)

    except WebSocketDisconnect:
        await cleanup_recording()
        return
    except Exception as e:
        await cleanup_recording()
        try:
            await send_info(f"Server error: {str(e)}", force=True)
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=5050,
        reload=True,
        log_level="info"
    )
