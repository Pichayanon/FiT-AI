"""
Extract bottom-phase snippets from lunges video(s) (around knee-angle minima).

--video can be a single file OR a folder containing videos.

Labels:
  correct             - correct lunge form
  torso_lean_forward  - torso leans forward too much
  knee_over_toe       - front knee extends past toes
  not_deep_enough     - rear knee doesn't go low enough

Usage:
  python lunges/extract_bottom_lunges.py --video path.mp4 --label correct --show
  python lunges/extract_bottom_lunges.py --video dataset/lunges/correct/ --label correct

Output: dataset/lunges/dataset_bottom/<vidname>_<label>_snip<NNN>.npz
"""
from __future__ import annotations

import argparse
import glob
import os
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np


# -----------------------------
# Small math / pose helpers
# -----------------------------

def angle_3pts(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> float:
    """Compute the angle (in degrees) at point b formed by points a-b-c."""
    a_np = np.array(a, dtype=np.float32)
    b_np = np.array(b, dtype=np.float32)
    c_np = np.array(c, dtype=np.float32)

    ba = a_np - b_np
    bc = c_np - b_np

    denom = (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    cosang = np.clip(float(np.dot(ba, bc) / denom), -1.0, 1.0)

    return float(np.degrees(np.arccos(cosang)))


def landmarks_to_array(lm) -> np.ndarray:
    """Convert MediaPipe pose landmarks into a fixed (33, 4) array."""
    arr = np.zeros((33, 4), dtype=np.float32)
    for i in range(33):
        arr[i, 0] = lm[i].x
        arr[i, 1] = lm[i].y
        arr[i, 2] = lm[i].z
        arr[i, 3] = lm[i].visibility
    return arr


def overlay_text(frame: np.ndarray, lines: List[str], x: int = 20, y: int = 30, lh: int = 22) -> None:
    """Draw multiple lines of text on a frame."""
    for i, t in enumerate(lines):
        cv2.putText(
            frame,
            t,
            (x, y + i * lh),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


# -----------------------------
# Bottom detection
# -----------------------------

class KneeMinimaDetector:
    """Detect lunge bottom using local minima of EMA-smoothed knee angle.
    
    For Lunges, we use the average knee angle (similar to squat) because
    both knees flex significantly during a deep lunge.
    """

    def __init__(self, ema_alpha: float = 0.3, max_bottom_deg: float = 140.0, min_gap: int = 20):
        self.ema_alpha = float(ema_alpha)
        self.max_bottom_deg = float(max_bottom_deg)
        self.min_gap = int(min_gap)

        self.knee_ema: Optional[float] = None
        self.last_event_frame: int = -10**9

        self.k_hist: Deque[float] = deque(maxlen=3)
        self.f_hist: Deque[int] = deque(maxlen=3)

    def update(self, knee_deg: Optional[float], frame_idx: int) -> Tuple[Optional[int], Optional[float]]:
        if knee_deg is None:
            return None, self.knee_ema

        k_raw = float(knee_deg)

        if self.knee_ema is None:
            self.knee_ema = k_raw
        else:
            a = self.ema_alpha
            self.knee_ema = a * k_raw + (1.0 - a) * self.knee_ema

        self.k_hist.append(self.knee_ema)
        self.f_hist.append(frame_idx)

        if len(self.k_hist) < 3:
            return None, self.knee_ema

        k0, k1, k2 = self.k_hist[0], self.k_hist[1], self.k_hist[2]
        f1 = self.f_hist[1]

        # Local minima
        is_min = (k1 < k0) and (k1 < k2)

        if is_min and (k1 <= self.max_bottom_deg) and (f1 - self.last_event_frame >= self.min_gap):
            self.last_event_frame = f1
            return f1, self.knee_ema

        return None, self.knee_ema


# -----------------------------
# Snippet capture structures
# -----------------------------

@dataclass
class PendingSnippet:
    snip_id: str
    frames: List[np.ndarray]
    kpts: List[Optional[np.ndarray]]
    need_post: int
    event_frame: int
    start_frame: int
    end_frame: int


# -----------------------------
# IO helpers
# -----------------------------

def write_snippet_mp4(
    out_path: str,
    frames: List[np.ndarray],
    fps: float,
    size_wh: Tuple[int, int],
    label: str,
    snip_id: str,
) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, float(fps), size_wh)

    if not writer.isOpened():
        raise RuntimeError("Cannot open VideoWriter: " + out_path)

    for f in frames:
        cv2.putText(
            f,
            f"{label} | {snip_id}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        writer.write(f)

    writer.release()


def write_snippet_npz(
    out_path: str,
    kpts_list: List[Optional[np.ndarray]],
    label: str,
    fps: float,
    event_frame: int,
    start_frame: int,
    end_frame: int,
) -> None:
    T = len(kpts_list)
    kp_seq = np.zeros((T, 33, 4), dtype=np.float32)
    mask = np.zeros((T,), dtype=np.float32)

    for i, k in enumerate(kpts_list):
        if k is not None:
            kp_seq[i] = k
            mask[i] = 1.0

    np.savez_compressed(
        out_path,
        keypoints=kp_seq,
        mask=mask,
        label=label,
        fps=float(fps),
        event_frame=int(event_frame),
        start_frame=int(start_frame),
        end_frame=int(end_frame),
    )


# -----------------------------
# Core pipeline
# -----------------------------

def compute_knee_angle_avg(lm, mp_pose) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Compute average knee angle. For lunges, this is a decent proxy for depth."""
    LHIP = mp_pose.PoseLandmark.LEFT_HIP.value
    RHIP = mp_pose.PoseLandmark.RIGHT_HIP.value
    LKNEE = mp_pose.PoseLandmark.LEFT_KNEE.value
    RKNEE = mp_pose.PoseLandmark.RIGHT_KNEE.value
    LANK = mp_pose.PoseLandmark.LEFT_ANKLE.value
    RANK = mp_pose.PoseLandmark.RIGHT_ANKLE.value

    lhip = (lm[LHIP].x, lm[LHIP].y)
    rhip = (lm[RHIP].x, lm[RHIP].y)
    lknee = (lm[LKNEE].x, lm[LKNEE].y)
    rknee = (lm[RKNEE].x, lm[RKNEE].y)
    lank = (lm[LANK].x, lm[LANK].y)
    rank = (lm[RANK].x, lm[RANK].y)

    knee_l = angle_3pts(lhip, lknee, lank)
    knee_r = angle_3pts(rhip, rknee, rank)
    knee_deg = float((knee_l + knee_r) * 0.5)

    return knee_deg, knee_l, knee_r


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="single video file OR folder of videos")
    ap.add_argument(
        "--label", required=True,
        choices=["correct", "torso_lean_forward", "knee_over_toe", "not_deep_enough"],
        help="Lunge labels",
    )
    ap.add_argument("--outdir", default="dataset/lunges/dataset_bottom")
    ap.add_argument("--pre", type=int, default=15)
    ap.add_argument("--post", type=int, default=15)
    ap.add_argument("--ema_alpha", type=float, default=0.3)
    ap.add_argument("--max_bottom_deg", type=float, default=130.0, help="Lower threshold for lunge depth")
    ap.add_argument("--min_gap", type=int, default=18)
    ap.add_argument("--show", action="store_true")
    return ap


VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v")


def collect_video_paths(path: str) -> List[str]:
    if os.path.isfile(path):
        return [path]
    if os.path.isdir(path):
        files: List[str] = []
        for ext in VIDEO_EXTS:
            files.extend(glob.glob(os.path.join(path, f"*{ext}")))
            files.extend(glob.glob(os.path.join(path, f"*{ext.upper()}")))
        files = sorted(set(files))
        if not files:
            raise FileNotFoundError(f"No video files found in: {path}")
        return files
    raise FileNotFoundError(f"Path does not exist: {path}")


def process_one_video(video_path: str, args: argparse.Namespace) -> int:
    vidname = os.path.splitext(os.path.basename(video_path))[0]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[WARN] Cannot open: {video_path}, skipping.")
        return 0

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    mp_pose = mp.solutions.pose
    mp_draw = mp.solutions.drawing_utils

    det = KneeMinimaDetector(
        ema_alpha=args.ema_alpha,
        max_bottom_deg=args.max_bottom_deg,
        min_gap=args.min_gap,
    )

    hist_len = args.pre + 60
    frame_hist: Deque[np.ndarray] = deque(maxlen=hist_len)
    kpt_hist: Deque[Optional[np.ndarray]] = deque(maxlen=hist_len)
    idx_hist: Deque[int] = deque(maxlen=hist_len)

    pending: Optional[PendingSnippet] = None
    snip_count = 0

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:

        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)

            knee_deg: Optional[float] = None
            knee_l: Optional[float] = None
            knee_r: Optional[float] = None
            kpts: Optional[np.ndarray] = None

            if res.pose_landmarks:
                lm = res.pose_landmarks.landmark
                kpts = landmarks_to_array(lm)
                knee_deg, knee_l, knee_r = compute_knee_angle_avg(lm, mp_pose)

                mp_draw.draw_landmarks(
                    frame,
                    res.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_draw.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2),
                    connection_drawing_spec=mp_draw.DrawingSpec(color=(255, 255, 255), thickness=2),
                )

            event_frame, knee_ema = det.update(knee_deg, frame_idx)

            frame_hist.append(frame.copy())
            kpt_hist.append(kpts)
            idx_hist.append(frame_idx)

            if pending is not None:
                pending.frames.append(frame.copy())
                pending.kpts.append(kpts)
                pending.need_post -= 1

                if pending.need_post <= 0:
                    snip_id = pending.snip_id
                    out_mp4 = os.path.join(args.outdir, snip_id + ".mp4")
                    out_npz = os.path.join(args.outdir, snip_id + ".npz")

                    write_snippet_mp4(out_mp4, pending.frames, fps, (W, H), args.label, snip_id)
                    write_snippet_npz(
                        out_npz, pending.kpts, args.label, fps,
                        pending.event_frame, pending.start_frame, pending.end_frame
                    )

                    print("Saved:", out_mp4)
                    print("Saved:", out_npz)

                    snip_count += 1
                    pending = None

            if event_frame is not None and pending is None:
                start_frame = event_frame - args.pre
                end_frame = event_frame + args.post

                frames_init: List[np.ndarray] = []
                kpts_init: List[Optional[np.ndarray]] = []

                if len(idx_hist) > 0 and start_frame >= idx_hist[0]:
                    for f, k, idx in zip(frame_hist, kpt_hist, idx_hist):
                        if start_frame <= idx <= frame_idx:
                            frames_init.append(f.copy())
                            kpts_init.append(k)

                    need_post = end_frame - frame_idx
                    if need_post < 0: need_post = 0

                    snip_id = f"{vidname}_{args.label}_snip{snip_count:03d}"
                    pending = PendingSnippet(
                        snip_id=snip_id,
                        frames=frames_init,
                        kpts=kpts_init,
                        need_post=need_post,
                        event_frame=int(event_frame),
                        start_frame=int(start_frame),
                        end_frame=int(end_frame),
                    )

            if args.show:
                dbg = frame.copy()
                lines = [
                    f"label={args.label} frame={frame_idx}",
                    (f"knee_raw(avg)={knee_deg:.1f}" if knee_deg else "knee_raw=NA"),
                    (f"knee_ema={knee_ema:.1f}" if knee_ema else "knee_ema=NA"),
                ]
                if event_frame is not None:
                    lines.append(f"EVENT: BOTTOM at {event_frame}")
                overlay_text(dbg, lines)
                cv2.imshow("extract_bottom_lunges", dbg)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

            frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    return snip_count


def main() -> None:
    args = build_arg_parser().parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    video_paths = collect_video_paths(args.video)
    print(f"[BATCH] {len(video_paths)} video(s) to process, label={args.label}")

    total_snips = 0
    for i, vp in enumerate(video_paths, 1):
        print(f"\n[{i}/{len(video_paths)}] {os.path.basename(vp)}")
        n = process_one_video(vp, args)
        total_snips += n

    print(f"\n[DONE] Total: {total_snips} snippets.")

if __name__ == "__main__":
    main()
