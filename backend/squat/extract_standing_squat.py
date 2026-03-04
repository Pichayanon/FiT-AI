"""
Extract standing-phase segments from squat video(s).

--video can be a single file OR a folder containing videos.

Labels:
  good_stand       — correct stance width
  stand_too_narrow — feet too close together
  stand_too_wide   — feet too far apart

Usage:
  python squat/extract_standing_squat.py --video path.mp4 --label good_stand --show
  python squat/extract_standing_squat.py --video dataset/squat/good_stand/ --label good_stand

Output: dataset/squat/dataset_standing/<vidname>_<label>_stand<NNN>_f<start>-<end>.npz
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
    """Compute the angle (in degrees) at point b formed by points a-b-c.

    :param a: point (x, y)
    :param b: vertex point (x, y)
    :param c: point (x, y)
    :returns: angle at b in degrees
    """
    a_np = np.array(a, dtype=np.float32)
    b_np = np.array(b, dtype=np.float32)
    c_np = np.array(c, dtype=np.float32)

    ba = a_np - b_np
    bc = c_np - b_np

    denom = (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    cosang = np.clip(float(np.dot(ba, bc) / denom), -1.0, 1.0)

    return float(np.degrees(np.arccos(cosang)))


def landmarks_to_array(lm) -> np.ndarray:
    """Convert MediaPipe pose landmarks into a fixed (33, 4) array.

    Output format per landmark:
      [x, y, z, visibility]

    :param lm: MediaPipe landmarks list (length 33)
    :returns: numpy array of shape (33, 4), dtype float32
    """
    arr = np.zeros((33, 4), dtype=np.float32)
    for i in range(33):
        arr[i, 0] = lm[i].x
        arr[i, 1] = lm[i].y
        arr[i, 2] = lm[i].z
        arr[i, 3] = lm[i].visibility
    return arr


def overlay_text(frame: np.ndarray, lines: List[str], x: int = 20, y: int = 30, lh: int = 22) -> None:
    """Draw multiple lines of text on a frame.

    :param frame: image frame (BGR)
    :param lines: list of strings to draw
    :param x: left position
    :param y: top position
    :param lh: line height
    :returns: None
    """
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


def compute_knee_angle_avg(lm, mp_pose) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Compute average knee angle from MediaPipe landmarks.

    :param lm: MediaPipe landmarks list
    :param mp_pose: mediapipe pose module (for landmark indices)
    :returns: (knee_deg_avg, knee_left, knee_right)
              Any of them can be None if something is missing.
    """
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


# -----------------------------
# EMA helper (same idea as bottom script)
# -----------------------------

class EMA:
    """Simple EMA (exponential moving average) helper.

    :param alpha: smoothing factor (0..1)
    """

    def __init__(self, alpha: float = 0.30):
        self.alpha = float(alpha)
        self.v: Optional[float] = None

    def update(self, x: Optional[float]) -> Optional[float]:
        """Update EMA with a new value.

        :param x: new value (or None)
        :returns: updated EMA (or last EMA if x is None)
        """
        if x is None:
            return self.v

        x_f = float(x)
        if self.v is None:
            self.v = x_f
        else:
            a = self.alpha
            self.v = a * x_f + (1.0 - a) * self.v

        return self.v


# -----------------------------
# Snippet capture structures
# -----------------------------

@dataclass
class PendingSegment:
    """Hold segment data while we optionally collect post-padding frames.

    :param seg_id: segment identifier (used for filenames)
    :param frames: list of BGR frames (copied)
    :param kpts: list of keypoint arrays or None (aligned with frames)
    :param need_post: how many more frames to collect after segment end
    :param start_frame: segment start frame index (including pad_pre if used)
    :param end_frame: segment end frame index (without pad_post yet)
    """
    seg_id: str
    frames: List[np.ndarray]
    kpts: List[Optional[np.ndarray]]
    need_post: int
    start_frame: int
    end_frame: int


# -----------------------------
# IO helpers (same pattern as bottom script)
# -----------------------------

def write_snippet_mp4(
    out_path: str,
    frames: List[np.ndarray],
    fps: float,
    size_wh: Tuple[int, int],
    label: str,
    snip_id: str,
) -> None:
    """Write a list of frames to an MP4 file.

    :param out_path: output .mp4 path
    :param frames: list of BGR frames
    :param fps: frames per second
    :param size_wh: (W, H)
    :param label: label text to overlay
    :param snip_id: snippet/segment id to overlay
    :returns: None
    :raises: RuntimeError if VideoWriter fails to open
    """
    W, H = size_wh
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, float(fps), (W, H))

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
    start_frame: int,
    end_frame: int,
) -> None:
    """Write segment keypoints + metadata into a compressed NPZ file.

    :param out_path: output .npz path
    :param kpts_list: list of (33, 4) arrays or None (aligned with frames)
    :param label: segment label
    :param fps: video fps
    :param start_frame: segment start frame index
    :param end_frame: segment end frame index
    :returns: None
    """
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
        start_frame=int(start_frame),
        end_frame=int(end_frame),
    )


# -----------------------------
# CLI
# -----------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser.

    :returns: configured ArgumentParser
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="single video file OR folder of videos")
    ap.add_argument(
        "--label", required=True,
        choices=["good_stand", "stand_too_narrow", "stand_too_wide"],
        help="good_stand | stand_too_narrow | stand_too_wide",
    )
    ap.add_argument("--outdir", default="dataset/squat/dataset_standing")

    # Standing definition
    ap.add_argument("--stand_th", type=float, default=155.0, help="knee >= this is standing")
    ap.add_argument("--stand_delta", type=float, default=5.0, help="knee change <= this")
    ap.add_argument("--streak", type=int, default=3, help="need this many stable frames")

    # Segment trimming
    ap.add_argument("--pad_pre", type=int, default=0, help="include extra frames before stand start (if available)")
    ap.add_argument("--pad_post", type=int, default=0, help="include extra frames after stand end")

    # MediaPipe
    ap.add_argument("--det", type=float, default=0.6)
    ap.add_argument("--track", type=float, default=0.6)

    ap.add_argument("--show", action="store_true")
    return ap


# -----------------------------
# Video file extensions
# -----------------------------
VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v")


def collect_video_paths(path: str) -> List[str]:
    """Return a sorted list of video file paths.

    If *path* is a file, return it as a single-element list.
    If *path* is a directory, glob all video files inside (non-recursive).

    :param path: file or directory path
    :returns: list of absolute video paths
    :raises: FileNotFoundError if path does not exist
    """
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


# -----------------------------
# Process one video
# -----------------------------

def process_one_video(video_path: str, args: argparse.Namespace) -> int:
    """Extract standing segments from a single video.

    :param video_path: path to the video file
    :param args: CLI arguments
    :returns: number of segments saved
    """
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

    ema = EMA(alpha=args.ema_alpha)

    # Segment state
    in_stand = False
    stand_frames: List[np.ndarray] = []
    stand_kpts: List[Optional[np.ndarray]] = []
    stand_start_idx: Optional[int] = None

    # Pre-padding history buffer
    hist_len = max(1, args.pad_pre) + 5
    frame_hist: Deque[np.ndarray] = deque(maxlen=hist_len)
    kpt_hist: Deque[Optional[np.ndarray]] = deque(maxlen=hist_len)
    idx_hist: Deque[int] = deque(maxlen=hist_len)

    pending: Optional[PendingSegment] = None
    seg_count = 0

    def reset_segment() -> None:
        """Reset current standing segment state.

        :returns: None
        """
        nonlocal in_stand, stand_frames, stand_kpts, stand_start_idx, pending
        in_stand = False
        stand_frames = []
        stand_kpts = []
        stand_start_idx = None
        pending = None

    def finalize_segment(end_idx_exclusive: int, post_frames: List[np.ndarray], post_kpts: List[Optional[np.ndarray]]) -> None:
        """Finalize and save a standing segment.

        :param end_idx_exclusive: index after the last "standing" frame
        :param post_frames: post padding frames to append (already limited if needed)
        :param post_kpts: post padding keypoints aligned with post_frames
        :returns: None
        """
        nonlocal seg_count, pending, in_stand, stand_frames, stand_kpts, stand_start_idx

        if stand_start_idx is None:
            reset_segment()
            return

        # Add post padding if requested.
        if args.pad_post > 0 and len(post_frames) > 0:
            stand_frames.extend(post_frames[:args.pad_post])
            stand_kpts.extend(post_kpts[:args.pad_post])

        # Enforce minimum length.
        if len(stand_frames) < args.min_stand_frames:
            reset_segment()
            return

        seg_id = f"{vidname}_{args.label}_stand{seg_count:03d}_f{stand_start_idx:06d}-{end_idx_exclusive-1:06d}"
        out_mp4 = os.path.join(args.outdir, seg_id + ".mp4")
        out_npz = os.path.join(args.outdir, seg_id + ".npz")

        write_snippet_mp4(
            out_path=out_mp4,
            frames=[f.copy() for f in stand_frames],
            fps=float(fps),
            size_wh=(W, H),
            label=args.label,
            snip_id=seg_id,
        )
        write_snippet_npz(
            out_path=out_npz,
            kpts_list=stand_kpts,
            label=args.label,
            fps=float(fps),
            start_frame=int(stand_start_idx),
            end_frame=int(end_idx_exclusive - 1),
        )

        print("Saved:", out_mp4)
        print("Saved:", out_npz)

        seg_count += 1
        reset_segment()

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=args.det,
        min_tracking_confidence=args.track,
    ) as pose:

        # When we leave standing, we may collect pad_post frames.
        post_buffer_frames: List[np.ndarray] = []
        post_buffer_kpts: List[Optional[np.ndarray]] = []
        post_collect_left = 0

        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                # If the video ends while standing, finalize with whatever we have.
                if in_stand:
                    finalize_segment(frame_idx, [], [])
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

            knee_ema = ema.update(knee_deg)
            is_standing = (knee_ema is not None) and (knee_ema >= args.stand_deg)

            # Keep history for pre padding.
            frame_hist.append(frame.copy())
            kpt_hist.append(kpts)
            idx_hist.append(frame_idx)

            # If we are collecting post padding after leaving stand.
            if post_collect_left > 0:
                post_buffer_frames.append(frame.copy())
                post_buffer_kpts.append(kpts)
                post_collect_left -= 1

                if post_collect_left == 0 and in_stand:
                    # After collecting enough post frames, finalize.
                    finalize_segment(frame_idx + 1, post_buffer_frames, post_buffer_kpts)
                    post_buffer_frames = []
                    post_buffer_kpts = []

            # Transitions (start / continue / end)
            if is_standing and not in_stand:
                # START standing segment.
                in_stand = True
                stand_frames = []
                stand_kpts = []
                stand_start_idx = frame_idx

                # Add pre padding from history if requested.
                if args.pad_pre > 0:
                    take_n = min(args.pad_pre, len(frame_hist))
                    for f, k, idx in list(zip(frame_hist, kpt_hist, idx_hist))[-take_n:]:
                        stand_frames.append(f.copy())
                        stand_kpts.append(k)
                    stand_start_idx = idx_hist[-take_n]

                # Include current frame.
                stand_frames.append(frame.copy())
                stand_kpts.append(kpts)

            elif is_standing and in_stand:
                # CONTINUE standing segment.
                stand_frames.append(frame.copy())
                stand_kpts.append(kpts)

            elif (not is_standing) and in_stand:
                # END standing segment.
                if args.pad_post > 0:
                    post_buffer_frames = []
                    post_buffer_kpts = []
                    post_collect_left = args.pad_post
                else:
                    finalize_segment(frame_idx, [], [])

            # Optional debug window.
            if args.show:
                dbg = frame.copy()
                lines = [
                    f"label={args.label} frame={frame_idx}",
                    (
                        f"knee_raw(avg)={knee_deg:.1f} (L={knee_l:.1f} R={knee_r:.1f})"
                        if knee_deg is not None
                        else "knee_raw(avg)=NA"
                    ),
                    (f"knee_ema={knee_ema:.1f}" if knee_ema is not None else "knee_ema=NA"),
                    f"STAND if knee_ema >= {args.stand_deg:.1f} -> {is_standing}",
                    f"in_stand={in_stand} seg_len={len(stand_frames)} min={args.min_stand_frames}",
                    f"pad_pre={args.pad_pre} pad_post={args.pad_post}",
                ]
                overlay_text(dbg, lines, 20, 30, 22)
                cv2.imshow("extract_standing_squat", dbg)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

            frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    return seg_count


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    """Run standing segment extraction from one or more videos.

    :returns: None
    """
    args = build_arg_parser().parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    video_paths = collect_video_paths(args.video)
    print(f"[BATCH] {len(video_paths)} video(s) to process, label={args.label}")
    print(f"[BATCH] output: {args.outdir}")

    total_segs = 0
    for i, vp in enumerate(video_paths, 1):
        print(f"\n[{i}/{len(video_paths)}] {os.path.basename(vp)}")
        n = process_one_video(vp, args)
        print(f"  -> {n} segment(s)")
        total_segs += n

    print(f"\n[DONE] Total: {total_segs} standing segments from {len(video_paths)} video(s)")
    print(f"[DONE] Output folder: {args.outdir}")


if __name__ == "__main__":
    main()
