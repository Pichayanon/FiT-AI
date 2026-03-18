"""
Extract bottom-phase snippets from lunge video(s) using a trained PhaseTCN model.

Replaces the rule-based KneeMinimaDetector with model-based phase detection.
The bottom event is identified at the frame where the predicted phase transitions
from eccentric (0) to concentric (1).

Output format is identical to extract_bottom_lunges.py so that train_lunges_bottom.py
can consume the saved .npz files without modification.

Usage:
  python lunges/extract_bottom_lunges_phase.py \
      --video path.mp4 \
      --label correct \
      --model lunges/models/lunges_phase_tcn.pt

  python lunges/extract_bottom_lunges_phase.py \
      --video dataset/lunges/correct/ \
      --label correct \
      --model lunges/models/lunges_phase_tcn.pt

Output: dataset/lunges/dataset_bottom/<vidname>_<label>_snip<NNN>.npz
"""

from __future__ import annotations

if __package__ in {None, ""}:
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import argparse
import glob
import os
from typing import List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
import torch

from shared.tcn_models import PhaseTCN
from lunges.features import extract_phase_features, LandmarkSmoother


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v")

ECCENTRIC  = 0
CONCENTRIC = 1


# ─────────────────────────────────────────────────────────────────────────────
# Model loader
# ─────────────────────────────────────────────────────────────────────────────

def load_phase_model(model_path: str, device: str) -> Tuple[PhaseTCN, int, int]:
    """Load a saved PhaseTCN checkpoint.

    :param model_path: path to .pt checkpoint saved by train_lunge_phase.py
    :param device: 'cpu' or 'cuda'
    :returns: (model, in_dim, window_size)
    """
    ckpt = torch.load(model_path, map_location=device)
    in_dim      = int(ckpt["in_dim"])
    num_classes = int(ckpt.get("num_classes", 2))
    window      = int(ckpt.get("window", 30))

    model = PhaseTCN(in_dim=in_dim, num_classes=num_classes).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, in_dim, window


# ─────────────────────────────────────────────────────────────────────────────
# Phase inference
# ─────────────────────────────────────────────────────────────────────────────

def predict_phases(
    feature_seq: np.ndarray,   # (T, F)
    model: PhaseTCN,
    window: int,
    device: str,
) -> np.ndarray:               # (T,) int64, per-frame predicted phase
    """Run sliding-window phase inference and return per-frame predictions.

    Each frame's prediction is determined by majority vote across all windows
    that contain that frame.  Frames not covered by any window (at the tail)
    inherit the last known prediction.

    :param feature_seq: feature matrix of shape (T, F)
    :param model: loaded PhaseTCN in eval mode
    :param window: window size used during training (e.g. 30)
    :param device: torch device string
    :returns: integer array of shape (T,) with predicted phase per frame
    """
    T = len(feature_seq)
    vote_counts = np.zeros((T, 2), dtype=np.int32)

    with torch.no_grad():
        for start in range(0, max(1, T - window + 1)):
            end = start + window
            chunk = feature_seq[start:end]
            if len(chunk) < window:
                break

            x = torch.from_numpy(chunk).float().unsqueeze(0).to(device)  # (1, W, F)
            logits = model(x)                                              # (1, W, 2)
            preds  = logits.argmax(dim=-1).squeeze(0).cpu().numpy()       # (W,)

            for offset, pred in enumerate(preds):
                frame_idx = start + offset
                if frame_idx < T:
                    vote_counts[frame_idx, int(pred)] += 1

    # Frames with no votes default to eccentric (0)
    per_frame = vote_counts.argmax(axis=1).astype(np.int64)

    covered = vote_counts.sum(axis=1) > 0
    for t in range(T):
        if not covered[t]:
            per_frame[t] = per_frame[t - 1] if t > 0 else ECCENTRIC

    return per_frame


def find_bottom_events(
    per_frame: np.ndarray,
    min_gap: int = 18,
) -> List[int]:
    """Find frame indices where phase transitions from eccentric → concentric.

    :param per_frame: per-frame phase predictions
    :param min_gap: minimum frames between consecutive bottom events
    :returns: list of transition frame indices
    """
    events: List[int] = []
    last_event = -min_gap - 1

    for t in range(1, len(per_frame)):
        if (
            per_frame[t - 1] == ECCENTRIC
            and per_frame[t] == CONCENTRIC
            and (t - last_event) >= min_gap
        ):
            events.append(t)
            last_event = t

    return events


# ─────────────────────────────────────────────────────────────────────────────
# NPZ / MP4 writers  (identical format to extract_bottom_lunges.py)
# ─────────────────────────────────────────────────────────────────────────────

def write_snippet_npz(
    out_path: str,
    kpts_list: List[Optional[np.ndarray]],
    label: str,
    fps: float,
    event_frame: int,
    start_frame: int,
    end_frame: int,
) -> None:
    T      = len(kpts_list)
    kp_seq = np.zeros((T, 33, 4), dtype=np.float32)
    mask   = np.zeros((T,),       dtype=np.float32)

    for i, k in enumerate(kpts_list):
        if k is not None:
            kp_seq[i] = k
            mask[i]   = 1.0

    np.savez_compressed(
        out_path,
        keypoints   = kp_seq,
        mask        = mask,
        label       = label,
        fps         = float(fps),
        event_frame = int(event_frame),
        start_frame = int(start_frame),
        end_frame   = int(end_frame),
    )


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
            f, f"{label} | {snip_id}",
            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
            (255, 255, 255), 2, cv2.LINE_AA,
        )
        writer.write(f)

    writer.release()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def landmarks_to_array(lm) -> np.ndarray:
    arr = np.zeros((33, 4), dtype=np.float32)
    for i in range(33):
        arr[i] = [lm[i].x, lm[i].y, lm[i].z, lm[i].visibility]
    return arr


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


# ─────────────────────────────────────────────────────────────────────────────
# Per-video processing
# ─────────────────────────────────────────────────────────────────────────────

def process_one_video(
    video_path: str,
    args: argparse.Namespace,
    model: PhaseTCN,
    window: int,
    device: str,
) -> int:
    """Extract bottom snippets from a single video using phase model.

    Pass 1: extract MediaPipe landmarks + phase features for every frame.
    Pass 2: run phase model to get per-frame predictions.
    Pass 3: detect eccentric→concentric transitions and save snippets.

    :param video_path: input video path
    :param args: parsed CLI args
    :param model: loaded PhaseTCN
    :param window: model window size
    :param device: torch device string
    :returns: number of snippets saved
    """
    vidname = os.path.splitext(os.path.basename(video_path))[0]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[WARN] Cannot open: {video_path}, skipping.")
        return 0

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    mp_pose = mp.solutions.pose
    mp_draw = mp.solutions.drawing_utils

    smoother = LandmarkSmoother(alpha=0.6)

    # ── Pass 1: collect per-frame data ─────────────────────────────────────
    all_frames:      List[np.ndarray]           = []
    all_kpts:        List[Optional[np.ndarray]] = []
    all_phase_feats: List[np.ndarray]           = []

    previous_heights = None  # stateful: (hip_h, shoulder_h, knee_h)

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.8,
        min_tracking_confidence=0.8,
    ) as pose:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)

            kpts        = None
            phase_feat  = np.zeros(6, dtype=np.float32)  # PHASE_FEATURE_DIM = 6

            if res.pose_landmarks:
                lm   = res.pose_landmarks.landmark
                kpts = landmarks_to_array(lm)
                kpts = smoother.update(kpts)

                # lunges phase features are stateful (need previous heights for velocity)
                phase_feat, previous_heights = extract_phase_features(lm, previous_heights)

                if args.show:
                    mp_draw.draw_landmarks(
                        frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_draw.DrawingSpec(
                            color=(255, 255, 255), thickness=2, circle_radius=2),
                        connection_drawing_spec=mp_draw.DrawingSpec(
                            color=(255, 255, 255), thickness=2),
                    )
            else:
                # Reset velocity state on lost detection to avoid stale deltas
                previous_heights = None

            all_frames.append(frame.copy())
            all_kpts.append(kpts)
            all_phase_feats.append(phase_feat)

    cap.release()

    if len(all_frames) == 0:
        print(f"[WARN] No frames read from: {video_path}")
        return 0

    feature_seq = np.stack(all_phase_feats, axis=0)   # (T, 6)
    T = len(all_frames)

    # ── Pass 2: phase model inference ──────────────────────────────────────
    per_frame = predict_phases(feature_seq, model, window, device)

    # ── Pass 3: detect transitions & save snippets ─────────────────────────
    events     = find_bottom_events(per_frame, min_gap=args.min_gap)
    snip_count = 0

    for event_frame in events:
        start_frame = max(0, event_frame - args.pre)
        end_frame   = min(T - 1, event_frame + args.post)

        snippet_kpts   = all_kpts[start_frame   : end_frame + 1]
        snippet_frames = all_frames[start_frame : end_frame + 1]

        snip_id = f"{vidname}_{args.label}_snip{snip_count:03d}"
        out_npz = os.path.join(args.outdir, snip_id + ".npz")
        out_mp4 = os.path.join(args.outdir, snip_id + ".mp4")

        write_snippet_npz(
            out_path    = out_npz,
            kpts_list   = snippet_kpts,
            label       = args.label,
            fps         = float(fps),
            event_frame = event_frame,
            start_frame = start_frame,
            end_frame   = end_frame,
        )
        write_snippet_mp4(
            out_path = out_mp4,
            frames   = [f.copy() for f in snippet_frames],
            fps      = float(fps),
            size_wh  = (W, H),
            label    = args.label,
            snip_id  = snip_id,
        )

        print(f"  Saved: {out_npz}")
        print(f"  Saved: {out_mp4}")
        snip_count += 1

    # ── Optional debug visualisation ───────────────────────────────────────
    if args.show:
        event_set = set(events)
        for t, frame in enumerate(all_frames):
            phase_name = "concentric" if per_frame[t] == CONCENTRIC else "eccentric"
            color      = (0, 255, 0) if per_frame[t] == CONCENTRIC else (0, 100, 255)
            tag        = " ← BOTTOM" if t in event_set else ""
            cv2.putText(
                frame,
                f"frame={t}  phase={phase_name}{tag}",
                (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA,
            )
            cv2.imshow("extract_bottom_lunges_phase", frame)
            if cv2.waitKey(30) & 0xFF == 27:
                break
        cv2.destroyAllWindows()

    return snip_count


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Extract lunge bottom snippets using a trained PhaseTCN model."
    )
    ap.add_argument("--video",  required=True,
                    help="Single video file or folder of videos.")
    ap.add_argument("--label",  required=True,
                    choices=["correct", "torso_lean_forward", "knee_over_toe", "not_deep_enough"],
                    help="Form label for all snippets from this video/folder.")
    ap.add_argument("--model",  required=True,
                    help="Path to trained lunge phase model checkpoint (.pt).")
    ap.add_argument("--outdir", default="dataset/lunges/dataset_bottom",
                    help="Output directory for .npz and .mp4 snippets.")
    ap.add_argument("--pre",     type=int,   default=15,
                    help="Frames to include before the bottom event.")
    ap.add_argument("--post",    type=int,   default=15,
                    help="Frames to include after the bottom event.")
    ap.add_argument("--min_gap", type=int,   default=18,
                    help="Minimum frames between consecutive bottom detections.")
    ap.add_argument("--show",    action="store_true",
                    help="Display debug visualisation (requires display).")
    return ap


def main() -> None:
    args   = build_arg_parser().parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[INFO] Loading phase model from: {args.model}")
    model, in_dim, window = load_phase_model(args.model, device)
    print(f"[INFO] Model loaded — in_dim={in_dim}, window={window}, device={device}")

    os.makedirs(args.outdir, exist_ok=True)
    video_paths = collect_video_paths(args.video)
    print(f"[BATCH] {len(video_paths)} video(s), label={args.label}, outdir={args.outdir}")

    total = 0
    for i, vp in enumerate(video_paths, 1):
        print(f"\n[{i}/{len(video_paths)}] {os.path.basename(vp)}")
        n = process_one_video(vp, args, model, window, device)
        print(f"  → {n} snippet(s) saved")
        total += n

    print(f"\n[DONE] {total} bottom snippet(s) from {len(video_paths)} video(s)")
    print(f"[DONE] Output: {args.outdir}")


if __name__ == "__main__":
    main()
