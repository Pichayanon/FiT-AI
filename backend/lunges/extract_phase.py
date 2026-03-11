from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import cv2
import mediapipe as mp
import numpy as np


# -----------------------------
# Config
# -----------------------------

PHASE_MAP = {
    "eccentric": 0,
    "concentric": 1,
}

IMPORTANT_LANDMARKS = {
    "l_shoulder": 11,
    "r_shoulder": 12,
    "l_hip": 23,
    "r_hip": 24,
    "l_knee": 25,
    "r_knee": 26,
    "l_ankle": 27,
    "r_ankle": 28,
    "l_heel": 29,
    "r_heel": 30,
    "l_foot": 31,
    "r_foot": 32,
}


# -----------------------------
# Geometry Helpers
# -----------------------------


def angle_2d(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ba = a - b
    bc = c - b

    cos_angle = np.dot(ba, bc) / (
        np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6
    )

    return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))


# -----------------------------
# Feature Extraction
# -----------------------------


def extract_features(lm, prev_hip_y=None):

    pts = {}

    for name, idx in IMPORTANT_LANDMARKS.items():
        pts[name] = np.array([lm[idx].x, lm[idx].y], dtype=np.float32)

    # -------------------------
    # Torso normalization
    # -------------------------

    mid_hip = (pts["l_hip"] + pts["r_hip"]) / 2
    mid_shoulder = (pts["l_shoulder"] + pts["r_shoulder"]) / 2

    torso_len = abs(mid_shoulder[1] - mid_hip[1]) + 1e-6

    def ny(p):
        return (p[1] - mid_hip[1]) / torso_len

    # -------------------------
    # Knee angles
    # -------------------------

    l_knee_angle = angle_2d(
        pts["l_hip"],
        pts["l_knee"],
        pts["l_ankle"],
    )

    r_knee_angle = angle_2d(
        pts["r_hip"],
        pts["r_knee"],
        pts["r_ankle"],
    )

    # -------------------------
    # Detect front leg
    # -------------------------

    if pts["l_ankle"][0] > pts["r_ankle"][0]:

        front_knee_angle = l_knee_angle
        back_knee_angle = r_knee_angle

        front_knee = pts["l_knee"]
        front_ankle = pts["l_ankle"]

    else:

        front_knee_angle = r_knee_angle
        back_knee_angle = l_knee_angle

        front_knee = pts["r_knee"]
        front_ankle = pts["r_ankle"]

    # -------------------------
    # Step length
    # -------------------------

    step_length = abs(
        pts["l_ankle"][0] - pts["r_ankle"][0]
    ) / torso_len

    # -------------------------
    # Knee forward travel
    # -------------------------

    knee_forward = (
        front_knee[0] - front_ankle[0]
    ) / torso_len

    # -------------------------
    # Hip height
    # -------------------------

    hip_y = mid_hip[1]

    hip_height_norm = ny(mid_hip)

    # -------------------------
    # Hip velocity
    # -------------------------

    if prev_hip_y is None:
        hip_velocity = 0.0
    else:
        hip_velocity = (hip_y - prev_hip_y) / torso_len

    # -------------------------
    # Torso lean
    # -------------------------

    torso_angle = angle_2d(
        mid_shoulder,
        mid_hip,
        front_knee,
    )

    # -------------------------
    # Shoulder height
    # -------------------------

    shoulder_height = ny(mid_shoulder)

    # -------------------------
    # Knee asymmetry
    # -------------------------

    knee_angle_diff = abs(
        l_knee_angle - r_knee_angle
    )

    # -------------------------
    # Feature vector
    # -------------------------

    features = np.array(
        [
            hip_height_norm,
            shoulder_height,
            front_knee_angle / 180.0,
            back_knee_angle / 180.0,
            knee_angle_diff / 180.0,
            step_length,
            knee_forward,
            torso_angle / 180.0,
            hip_velocity,
        ],
        dtype=np.float32,
    )

    return features, hip_y


# -----------------------------
# Visualization
# -----------------------------


def overlay_text(frame, text, x=20, y=30):

    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )


# -----------------------------
# CLI
# -----------------------------


def build_arg_parser():

    ap = argparse.ArgumentParser(
        description="Extract lunge side-view phase features"
    )

    ap.add_argument(
        "--video",
        required=True,
        help="Input video path",
    )

    ap.add_argument(
        "--labels",
        required=True,
        help="JSON label file",
    )

    ap.add_argument(
        "--outdir",
        default="lunge_side_view_phase_clean",
    )

    ap.add_argument(
        "--show",
        action="store_true",
    )

    return ap


# -----------------------------
# Main
# -----------------------------


def main():

    args = build_arg_parser().parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    vidname = os.path.splitext(
        os.path.basename(args.video)
    )[0]

    with open(args.labels, "r") as f:
        raw_ranges: Dict[str, List[List[int]]] = json.load(f)

    label_ranges = {
        "eccentric": [],
        "concentric": raw_ranges.get("concentric", []),
    }

    label_ranges["eccentric"].extend(
        raw_ranges.get("eccentric", [])
    )

    label_ranges["eccentric"].extend(
        raw_ranges.get("stand", [])
    )

    def frame_to_label(frame_idx: int) -> int:

        for phase, ranges in label_ranges.items():

            for start, end in ranges:

                if start <= frame_idx <= end:
                    return PHASE_MAP[phase]

        raise RuntimeError(
            f"Frame {frame_idx} not labeled"
        )

    cap = cv2.VideoCapture(args.video)

    if not cap.isOpened():
        raise RuntimeError("Cannot open video")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    mp_pose = mp.solutions.pose
    mp_draw = mp.solutions.drawing_utils

    features_seq = []
    labels_seq = []
    mask_seq = []

    prev_hip_y = None

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:

        frame_idx = 0

        while True:

            ok, frame = cap.read()

            if not ok:
                break

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB,
            )

            res = pose.process(rgb)

            label = frame_to_label(frame_idx)

            if res.pose_landmarks:

                feats, prev_hip_y = extract_features(
                    res.pose_landmarks.landmark,
                    prev_hip_y,
                )

                mask = 1

                mp_draw.draw_landmarks(
                    frame,
                    res.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                )

            else:

                feats = np.zeros((9,), dtype=np.float32)
                mask = 0

            features_seq.append(feats)
            labels_seq.append(label)
            mask_seq.append(mask)

            if args.show:

                phase_name = list(
                    PHASE_MAP.keys()
                )[label]

                overlay_text(
                    frame,
                    f"phase={phase_name}",
                )

                cv2.imshow(
                    "lunge_phase_extractor",
                    frame,
                )

                if cv2.waitKey(1) & 0xFF == 27:
                    break

            frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()

    features = np.asarray(features_seq, dtype=np.float32)
    labels = np.asarray(labels_seq, dtype=np.int64)
    mask = np.asarray(mask_seq, dtype=np.float32)

    out_path = os.path.join(
        args.outdir,
        vidname + ".npz",
    )

    np.savez_compressed(
        out_path,
        features=features,
        labels=labels,
        mask=mask,
        fps=float(fps),
        phase_map=PHASE_MAP,
    )

    print("Saved:", out_path)
    print("features:", features.shape)
    print("labels:", labels.shape)


if __name__ == "__main__":
    main()