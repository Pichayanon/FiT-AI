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
    "l_hip": 23,
    "l_knee": 25,
    "l_ankle": 27,
    "r_shoulder": 12,
    "r_hip": 24,
    "r_knee": 26,
    "r_ankle": 28,
}


# -----------------------------
# Helpers
# -----------------------------

def angle_2d(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (
        np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6
    )
    return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))


def extract_features(lm):
    pts = {}
    for name, idx in IMPORTANT_LANDMARKS.items():
        pts[name] = np.array([lm[idx].x, lm[idx].y], dtype=np.float32)

    mid_hip_y = (pts["l_hip"][1] + pts["r_hip"][1]) / 2
    mid_shoulder_y = (pts["l_shoulder"][1] + pts["r_shoulder"][1]) / 2
    torso_len = abs(mid_shoulder_y - mid_hip_y) + 1e-6

    def ny(p):
        return (p[1] - mid_hip_y) / torso_len

    l_knee_angle = angle_2d(
        pts["l_hip"], pts["l_knee"], pts["l_ankle"]
    )
    r_knee_angle = angle_2d(
        pts["r_hip"], pts["r_knee"], pts["r_ankle"]
    )

    return np.array([
        ny(pts["l_shoulder"]),
        ny(pts["r_shoulder"]),
        ny(pts["l_hip"]),
        ny(pts["r_hip"]),
        ny(pts["l_knee"]),
        ny(pts["r_knee"]),
        ny(pts["l_ankle"]),
        ny(pts["r_ankle"]),
        l_knee_angle / 180.0,
        r_knee_angle / 180.0,
    ], dtype=np.float32)



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


def build_arg_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--outdir", default="dataset_phase_clean")
    ap.add_argument("--show", action="store_true")
    return ap


# -----------------------------
# Main
# -----------------------------

def main():
    args = build_arg_parser().parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    vidname = os.path.splitext(os.path.basename(args.video))[0]

    with open(args.labels, "r") as f:
        raw_ranges: Dict[str, List[List[int]]] = json.load(f)

    # 🔥 Merge stand → eccentric
    label_ranges = {
        "eccentric": [],
        "concentric": raw_ranges.get("concentric", []),
    }

    label_ranges["eccentric"].extend(raw_ranges.get("eccentric", []))
    label_ranges["eccentric"].extend(raw_ranges.get("stand", []))

    def frame_to_label(frame_idx: int) -> int:
        for phase, ranges in label_ranges.items():
            for start, end in ranges:
                if start <= frame_idx <= end:
                    return PHASE_MAP[phase]
        raise RuntimeError(f"Frame {frame_idx} is not labeled")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    mp_pose = mp.solutions.pose
    mp_draw = mp.solutions.drawing_utils

    features_seq = []
    labels_seq = []
    mask_seq = []

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

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)

            label = frame_to_label(frame_idx)

            if res.pose_landmarks:
                feats = extract_features(res.pose_landmarks.landmark)
                mask = 1
                mp_draw.draw_landmarks(
                    frame,
                    res.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                )
            else:
                feats = np.zeros((10,), dtype=np.float32)
                mask = 0

            features_seq.append(feats)
            labels_seq.append(label)
            mask_seq.append(mask)

            if args.show:
                phase_name = list(PHASE_MAP.keys())[label]
                overlay_text(frame, f"phase={phase_name}")
                cv2.imshow("extract_phase_clean", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

            frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()

    features = np.asarray(features_seq, dtype=np.float32)
    labels = np.asarray(labels_seq, dtype=np.int64)
    mask = np.asarray(mask_seq, dtype=np.float32)

    out_path = os.path.join(args.outdir, vidname + ".npz")
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
    print("classes:", PHASE_MAP)


if __name__ == "__main__":
    main()
