from __future__ import annotations

if __package__ in {None, ""}:
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import argparse
import json
import os
from typing import Dict, List

import cv2
import mediapipe as mp
import numpy as np

from lunges.features import PHASE_FEATURE_DIM, extract_phase_features


# -----------------------------
# Config
# -----------------------------

PHASE_MAP = {
    "eccentric": 0,
    "concentric": 1,
}

# Keep this consistent with lunges.features.extract_phase_features().
FEATURE_DIM = PHASE_FEATURE_DIM


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

    with open(args.labels, "r") as label_file:
        raw_ranges: Dict[str, List[List[int]]] = json.load(label_file)

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

    previous_height_values = None

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

            pose_result = pose.process(rgb)

            label = frame_to_label(frame_idx)

            if pose_result.pose_landmarks:

                frame_feature_vector, previous_height_values = extract_phase_features(
                    pose_result.pose_landmarks.landmark,
                    previous_height_values,
                )

                mask = 1

                mp_draw.draw_landmarks(
                    frame,
                    pose_result.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                )

            else:

                frame_feature_vector = np.zeros((FEATURE_DIM,), dtype=np.float32)
                mask = 0

            features_seq.append(frame_feature_vector)
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
