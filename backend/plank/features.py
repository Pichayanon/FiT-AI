"""Shared plank feature extraction helpers.

Used by both training and streaming so plank feature computation stays
consistent across offline and realtime paths.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from shared.math_utils import angle_3pts, dist, get_xyz, position_normalize


LEFT_SHOULDER_INDEX = 11
RIGHT_SHOULDER_INDEX = 12
LEFT_HIP_INDEX = 23
RIGHT_HIP_INDEX = 24
LEFT_ANKLE_INDEX = 27
RIGHT_ANKLE_INDEX = 28


def choose_visible_side(lm: list) -> str:
    """Select the body side with the more visible hip landmark."""
    right_hip_visibility = lm[RIGHT_HIP_INDEX].visibility
    left_hip_visibility = lm[LEFT_HIP_INDEX].visibility
    return "right" if right_hip_visibility > left_hip_visibility else "left"


def extract_frame_features(lm: list, side: str) -> Tuple[float, float, float]:
    """Extract normalized plank features from one landmark frame.

    The selected shoulder, hip, and ankle are first normalized in xyz space
    relative to the chosen-side hip. The resulting features are:
        - hip_signed_dist: signed hip distance from shoulder-ankle line
        - hip_height_norm: hip vertical offset from shoulder/ankle midpoint
        - body_angle: shoulder-hip-ankle angle in degrees
    """
    landmark_xyz = get_xyz(lm)

    if side == "right":
        shoulder_index = RIGHT_SHOULDER_INDEX
        hip_index = RIGHT_HIP_INDEX
        ankle_index = RIGHT_ANKLE_INDEX
    else:
        shoulder_index = LEFT_SHOULDER_INDEX
        hip_index = LEFT_HIP_INDEX
        ankle_index = LEFT_ANKLE_INDEX

    selected_shoulder = landmark_xyz[shoulder_index]
    selected_hip = landmark_xyz[hip_index]
    selected_ankle = landmark_xyz[ankle_index]

    torso_length = dist(selected_shoulder, selected_hip)
    body_length = dist(selected_shoulder, selected_ankle)
    hip_width = dist(
        landmark_xyz[LEFT_HIP_INDEX],
        landmark_xyz[RIGHT_HIP_INDEX],
    )
    shoulder_width = dist(
        landmark_xyz[LEFT_SHOULDER_INDEX],
        landmark_xyz[RIGHT_SHOULDER_INDEX],
    )
    reference_scale = torso_length
    if reference_scale <= 1e-4:
        reference_scale = body_length
    if reference_scale <= 1e-4:
        reference_scale = hip_width
    if reference_scale <= 1e-4:
        reference_scale = shoulder_width
    if reference_scale <= 1e-4:
        reference_scale = 1e-6

    normalized_shoulder = position_normalize(
        selected_shoulder,
        center=selected_hip,
        scale=reference_scale,
    )
    normalized_hip = position_normalize(
        selected_hip,
        center=selected_hip,
        scale=reference_scale,
    )
    normalized_ankle = position_normalize(
        selected_ankle,
        center=selected_hip,
        scale=reference_scale,
    )

    normalized_shoulder_xy = normalized_shoulder[:2]
    normalized_hip_xy = normalized_hip[:2]
    normalized_ankle_xy = normalized_ankle[:2]

    shoulder_to_ankle_vector = normalized_ankle_xy - normalized_shoulder_xy
    shoulder_to_hip_vector = normalized_hip_xy - normalized_shoulder_xy
    hip_signed_dist = np.cross(
        shoulder_to_ankle_vector,
        shoulder_to_hip_vector,
    ) / (np.linalg.norm(shoulder_to_ankle_vector) + 1e-6)

    body_angle = angle_3pts(
        normalized_shoulder,
        normalized_hip,
        normalized_ankle,
    )

    normalized_body_length = (
        np.linalg.norm(normalized_ankle_xy - normalized_shoulder_xy) + 1e-6
    )
    hip_height_norm = (
        normalized_hip_xy[1]
        - 0.5 * (normalized_shoulder_xy[1] + normalized_ankle_xy[1])
    ) / normalized_body_length

    return float(hip_signed_dist), float(hip_height_norm), float(body_angle)


def aggregate_window(
    frame_values: List[Tuple[float, float, float]],
) -> np.ndarray:
    """Aggregate per-frame tuples into the model's 6-D plank feature vector.

    Feature order:
        [0] mean hip signed distance
        [1] std hip signed distance
        [2] mean hip height offset
        [3] std hip height offset
        [4] mean body angle
        [5] std body angle
    """
    hip_signed_distances = [frame_value[0] for frame_value in frame_values]
    hip_height_offsets = [frame_value[1] for frame_value in frame_values]
    body_angles = [frame_value[2] for frame_value in frame_values]

    return np.array(
        [
            np.mean(hip_signed_distances),
            np.std(hip_signed_distances),
            np.mean(hip_height_offsets),
            np.std(hip_height_offsets),
            np.mean(body_angles),
            np.std(body_angles),
        ],
        dtype=np.float32,
    )
