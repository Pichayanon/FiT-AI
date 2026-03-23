"""Shared lunge feature extraction used by training and realtime inference.

Each extractor accepts either:
    - a single MediaPipe landmark list
    - a single numpy landmark frame shaped like (33, 3/4)
    - a numpy sequence shaped like (T, 33, 3/4) for bottom features
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np

from shared.math_utils import (
    angle_from_points,
    landmarks_to_xyz,
    point_distance,
    safe_norm,
)


# ---------------------------------------------------------------
# Constants (shared between training and streaming)
# ---------------------------------------------------------------

BOTTOM_FEATURE_DIM = 42
PHASE_FEATURE_DIM = 6

LEFT_EAR_INDEX = 7
RIGHT_EAR_INDEX = 8
LEFT_SHOULDER_INDEX = 11
RIGHT_SHOULDER_INDEX = 12
LEFT_HIP_INDEX = 23
RIGHT_HIP_INDEX = 24
LEFT_KNEE_INDEX = 25
RIGHT_KNEE_INDEX = 26
LEFT_ANKLE_INDEX = 27
RIGHT_ANKLE_INDEX = 28
LEFT_HEEL_INDEX = 29
RIGHT_HEEL_INDEX = 30
LEFT_FOOT_INDEX = 31
RIGHT_FOOT_INDEX = 32

# ---------------------------------------------------------------
# Shared landmark parsing
# ---------------------------------------------------------------

def _as_xyz_frame(landmarks: Any) -> np.ndarray:
    """Return a single frame of xyz landmarks as shape (33, 3)."""
    if isinstance(landmarks, np.ndarray):
        landmark_array = np.asarray(landmarks, dtype=np.float32)
        if (
            landmark_array.ndim != 2
            or landmark_array.shape[0] != 33
            or landmark_array.shape[1] < 3
        ):
            raise ValueError("Expected landmark frame with shape (33, 3/4).")
        return landmark_array[:, :3].astype(np.float32)
    return landmarks_to_xyz(landmarks)


def _as_xyz_sequence(landmark_sequence: np.ndarray) -> np.ndarray:
    """Return a landmark sequence as shape (T, 33, 3)."""
    landmark_array = np.asarray(landmark_sequence, dtype=np.float32)
    if (
        landmark_array.ndim != 3
        or landmark_array.shape[1] != 33
        or landmark_array.shape[2] < 3
    ):
        raise ValueError("Expected landmark sequence with shape (T, 33, 3/4).")
    return landmark_array[..., :3].astype(np.float32)


def _as_xy_frame(landmarks: Any) -> np.ndarray:
    """Return a single frame of xy landmarks as shape (33, 2)."""
    if isinstance(landmarks, np.ndarray):
        landmark_array = np.asarray(landmarks, dtype=np.float32)
        if (
            landmark_array.ndim != 2
            or landmark_array.shape[0] != 33
            or landmark_array.shape[1] < 2
        ):
            raise ValueError("Expected landmark frame with shape (33, 2/3/4).")
        return landmark_array[:, :2].astype(np.float32)
    return landmarks_to_xyz(landmarks)[:, :2]


# ---------------------------------------------------------------
# Bottom features (42-D)
# ---------------------------------------------------------------

def _normalize_facing_direction(frame_xyz: np.ndarray) -> np.ndarray:
    """Flip X so the athlete always appears to face right."""
    average_foot_direction = (
        frame_xyz[LEFT_FOOT_INDEX][0] - frame_xyz[LEFT_HEEL_INDEX][0]
    ) + (
        frame_xyz[RIGHT_FOOT_INDEX][0] - frame_xyz[RIGHT_HEEL_INDEX][0]
    )
    facing_right = average_foot_direction >= 0

    orientation_normalized_xyz = frame_xyz.copy()
    if not facing_right:
        orientation_normalized_xyz[:, 0] = -orientation_normalized_xyz[:, 0]
    return orientation_normalized_xyz


def _select_front_back_joint_indices(
    orientation_normalized_xyz: np.ndarray,
) -> dict[str, int]:
    """Return joint indices after ordering the body into front/back sides."""
    left_ankle_x = orientation_normalized_xyz[LEFT_ANKLE_INDEX][0]
    right_ankle_x = orientation_normalized_xyz[RIGHT_ANKLE_INDEX][0]
    is_left_leg_front = left_ankle_x > right_ankle_x

    if is_left_leg_front:
        return {
            "front_ear": LEFT_EAR_INDEX,
            "back_ear": RIGHT_EAR_INDEX,
            "front_shoulder": LEFT_SHOULDER_INDEX,
            "back_shoulder": RIGHT_SHOULDER_INDEX,
            "front_hip": LEFT_HIP_INDEX,
            "back_hip": RIGHT_HIP_INDEX,
            "front_knee": LEFT_KNEE_INDEX,
            "back_knee": RIGHT_KNEE_INDEX,
            "front_ankle": LEFT_ANKLE_INDEX,
            "back_ankle": RIGHT_ANKLE_INDEX,
        }

    return {
        "front_ear": RIGHT_EAR_INDEX,
        "back_ear": LEFT_EAR_INDEX,
        "front_shoulder": RIGHT_SHOULDER_INDEX,
        "back_shoulder": LEFT_SHOULDER_INDEX,
        "front_hip": RIGHT_HIP_INDEX,
        "back_hip": LEFT_HIP_INDEX,
        "front_knee": RIGHT_KNEE_INDEX,
        "back_knee": LEFT_KNEE_INDEX,
        "front_ankle": RIGHT_ANKLE_INDEX,
        "back_ankle": LEFT_ANKLE_INDEX,
    }


def _extract_bottom_features_from_single_frame_xyz(frame_xyz: np.ndarray) -> np.ndarray:
    """Extract 42-dim features for the lunge bottom TCN.

    Feature order:
        [0-29]  front/back ear, shoulder, hip, knee, ankle xyz
        [30]    front knee angle / 180
        [31]    back knee angle / 180
        [32]    front hip angle / 180
        [33]    back hip angle / 180
        [34]    torso tilt / 180
        [35]    stride length ratio
        [36]    front knee-over-toe
        [37]    back knee-over-toe
        [38]    front knee height
        [39]    back knee height
        [40]    spine angle / 180
        [41]    hip drop / scale
    """
    feature_vector = np.zeros(BOTTOM_FEATURE_DIM, dtype=np.float32)

    orientation_normalized_xyz = _normalize_facing_direction(frame_xyz)
    joint_indices = _select_front_back_joint_indices(orientation_normalized_xyz)

    front_ear = orientation_normalized_xyz[joint_indices["front_ear"]]
    back_ear = orientation_normalized_xyz[joint_indices["back_ear"]]
    front_shoulder = orientation_normalized_xyz[joint_indices["front_shoulder"]]
    back_shoulder = orientation_normalized_xyz[joint_indices["back_shoulder"]]
    front_hip = orientation_normalized_xyz[joint_indices["front_hip"]]
    back_hip = orientation_normalized_xyz[joint_indices["back_hip"]]
    front_knee = orientation_normalized_xyz[joint_indices["front_knee"]]
    back_knee = orientation_normalized_xyz[joint_indices["back_knee"]]
    front_ankle = orientation_normalized_xyz[joint_indices["front_ankle"]]
    back_ankle = orientation_normalized_xyz[joint_indices["back_ankle"]]

    hip_midpoint = 0.5 * (front_hip + back_hip)
    shoulder_midpoint = 0.5 * (front_shoulder + back_shoulder)
    ear_midpoint = 0.5 * (front_ear + back_ear)

    torso_length = point_distance(hip_midpoint, shoulder_midpoint)
    reference_scale = torso_length if torso_length > 1e-4 else 1.0

    ordered_joint_indices = [
        joint_indices["front_ear"],
        joint_indices["back_ear"],
        joint_indices["front_shoulder"],
        joint_indices["back_shoulder"],
        joint_indices["front_hip"],
        joint_indices["back_hip"],
        joint_indices["front_knee"],
        joint_indices["back_knee"],
        joint_indices["front_ankle"],
        joint_indices["back_ankle"],
    ]
    for joint_offset, joint_index in enumerate(ordered_joint_indices):
        start_index = joint_offset * 3
        end_index = start_index + 3
        feature_vector[start_index:end_index] = (
            orientation_normalized_xyz[joint_index] - hip_midpoint
        ) / reference_scale

    feature_vector[30] = (
        angle_from_points(front_hip, front_knee, front_ankle) / 180.0
    )
    feature_vector[31] = (
        angle_from_points(back_hip, back_knee, back_ankle) / 180.0
    )
    feature_vector[32] = (
        angle_from_points(front_shoulder, front_hip, front_knee) / 180.0
    )
    feature_vector[33] = (
        angle_from_points(back_shoulder, back_hip, back_knee) / 180.0
    )

    torso_vector = shoulder_midpoint - hip_midpoint
    vertical_up_vector = np.array([0.0, -1.0, 0.0], dtype=np.float32)
    torso_tilt_denominator = (
        safe_norm(torso_vector) * safe_norm(vertical_up_vector)
    ) + 1e-6
    torso_tilt_cosine = float(
        np.clip(
            np.dot(torso_vector, vertical_up_vector) / torso_tilt_denominator,
            -1.0,
            1.0,
        )
    )
    feature_vector[34] = float(np.degrees(np.arccos(torso_tilt_cosine))) / 180.0

    feature_vector[35] = point_distance(front_ankle, back_ankle) / reference_scale
    feature_vector[36] = front_knee[0] - front_ankle[0]
    feature_vector[37] = back_knee[0] - back_ankle[0]

    ground_y = max(front_ankle[1], back_ankle[1])
    feature_vector[38] = ground_y - front_knee[1]
    feature_vector[39] = ground_y - back_knee[1]
    feature_vector[40] = (
        angle_from_points(ear_midpoint, shoulder_midpoint, hip_midpoint) / 180.0
    )
    feature_vector[41] = (ground_y - hip_midpoint[1]) / reference_scale

    return feature_vector


def extract_bottom_features(landmarks: Any) -> np.ndarray:
    """Extract 42-dim bottom features from one frame or a sequence."""
    if isinstance(landmarks, np.ndarray) and landmarks.ndim == 3:
        landmark_sequence_xyz = _as_xyz_sequence(landmarks)
        return np.stack(
            [
                _extract_bottom_features_from_single_frame_xyz(
                    landmark_sequence_xyz[frame_index]
                )
                for frame_index in range(landmark_sequence_xyz.shape[0])
            ],
            axis=0,
        )
    return _extract_bottom_features_from_single_frame_xyz(
        _as_xyz_frame(landmarks)
    )


# ---------------------------------------------------------------
# Phase features (6-D)
# ---------------------------------------------------------------

def extract_phase_features(
    landmarks: Any,
    previous_heights: Optional[Tuple[float, float, float]],
) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    """Extract 6-dim features for the lunge phase TCN.

    Feature order:
        [0] hip height
        [1] shoulder height
        [2] knee height
        [3] hip velocity
        [4] shoulder velocity
        [5] knee velocity
    """
    frame_xy = _as_xy_frame(landmarks)

    hip_midpoint = (frame_xy[LEFT_HIP_INDEX] + frame_xy[RIGHT_HIP_INDEX]) * 0.5
    shoulder_midpoint = (
        frame_xy[LEFT_SHOULDER_INDEX] + frame_xy[RIGHT_SHOULDER_INDEX]
    ) * 0.5
    knee_midpoint = (frame_xy[LEFT_KNEE_INDEX] + frame_xy[RIGHT_KNEE_INDEX]) * 0.5

    torso_length = float(abs(shoulder_midpoint[1] - hip_midpoint[1]) + 1e-6)

    def normalize_vertical(point_xy: np.ndarray) -> float:
        return float((point_xy[1] - hip_midpoint[1]) / torso_length)

    hip_height = normalize_vertical(hip_midpoint)
    shoulder_height = normalize_vertical(shoulder_midpoint)
    knee_height = normalize_vertical(knee_midpoint)

    if previous_heights is None:
        hip_velocity = 0.0
        shoulder_velocity = 0.0
        knee_velocity = 0.0
    else:
        hip_velocity = hip_height - previous_heights[0]
        shoulder_velocity = shoulder_height - previous_heights[1]
        knee_velocity = knee_height - previous_heights[2]

    feature_vector = np.array(
        [
            hip_height,
            shoulder_height,
            knee_height,
            hip_velocity,
            shoulder_velocity,
            knee_velocity,
        ],
        dtype=np.float32,
    )
    current_heights = (hip_height, shoulder_height, knee_height)
    return feature_vector, current_heights


# ---------------------------------------------------------------
# Landmark smoother (used by streaming, not training)
# ---------------------------------------------------------------

class LandmarkSmoother:
    """Exponential moving average smoother for MediaPipe landmarks."""

    def __init__(self, alpha: float = 0.6) -> None:
        self.alpha = alpha
        self.previous_landmarks: Optional[np.ndarray] = None

    def reset(self) -> None:
        self.previous_landmarks = None

    def update(self, current_landmarks: np.ndarray) -> np.ndarray:
        if self.previous_landmarks is None:
            self.previous_landmarks = current_landmarks
            return current_landmarks

        self.previous_landmarks = (
            self.alpha * current_landmarks
            + (1.0 - self.alpha) * self.previous_landmarks
        )
        return self.previous_landmarks
