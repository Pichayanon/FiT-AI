from __future__ import annotations

from typing import Any, Callable

import numpy as np

from shared.math_utils import (
    angle_from_points,
    angle_to_direction,
    as_xy_frame,
    as_xyz_frame,
    as_xyz_sequence,
    pick_scale,
    point_distance,
)

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

PhaseHeights = tuple[float, float, float]


def normalize_facing_direction(frame_xyz: np.ndarray) -> np.ndarray:
    average_foot_direction = (
        frame_xyz[LEFT_FOOT_INDEX][0] - frame_xyz[LEFT_HEEL_INDEX][0]
    ) + (frame_xyz[RIGHT_FOOT_INDEX][0] - frame_xyz[RIGHT_HEEL_INDEX][0])
    facing_right = average_foot_direction >= 0

    orientation_normalized_xyz = frame_xyz.copy()
    if not facing_right:
        orientation_normalized_xyz[:, 0] = -orientation_normalized_xyz[:, 0]
    return orientation_normalized_xyz


def select_front_back_joint_indices(
    orientation_normalized_xyz: np.ndarray,
) -> dict[str, int]:
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


def map_xyz_sequence_features(
    landmarks: np.ndarray,
    frame_extractor: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    frame_sequence_xyz = as_xyz_sequence(landmarks)
    return np.stack(
        [frame_extractor(frame_xyz) for frame_xyz in frame_sequence_xyz],
        axis=0,
    )


def bottom_frame_features(frame_xyz: np.ndarray) -> np.ndarray:
    features = np.zeros(BOTTOM_FEATURE_DIM, dtype=np.float32)

    orientation_normalized_xyz = normalize_facing_direction(frame_xyz)
    joint_indices = select_front_back_joint_indices(orientation_normalized_xyz)

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
    reference_scale = pick_scale(torso_length)

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
        features[start_index:end_index] = (
            orientation_normalized_xyz[joint_index] - hip_midpoint
        ) / reference_scale

    features[30] = angle_from_points(front_hip, front_knee, front_ankle) / 180.0
    features[31] = angle_from_points(back_hip, back_knee, back_ankle) / 180.0
    features[32] = angle_from_points(front_shoulder, front_hip, front_knee) / 180.0
    features[33] = angle_from_points(back_shoulder, back_hip, back_knee) / 180.0

    torso_vector = shoulder_midpoint - hip_midpoint
    vertical_up = np.array([0.0, -1.0, 0.0], dtype=np.float32)
    features[34] = angle_to_direction(torso_vector, vertical_up) / 180.0

    features[35] = point_distance(front_ankle, back_ankle) / reference_scale
    features[36] = front_knee[0] - front_ankle[0]
    features[37] = back_knee[0] - back_ankle[0]

    ground_y = max(front_ankle[1], back_ankle[1])
    features[38] = ground_y - front_knee[1]
    features[39] = ground_y - back_knee[1]
    features[40] = angle_from_points(ear_midpoint, shoulder_midpoint, hip_midpoint) / (
        180.0
    )
    features[41] = (ground_y - hip_midpoint[1]) / reference_scale

    return features


def extract_bottom_features(landmarks: Any) -> np.ndarray:
    if isinstance(landmarks, np.ndarray) and landmarks.ndim == 3:
        return map_xyz_sequence_features(landmarks, bottom_frame_features)
    return bottom_frame_features(as_xyz_frame(landmarks))


def extract_phase_features(
    landmarks: Any,
    previous_heights: PhaseHeights | None,
) -> tuple[np.ndarray, PhaseHeights]:
    frame_xy = as_xy_frame(landmarks)

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

    features = np.array(
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
    return features, current_heights


class LandmarkSmoother:
    def __init__(self, alpha: float = 0.6) -> None:
        self.alpha = alpha
        self.previous_landmarks: np.ndarray | None = None

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
