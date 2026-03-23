from __future__ import annotations

from typing import Any, Callable

import numpy as np

from shared.math_utils import (
    angle_from_points,
    angle_to_direction,
    as_xyz_frame,
    as_xyz_sequence,
    pick_scale,
    point_distance,
)

PHASE_FEATURE_DIM = 10
STAND_FEATURE_DIM = 16
BOTTOM_FEATURE_DIM = 41

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

KEY_JOINT_INDICES = [
    LEFT_EAR_INDEX,
    RIGHT_EAR_INDEX,
    LEFT_SHOULDER_INDEX,
    RIGHT_SHOULDER_INDEX,
    LEFT_HIP_INDEX,
    RIGHT_HIP_INDEX,
    LEFT_KNEE_INDEX,
    RIGHT_KNEE_INDEX,
    LEFT_ANKLE_INDEX,
    RIGHT_ANKLE_INDEX,
]

KEY_JOINTS = KEY_JOINT_INDICES
L_SHO, R_SHO = LEFT_SHOULDER_INDEX, RIGHT_SHOULDER_INDEX
L_HIP, R_HIP = LEFT_HIP_INDEX, RIGHT_HIP_INDEX
L_KNE, R_KNE = LEFT_KNEE_INDEX, RIGHT_KNEE_INDEX
L_ANK, R_ANK = LEFT_ANKLE_INDEX, RIGHT_ANKLE_INDEX


def map_xyz_sequence_features(
    landmarks: np.ndarray,
    frame_extractor: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    frame_sequence_xyz = as_xyz_sequence(landmarks)
    return np.stack(
        [frame_extractor(frame_xyz) for frame_xyz in frame_sequence_xyz],
        axis=0,
    )


def phase_frame_features(frame_xyz: np.ndarray) -> np.ndarray:
    left_shoulder_xy = frame_xyz[LEFT_SHOULDER_INDEX, :2]
    right_shoulder_xy = frame_xyz[RIGHT_SHOULDER_INDEX, :2]
    left_hip_xy = frame_xyz[LEFT_HIP_INDEX, :2]
    right_hip_xy = frame_xyz[RIGHT_HIP_INDEX, :2]
    left_knee_xy = frame_xyz[LEFT_KNEE_INDEX, :2]
    right_knee_xy = frame_xyz[RIGHT_KNEE_INDEX, :2]
    left_ankle_xy = frame_xyz[LEFT_ANKLE_INDEX, :2]
    right_ankle_xy = frame_xyz[RIGHT_ANKLE_INDEX, :2]

    hip_midpoint_y = (left_hip_xy[1] + right_hip_xy[1]) / 2.0
    shoulder_midpoint_y = (left_shoulder_xy[1] + right_shoulder_xy[1]) / 2.0
    torso_length = abs(shoulder_midpoint_y - hip_midpoint_y) + 1e-6

    def normalize_vertical(point_xy: np.ndarray) -> float:
        return float((point_xy[1] - hip_midpoint_y) / torso_length)

    left_knee_angle = angle_from_points(left_hip_xy, left_knee_xy, left_ankle_xy)
    right_knee_angle = angle_from_points(
        right_hip_xy,
        right_knee_xy,
        right_ankle_xy,
    )

    return np.array(
        [
            normalize_vertical(left_shoulder_xy),
            normalize_vertical(right_shoulder_xy),
            normalize_vertical(left_hip_xy),
            normalize_vertical(right_hip_xy),
            normalize_vertical(left_knee_xy),
            normalize_vertical(right_knee_xy),
            normalize_vertical(left_ankle_xy),
            normalize_vertical(right_ankle_xy),
            left_knee_angle / 180.0,
            right_knee_angle / 180.0,
        ],
        dtype=np.float32,
    )


def extract_phase_features(landmarks: Any) -> np.ndarray:
    if isinstance(landmarks, np.ndarray) and landmarks.ndim == 3:
        return map_xyz_sequence_features(landmarks, phase_frame_features)
    return phase_frame_features(as_xyz_frame(landmarks))


def stand_frame_features(frame_xyz: np.ndarray) -> np.ndarray:
    left_hip = frame_xyz[LEFT_HIP_INDEX]
    right_hip = frame_xyz[RIGHT_HIP_INDEX]
    left_shoulder = frame_xyz[LEFT_SHOULDER_INDEX]
    right_shoulder = frame_xyz[RIGHT_SHOULDER_INDEX]
    left_knee = frame_xyz[LEFT_KNEE_INDEX]
    right_knee = frame_xyz[RIGHT_KNEE_INDEX]
    left_ankle = frame_xyz[LEFT_ANKLE_INDEX]
    right_ankle = frame_xyz[RIGHT_ANKLE_INDEX]

    hip_midpoint = 0.5 * (left_hip + right_hip)
    hip_width = point_distance(left_hip, right_hip)
    shoulder_width = point_distance(left_shoulder, right_shoulder)
    ankle_width = point_distance(left_ankle, right_ankle)
    knee_width = point_distance(left_knee, right_knee)
    reference_scale = pick_scale(hip_width, shoulder_width)

    features = np.zeros(STAND_FEATURE_DIM, dtype=np.float32)
    features[0] = ankle_width / (hip_width + 1e-6)
    features[1] = ankle_width / (shoulder_width + 1e-6)
    features[2] = knee_width / (hip_width + 1e-6)
    features[3] = knee_width / (shoulder_width + 1e-6)
    features[4] = (left_ankle[0] - hip_midpoint[0]) / (reference_scale + 1e-6)
    features[5] = (right_ankle[0] - hip_midpoint[0]) / (reference_scale + 1e-6)
    features[6] = (left_knee[0] - hip_midpoint[0]) / (reference_scale + 1e-6)
    features[7] = (right_knee[0] - hip_midpoint[0]) / (reference_scale + 1e-6)
    features[8] = (left_hip[0] - hip_midpoint[0]) / (reference_scale + 1e-6)
    features[9] = (right_hip[0] - hip_midpoint[0]) / (reference_scale + 1e-6)
    features[10] = angle_from_points(left_hip, left_knee, left_ankle) / 180.0
    features[11] = angle_from_points(right_hip, right_knee, right_ankle) / 180.0

    shoulder_midpoint = 0.5 * (left_shoulder + right_shoulder)
    torso_vector = (shoulder_midpoint - hip_midpoint).astype(np.float32)
    vertical_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    features[12] = angle_to_direction(torso_vector, vertical_up) / 180.0
    features[13] = ankle_width / (reference_scale + 1e-6)
    features[14] = shoulder_width / (reference_scale + 1e-6)
    features[15] = ankle_width / (shoulder_width + 1e-6)
    return features


def extract_stand_features(landmarks: Any) -> np.ndarray:
    if isinstance(landmarks, np.ndarray) and landmarks.ndim == 3:
        return map_xyz_sequence_features(landmarks, stand_frame_features)
    return stand_frame_features(as_xyz_frame(landmarks))


def bottom_frame_features(frame_xyz: np.ndarray) -> np.ndarray:
    left_ear = frame_xyz[LEFT_EAR_INDEX]
    right_ear = frame_xyz[RIGHT_EAR_INDEX]
    left_hip = frame_xyz[LEFT_HIP_INDEX]
    right_hip = frame_xyz[RIGHT_HIP_INDEX]
    left_shoulder = frame_xyz[LEFT_SHOULDER_INDEX]
    right_shoulder = frame_xyz[RIGHT_SHOULDER_INDEX]
    left_knee = frame_xyz[LEFT_KNEE_INDEX]
    right_knee = frame_xyz[RIGHT_KNEE_INDEX]
    left_ankle = frame_xyz[LEFT_ANKLE_INDEX]
    right_ankle = frame_xyz[RIGHT_ANKLE_INDEX]

    hip_midpoint = 0.5 * (left_hip + right_hip)
    hip_width = point_distance(left_hip, right_hip)
    shoulder_width = point_distance(left_shoulder, right_shoulder)
    reference_scale = pick_scale(hip_width, shoulder_width)

    features = np.zeros(BOTTOM_FEATURE_DIM, dtype=np.float32)

    for joint_offset, joint_index in enumerate(KEY_JOINT_INDICES):
        start_index = joint_offset * 3
        end_index = start_index + 3
        features[start_index:end_index] = (frame_xyz[joint_index] - hip_midpoint) / (
            reference_scale + 1e-6
        )

    features[30] = angle_from_points(left_hip, left_knee, left_ankle) / 180.0
    features[31] = angle_from_points(right_hip, right_knee, right_ankle) / 180.0
    features[32] = angle_from_points(left_shoulder, left_hip, left_knee) / 180.0
    features[33] = angle_from_points(right_shoulder, right_hip, right_knee) / 180.0

    shoulder_midpoint = 0.5 * (left_shoulder + right_shoulder)
    ear_midpoint = 0.5 * (left_ear + right_ear)
    vertical_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    torso_vector = (shoulder_midpoint - hip_midpoint).astype(np.float32)
    features[34] = angle_to_direction(torso_vector, vertical_up) / 180.0

    neck_vector = (ear_midpoint - shoulder_midpoint).astype(np.float32)
    features[35] = angle_to_direction(neck_vector, vertical_up) / 180.0

    features[36] = angle_from_points(ear_midpoint, shoulder_midpoint, hip_midpoint) / (
        180.0
    )

    knee_width = point_distance(left_knee, right_knee)
    ankle_width = point_distance(left_ankle, right_ankle)
    features[37] = knee_width / (hip_width + 1e-6)
    features[38] = knee_width / (ankle_width + 1e-6)
    features[39] = ankle_width / (hip_width + 1e-6)
    features[40] = shoulder_width / (hip_width + 1e-6)

    return features


def extract_bottom_features(landmarks: Any) -> np.ndarray:
    if isinstance(landmarks, np.ndarray) and landmarks.ndim == 3:
        return map_xyz_sequence_features(landmarks, bottom_frame_features)
    return bottom_frame_features(as_xyz_frame(landmarks))
