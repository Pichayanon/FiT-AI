"""Shared squat feature extraction functions.

Single source of truth for both training scripts and realtime streaming.
Each extractor accepts either:
    - a single MediaPipe landmark list
    - a single numpy landmark frame shaped like (33, 3/4)
    - a numpy sequence shaped like (T, 33, 3/4)
"""

from __future__ import annotations

from typing import Any

import numpy as np

from shared.math_utils import angle_3pts, dist, get_xyz, safe_norm


# ---------------------------------------------------------------
# Constants (shared between training and streaming)
# ---------------------------------------------------------------

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

# Backward-compatible aliases used by other squat modules.
KEY_JOINTS = KEY_JOINT_INDICES
L_SHO, R_SHO = LEFT_SHOULDER_INDEX, RIGHT_SHOULDER_INDEX
L_HIP, R_HIP = LEFT_HIP_INDEX, RIGHT_HIP_INDEX
L_KNE, R_KNE = LEFT_KNEE_INDEX, RIGHT_KNEE_INDEX
L_ANK, R_ANK = LEFT_ANKLE_INDEX, RIGHT_ANKLE_INDEX


# ---------------------------------------------------------------
# Shared landmark parsing
# ---------------------------------------------------------------

def _to_single_frame_xyz(landmarks: Any) -> np.ndarray:
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
    return get_xyz(landmarks)


def _to_landmark_sequence_xyz(landmark_sequence: np.ndarray) -> np.ndarray:
    """Return a sequence of xyz landmarks as shape (T, 33, 3)."""
    landmark_array = np.asarray(landmark_sequence, dtype=np.float32)
    if (
        landmark_array.ndim != 3
        or landmark_array.shape[1] != 33
        or landmark_array.shape[2] < 3
    ):
        raise ValueError("Expected landmark sequence with shape (T, 33, 3/4).")
    return landmark_array[..., :3].astype(np.float32)


# ---------------------------------------------------------------
# Phase features (10-D) — matches squat/extract_phase.py
# ---------------------------------------------------------------

def _extract_phase_features_from_single_frame_xyz(frame_xyz: np.ndarray) -> np.ndarray:
    """Extract 10-dim phase features from a single xyz landmark frame.

    Feature order:
        [0] left shoulder height
        [1] right shoulder height
        [2] left hip height
        [3] right hip height
        [4] left knee height
        [5] right knee height
        [6] left ankle height
        [7] right ankle height
        [8] left knee angle / 180
        [9] right knee angle / 180
    """
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

    left_knee_angle = angle_3pts(left_hip_xy, left_knee_xy, left_ankle_xy)
    right_knee_angle = angle_3pts(right_hip_xy, right_knee_xy, right_ankle_xy)

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
    """Extract 10-dim phase features from one frame or a sequence."""
    if isinstance(landmarks, np.ndarray) and landmarks.ndim == 3:
        landmark_sequence_xyz = _to_landmark_sequence_xyz(landmarks)
        return np.stack(
            [
                _extract_phase_features_from_single_frame_xyz(
                    landmark_sequence_xyz[frame_index]
                )
                for frame_index in range(landmark_sequence_xyz.shape[0])
            ],
            axis=0,
        )
    return _extract_phase_features_from_single_frame_xyz(
        _to_single_frame_xyz(landmarks)
    )


# ---------------------------------------------------------------
# Stand features (16-D) — matches squat/extract_standing_squat.py
# ---------------------------------------------------------------

def _extract_stand_features_from_single_frame_xyz(frame_xyz: np.ndarray) -> np.ndarray:
    """Extract 16-dim stand features from a single xyz landmark frame.

    Feature order:
        [0] ankle width / hip width
        [1] ankle width / shoulder width
        [2] knee width / hip width
        [3] knee width / shoulder width
        [4] left ankle x offset
        [5] right ankle x offset
        [6] left knee x offset
        [7] right knee x offset
        [8] left hip x offset
        [9] right hip x offset
        [10] left knee angle / 180
        [11] right knee angle / 180
        [12] torso tilt / 180
        [13] ankle width / scale
        [14] shoulder width / scale
        [15] ankle width / shoulder width
    """
    left_hip = frame_xyz[LEFT_HIP_INDEX]
    right_hip = frame_xyz[RIGHT_HIP_INDEX]
    left_shoulder = frame_xyz[LEFT_SHOULDER_INDEX]
    right_shoulder = frame_xyz[RIGHT_SHOULDER_INDEX]
    left_knee = frame_xyz[LEFT_KNEE_INDEX]
    right_knee = frame_xyz[RIGHT_KNEE_INDEX]
    left_ankle = frame_xyz[LEFT_ANKLE_INDEX]
    right_ankle = frame_xyz[RIGHT_ANKLE_INDEX]

    hip_midpoint = 0.5 * (left_hip + right_hip)
    hip_width = dist(left_hip, right_hip)
    shoulder_width = dist(left_shoulder, right_shoulder)
    ankle_width = dist(left_ankle, right_ankle)
    knee_width = dist(left_knee, right_knee)
    reference_scale = (
        hip_width if hip_width > 1e-4
        else (shoulder_width if shoulder_width > 1e-4 else 1.0)
    )

    feature_vector = np.zeros(STAND_FEATURE_DIM, dtype=np.float32)
    feature_vector[0] = ankle_width / (hip_width + 1e-6)
    feature_vector[1] = ankle_width / (shoulder_width + 1e-6)
    feature_vector[2] = knee_width / (hip_width + 1e-6)
    feature_vector[3] = knee_width / (shoulder_width + 1e-6)
    feature_vector[4] = (left_ankle[0] - hip_midpoint[0]) / (reference_scale + 1e-6)
    feature_vector[5] = (right_ankle[0] - hip_midpoint[0]) / (reference_scale + 1e-6)
    feature_vector[6] = (left_knee[0] - hip_midpoint[0]) / (reference_scale + 1e-6)
    feature_vector[7] = (right_knee[0] - hip_midpoint[0]) / (reference_scale + 1e-6)
    feature_vector[8] = (left_hip[0] - hip_midpoint[0]) / (reference_scale + 1e-6)
    feature_vector[9] = (right_hip[0] - hip_midpoint[0]) / (reference_scale + 1e-6)
    feature_vector[10] = angle_3pts(left_hip, left_knee, left_ankle) / 180.0
    feature_vector[11] = angle_3pts(right_hip, right_knee, right_ankle) / 180.0

    shoulder_midpoint = 0.5 * (left_shoulder + right_shoulder)
    torso_vector = (shoulder_midpoint - hip_midpoint).astype(np.float32)
    vertical_up_vector = np.array([0.0, 1.0, 0.0], dtype=np.float32)
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
    feature_vector[12] = float(np.degrees(np.arccos(torso_tilt_cosine))) / 180.0
    feature_vector[13] = ankle_width / (reference_scale + 1e-6)
    feature_vector[14] = shoulder_width / (reference_scale + 1e-6)
    feature_vector[15] = ankle_width / (shoulder_width + 1e-6)
    return feature_vector


def extract_stand_features(landmarks: Any) -> np.ndarray:
    """Extract 16-dim stand features from one frame or a sequence."""
    if isinstance(landmarks, np.ndarray) and landmarks.ndim == 3:
        landmark_sequence_xyz = _to_landmark_sequence_xyz(landmarks)
        return np.stack(
            [
                _extract_stand_features_from_single_frame_xyz(
                    landmark_sequence_xyz[frame_index]
                )
                for frame_index in range(landmark_sequence_xyz.shape[0])
            ],
            axis=0,
        )
    return _extract_stand_features_from_single_frame_xyz(
        _to_single_frame_xyz(landmarks)
    )


# ---------------------------------------------------------------
# Bottom features (41-D) — matches squat/extract_bottom_squat.py
# ---------------------------------------------------------------

def _extract_bottom_features_from_single_frame_xyz(frame_xyz: np.ndarray) -> np.ndarray:
    """Extract 41-dim bottom features from a single xyz landmark frame.

    Feature order:
        [0-29]  left/right ear, shoulder, hip, knee, ankle xyz
        [30]    left knee angle / 180
        [31]    right knee angle / 180
        [32]    left hip angle / 180
        [33]    right hip angle / 180
        [34]    torso tilt / 180
        [35]    neck tilt / 180
        [36]    spine angle / 180
        [37]    knee width / hip width
        [38]    knee width / ankle width
        [39]    ankle width / hip width
        [40]    shoulder width / hip width
    """
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
    hip_width = dist(left_hip, right_hip)
    shoulder_width = dist(left_shoulder, right_shoulder)
    reference_scale = (
        hip_width if hip_width > 1e-4
        else (shoulder_width if shoulder_width > 1e-4 else 1.0)
    )

    feature_vector = np.zeros(BOTTOM_FEATURE_DIM, dtype=np.float32)

    for joint_offset, joint_index in enumerate(KEY_JOINT_INDICES):
        start_index = joint_offset * 3
        end_index = start_index + 3
        feature_vector[start_index:end_index] = (
            frame_xyz[joint_index] - hip_midpoint
        ) / (reference_scale + 1e-6)

    feature_vector[30] = angle_3pts(left_hip, left_knee, left_ankle) / 180.0
    feature_vector[31] = angle_3pts(right_hip, right_knee, right_ankle) / 180.0
    feature_vector[32] = angle_3pts(left_shoulder, left_hip, left_knee) / 180.0
    feature_vector[33] = angle_3pts(right_shoulder, right_hip, right_knee) / 180.0

    shoulder_midpoint = 0.5 * (left_shoulder + right_shoulder)
    ear_midpoint = 0.5 * (left_ear + right_ear)
    vertical_up_vector = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    torso_vector = (shoulder_midpoint - hip_midpoint).astype(np.float32)
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

    neck_vector = (ear_midpoint - shoulder_midpoint).astype(np.float32)
    neck_tilt_denominator = (
        safe_norm(neck_vector) * safe_norm(vertical_up_vector)
    ) + 1e-6
    neck_tilt_cosine = float(
        np.clip(
            np.dot(neck_vector, vertical_up_vector) / neck_tilt_denominator,
            -1.0,
            1.0,
        )
    )
    feature_vector[35] = float(np.degrees(np.arccos(neck_tilt_cosine))) / 180.0

    feature_vector[36] = angle_3pts(ear_midpoint, shoulder_midpoint, hip_midpoint) / 180.0

    knee_width = dist(left_knee, right_knee)
    ankle_width = dist(left_ankle, right_ankle)
    feature_vector[37] = knee_width / (hip_width + 1e-6)
    feature_vector[38] = knee_width / (ankle_width + 1e-6)
    feature_vector[39] = ankle_width / (hip_width + 1e-6)
    feature_vector[40] = shoulder_width / (hip_width + 1e-6)

    return feature_vector


def extract_bottom_features(landmarks: Any) -> np.ndarray:
    """Extract 41-dim bottom features from one frame or a sequence."""
    if isinstance(landmarks, np.ndarray) and landmarks.ndim == 3:
        landmark_sequence_xyz = _to_landmark_sequence_xyz(landmarks)
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
        _to_single_frame_xyz(landmarks)
    )
