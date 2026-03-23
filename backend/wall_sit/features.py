"""Shared wall-sit feature extraction helpers.

Used by both training and streaming so wall-sit feature computation stays
consistent across offline and realtime paths.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from shared.math_utils import (
    angle_from_points,
    landmarks_to_xyz,
    point_distance,
    position_normalize,
)


LEFT_SHOULDER_INDEX = 11
RIGHT_SHOULDER_INDEX = 12
LEFT_HIP_INDEX = 23
RIGHT_HIP_INDEX = 24
LEFT_KNEE_INDEX = 25
RIGHT_KNEE_INDEX = 26
LEFT_ANKLE_INDEX = 27
RIGHT_ANKLE_INDEX = 28


def extract_frame_features(landmarks: list, side: str) -> Tuple[float, float, float]:
    """Extract normalized wall-sit features from one landmark frame.

    All selected joints are first normalized in xyz space relative to the
    chosen-side hip using a shared reference scale. The resulting
    features are:
        - foot_wall_norm: horizontal ankle offset from hip
        - knee_angle_norm: hip-knee-ankle angle divided by 180
        - torso_alignment: horizontal shoulder offset from hip
    """
    landmark_xyz = landmarks_to_xyz(landmarks)

    if side == "right":
        hip_index = RIGHT_HIP_INDEX
        knee_index = RIGHT_KNEE_INDEX
        ankle_index = RIGHT_ANKLE_INDEX
        shoulder_index = RIGHT_SHOULDER_INDEX
    else:
        hip_index = LEFT_HIP_INDEX
        knee_index = LEFT_KNEE_INDEX
        ankle_index = LEFT_ANKLE_INDEX
        shoulder_index = LEFT_SHOULDER_INDEX

    selected_hip = landmark_xyz[hip_index]
    selected_knee = landmark_xyz[knee_index]
    selected_ankle = landmark_xyz[ankle_index]
    selected_shoulder = landmark_xyz[shoulder_index]

    shoulder_width = point_distance(
        landmark_xyz[LEFT_SHOULDER_INDEX],
        landmark_xyz[RIGHT_SHOULDER_INDEX],
    )
    hip_width = point_distance(
        landmark_xyz[LEFT_HIP_INDEX],
        landmark_xyz[RIGHT_HIP_INDEX],
    )
    reference_scale = (
        hip_width if hip_width > 1e-4
        else (shoulder_width if shoulder_width > 1e-4 else 1e-6)
    )

    normalized_hip = position_normalize(
        selected_hip,
        center=selected_hip,
        scale=reference_scale,
    )
    normalized_knee = position_normalize(
        selected_knee,
        center=selected_hip,
        scale=reference_scale,
    )
    normalized_ankle = position_normalize(
        selected_ankle,
        center=selected_hip,
        scale=reference_scale,
    )
    normalized_shoulder = position_normalize(
        selected_shoulder,
        center=selected_hip,
        scale=reference_scale,
    )

    foot_wall_norm = abs(normalized_ankle[0] - normalized_hip[0])
    knee_angle_norm = (
        angle_from_points(normalized_hip, normalized_knee, normalized_ankle) / 180.0
    )
    torso_alignment = abs(normalized_shoulder[0] - normalized_hip[0])

    return float(foot_wall_norm), float(knee_angle_norm), float(torso_alignment)


def aggregate_window_features(
    frame_values: List[Tuple[float, float, float]],
) -> np.ndarray:
    """Aggregate per-frame tuples into the model's 5-D wall-sit feature vector.

    Feature order:
        [0] mean foot-wall distance
        [1] std foot-wall distance
        [2] mean knee angle
        [3] min knee angle
        [4] mean torso alignment
    """
    foot_wall_values = [frame_value[0] for frame_value in frame_values]
    knee_angle_values = [frame_value[1] for frame_value in frame_values]
    torso_alignment_values = [frame_value[2] for frame_value in frame_values]

    return np.array(
        [
            np.mean(foot_wall_values),
            np.std(foot_wall_values),
            np.mean(knee_angle_values),
            np.min(knee_angle_values),
            np.mean(torso_alignment_values),
        ],
        dtype=np.float32,
    )


# ---------------------------------------------------------------
# Feature extractor (streaming adapter)
# ---------------------------------------------------------------

from dataclasses import dataclass, field
from typing import Any, Optional


class WallSitFeatureExtractor:
    """Adapter that wraps wall-sit feature helpers for use in the streaming session."""

    def __init__(self, mp_pose: Any) -> None:
        self.mp_pose = mp_pose

    def extract_features(
        self,
        pose_result: Any,
        side: str,
    ) -> Optional[Tuple[float, float, float]]:
        """Extract a single-frame wall-sit feature tuple for the given side."""
        if not pose_result.pose_landmarks:
            return None
        return extract_frame_features(pose_result.pose_landmarks.landmark, side)

    @staticmethod
    def aggregate_window(
        frame_features: List[Tuple[float, float, float]],
    ) -> np.ndarray:
        """Aggregate a window of per-frame tuples into a 5-D feature vector."""
        return aggregate_window_features(frame_features)


# ---------------------------------------------------------------
# Stream state
# ---------------------------------------------------------------

@dataclass
class StreamState:
    """Session-level state for a wall sit streaming WebSocket connection."""

    started: bool = False
    frame_features: List[Tuple[float, float, float]] = field(default_factory=list)
    stand_streak: int = 0

    ready: bool = False
    ready_streak: int = 0
    chosen_side: Optional[str] = None

    session_id: str = ""

    last_status: str = ""
    status_tick: int = 0

    last_prediction_label: str = ""
    last_prediction_confidence: Optional[float] = None

    last_sent_label: str = ""
    last_sent_confidence: Optional[float] = None

    frame_count: int = 0

    no_pose_since: Optional[float] = None
    no_pose_alerted: bool = False

    dark_since: Optional[float] = None
    dark_alerted: bool = False


# Backward-compatible alias for older imports.
aggregate_window = aggregate_window_features
