from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from shared.math_utils import (
    angle_from_points,
    as_xy_frame,
)

LEFT_SHOULDER_INDEX = 11
RIGHT_SHOULDER_INDEX = 12
LEFT_HIP_INDEX = 23
RIGHT_HIP_INDEX = 24
LEFT_KNEE_INDEX = 25
RIGHT_KNEE_INDEX = 26
LEFT_ANKLE_INDEX = 27
RIGHT_ANKLE_INDEX = 28

FrameFeatures = tuple[float, float, float]


def side_indices(side: str) -> tuple[int, int, int, int]:
    if side == "right":
        return (
            RIGHT_SHOULDER_INDEX,
            RIGHT_HIP_INDEX,
            RIGHT_KNEE_INDEX,
            RIGHT_ANKLE_INDEX,
        )
    return (
        LEFT_SHOULDER_INDEX,
        LEFT_HIP_INDEX,
        LEFT_KNEE_INDEX,
        LEFT_ANKLE_INDEX,
    )


def extract_frame_features(landmarks: list, side: str) -> FrameFeatures:
    frame_xy = as_xy_frame(landmarks)
    shoulder_index, hip_index, knee_index, ankle_index = side_indices(side)

    hip = frame_xy[hip_index]
    knee = frame_xy[knee_index]
    ankle = frame_xy[ankle_index]
    shoulder = frame_xy[shoulder_index]

    shoulder_width = abs(
        frame_xy[LEFT_SHOULDER_INDEX][0] - frame_xy[RIGHT_SHOULDER_INDEX][0]
    )
    foot_wall_dist = abs(ankle[0] - hip[0])
    foot_wall_norm = foot_wall_dist / (shoulder_width + 1e-6)
    knee_angle = angle_from_points(hip, knee, ankle)
    torso_alignment = abs(shoulder[0] - hip[0])

    return float(foot_wall_norm), float(knee_angle), float(torso_alignment)


def aggregate_window_features(
    frame_values: list[FrameFeatures],
) -> np.ndarray:
    foot_wall_values = [foot_wall for foot_wall, _, _ in frame_values]
    knee_angle_values = [knee_angle for _, knee_angle, _ in frame_values]
    torso_alignment_values = [torso_align for _, _, torso_align in frame_values]

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


class WallSitFeatureExtractor:
    def __init__(self, mp_pose: Any) -> None:
        self.mp_pose = mp_pose

    def extract_features(
        self,
        pose_result: Any,
        side: str,
    ) -> Optional[FrameFeatures]:
        if not pose_result.pose_landmarks:
            return None
        return extract_frame_features(pose_result.pose_landmarks.landmark, side)

    @staticmethod
    def aggregate_window(
        frame_features: list[FrameFeatures],
    ) -> np.ndarray:
        return aggregate_window_features(frame_features)


@dataclass
class StreamState:
    started: bool = False
    frame_features: list[FrameFeatures] = field(default_factory=list)
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


aggregate_window = aggregate_window_features
