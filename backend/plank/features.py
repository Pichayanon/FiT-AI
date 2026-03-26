from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from shared.math_utils import (
    angle_from_points,
    as_xyz_frame,
    pick_scale,
    point_distance,
    position_normalize,
)

LEFT_SHOULDER_INDEX = 11
RIGHT_SHOULDER_INDEX = 12
LEFT_HIP_INDEX = 23
RIGHT_HIP_INDEX = 24
LEFT_ANKLE_INDEX = 27
RIGHT_ANKLE_INDEX = 28

FrameFeatures = tuple[float, float, float]


def side_indices(side: str) -> tuple[int, int, int]:
    if side == "right":
        return RIGHT_SHOULDER_INDEX, RIGHT_HIP_INDEX, RIGHT_ANKLE_INDEX
    return LEFT_SHOULDER_INDEX, LEFT_HIP_INDEX, LEFT_ANKLE_INDEX


def horizontal_angle(start_xy: np.ndarray, end_xy: np.ndarray) -> float:
    delta = np.asarray(end_xy, dtype=np.float32) - np.asarray(
        start_xy,
        dtype=np.float32,
    )
    angle_deg = abs(float(np.degrees(np.arctan2(delta[1], delta[0]))))
    if angle_deg > 90.0:
        angle_deg = 180.0 - angle_deg
    return angle_deg


def choose_visible_side(landmarks: list) -> str:
    right_hip_visibility = landmarks[RIGHT_HIP_INDEX].visibility
    left_hip_visibility = landmarks[LEFT_HIP_INDEX].visibility
    return "right" if right_hip_visibility > left_hip_visibility else "left"


def extract_posture_metrics(landmarks: list, side: str) -> dict[str, float]:
    frame_xyz = as_xyz_frame(landmarks)
    shoulder_index, hip_index, ankle_index = side_indices(side)

    shoulder = frame_xyz[shoulder_index]
    hip = frame_xyz[hip_index]
    ankle = frame_xyz[ankle_index]

    torso_angle = horizontal_angle(shoulder[:2], hip[:2])
    leg_angle = horizontal_angle(hip[:2], ankle[:2])
    body_axis_angle = horizontal_angle(shoulder[:2], ankle[:2])
    torso_dx = abs(float(shoulder[0] - hip[0]))
    torso_dy = abs(float(shoulder[1] - hip[1]))
    leg_dx = abs(float(ankle[0] - hip[0]))
    leg_dy = abs(float(ankle[1] - hip[1]))
    body_x_span = float(
        max(shoulder[0], hip[0], ankle[0]) - min(shoulder[0], hip[0], ankle[0])
    )
    body_y_span = float(
        max(shoulder[1], hip[1], ankle[1]) - min(shoulder[1], hip[1], ankle[1])
    )

    return {
        "torso_angle_deg": float(torso_angle),
        "leg_angle_deg": float(leg_angle),
        "body_axis_angle_deg": float(body_axis_angle),
        "torso_horizontal_ratio": torso_dx / (torso_dy + 1e-6),
        "leg_horizontal_ratio": leg_dx / (leg_dy + 1e-6),
        "body_horizontal_ratio": body_x_span / (body_y_span + 1e-6),
        "body_x_span": body_x_span,
        "body_y_span": body_y_span,
    }


def is_plank_ready_pose(
    landmarks: list,
    side: str,
    *,
    max_body_axis_angle_deg: float = 35.0,
    max_torso_angle_deg: float = 45.0,
    max_leg_angle_deg: float = 45.0,
    min_body_horizontal_ratio: float = 1.15,
    min_torso_horizontal_ratio: float = 0.75,
    min_leg_horizontal_ratio: float = 0.75,
) -> tuple[bool, dict[str, float]]:
    metrics = extract_posture_metrics(landmarks, side)
    is_ready = (
        metrics["body_axis_angle_deg"] <= max_body_axis_angle_deg
        and metrics["torso_angle_deg"] <= max_torso_angle_deg
        and metrics["leg_angle_deg"] <= max_leg_angle_deg
        and metrics["body_horizontal_ratio"] >= min_body_horizontal_ratio
        and metrics["torso_horizontal_ratio"] >= min_torso_horizontal_ratio
        and metrics["leg_horizontal_ratio"] >= min_leg_horizontal_ratio
    )
    metrics.update(
        {
            "plank_pose_ok": is_ready,
            "max_body_axis_angle_deg": float(max_body_axis_angle_deg),
            "max_torso_angle_deg": float(max_torso_angle_deg),
            "max_leg_angle_deg": float(max_leg_angle_deg),
            "min_body_horizontal_ratio": float(min_body_horizontal_ratio),
            "min_torso_horizontal_ratio": float(min_torso_horizontal_ratio),
            "min_leg_horizontal_ratio": float(min_leg_horizontal_ratio),
        }
    )
    return is_ready, metrics


def extract_frame_features(
    landmarks: list,
    side: str,
) -> FrameFeatures:
    frame_xyz = as_xyz_frame(landmarks)

    shoulder_index, hip_index, ankle_index = side_indices(side)

    shoulder = frame_xyz[shoulder_index]
    hip = frame_xyz[hip_index]
    ankle = frame_xyz[ankle_index]

    torso_length = point_distance(shoulder, hip)
    body_length = point_distance(shoulder, ankle)
    hip_width = point_distance(
        frame_xyz[LEFT_HIP_INDEX],
        frame_xyz[RIGHT_HIP_INDEX],
    )
    shoulder_width = point_distance(
        frame_xyz[LEFT_SHOULDER_INDEX],
        frame_xyz[RIGHT_SHOULDER_INDEX],
    )
    reference_scale = pick_scale(
        torso_length,
        body_length,
        hip_width,
        shoulder_width,
        fallback=1e-6,
    )

    normalized_shoulder = position_normalize(
        shoulder,
        center=hip,
        scale=reference_scale,
    )
    normalized_hip = position_normalize(
        hip,
        center=hip,
        scale=reference_scale,
    )
    normalized_ankle = position_normalize(
        ankle,
        center=hip,
        scale=reference_scale,
    )

    normalized_shoulder_xy = normalized_shoulder[:2]
    normalized_hip_xy = normalized_hip[:2]
    normalized_ankle_xy = normalized_ankle[:2]

    shoulder_to_ankle = normalized_ankle_xy - normalized_shoulder_xy
    shoulder_to_hip = normalized_hip_xy - normalized_shoulder_xy
    hip_signed_dist = (
        shoulder_to_ankle[0] * shoulder_to_hip[1]
        - shoulder_to_ankle[1] * shoulder_to_hip[0]
    ) / (np.linalg.norm(shoulder_to_ankle) + 1e-6)

    body_angle = angle_from_points(
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


def _exp_weights(n: int, decay: float = 0.15) -> np.ndarray:
    """Exponential weights that sum to 1, biased toward the most recent frame."""
    w = np.exp(decay * np.arange(n))
    return w / w.sum()


def aggregate_window_features(
    frame_values: list[FrameFeatures],
) -> np.ndarray:
    hip_signed_distances = [hip_signed_dist for hip_signed_dist, _, _ in frame_values]
    hip_height_offsets = [hip_height for _, hip_height, _ in frame_values]
    body_angles = [body_angle for _, _, body_angle in frame_values]

    w = _exp_weights(len(hip_signed_distances))

    return np.array(
        [
            np.average(hip_signed_distances, weights=w),
            np.std(hip_signed_distances),
            np.average(hip_height_offsets, weights=w),
            np.std(hip_height_offsets),
            np.average(body_angles, weights=w),
            np.std(body_angles),
        ],
        dtype=np.float32,
    )


class PlankFeatureExtractor:
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


FeatureExtractor = PlankFeatureExtractor


@dataclass
class StreamState:
    started: bool = False

    frame_features: list[FrameFeatures] = field(default_factory=list)

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
